"""Command-line research pipeline. `--help` lists supported entry points."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from backtesting.engine import risk_managed_backtest
from backtesting.rolling_walk_forward_backtest import run_rolling_walk_forward_backtest
from data.processor import DataProcessor
from data.sample_generator import make_synthetic_ohlcv
from features.feature_engineer import create_features, get_default_feature_cols
from features.label_generator import add_binary_label
from features.preprocess import build_sequences, fit_scaler, transform_features, save_scaler_and_features
from models.lstm import LSTMClassifier
from models.hybrid import TransformerLSTMClassifier
from models.cnn import CNN1DClassifier
from models.mlp import MLPClassifier
from training.cv import expanding_window_split
from training.trainer import train_one_fold, save_model

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ["lstm", "hybrid", "cnn", "mlp"]


def build_model(model_name: str, input_size: int, seq_len: int, params: dict | None = None):
    p = params or {}
    if model_name == "lstm":
        return LSTMClassifier(input_size, hidden_size=p.get("hidden_size", 64),
                              num_layers=p.get("num_layers", 2), dropout=p.get("dropout", .2))
    if model_name == "hybrid":
        return TransformerLSTMClassifier(input_size,
                   d_model=p.get("d_model",64), nhead=p.get("nhead",4),
                   num_transformer_layers=p.get("num_transformer_layers",2),
                   lstm_hidden_size=p.get("lstm_hidden_size",64),
                   lstm_num_layers=p.get("lstm_num_layers",1), dropout=p.get("dropout",.2))
    if model_name == "cnn":
        return CNN1DClassifier(input_size, num_channels=p.get("num_channels",64),
                               dropout=p.get("dropout", .2))
    if model_name == "mlp":
        return MLPClassifier(input_size, seq_len,
                             hidden_size=p.get("hidden_size",128), dropout=p.get("dropout",.2))
    raise ValueError(f"Unknown model: {model_name}")


def load_data(args):
    if args.demo:
        print("SYNTHETIC DEMO DATA — NOT REAL MARKET PERFORMANCE")
        raw = make_synthetic_ohlcv(args.demo_bars, args.seed)
    elif args.data_csv:
        raw = pd.read_csv(args.data_csv)
    else:
        from data.loader import TushareLoader
        raw = TushareLoader().load_daily(args.ticker, args.start, args.end)
    df = DataProcessor().standardize_ohlcv(raw)
    df = add_binary_label(create_features(df), horizon=1)
    cols = get_default_feature_cols()
    df = df.dropna(subset=cols+["label"]).reset_index(drop=True)
    if len(df) < max(120, args.window+60):
        raise ValueError("Insufficient cleaned observations; please provide >= 120 samples")
    return df, cols


def write_csv(frame, output, name):
    frame.to_csv(output / name, index=False, encoding="utf-8")


def write_json(value, output, name):
    (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def run_walk_forward(df, cols, args, model_name=None):
    model_name = model_name or args.model
    preds, folds, bt, stats = run_rolling_walk_forward_backtest(
        df, cols, model_name, build_model, window=args.window,
        train_ratio=args.train_ratio, test_window=args.test_bars,
        epochs=args.epochs, patience=args.patience, batch_size=args.batch_size,
        device=args.device, commission=args.commission,
        slippage=args.slippage, stop_loss_pct=args.stop_loss)
    return preds, folds, bt, stats


def save_walk_forward(preds, folds, bt, stats, output, name):
    write_csv(preds, output, f"{name}_walk_forward_predictions.csv")
    write_csv(folds, output, f"{name}_walk_forward_folds.csv")
    write_csv(bt, output, f"{name}_walk_forward_backtest.csv")
    write_json(stats, output, f"{name}_walk_forward_metrics.json")
    signal = preds[["datetime", "prob"]].copy()
    signal["signal"] = np.select([signal.prob >= .55, signal.prob <= .45], [1, -1], default=0)
    write_csv(signal, output, f"{name}_signals.csv")


def _cv_arrays(raw_X, raw_y, idx_train, idx_val):
    """Scaler fit on training sample time steps only (never validation)."""
    scaler = fit_scaler(pd.DataFrame(raw_X[idx_train].reshape(-1, raw_X.shape[-1])),
                        list(range(raw_X.shape[-1])))
    tr_shape, va_shape = raw_X[idx_train].shape, raw_X[idx_val].shape
    Xt = scaler.transform(raw_X[idx_train].reshape(-1, raw_X.shape[-1])).reshape(tr_shape).astype("float32")
    Xv = scaler.transform(raw_X[idx_val].reshape(-1, raw_X.shape[-1])).reshape(va_shape).astype("float32")
    return Xt, raw_y[idx_train], Xv, raw_y[idx_val], scaler


def run_development_cv(df, cols, args, output):
    """Development-only purged CV. NO backtest or claim of OOS returns."""
    X, y, _ = build_sequences(df, cols, args.window)
    idx_splits = list(expanding_window_split(len(X), n_splits=3, purge=1))
    if not idx_splits:
        raise ValueError("Not enough data for purged CV")
    fold_data = [_cv_arrays(X,y,tr,va) for tr,va in idx_splits]
    params = {}
    if args.optuna_trials:
        from training.optuna_tuner import run_optuna_study
        params, study = run_optuna_study(args.model, build_model, X, y,
                input_size=len(cols), seq_len=args.window, n_trials=args.optuna_trials,
                n_splits=3, epochs=args.epochs, patience=args.patience,
                device=args.device, seed=args.seed,
                fold_data=[(a,b,c,d) for a,b,c,d,_ in fold_data])
        write_json(params, output, f"{args.model}_optuna_best_params.json")
        write_csv(study.trials_dataframe(), output, f"{args.model}_optuna_trials.csv")
    metrics = []
    best = None
    best_scaler = None
    best_val = -1
    best_explain = None
    for fold, (Xt,yt,Xv,yv,scaler) in enumerate(fold_data,1):
        torch.manual_seed(args.seed + fold)
        model = build_model(args.model, len(cols), args.window, params)
        model, history = train_one_fold(model, Xt, yt, Xv, yv,
            lr=params.get("lr",.001), weight_decay=params.get("weight_decay",.0001),
            batch_size=params.get("batch_size",args.batch_size), epochs=args.epochs,
            patience=args.patience, device=args.device)
        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(Xv, device=args.device)).cpu().numpy()
        try: auc = float(roc_auc_score(yv,probs))
        except ValueError: auc = .5
        metrics.append({"fold":fold, "validation_auc":auc, "epochs_run":len(history["val_auc"]),
                        "train_n":len(Xt), "val_n":len(Xv)})
        write_csv(pd.DataFrame(history), output, f"{args.model}_fold{fold}_history.csv")
        if auc > best_val:
            best, best_val, best_scaler = model, auc, scaler
            best_explain = (Xv[:min(160,len(Xv))], yv[:min(160,len(yv))])
    write_json({"status":"development_cv_only", "folds":metrics,
                "note":"Best fold chosen using validation AUC: NOT OOS PnL."},
                output, f"{args.model}_cv_results.json")
    save_model(best, output, filename=f"{args.model}_model.pth")
    save_scaler_and_features(best_scaler, cols, output, prefix=f"{args.model}_")
    if args.explain:
        from explainability.importance import (gradient_importance, permutation_importance,
                                                shap_importance, plot_feature_importance,
                                                feature_stability)
        val_x, val_y = best_explain
        grad = gradient_importance(best, val_x, cols, args.device,
                          output/f"{args.model}_gradient_importance.csv")
        perm = permutation_importance(best, val_x, val_y, cols, args.device,
                          n_repeats=2, save_path=output/f"{args.model}_permutation_importance.csv")
        plot_feature_importance(grad, "gradient_importance", "Gradient attribution",
                                save_path=output/f"{args.model}_gradient_importance.png")
        plot_feature_importance(perm, "importance_mean", "Permutation importance",
                                save_path=output/f"{args.model}_permutation_importance.png")
        sources = {"gradient": grad.rename(columns={"gradient_importance":"importance"}),
                   "permutation": perm.rename(columns={"importance_mean":"importance"})}
        if args.shap:
            try:
                shap_df = shap_importance(best, val_x[:min(30,len(val_x))],
                            val_x[:min(40,len(val_x))], cols, args.device,
                            output/f"{args.model}_shap_importance.csv")
                sources["shap"] = shap_df.rename(columns={"shap_importance":"importance"})
                plot_feature_importance(shap_df,"shap_importance","SHAP importance",
                                        save_path=output/f"{args.model}_shap_importance.png")
            except Exception as exc:
                print("Optional SHAP attribution unavailable:", exc)
        feature_stability(sources,"importance",save_path=output/f"{args.model}_feature_stability.csv")
    print("Development CV AUCs:", [round(v['validation_auc'],4) for v in metrics])
    return metrics


def run_ensemble(df, cols, args, output):
    """Equal-weight average of four independently generated OOS fold predictions."""
    per_model = []
    for model_name in MODEL_NAMES:
        preds, folds, bt, stats = run_walk_forward(df,cols,args,model_name)
        save_walk_forward(preds,folds,bt,stats,output,model_name)
        per_model.append(preds[["datetime","close","prob"]].rename(columns={"prob":model_name}))
    merged = per_model[0]
    for frame in per_model[1:]:
        merged = merged.merge(frame[["datetime", frame.columns[-1]]], on="datetime", validate="one_to_one")
    merged["prob"] = merged[MODEL_NAMES].mean(axis=1)
    bt, stats = risk_managed_backtest(merged.close,merged.prob,
                                     commission=args.commission, slippage=args.slippage,
                                     stop_loss_pct=args.stop_loss)
    bt.insert(0,"datetime",merged.datetime.to_numpy())
    write_csv(merged,output,"ensemble_oos_predictions.csv")
    write_csv(bt,output,"ensemble_walk_forward_backtest.csv")
    write_json({"method":"equal_weight_OOS_pred_average", **stats},output,"ensemble_walk_forward_metrics.json")
    print("Ensemble OOS (idealized research assumptions):",stats)


def make_parser():
    p = argparse.ArgumentParser(description="A-share deep learning research — data, model, purged CV and WF backtesting")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--demo",action="store_true",help="Offline synthetic-data smoke run, NOT real performance")
    source.add_argument("--data-csv",type=Path,help="Locally provided OHLCV or Tushare daily CSV")
    p.add_argument("--source",choices=["tushare"], default="tushare",help="Default data provider if no demo / CSV")
    p.add_argument("--ticker",default="688981.SH")
    p.add_argument("--start",default="20220101")
    p.add_argument("--end",default="20251231")
    p.add_argument("--demo-bars",type=int,default=520)
    p.add_argument("--mode",choices=["walk_forward","full","ensemble"], default="walk_forward")
    p.add_argument("--model",choices=MODEL_NAMES,default="lstm")
    p.add_argument("--window",type=int,default=30)
    p.add_argument("--train-ratio",type=float,default=.6)
    p.add_argument("--test-bars",type=int,default=60)
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--patience",type=int,default=5)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--optuna-trials",type=int,default=0,help="Only applies in development-only full CV mode")
    p.add_argument("--explain", action="store_true",help="Gradient and permutation importance for full CV")
    p.add_argument("--shap", action="store_true",help="Optional SHAP, requires --explain (and shap)")
    p.add_argument("--commission",type=float,default=.001)
    p.add_argument("--slippage",type=float,default=.0005)
    p.add_argument("--stop-loss",type=float,default=.03)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--output",type=Path,default=ROOT/"artifacts")
    return p


def main():
    args = make_parser().parse_args()
    if args.device == "auto": args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device=="cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available; use --device cpu")
    if args.shap and not args.explain:
        raise ValueError("--shap requires --explain")
    if args.epochs < 1 or args.batch_size < 1 or args.window < 1:
        raise ValueError("epochs, batch-size and window must all be positive")
    torch.set_num_threads(min(torch.get_num_threads(),4))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output = args.output.expanduser().resolve();output.mkdir(parents=True,exist_ok=True)
    df, cols = load_data(args)
    write_json({"data_source": "synthetic demo" if args.demo else ("user CSV" if args.data_csv else "Tushare daily"),
                "model":args.model,"mode":args.mode,"num_bars_after_cleaning":len(df),
                "features":cols,"label":"close(t+1)>close(t)",
                "lookback_bars":args.window, "seed":args.seed, "test_bars":args.test_bars,
                "train_ratio":args.train_ratio, "epochs":args.epochs,
                "commission":args.commission, "slippage":args.slippage,
                "stop_loss":args.stop_loss, "model_frequency":"daily",
                "caveat":"Idealized close-to-close fill; not an executable A-share trading strategy"},
                output,"run_metadata.json")
    if args.mode=="full":
        run_development_cv(df,cols,args,output)
    elif args.mode=="ensemble":
        run_ensemble(df,cols,args,output)
    else:
        preds,folds,bt,stats=run_walk_forward(df,cols,args)
        save_walk_forward(preds,folds,bt,stats,output,args.model)
        print("Walk-forward research metrics:",stats)
    print("Artifacts:",output)

if __name__=="__main__": main()
