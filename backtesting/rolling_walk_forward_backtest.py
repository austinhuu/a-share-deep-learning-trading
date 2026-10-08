"""Leakage-aware expanding-window forecasts with one-bar label purge.

Data sample at t includes indicators through bar t; target is t+1 direction.
Past labels that would end inside the validation/test segment are excluded.
The scaler is fit exclusively using the inner-training segment of each fold.
"""
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from features.preprocess import fit_scaler, transform_features, build_sequences
from training.trainer import train_one_fold
from backtesting.engine import risk_managed_backtest


def _auc(labels, probs):
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:
        return 0.5


def make_fold_arrays(clean_df, features, window, val_start, train_end, test_end, purge=1):
    """Return scaled arrays and sample date metadata for ONE walk-forward fold.

    Row t represents the features known at close(t); label horizon is `purge`.
    Validation begins at val_start, testing at train_end.
    """
    if not (window < val_start - purge < train_end - purge < test_end <= len(clean_df)):
        raise ValueError("Insufficient data or invalid fold boundary")
    # Only rows strictly preceding validation are allowed to fit the scaler.
    scaler = fit_scaler(clean_df.iloc[:val_start - purge], features)
    scaled = clean_df.iloc[:test_end].copy()
    scaled.loc[:, features] = transform_features(scaled, scaler, features)
    X, y, meta = build_sequences(scaled, features, window)
    rows = meta["row_index"].to_numpy(dtype=int)
    # label at t requires close(t+purge): ensure no boundary crossing
    tr = rows < val_start - purge
    va = (rows >= val_start) & (rows < train_end - purge)
    te = (rows >= train_end) & (rows < test_end)
    if not (tr.any() and va.any() and te.any()):
        raise ValueError("Empty train/validation/test split")
    return (X[tr], y[tr], X[va], y[va], X[te], y[te], meta.loc[te].reset_index(drop=True), scaler)


def rolling_walk_forward_predict(
    df: pd.DataFrame, feature_cols: List[str], model_name: str, model_builder: Callable,
    window: int = 30, train_ratio: float = 0.6, test_window: int = 60,
    model_params: Dict | None = None, lr: float = 1e-3, batch_size: int = 64,
    epochs: int = 30, patience: int = 5, device: str = "cpu", horizon: int = 1,
):
    """Produce genuinely chronological fold predictions; no broker/exchange simulation."""
    if horizon != 1:
        raise ValueError("Current label generator and runner support horizon=1 only")
    if not (0.3 < train_ratio < 1) or test_window < 1:
        raise ValueError("Invalid walk-forward train_ratio or test_window")
    clean_df = df.dropna(subset=feature_cols + ["label"]).sort_values("datetime").reset_index(drop=True)
    n = len(clean_df)
    train_end = max(int(n * train_ratio), window + 32)
    fold_id, records, folds = 0, [], []
    while train_end < n:
        fold_id += 1
        test_end = min(n, train_end + test_window)
        val_start = max(int(train_end * 0.8), window + 4)
        X_tr, y_tr, X_va, y_va, X_te, y_te, meta, _ = make_fold_arrays(
            clean_df, feature_cols, window, val_start, train_end, test_end, purge=horizon)
        torch.manual_seed(42 + fold_id)
        model = model_builder(model_name, len(feature_cols), window, model_params or {})
        model, history = train_one_fold(model, X_tr, y_tr, X_va, y_va,
                                        lr=lr, batch_size=batch_size, epochs=epochs,
                                        patience=patience, device=device)
        model.eval()
        with torch.no_grad():
            probs = model(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()
        meta["prob"] = probs
        meta["fold"] = fold_id
        records.append(meta)
        folds.append({"fold": fold_id,
                      "train_end_date": str(clean_df.iloc[val_start - horizon - 1]["datetime"]),
                      "validation_start": str(clean_df.iloc[val_start]["datetime"]),
                      "validation_end": str(clean_df.iloc[train_end - horizon - 1]["datetime"]),
                      "test_start": str(meta.datetime.iloc[0]), "test_end": str(meta.datetime.iloc[-1]),
                      "test_auc": _auc(y_te, probs), "epochs_run": len(history["val_auc"]),
                      "scaler_fit_through": str(clean_df.iloc[val_start - horizon - 1]["datetime"]),
                      "train_samples": int(len(X_tr)), "val_samples": int(len(X_va)),
                      "test_samples": int(len(X_te))})
        print(f"Fold {fold_id}: {len(X_tr)} train / {len(X_va)} val / {len(X_te)} test, AUC={_auc(y_te, probs):.4f}")
        train_end = test_end
    if not records:
        raise ValueError("No walk-forward folds created: supply a longer time series")
    pred = pd.concat(records, ignore_index=True)
    assert pred["datetime"].is_unique
    return pred, pd.DataFrame(folds)


def run_rolling_walk_forward_backtest(
    df: pd.DataFrame, feature_cols: List[str], model_name: str, model_builder: Callable,
    window: int = 30, train_ratio: float = 0.6, test_window: int = 60,
    model_params: Dict | None = None, lr: float = 1e-3, batch_size: int = 64,
    epochs: int = 30, patience: int = 5, device: str = "cpu",
    commission: float = .001, slippage: float = .0005, stop_loss_pct: float = .03,
):
    preds, folds = rolling_walk_forward_predict(df, feature_cols, model_name, model_builder,
        window=window, train_ratio=train_ratio, test_window=test_window,
        model_params=model_params, lr=lr, batch_size=batch_size, epochs=epochs,
        patience=patience, device=device)
    bt, stats = risk_managed_backtest(preds.close, preds.prob, commission=commission,
                                     slippage=slippage, stop_loss_pct=stop_loss_pct)
    bt.insert(0, "datetime", preds.datetime.to_numpy())
    bt.insert(1, "fold", preds.fold.to_numpy())
    return preds, folds, bt, stats
