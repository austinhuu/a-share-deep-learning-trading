from typing import Callable, Dict, Tuple

import numpy as np
import optuna

from training.cv import expanding_window_split
from training.trainer import train_one_fold, evaluate, create_dataloaders


def suggest_params(trial: optuna.Trial, model_name: str) -> Dict:
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 3e-3, log=True),
        "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
    }

    if model_name == "lstm":
        params.update({
            "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
            "num_layers": trial.suggest_int("num_layers", 1, 3),
        })

    elif model_name == "hybrid":
        params.update({
            "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
            "nhead": trial.suggest_categorical("nhead", [2, 4, 8]),
            "num_transformer_layers": trial.suggest_int("num_transformer_layers", 1, 3),
            "lstm_hidden_size": trial.suggest_categorical("lstm_hidden_size", [32, 64, 128]),
            "lstm_num_layers": trial.suggest_int("lstm_num_layers", 1, 2),
        })

        if params["d_model"] % params["nhead"] != 0:
            raise optuna.TrialPruned()

    elif model_name == "cnn":
        params.update({
            "num_channels": trial.suggest_categorical("num_channels", [32, 64, 128]),
        })

    elif model_name == "mlp":
        params.update({
            "hidden_size": trial.suggest_categorical("hidden_size", [64, 128, 256]),
        })

    else:
        raise ValueError(f"不支持的模型类型: {model_name}")

    return params


def run_optuna_study(
    model_name: str,
    model_builder: Callable[[str, int, int, Dict], object],
    X,
    y,
    input_size: int,
    seq_len: int,
    n_trials: int = 10,
    n_splits: int = 3,
    epochs: int = 30,
    patience: int = 5,
    device: str = "cpu",
    seed: int = 42,
    fold_data=None,
) -> Tuple[Dict, optuna.Study]:

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, model_name)
        fold_aucs = []

        if fold_data is None:
            raise ValueError("fold_data must be provided: fold-specific scaling prevents lookahead")
        for X_train, y_train, X_val, y_val in fold_data:

            model = model_builder(
                model_name=model_name,
                input_size=input_size,
                seq_len=seq_len,
                params=params,
            )

            model, history = train_one_fold(
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                lr=params["lr"],
                batch_size=params["batch_size"],
                epochs=epochs,
                patience=patience,
                device=device,
                weight_decay=params["weight_decay"],
            )

            _, val_auc = evaluate(
                model,
                create_dataloaders(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    batch_size=params["batch_size"],
                )[1],
                device=device,
            )

            fold_aucs.append(val_auc)

        return float(np.mean(fold_aucs)) if fold_aucs else 0.5

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study