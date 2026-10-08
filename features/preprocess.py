import json
from pathlib import Path
from typing import List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def fit_scaler(train_df: pd.DataFrame, feature_cols: List[str]) -> StandardScaler:
    if train_df.empty:
        raise ValueError("Scaler training partition is empty")
    return StandardScaler().fit(train_df[feature_cols].to_numpy())

def transform_features(df: pd.DataFrame, scaler: StandardScaler, feature_cols: List[str]) -> np.ndarray:
    return scaler.transform(df[feature_cols].to_numpy())

def save_scaler_and_features(scaler: StandardScaler, feature_cols: List[str], artifact_dir: str, prefix: str = "") -> None:
    p = Path(artifact_dir); p.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, p / f"{prefix}scaler.pkl")
    (p / f"{prefix}feature_cols.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

def load_scaler_and_features(artifact_dir: str, prefix: str = ""):
    p = Path(artifact_dir)
    return joblib.load(p / f"{prefix}scaler.pkl"), json.loads((p / f"{prefix}feature_cols.json").read_text(encoding="utf-8"))

def build_sequences(df: pd.DataFrame, feature_cols: List[str], window: int) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Sample dated t uses features through t, predicts close(t+h)/close(t).
    Assumes a chronologically ordered, NaN-free feature/label frame.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    clean = df.dropna(subset=feature_cols + ["label"]).reset_index(drop=True)
    X, y, meta = [], [], []
    for end in range(window - 1, len(clean)):
        hist = clean.iloc[end - window + 1:end + 1]
        row = clean.iloc[end]
        X.append(hist[feature_cols].to_numpy(dtype=np.float32))
        y.append(float(row["label"]))
        meta.append({"datetime": row["datetime"], "close": row["close"], "label": float(row["label"]), "row_index": end})
    return (np.asarray(X, dtype=np.float32).reshape(-1, window, len(feature_cols)),
            np.asarray(y, dtype=np.float32), pd.DataFrame(meta))
