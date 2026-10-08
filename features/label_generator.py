import pandas as pd

def add_binary_label(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """Next-horizon close-to-close direction, with unknown final labels left NaN."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    out = df.copy()
    out["future_close"] = out["close"].shift(-horizon)
    out["label"] = (out["future_close"] > out["close"]).astype("float64")
    out.loc[out["future_close"].isna(), "label"] = float("nan")
    return out
