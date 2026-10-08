"""Deterministic fake prices exclusively for smoke testing; not market data."""
import numpy as np
import pandas as pd

def make_synthetic_ohlcv(n: int = 520, seed: int = 42):
    if n < 90:
        raise ValueError("At least 90 bars required for indicators and sequence windows")
    rng = np.random.default_rng(seed)
    daily = rng.normal(0.0002, 0.019, n)
    close = 50.0 * np.exp(np.cumsum(daily))
    opened = close * np.exp(rng.normal(0, .005, n))
    high = np.maximum(opened, close) * (1 + rng.uniform(0, .015, n))
    low = np.minimum(opened, close) * (1 - rng.uniform(0, .015, n))
    return pd.DataFrame({"datetime": pd.bdate_range('2022-01-03', periods=n),
                         "open": opened, "high": high, "low": low,
                         "close": close, "volume": rng.integers(100000, 1000000, n)})
