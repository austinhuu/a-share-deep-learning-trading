"""Optional Tushare adapter. Never embed API keys in source control."""
import os
from pathlib import Path
from typing import Optional
import pandas as pd

class TushareLoader:
    def __init__(self, token: Optional[str] = None):
        token = token or os.getenv("TUSHARE_TOKEN")
        if not token:
            raise ValueError("TUSHARE_TOKEN not set. Set it or use --data-csv / --demo.")
        try:
            import tushare as ts
        except ImportError as exc:
            raise ImportError("pip install tushare (optional data provider)") from exc
        ts.set_token(token)
        self.pro = ts.pro_api()

    def load_daily(self, ts_code: str, start_date: str, end_date: str, save_path=None) -> pd.DataFrame:
        df = self.pro.daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
        if df is None or df.empty:
            raise ValueError("No daily bars returned. Check ticker, dates and Tushare permissions.")
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
        return df

    @staticmethod
    def load_csv(path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError(f"CSV is empty: {path}")
        return df
