import pandas as pd


class DataProcessor:
    """
    负责把原始数据整理成项目统一格式：
    datetime, open, high, low, close, volume
    """

    REQUIRED_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]

    def standardize_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        rename_map = {
            "trade_date": "datetime",
            "vol": "volume",
        }
        df = df.rename(columns=rename_map)

        missing = [c for c in ["datetime", "open", "high", "low", "close"] if c not in df.columns]
        if missing:
            raise ValueError(f"缺少必要列: {missing}")

        if "volume" not in df.columns:
            raise ValueError("缺少 volume/vol 列")

        # Tushare trade_date often arrives as an integer YYYYMMDD when read from CSV.
        # pd.to_datetime(integer) otherwise treats it as epoch nanoseconds.
        raw_dates = df["datetime"]
        as_str = raw_dates.astype(str).str.strip()
        if as_str.str.fullmatch(r"\d{8}").all():
            df["datetime"] = pd.to_datetime(as_str, format="%Y%m%d", errors="raise")
        else:
            df["datetime"] = pd.to_datetime(raw_dates, errors="raise")
        df = df.sort_values("datetime").reset_index(drop=True)

        # 保留统一字段
        df = df[self.REQUIRED_COLUMNS].copy()

        # 基础类型整理
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna().drop_duplicates(subset=["datetime"]).reset_index(drop=True)
        return df
