import numpy as np
import pandas as pd

from .indicators import sma, ema, rolling_std, rsi, macd, bollinger_bands, atr


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    训练/验证/回测统一特征入口。
    后续 VeighNa 实盘也应该复用同样逻辑。
    """
    df = df.copy()

    # 收益率与价格变化
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["log_return_1"] = np.log(df["close"] / df["close"].shift(1))

    # 均线与偏离
    df["ma_5"] = sma(df["close"], 5)
    df["ma_10"] = sma(df["close"], 10)
    df["ma_20"] = sma(df["close"], 20)
    df["ema_12"] = ema(df["close"], 12)
    df["ema_26"] = ema(df["close"], 26)
    df["close_ma5_bias"] = df["close"] / (df["ma_5"] + 1e-8) - 1
    df["close_ma10_bias"] = df["close"] / (df["ma_10"] + 1e-8) - 1

    # 波动率
    df["std_5"] = rolling_std(df["close"], 5)
    df["std_10"] = rolling_std(df["close"], 10)
    df["hl_spread"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)

    # 成交量
    df["vol_ma_5"] = sma(df["volume"], 5)
    df["vol_ma_10"] = sma(df["volume"], 10)
    df["vol_ratio_5"] = df["volume"] / (df["vol_ma_5"] + 1e-8)
    df["vol_ratio_10"] = df["volume"] / (df["vol_ma_10"] + 1e-8)

    # RSI / MACD / BOLL / ATR
    df["rsi_14"] = rsi(df["close"], 14)
    macd_dif, macd_dea, macd_hist = macd(df["close"])
    df["macd_dif"] = macd_dif
    df["macd_dea"] = macd_dea
    df["macd_hist"] = macd_hist

    boll_up, boll_mid, boll_low = bollinger_bands(df["close"], 20, 2)
    df["boll_up"] = boll_up
    df["boll_mid"] = boll_mid
    df["boll_low"] = boll_low
    df["boll_width"] = (boll_up - boll_low) / (boll_mid + 1e-8)

    df["atr_14"] = atr(df["high"], df["low"], df["close"], 14)

    return df


def get_default_feature_cols():
    return [
        "return_1", "return_3", "return_5", "log_return_1",
        "ma_5", "ma_10", "ma_20", "ema_12", "ema_26",
        "close_ma5_bias", "close_ma10_bias",
        "std_5", "std_10", "hl_spread",
        "vol_ma_5", "vol_ma_10", "vol_ratio_5", "vol_ratio_10",
        "rsi_14", "macd_dif", "macd_dea", "macd_hist",
        "boll_up", "boll_mid", "boll_low", "boll_width",
        "atr_14"
    ]


def create_features_for_inference(df: pd.DataFrame) -> pd.DataFrame:
    """
    为后续策略层预留的推理特征函数。
    当前阶段 A 训练时也可直接复用 create_features。
    """
    return create_features(df)
