"""Idealized close-to-close single-asset backtest, for research only.

Assumption: signal for bar t is computable using that bar's closing values,
and its target position is filled at close(t). The target contributes only
to returns close(t)->close(t+1), never to the contemporaneous return.
Costs are charged on changes in target position; real A-share fillability,
T+1 restrictions, price limits, order-book liquidity and taxes are NOT modeled.
"""
import numpy as np
import pandas as pd
from .metrics import sharpe_ratio, sortino_ratio, max_drawdown, calmar_ratio

def risk_managed_backtest(close_series: pd.Series, prob_series: pd.Series,
                          buy_threshold=0.55, sell_threshold=0.45,
                          commission=0.001, slippage=0.0005,
                          max_position=1.0, min_position=0.0, stop_loss_pct=0.03):
    close = pd.Series(close_series, dtype=float).reset_index(drop=True)
    prob = pd.Series(prob_series, dtype=float).reset_index(drop=True)
    if not len(close) or len(close) != len(prob):
        raise ValueError("Close/prob must be equally sized, non-empty arrays")
    if close.isna().any() or prob.isna().any():
        raise ValueError("NaN in input")
    if not (0 <= sell_threshold < buy_threshold <= 1):
        raise ValueError("Thresholds must satisfy 0<=sell<buy<=1")
    if min_position < 0 or max_position < min_position:
        raise ValueError("Invalid position bounds")
    df = pd.DataFrame({"close": close, "prob": prob})
    df["asset_return"] = df.close.pct_change().fillna(0.0)
    targets, stopped = [], []
    pos, entry_price = 0.0, None
    for price, p in zip(df.close, df.prob):
        stop = bool(pos > 0 and entry_price is not None and price <= entry_price * (1-stop_loss_pct))
        target = pos
        if stop or p <= sell_threshold:
            target = 0.0
        elif p >= buy_threshold:
            confidence = (p - buy_threshold) / max(1.0 - buy_threshold, 1e-8)
            target = min_position + confidence * (max_position - min_position)
            target = float(np.clip(target, min_position, max_position))
        if target == 0.0:
            entry_price = None
        elif pos == 0.0:
            entry_price = float(price)
        targets.append(target)
        stopped.append(stop)
        pos = target
    df["position"] = targets
    df["stop_loss"] = stopped
    df["trade"] = df.position.diff().abs().fillna(df.position)
    df["trade_cost"] = df.trade * (commission + slippage)
    df["strategy_return"] = df.position.shift(1).fillna(0.0) * df.asset_return - df.trade_cost
    df["equity"] = (1 + df.strategy_return).cumprod()
    metrics = {
        "sharpe": sharpe_ratio(df.strategy_return),
        "sortino": sortino_ratio(df.strategy_return),
        "max_drawdown": max_drawdown(df.equity),
        "calmar": calmar_ratio(df.strategy_return, df.equity),
        "total_return": float(df.equity.iloc[-1]-1),
        "turnover": float(df.trade.sum()),
        "stop_loss_count": int(df.stop_loss.sum()),
        "avg_position": float(df.position.mean()),
    }
    return df, metrics

def simple_threshold_backtest(close_series, prob_series, buy_threshold=0.55, sell_threshold=0.45,
                              commission=0.001, slippage=0.0005):
    df, metrics = risk_managed_backtest(close_series, prob_series,
                                        buy_threshold=buy_threshold, sell_threshold=sell_threshold,
                                        commission=commission, slippage=slippage,
                                        max_position=1.0, min_position=1.0, stop_loss_pct=1.0)
    return df, {k: metrics[k] for k in ["sharpe", "sortino", "max_drawdown", "calmar", "total_return"]}
