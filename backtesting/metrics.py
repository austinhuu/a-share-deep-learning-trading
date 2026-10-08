import numpy as np

def sharpe_ratio(returns, annualization: int = 252):
    x = np.asarray(returns, dtype=float)
    if len(x) < 2 or np.std(x, ddof=1) < 1e-12:
        return 0.0
    return float(np.sqrt(annualization) * x.mean() / np.std(x, ddof=1))

def sortino_ratio(returns, annualization: int = 252):
    x = np.asarray(returns, dtype=float)
    if not len(x):
        return 0.0
    downside_dev = float(np.sqrt(np.mean(np.minimum(x, 0) ** 2)))
    return float(np.sqrt(annualization) * x.mean() / downside_dev) if downside_dev > 1e-12 else 0.0

def max_drawdown(equity_curve):
    eq = np.asarray(equity_curve, dtype=float)
    if not len(eq):
        return 0.0
    peaks = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return float(np.min(eq / peaks - 1))

def calmar_ratio(returns, equity_curve, annualization: int = 252):
    x = np.asarray(returns, dtype=float)
    if len(x) == 0:
        return 0.0
    gross = float(np.prod(1 + x))
    if gross <= 0:
        return 0.0
    annual_return = gross ** (annualization / len(x)) - 1
    drawdown = abs(max_drawdown(equity_curve))
    return float(annual_return / drawdown) if drawdown > 1e-12 else 0.0
