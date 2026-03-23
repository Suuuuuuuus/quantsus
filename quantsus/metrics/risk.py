import numpy as np

__all__ = [
    "max_drawdown", "volatility", "calmar_ratio"
]

def max_drawdown(returns, initial_cash):
    cumulative = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / initial_cash
    return drawdown.min()

def volatility(returns):
    return np.std(returns)

def calmar_ratio(returns, initial_cash):
    return returns.sum() / abs(max_drawdown(returns, initial_cash) + 1e-8)