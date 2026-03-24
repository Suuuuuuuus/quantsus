import numpy as np

__all__ = [
    "max_drawdown", "volatility", "calmar_ratio"
]
def max_drawdown(returns):
    cumulative = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    return drawdown.min()

def volatility(returns):
    return np.std(returns)

def total_return(returns):
    return np.prod(1 + returns) - 1

def calmar_ratio(returns):
    mdd = max_drawdown(returns)
    return total_return(returns) / abs(mdd + 1e-8)