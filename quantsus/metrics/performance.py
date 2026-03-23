import numpy as np
from .variable import *

__all__ = [
    "sharpe_ratio", "win_rate"
]

    
def sharpe_ratio(returns, risk_free=RISK_FREE_RATE):
    excess = returns - risk_free
    return np.mean(excess) / (np.std(excess) + 1e-8)

def win_rate(returns):
    return np.mean(returns > 0)