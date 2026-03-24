import numpy as np
from .variable import *

__all__ = [
    "sharpe_ratio", "win_rate", "final_pnl"
]

def sharpe_ratio(returns, annual_rf=ANNUAL_RISK_FREE_RATE, periods_per_year=ANNUAL_FX_TRADING_HOURS):
    rf_per_period = (1 + annual_rf)**(1 / periods_per_year) - 1
    excess = returns - rf_per_period
    
    mean_excess = np.mean(excess)
    std_excess = np.std(excess, ddof=1)
    
    return (mean_excess / (std_excess + 1e-8)) * np.sqrt(periods_per_year)


def win_rate(returns):
    return np.mean(returns > 0)

def final_pnl(returns):
    return ((1 + returns).cumprod())[-1]