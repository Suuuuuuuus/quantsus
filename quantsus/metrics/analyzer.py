import numpy as np
from .performance import *
from .risk import *


class SusPerformanceAnalyzer:
    def __init__(self):
        self.history = []

    def evaluate(self, returns, initial_cash):
        returns = np.array(returns)
        metrics = {
            "sharpe": sharpe_ratio(returns),
            "max_drawdown": max_drawdown(returns, initial_cash),
            "win_rate": win_rate(returns),
            "calmar_ratio": calmar_ratio(returns, initial_cash),
            "volatility": volatility(returns),
            "total_return": returns.sum()
        }
        self.history.append(metrics)
        return metrics