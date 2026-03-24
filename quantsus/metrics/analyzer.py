import numpy as np
from .performance import *
from .risk import *


import pandas as pd
import numpy as np

class SusPerformanceAnalyzer:
    def __init__(self):
        # Store history as a list of dicts initially
        self.history = []

    def evaluate(self, returns):
        """
        Compute performance metrics for a single episode and store in history.
        
        Args:
            returns: list or array of per-step pct_pnl
        Returns:
            metrics: dict with calculated metrics
        """
        returns = np.array(returns)
        metrics = {
            "final_pnl": final_pnl(returns),
            "sharpe": sharpe_ratio(returns),
            "max_drawdown": max_drawdown(returns),
            "win_rate": win_rate(returns),
            "volatility": volatility(returns),
            "calmar_ratio": calmar_ratio(returns)
        }

        self.history.append(metrics)
        return metrics

    def as_df(self):
        """
        Return all recorded metrics as a pandas DataFrame.
        Each row corresponds to one evaluation call (e.g., one episode or epoch)
        """
        if not self.history:
            return pd.DataFrame()
        return pd.DataFrame(self.history)

    def reset(self):
        """
        Clear the stored history
        """
        self.history = []

    def last(self):
        """
        Return the last computed metrics
        """
        if not self.history:
            return None
        return self.history[-1]