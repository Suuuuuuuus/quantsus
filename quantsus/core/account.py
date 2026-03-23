import numpy as np
from .variable import *

class SusAccount:
    def __init__(self, initial_cash=INITIAL_CASH):
        self.initial_cash = initial_cash

    def reset(self, n_assets):
        self.cash = float(self.initial_cash)
        self.positions = np.zeros(n_assets)
        self.used_margin = 0.0
        self.available_margin = self.cash

    def equity(self):
        return self.cash

    def update_margin(self, prices, contract_multipliers, margin_requirements):
        """
        margin_requirements: array of per-asset margin rates
        """
        position_values = np.abs(self.positions * prices * contract_multipliers)

        # Elementwise margin
        self.used_margin = (position_values * margin_requirements).sum()

        eq = self.equity(prices, contract_multipliers)
        self.available_margin = eq - self.used_margin

    def can_trade(self, trade_sizes, prices, contract_multipliers, margin_requirements):
        """
        trade_sizes: vector of trades per asset
        """
        required_margin = np.abs(trade_sizes * prices * contract_multipliers) * margin_requirements
        total_required = required_margin.sum()

        return self.available_margin >= total_required