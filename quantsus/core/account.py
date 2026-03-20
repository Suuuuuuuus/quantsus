import numpy as np
from .variable import *

class SusAccount:
    def __init__(self, initial_cash = INITIAL_CASH):
        self.initial_cash = initial_cash

    def reset(self, n_assets):
        self.cash = float(self.initial_cash)
        self.positions = np.zeros(n_assets)

    def equity(self, prices, contract_multipliers):
        return self.cash + (self.positions * prices * contract_multipliers).sum()