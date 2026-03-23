import numpy as np
from .variable import *

class SusTradingEnv:
    def __init__(self,
                 data,
                 exec_engine,
                 feature_engine,
                 leverage=LEVERAGE,
                 liquidation_level=LIQUIDATION_LEVEL,
                 position_change_penalty=0.0):

        self.data = data
        self.exec_engine = exec_engine
        self.feature_engine = feature_engine
        self.account = exec_engine.account

        self.n_assets = len(exec_engine.assets)
        self.leverage = leverage
        self.liquidation_level = liquidation_level
        self.position_change_penalty = position_change_penalty

        self.t = 0

    # ---------- action handling ----------

    def action_to_positions(self, action, prices, risk_scale=1):
        equity = self.account.cash

        action = np.tanh(action)
        target_notional = action * equity * risk_scale * self.leverage
        positions = target_notional / (prices * self.exec_engine.multipliers)

        return positions

    # ---------- state ----------

    def get_state(self):
        return self.feature_engine.get_state(self.t)

    # ---------- step ----------

    def step(self, action, liquidation_reward=LIQUIDATION_REWARD):
        prices = self.data.close.iloc[self.t].values
        next_prices = self.data.close.iloc[self.t + 1].values

        # store equity BEFORE pnl
        equity_before = self.account.cash

        # 1. action → target positions
        target_positions = self.action_to_positions(action, prices)

        # 2. delta (for penalty)
        delta = target_positions - self.account.positions
        
        # 3. execute trade (this updates equity internally)
        result = self.exec_engine.rebalance(target_positions, prices, next_prices)

        pnl = result["net_pnl"]

        same_direction = np.sign(target_positions) == np.sign(self.account.positions)
        abs_delta = np.abs(delta)

        penalty = self.position_change_penalty * np.sum(
            same_direction * np.maximum(0, 1 - abs_delta)
        )
        # 4. reward

        reward = (pnl - penalty) / (equity_before + 1e-8)

        # 5. liquidation check
        liquidated = self.exec_engine.is_liquidated(
            self.account.positions,
            next_prices,
            self.liquidation_level
        )

        if liquidated and self.t < len(self.data.close) - 1:
            reward = liquidation_reward

        # 6. advance time
        self.t += 1
        if self.t >= len(self.data.close) - 1 - self.feature_engine.window_size:
            liquidated = True

        return self.get_state(), reward, liquidated, {
            "used_margin": self.account.used_margin,
            "available_margin": self.account.available_margin,
            "equity": self.account.cash,
            "pnl": pnl,
            "positions": target_positions
        }

    # ---------- reset ----------

    def reset(self):
        self.t = 0
        self.account.reset(self.n_assets)
        return self.get_state()