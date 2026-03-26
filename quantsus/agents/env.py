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

        prev_equity = self.account.cash
        prev_positions = self.account.positions


        target_positions = self.action_to_positions(action, prices)        
        result = self.exec_engine.rebalance(target_positions, prices, next_prices)

        target_positions = result['positions']
        pnl = result["net_pnl"]
        delta = target_positions - prev_positions

        same_direction = np.sign(target_positions) == np.sign(prev_positions)
        abs_delta = np.abs(delta)

        penalty = self.position_change_penalty * np.sum(
            same_direction * np.maximum(0, 1 - abs_delta)
        )

        reward = (pnl - penalty) / (prev_equity + 1e-8)

        liquidated = self.exec_engine.is_liquidated(
            self.account.positions,
            next_prices,
            self.liquidation_level
        )

        curr_timestamp = self.data.close.index[self.t]

        if liquidated and self.t < len(self.data.close) - 1:
            reward = liquidation_reward

        # 6. advance time
        self.t += 1
        # if self.t >= len(self.data.close) - 1 - self.feature_engine.window_size:
        if self.t >= len(self.data.close) - 1:
            liquidated = True

        return self.get_state(), reward, liquidated, {
            "time": curr_timestamp,
            "costs": result["costs"],
            "equity": self.account.cash,
            "penalty": penalty,
            "net_pnl": pnl,
            "pct_pnl": result["pct_pnl"],
            "positions": target_positions
        }

    # ---------- reset ----------

    def reset(self):
        self.t = 0
        self.account.reset(self.n_assets)
        return self.get_state()