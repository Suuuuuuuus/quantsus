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

    def action_to_positions(self, action, prices):
        equity = self.account.equity(prices, self.exec_engine.multipliers)
        max_notional = equity * self.leverage
        action = action / (np.sum(np.abs(action)) + 1e-8)
        target_notional = action * max_notional
        positions = target_notional / (prices * self.exec_engine.multipliers)
        return positions

    def scale_to_margin(self, target_positions, prices):
        margin = self.exec_engine.margin_used(target_positions, prices)
        equity = self.account.equity(prices, self.exec_engine.multipliers)

        if margin == 0:
            return target_positions

        scale = equity / margin
        return target_positions * scale

    # ---------- state ----------

    def get_state(self):
        feature_vec = self.feature_engine.get_state(self.t)

        positions = self.account.positions
        prices = self.data.close.iloc[self.t].values

        margin_ratio = self.exec_engine.margin_ratio(positions, prices)

        return np.concatenate([
            feature_vec,
            positions,
            [margin_ratio]
        ])

    # ---------- step ----------

    def step(self, action, liquidation_reward = LIQUIDATION_REWARD):
        prices = self.data.close.iloc[self.t].values
        next_prices = self.data.close.iloc[self.t + 1].values

        # 1. action → target positions
        target_positions = self.action_to_positions(action, prices)

        # 2. enforce margin (soft constraint)
        if not self.exec_engine.is_margin_safe(target_positions, prices):
            target_positions = self.scale_to_margin(target_positions, prices)

        # 3. delta (for penalty)
        delta = target_positions - self.account.positions

        # 4. execute trade
        result = self.exec_engine.rebalance(target_positions, prices, next_prices)

        pnl = result["net_pnl"]

        # 5. reward (PnL - trading penalty)
        penalty = self.position_change_penalty * np.sum(np.abs(delta))
        reward = pnl - penalty

        # 6. check liquidation (hard constraint)
        liquidated = self.exec_engine.is_liquidated(
            self.account.positions,
            next_prices,
            self.liquidation_level
        )

        if liquidated:
            reward = liquidation_reward

        # 7. advance time
        self.t += 1
        if self.t >= len(self.data.close) - 1:
            liquidated = True

        return self.get_state(), reward, liquidated, {}

    # ---------- reset ----------

    def reset(self):
        self.t = 0
        self.account.reset(self.n_assets)

        return self.get_state()