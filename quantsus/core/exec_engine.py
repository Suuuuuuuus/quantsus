import numpy as np


class SusExecutionEngine:
    def __init__(self, assets, account):
        """
        assets: list of SusAssetParameters objects
        account: a SusAccount object
        """
        self.assets = assets
        self.account = account
        self.n_assets = len(assets)

        self.multipliers = np.array([a.multiplier for a in assets])
        self.min_units = np.array([a.min_unit for a in assets])
        self.margin_rates = np.array([a.margin_rate for a in assets])
        self.tx_cost_bp = np.array([a.tx_cost_bp for a in assets])
        self.slippage_bp = np.array([a.slippage_bp for a in assets])


    def round_positions(self, target_positions):
        return np.round(target_positions / self.min_units) * self.min_units

    def compute_costs(self, delta, prices):
        notional = np.abs(delta * prices * self.multipliers)
        costs = np.sum(notional * (self.tx_cost_bp + self.slippage_bp) / 1e4)
        return costs

    def compute_pnl(self, positions, prices, next_prices):
        return np.sum(
            positions * (next_prices - prices) * self.multipliers
        )

    def margin_used(self, positions, prices):
        notional = np.abs(positions * prices * self.multipliers)
        return np.sum(notional * self.margin_rates)

    def margin_ratio(self, positions, prices):
        equity = self.account.equity(prices, self.multipliers)
        margin = self.margin_used(positions, prices)
        return 1 if margin == 0 else equity / margin

    def is_margin_safe(self, positions, prices):
        return self.margin_ratio(positions, prices) > 1.0

    def is_liquidated(self, positions, prices, liquidation_level):
        return self.margin_ratio(positions, prices) < liquidation_level

    # ---------- main execution ----------

    def rebalance(self, target_positions, prices, next_prices):
        target_positions = self.round_positions(target_positions)
        delta = target_positions - self.account.positions
        costs = self.compute_costs(delta, prices)
        pnl = self.compute_pnl(target_positions, prices, next_prices)
        net_pnl = pnl - costs

        self.account.cash += net_pnl
        self.account.positions = target_positions

        return {
            "pnl": pnl,
            "costs": costs,
            "net_pnl": net_pnl,
            "delta": delta
        }