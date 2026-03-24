import numpy as np


class SusExecutionEngine:
    def __init__(self, assets, account):
        """
        assets: list of SusAssetParameters objects
        account: a SusAccount object (cash == equity)
        """
        self.assets = assets
        self.account = account
        self.n_assets = len(assets)

        self.multipliers = np.array([a.multiplier for a in assets])
        self.min_units = np.array([a.min_unit for a in assets])
        self.margin_rates = np.array([a.margin_rate for a in assets])
        self.tx_cost_bp = np.array([a.tx_cost_bp for a in assets])
        self.slippage_bp = np.array([a.slippage_bp for a in assets])

    # ---------- helpers ----------

    def round_positions(self, target_positions):
        return np.round(target_positions / self.min_units) * self.min_units

    def compute_costs(self, delta, prices):
        notional = np.abs(delta * prices * self.multipliers)
        return np.sum(notional * (self.tx_cost_bp + self.slippage_bp) / 1e4)

    def compute_pnl(self, positions, prices, next_prices):
        # PnL from holding positions over the step
        return np.sum(
            positions * (next_prices - prices) * self.multipliers
        )

    def margin_used(self, positions, prices):
        notional = np.abs(positions * prices * self.multipliers)
        return np.sum(notional * self.margin_rates)

    def equity(self):
        # equity == cash in this model
        return self.account.cash

    def available_margin(self, positions, prices):
        return self.equity() - self.margin_used(positions, prices)

    def margin_ratio(self, positions, prices):
        margin = self.margin_used(positions, prices)
        eq = self.equity()
        return np.inf if margin == 0 else eq / margin

    def is_liquidated(self, positions, prices, liquidation_level):
        return self.margin_ratio(positions, prices) < liquidation_level

    # ---------- main execution ----------

    def rebalance(self, target_positions, prices, next_prices):
        old_positions = self.account.positions.copy()

        # IMPORTANT: PnL is earned BEFORE rebalancing
        pnl = self.compute_pnl(old_positions, prices, next_prices)

        # --- round + compute delta ---
        target_positions = self.round_positions(target_positions)
        delta = target_positions - old_positions

        # --- split into close / open ---
        same_sign = np.sign(old_positions) == np.sign(target_positions)
        close_qty = np.zeros_like(delta)

        close_qty[same_sign] = np.minimum(
            np.abs(old_positions[same_sign]),
            np.abs(delta[same_sign])
        )
        close_qty[~same_sign] = np.abs(old_positions[~same_sign])

        open_qty = np.abs(delta) - close_qty

        close_dir = -np.sign(old_positions)
        open_dir = np.sign(delta)

        # --- costs ---
        close_costs = self.compute_costs(close_qty * close_dir, prices)
        open_costs = self.compute_costs(open_qty * open_dir, prices)
        costs = close_costs + open_costs

        net_pnl = pnl - costs
        pct_pnl = net_pnl / self.account.cash

        # --- update equity (cash) ---
        self.account.cash += net_pnl

        # --- update positions AFTER pnl ---
        self.account.positions = target_positions

        # --- margin tracking ---
        used_margin = self.margin_used(target_positions, prices)
        equity = self.equity()
        available_margin = equity - used_margin

        self.account.used_margin = used_margin
        self.account.available_margin = available_margin

        return {
            "pnl": pnl,
            "pct_pnl": pct_pnl,
            "costs": costs,
            "positions": target_positions,
            "net_pnl": net_pnl,
            "delta": delta,
            "close_qty": close_qty,
            "open_qty": open_qty,
            "used_margin": used_margin,
            "available_margin": available_margin,
            "margin_ratio": np.inf if used_margin == 0 else equity / used_margin,
        }