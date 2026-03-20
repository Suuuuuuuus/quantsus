class SusAssetParameters:
    def __init__(self, name, 
                 multiplier = 1, 
                 min_unit = 1, 
                 tx_cost_bp = 0.0, 
                 slippage_bp = 0.0, 
                 margin_rate = 0.05):
        """
        tx_cost_bp / slippage_bp: in basis points (e.g., 0.01% = 1 bp)
        """
        self.name = name
        self.multiplier = multiplier
        self.min_unit = min_unit
        self.tx_cost_bp = tx_cost_bp
        self.slippage_bp = slippage_bp
        self.margin_rate = margin_rate