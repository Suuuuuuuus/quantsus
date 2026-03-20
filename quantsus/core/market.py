import pandas as pd

class SusMarketData:
    def __init__(self, df_dict):
        self.close  = self.build_field(df_dict, "Close")
        self.open   = self.build_field(df_dict, "Open")
        self.high   = self.build_field(df_dict, "High")
        self.low    = self.build_field(df_dict, "Low")
        self.volume = self.build_field(df_dict, "Volume")

    def build_field(self, df_dict, field):
        return pd.DataFrame({
            asset: df[field] for asset, df in df_dict.items()
        }).sort_index()

    def align(self):
        fields = ["close", "open", "high", "low", "volume"]

        # 1. Build common index (union)
        common_index = None
        for field in fields:
            idx = getattr(self, field).index
            common_index = idx if common_index is None else common_index.union(idx)

        # 2. Reindex everything
        for field in fields:
            df = getattr(self, field)
            setattr(self, field, df.reindex(common_index).sort_index())

        # 3. Remove timestamps where ALL assets have zero volume
        mask = ~(self.volume.fillna(0) == 0).all(axis=1)

        for field in fields:
            setattr(self, field, getattr(self, field).loc[mask])

    def slice(self, start=None, end=None):
        new = SusMarketData.__new__(SusMarketData)
        fields = ["close", "open", "high", "low", "volume"]

        for field in fields:
            df = getattr(self, field)

            if start is not None:
                df = df[df.index >= pd.to_datetime(start)]
            if end is not None:
                df = df[df.index < pd.to_datetime(end)]  # < instead of <=

            setattr(new, field, df)

        return new