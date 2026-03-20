import pandas as pd

class SusLoadCsvs:
    def __init__(self, path_dict):
        """
        path_dict: { 'AAPL': 'aapl.csv', 'MSFT': 'msft.csv' }
        """
        self.path_dict = path_dict

    def load(self):
        df_dict = {}
        for asset, path in self.path_dict.items():
            df = pd.read_csv(path, parse_dates=True, index_col=0)
            df.index = pd.to_datetime(df.index)
            df_dict[asset] = df
        return df_dict