import pandas as pd
import numpy as np
from .variable import *


def build_features(data, feature_specs):
    """
    feature_specs: list of (name, callable)
    callable must take (data) as input
    """
    return {
        name: func(data)
        for name, func in feature_specs
    }


def typical_price(data):
    return (data.high + data.low + data.close) / 3


def rolling_vwap(data, window):
    tp = typical_price(data)
    pv = tp * data.volume

    return (
        pv.rolling(window, min_periods=1).sum()
        / data.volume.rolling(window, min_periods=1).sum()
    )


def intraday_vwap(data):
    tp = typical_price(data)
    pv = tp * data.volume

    date_index = tp.index.date

    cum_pv = pv.groupby(date_index).cumsum()
    cum_vol = data.volume.groupby(date_index).cumsum()

    return cum_pv / cum_vol


def vwap(data, window, fvwap=rolling_vwap):
    vwaprice = fvwap(data, window)
    return (vwaprice - data.close) / data.close


def log_return(data):
    return np.log(data.close / data.close.shift(1)).fillna(0)


def normalized_log_volume(data, window):
    log_vol = np.log(data.volume + 1)
    mean = log_vol.rolling(window, min_periods=1).mean()
    return (log_vol - mean).fillna(0)


def time_sin_hour(data):
    hours = data.close.index.hour
    t = np.sin(2 * np.pi * hours / 24)

    return pd.DataFrame(
        np.tile(t.values.reshape(-1, 1), (1, data.close.shape[1])),
        index=data.close.index,
        columns=data.close.columns
    )