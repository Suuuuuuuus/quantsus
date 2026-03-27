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


def intraday_time_sin(data):
    index = data.close.index

    seconds = (
        index.hour * 3600 +
        index.minute * 60 +
        index.second
    )

    cycle = 24 * 3600

    sin_t = np.sin(2 * np.pi * seconds / cycle)

    return pd.DataFrame(
        np.tile(sin_t.to_numpy().reshape(-1, 1), (1, data.close.shape[1])),
        index=index,
        columns=data.close.columns
    )

def intraday_time_cos(data):
    index = data.close.index

    seconds = (
        index.hour * 3600 +
        index.minute * 60 +
        index.second
    )

    cycle = 24 * 3600

    cos_t = np.cos(2 * np.pi * seconds / cycle)

    return pd.DataFrame(
        np.tile(cos_t.to_numpy().reshape(-1, 1), (1, data.close.shape[1])),
        index=index,
        columns=data.close.columns
    )

def true_range(data):
    """Calculate True Range for ATR/ADX."""
    tr1 = data.high - data.low
    tr2 = (data.high - data.close.shift(1)).abs().fillna(0)
    tr3 = (data.low - data.close.shift(1)).abs().fillna(0)
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def average_true_range(data, window=14):
    tr = true_range(data)
    # Wilder smoothing
    atr = tr.rolling(window, min_periods=1).mean()
    return atr


def relative_strength_index(data, window=14):
    delta = data.close.diff().fillna(0)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window, min_periods=1).mean()
    avg_loss = loss.rolling(window, min_periods=1).mean()

    rs = avg_gain / (avg_loss + 1e-8)  # prevent division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)


def average_directional_index(data, window=14):
    up_move = data.high.diff().fillna(0)
    down_move = data.low.diff().abs().fillna(0)

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = true_range(data)
    tr_smooth = tr.rolling(window, min_periods=1).mean()
    plus_di = 100 * pd.Series(plus_dm.flatten(), index=data.close.index).rolling(window, min_periods=1).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm.flatten(), index=data.close.index).rolling(window, min_periods=1).mean() / tr_smooth

    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8))
    adx = dx.rolling(window, min_periods=1).mean()
    return adx.fillna(0)