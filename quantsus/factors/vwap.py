from typing import Callable
import pandas as pd
from .variable import *

def typical_price(high: pd.DataFrame,
                  low: pd.DataFrame,
                  close: pd.DataFrame) -> pd.DataFrame:
    """
    Typical price = (H + L + C) / 3

    Inputs:
        high, low, close: DataFrames (time * assets)

    Returns:
        DataFrame (time * assets)
    """
    return (high + low + close) / 3


def rolling_vwap(high: pd.DataFrame,
                 low: pd.DataFrame,
                 close: pd.DataFrame,
                 volume: pd.DataFrame,
                 window: int = VWAP_WINDOW_SIZE) -> pd.DataFrame:
    """
    Rolling VWAP using typical price.

    VWAP = sum(price * volume) / sum(volume)

    Inputs:
        high, low, close, volume: DataFrames (time * assets)
        window: rolling window

    Returns:
        DataFrame (time * assets)
    """
    tp = typical_price(high, low, close)
    pv = tp * volume

    vwap = pv.rolling(window, min_periods = 1).sum() / volume.rolling(window, min_periods = 1).sum()

    return vwap


def intraday_vwap(high: pd.DataFrame,
                  low: pd.DataFrame,
                  close: pd.DataFrame,
                  volume: pd.DataFrame) -> pd.DataFrame:
    """
    Intraday VWAP (resets each day).

    Assumes index is datetime.
    """
    tp = typical_price(high, low, close)
    pv = tp * volume

    date_index = tp.index.date

    cum_pv = pv.groupby(date_index).cumsum()
    cum_vol = volume.groupby(date_index).cumsum()

    return cum_pv / cum_vol


def vwap_signal(high: pd.DataFrame,
                low: pd.DataFrame,
                close: pd.DataFrame,
                volume: pd.DataFrame,
                window: int = VWAP_WINDOW_SIZE,
                fvwap: Callable = rolling_vwap) -> pd.DataFrame:
    """
    Normalized VWAP deviation signal.

    signal = (vwap - close) / close

    Returns:
        DataFrame (time * assets)
    """
    vwap = fvwap(high, low, close, volume, window)

    signal = (vwap - close) / close

    return signal