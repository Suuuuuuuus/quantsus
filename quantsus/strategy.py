import io
import os
import sys
import csv
import gzip
import time
import random
import secrets
import resource
import itertools
import multiprocessing
import pandas as pd
import numpy as np

from .auxiliary import *
from .variables import *

__all__ = [
    "open_position", "close_position", "is_margin_safe"
]

def open_margin_position(signal, cash, price, 
                  invest_fraction = INVEST_FRACTION,
                  leverage=LEVERAGE,
                  entry_cost=ENTRY_COST,
                  min_unit=MIN_UNIT):
    exec_price = price + signal * entry_cost
    margin_per_unit = exec_price / leverage
    max_units = cash / margin_per_unit
    invest_fraction = min(invest_fraction, 1)
    units = np.floor(max_units*invest_fraction / min_unit)*min_unit*signal
    margin_used = abs(units) * margin_per_unit
    return cash, units, exec_price


def close_margin_position(prev_spot, cash, price, units,
                 exit_cost = EXIT_COST):
    signal = np.sign(units)
    exit_spread = signal*exit_cost
    exec_price = price - exit_spread
    pnl = units * (exec_price - prev_spot)
    cash += pnl
    return cash, 0, 0


def is_margin_safe(prev_spot, cash, price, units,
                   leverage=LEVERAGE,
                   liquidation_level=LIQUIDATION_LEVEL):
    if units == 0:
        return True

    equity = cash + units * (price - prev_spot)
    initial_margin = abs(units) * prev_spot / leverage
    
    if (equity / initial_margin >= liquidation_level):
        return True
    else:
        print('Liquidation happened!')
        return False