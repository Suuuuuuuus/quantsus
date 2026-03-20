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

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import mplfinance as mpf


from .variable import *

__all__ = [
    "mpf_plot_day_candle"
]

def mpf_plot_day_candle(df, date, MPF_STYLE = MPF_STYLE):
    tmp = df[df.index.date == pd.Timestamp(date).date()]
    mpf.plot(
        tmp,
        type='candle',
        style=MPF_STYLE,
        ylabel='Price ($)',
        volume=False
    )
    return None