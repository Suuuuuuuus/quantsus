import os
import mplfinance as mpf
home_dir = os.environ.get("HOME_DIR")

ANNUAL_RISK_FREE_RATE = 0.05
ANNUAL_TRADING_DATE = 252
ANNUAL_TRADING_HOURS = ANNUAL_TRADING_DATE * 24
HOUR_RISK_FREE_RATE = ANNUAL_RISK_FREE_RATE / ANNUAL_TRADING_HOURS
SPREAD_BP = 0.00005
# SPREAD_BP = 0
ENTRY_COST = SPREAD_BP / 2
EXIT_COST  = SPREAD_BP / 2



MPF_MC = mpf.make_marketcolors(
    up='red',
    down='green',
    edge='inherit',
    wick='inherit',
    volume='inherit'
)

MPF_STYLE = mpf.make_mpf_style(
    base_mpf_style='charles',
    marketcolors=MPF_MC
)

# Backtest settings
PRESTART_DATE = '2025-05-01'
START_DATE = '2025-01-01'
END_DATE  = '2025-12-31'
POSTEND_DATE = '2026-01-01'
TICK_FREQ = '1h'

# Market settings
FORCE_CLOSE = '02:00'
MARKET_OPEN = '07:00'

# Account settings
INVEST_FRACTION = 0.95
INITIAL_CASH = 1e5
LEVERAGE = 3
LIQUIDATION_LEVEL = 0.5
MIN_UNIT = 0.01

# Strategy settings:
VWAP_WINDOW_SIZE = 24
VWAP_BAND_STD = 1
ER_ROLLING_WINDOWS = 5
ER_MA_WINDOWS = 1