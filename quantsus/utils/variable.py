import os
import mplfinance as mpf
home_dir = os.environ.get("HOME_DIR")

ANNUAL_RISK_FREE_RATE = 0.05
ANNUAL_TRADING_DATE = 252
ANNUAL_TRADING_HOURS = ANNUAL_TRADING_DATE * 24
HOUR_RISK_FREE_RATE = ANNUAL_RISK_FREE_RATE / ANNUAL_TRADING_HOURS


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