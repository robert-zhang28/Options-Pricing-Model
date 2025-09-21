import math
from  scipy.stats import stats
import numpy as np
import matplotlib as plt
import yfinance as yf
import pytz
from datetime import datetime

# 3 models: binomial, monte carlo, black-scholes
# will do everything on european options


# S0, T, K, C, p (risk neutral probability), u, d, N, j, sigma (for u)

def get_time_to_expiration():
    ticker = yf.Ticker("AAPL")
    expirations = ticker.options
    expiration = expirations[0]
    et = pytz.timezone('US/Eastern')
    expiry = datetime.strptime(expiration, "%Y-%m-%d")
    expiry_et = et.localize(expiry)
    au = pytz.timezone('Australia/Sydney')
    expiry_au = expiry_et.astimezone(au)
    today_au = datetime.now(au)
    return (expiry_au - today_au).days / 365

