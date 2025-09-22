import math
from  scipy.stats import stats
import numpy as np
import matplotlib as plt
import yfinance as yf
import pytz
from datetime import datetime

# 3 models: binomial, monte carlo, black-scholes
# implement on european call options first


# s0, T, K, p (risk neutral probability), u, d, N, j, sigma (for u)

ticker = yf.Ticker("AAPL")
N = 10

def get_time_to_expiration():
    
    expirations = ticker.options
    expiration = expirations[0]
    et = pytz.timezone('US/Eastern')
    expiry = datetime.strptime(expiration, "%Y-%m-%d")
    expiry_et = et.localize(expiry)
    au = pytz.timezone('Australia/Sydney')
    expiry_au = expiry_et.astimezone(au)
    today_au = datetime.now(au)
    t = (expiry_au - today_au).days / 365
    return t
    
def get_s0():
    
    date_str = "2025-09-15"
    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    data = ticker.history(start=date_str, end=date_str)
    from datetime import timedelta
    data = ticker.history(start=date_str, end=(date_obj + timedelta(days=1)).strftime("%Y-%m-%d"))
    s0 = data['Close'][0]
    print("Stock price on", date_str, "was", s0)
    return s0
    
def get_k():
    expirations = ticker.options
    print("Available expirations:", expirations)
    expiration = expirations[0]
    option_chain = ticker.option_chain(expiration)
    calls = option_chain.calls
    print(calls.head())
    strike_prices = calls['strike'].tolist()
    print("Call option strike prices:", strike_prices)
    return strike_prices[0]  

def get_up_and_down_factors():
    pass
    
def get_historical_volatility():
    ticker = yf.Ticker("AAPL")
    data = ticker.history(period="1y")
    close_prices = data['Close']
    

# implement binomial tree using dynamic programming

def build_binomial_pricing_tree(s0, t, k, c):
    # here we calculate C using dp
    dp = []
    for i in range(N + 1):
        for j in range(N + 1):
            dp[i][j] = 0
    # fill from bottom up so calculate C_N,j first where j is the number of u steps
    
