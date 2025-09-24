import math
from  scipy.stats import stats
import numpy as np
import matplotlib as plt
import yfinance as yf
import pytz
from datetime import datetime
import pandas as pd

# 3 models: binomial, monte carlo, black-scholes
# implement on european call options first


# s0, T, K, p (risk neutral probability), u, d, N, j, sigma (for u)

class BinomialTreeModel:

    def __init__(self):
        self.ticker = yf.Ticker("AAPL")
        self.n = 10
        self.r = 0.05  

    def get_time_to_expiration(self):
        expirations = self.ticker.options
        expiration = expirations[0]
        et = pytz.timezone('US/Eastern')
        expiry = datetime.strptime(expiration, "%Y-%m-%d")
        expiry_et = et.localize(expiry)
        au = pytz.timezone('Australia/Sydney')
        expiry_au = expiry_et.astimezone(au)
        today_au = datetime.now(au)
        t = (expiry_au - today_au).days / 365
        return t
        
    def get_s0(self):
        date_str = "2025-09-15"
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        data = self.ticker.history(start=date_str, end=date_str)
        from datetime import timedelta
        data = self.ticker.history(start=date_str, end=(date_obj + timedelta(days=1)).strftime("%Y-%m-%d"))
        s0 = data['Close'][0]
        print("Stock price on", date_str, "was", s0)
        return s0
        
    def get_k(self):
        expirations = self.ticker.options
        print("Available expirations:", expirations)
        expiration = expirations[0]
        option_chain = self.ticker.option_chain(expiration)
        calls = option_chain.calls
        print(calls.head())
        strike_prices = calls['strike'].tolist()
        print("Call option strike prices:", strike_prices)
        return strike_prices[0]  
        
    def get_historical_volatility(self):
        data = self.ticker.history(period="1y")
        close_prices = data['Close']
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        std_devs = log_returns.std()
        annualised_vol = std_devs * np.sqrt(252)
        return annualised_vol

    def get_up_and_down_factors(self, sigma, t):
        up = math.exp(sigma * math.sqrt(t/self.n))
        down = 1/up
        return [up, down]

    def get_risk_neutral_probabilities(self, u, d, t):
        dT = t / self.n
        p =  (np.exp(self.r * dT) - d) / (u - d)
        q = 1 - p
        return [p, q]
        

    # implement binomial tree using dynamic programming

    def get_call_option_price(self, s0, t, k):
        # here we calculate C using dp
        option_dp = []
        sigma = self.get_historical_volatility()
        t = self.get_time_to_expiration()
        u = self.get_up_and_down_factors(sigma, t)[0]
        d = self.get_up_and_down_factors(sigma, t)[1]
        p = self.get_risk_neutral_probabilities(u, d, t)[0]
        q = self.get_risk_neutral_probabilities(u, d, t)[1]
        dT = t / self.n
        
        for i in range(self.n + 1):
            for j in range(self.n + 1):
                option_dp[i][j] = 0.0
                
        stock = [[0.0 for j in range(i+1)] for i in range(self.n+1)]
        for i in range(self.n+1):
            for j in range(i+1):
                stock[i][j] = s0 * (u**j) * (d**(i-j))
                
        # fill from bottom up so calculate C_N,j first where j is the number of u steps
        # rows = time steps, cols = num of up moves
        for i in range(self.n + 1):
            option_dp[self.n][i] = max(0, stock[self.n][i] - k)
        
        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                option_dp[i][j] = math.exp(-self.r * dT) * (p * option_dp[i + 1][j + 1] + q * option_dp[i + 1][j])
        return option_dp[0][0]
        