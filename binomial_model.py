import math
import numpy as np
import matplotlib as plt
import yfinance as yf
import pytz
from datetime import datetime
import pandas as pd

# 3 models: binomial, monte carlo, black-scholes
# implement on european call options first

#TODO: need to allow for inputs and make consistent for inputs across other models: black scholes, monte carlo simulation


# s0, T, K, p (risk neutral probability), u, d, N, j, sigma (for u)

N = 10000

class BinomialTreeModel:

    def __init__(self, r, ticker=None, sigma=0, t=0, s0=0, k=0):
        self.ticker = ticker
        self.r = r
        self.sigma = sigma
        self.t = t
        self.s0 = s0
        self.k = k

    def set_time_to_expiration(self):
        expirations = self.ticker.options
        expiration = expirations[0]
        et = pytz.timezone('US/Eastern')
        expiry = datetime.strptime(expiration, "%Y-%m-%d")
        expiry_et = et.localize(expiry)
        au = pytz.timezone('Australia/Sydney')
        expiry_au = expiry_et.astimezone(au)
        today_au = datetime.now(au)
        t = (expiry_au - today_au).days / 365
        self.t = t
        
    def set_s0(self):
        # date_str = "2025-09-15"
        # date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        # data = self.ticker.history(start=date_str, end=date_str)
        data = self.ticker.history(period='1d')
        s0 = data['Close'].iloc[0]
        self.s0 = s0
        
    def set_k(self):
        expirations = self.ticker.options
        expiration = expirations[0]
        option_chain = self.ticker.option_chain(expiration)
        calls = option_chain.calls
        strike_prices = calls['strike'].tolist()
        self.k = strike_prices[0]  
    
        
    def set_historical_volatility(self):
        data = self.ticker.history(period="1y")
        close_prices = data['Close']
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        std_devs = log_returns.std()
        annualised_vol = std_devs * np.sqrt(252)
        self.sigma = annualised_vol

    def get_up_and_down_factors(self, sigma, t):
        up = math.exp(sigma * math.sqrt(t/N))
        down = 1/up
        return [up, down]

    def get_risk_neutral_probabilities(self, u, d, t):
        dT = t / N
        p =  (np.exp(self.r * dT) - d) / (u - d)
        q = 1 - p
        return [p, q]
        

    # implement binomial tree using dynamic programming

    def get_call_option_price(self):
        # here we calculate C using dp
        option_dp = []
        u = self.get_up_and_down_factors(self.sigma, self.t)[0]
        d = self.get_up_and_down_factors(self.sigma, self.t)[1]
        p = self.get_risk_neutral_probabilities(u, d, self.t)[0]
        q = self.get_risk_neutral_probabilities(u, d, self.t)[1]
        dT = self.t / N
        
        for i in range(N + 1):
            row = []
            for j in range(N + 1):
                row.append(0.0)
            option_dp.append(row)
                
        stock = [[0.0 for j in range(i+1)] for i in range(N+1)]
        for i in range(N+1):
            for j in range(i+1):
                stock[i][j] = self.s0 * (u**j) * (d**(i-j))
                
        # fill from bottom up so calculate C_N,j first where j is the number of u steps
        # rows = time steps, cols = num of up moves
        for i in range(N + 1):
            option_dp[N][i] = max(0, stock[N][i] - self.k)
        
        for i in range(N - 1, -1, -1):
            for j in range(i + 1):
                option_dp[i][j] = math.exp(-self.r * dT) * (p * option_dp[i + 1][j + 1] + q * option_dp[i + 1][j])
        return option_dp[0][0]
        
if __name__ == "__main__":
    ticker = yf.Ticker("AAPL")
    model = BinomialTreeModel(0.05, ticker)
    model.set_s0()
    model.set_time_to_expiration()
    model.set_k()
    model.set_historical_volatility()
    price = model.get_call_option_price()
    print(price)