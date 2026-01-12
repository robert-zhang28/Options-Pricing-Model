import math
import numpy as np
import yfinance as yf
import pytz
from datetime import datetime
from scipy.stats import norm
import pandas as pd

#TODO: fix the way to select s0, k, and expiration dates (mb)

class BlackScholesModel:
    
    def __init__(self, r, ticker=None, sigma=0, t=0, s0=0, k=0):
        self.ticker = ticker
        self.r = r
        self.sigma = sigma
        self.t = t
        self.s0 = s0
        self.k = k
        self.calls = None

    def set_time_to_expiration(self):
        expirations = self.ticker.options
        expiration = expirations[0]
        #print(expirations)
        et = pytz.timezone('US/Eastern')
        expiry = datetime.strptime(expiration, "%Y-%m-%d")
        expiry_et = et.localize(expiry)
        au = pytz.timezone('Australia/Sydney')
        expiry_au = expiry_et.astimezone(au)
        today_au = datetime.now(au)
        t = (expiry_au - today_au).days / 365
        self.t = t
        
    def set_s0(self):
        try:
            # Get the most recent closing price
            hist = self.ticker.history(period="1d")
            self.s0 = hist['Close'].iloc[-1]
        except Exception as e:
            print(f"Error fetching stock price: {e}")
            # Fallback to the last available price
            hist = self.ticker.history()
            if not hist.empty:
                self.s0 = hist['Close'].iloc[-1]
            else:
                raise ValueError("Could not fetch stock price data")
        
    def set_calls(self):
        expirations = self.ticker.options
        all_calls = []
        for exp in expirations:
            chain = self.ticker.option_chain(exp)
            calls = chain.calls.copy()
            calls['expiration'] = exp
            all_calls.append(calls)
        all_calls_df = pd.concat(all_calls, ignore_index=True)
        self.calls = all_calls_df
        
        
    def set_k(self):
        expirations = self.ticker.options
        expiration = expirations[0]
        print(expirations)
        option_chain = self.ticker.option_chain(expiration)
        # print(option_chain)
        calls = option_chain.calls
        strike_prices = calls['strike'].tolist()
        print(calls[['strike', 'lastPrice', 'impliedVolatility']].head())
        print(strike_prices)
        self.k = min(strike_prices, key=lambda x: abs(x - self.s0))
    
        
    def set_historical_volatility(self):
        data = self.ticker.history(period="1y")
        close_prices = data['Close']
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
        std_devs = log_returns.std()
        annualised_vol = std_devs * np.sqrt(252)
        self.sigma = annualised_vol

        
    def get_call_option_price(self, sigma=None):
        if sigma is None:
            sigma = self.sigma 
        d1 = (math.log(self.s0 / self.k) + (self.r + (((sigma) ** 2) / 2)) * self.t) / (sigma * math.sqrt(self.t))
        d2 = d1 - sigma * math.sqrt(self.t)
        c = norm.cdf(d1) * self.s0 - norm.cdf(d2) * self.k * math.exp(-self.r * self.t)
        return c
    
    def get_vega(self, sigma=None):
        if sigma is None:
            sigma = self.sigma 
        d1 = (math.log(self.s0 / self.k) + (self.r + (((sigma) ** 2) / 2)) * self.t) / (sigma * math.sqrt(self.t))
        return self.s0 * math.sqrt(self.t) * norm.pdf(d1) 
    
    def solve_for_iv(self, market_price, tol=0.00001):
        max_iterations = 1000
        vol_old = 0.2
        for i in range(max_iterations):
            bs_price = self.get_call_option_price(vol_old)
            vega = self.get_vega(vol_old)
            c = bs_price - market_price
            vol_new = vol_old - c / vega
            if (abs(vol_old - vol_new) < tol):
                return vol_new
            vol_old = vol_new
        return np.nan
    
if __name__ == "__main__":
    ticker = yf.Ticker("AAPL")
    model = BlackScholesModel(0.05, ticker)
    model.set_s0()
    model.set_time_to_expiration()
    model.set_k()
    model.set_historical_volatility()
    price = model.get_call_option_price()
    print(price)