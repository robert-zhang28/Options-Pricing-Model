import math
import numpy as np
import matplotlib as plt
import yfinance as yf
import binomial_model as bm
import black_scholes as bs
from datetime import datetime
from scipy.stats import norm


class VolatilitySurface:
    
    def __init__(self):
        pass
    
    
    def plot_curve(self, t, k):
        time_to_expiry_axis = np.linspace(0, t)
        strike_axis = np.linspace(0, k)



