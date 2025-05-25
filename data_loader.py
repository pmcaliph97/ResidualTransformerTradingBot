# data_loader.py
# Data loading, Kalman smoothing, and feature engineering

import pandas as pd
import numpy as np
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from pykalman import KalmanFilter
import alpaca_trade_api as tradeapi
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL)

def apply_kalman_filter(prices):
    kf = KalmanFilter(initial_state_mean=prices[0], n_dim_obs=1)
    state_means, _ = kf.smooth(prices)
    return state_means.flatten()

def fetch_data(stock_symbol, start_date, end_date):
    df = api.get_bars(stock_symbol, "1Day", start=start_date, end=end_date, feed="iex").df.copy()
    df.reset_index(inplace=True)
    df.rename(columns={"timestamp": "date", "close": "close", "volume": "volume"}, inplace=True)
    df = df[df["close"] > 0].dropna()
    df["Close_KF"] = apply_kalman_filter(df["close"].values)
    df["EMA10"] = EMAIndicator(close=df["Close_KF"], window=10).ema_indicator()
    df["EMA20"] = EMAIndicator(close=df["Close_KF"], window=20).ema_indicator()
    df["RSI"] = RSIIndicator(close=df["Close_KF"]).rsi()
    df["Previous_Close_KF"] = df["Close_KF"].shift(1)
    df["Target"] = df["Close_KF"].shift(-1)
    df.dropna(inplace=True)
    return df
