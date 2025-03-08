# options_chain.py

import yfinance as yf
import pandas as pd
from datetime import datetime

# Attempt to import py_vollib for Greeks
try:
    from py_vollib.black_scholes.greeks.analytical import delta, gamma, theta, vega
except ImportError:
    delta = gamma = theta = vega = None
    print("Warning: py_vollib is not installed. Greeks calculations will be unavailable.")

RISK_FREE_RATE = 0.05


def get_option_expirations(ticker_symbol: str) -> list:
    """
    Return a sorted list of available option expirations (YYYY-MM-DD).
    """
    ticker = yf.Ticker(ticker_symbol)
    if not ticker.options:
        return []
    raw_exps = ticker.options
    dt_exps = [datetime.strptime(e, "%Y-%m-%d") for e in raw_exps]
    dt_exps.sort()
    return [d.strftime("%Y-%m-%d") for d in dt_exps]


def fetch_option_chain(ticker_symbol: str, expiration: str):
    """
    Return (calls_df, puts_df) for the specified expiration.
    Each is a pandas DataFrame from yfinance.
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        chain = ticker.option_chain(expiration)
        return chain.calls, chain.puts
    except:
        return pd.DataFrame(), pd.DataFrame()


def calculate_greeks(option_type: str,
                     underlying_price: float,
                     strike: float,
                     days_to_expiry: int,
                     implied_vol: float) -> tuple:
    """
    Calculate (delta, gamma, theta, vega) using py_vollib, if installed.
    Otherwise, returns (None, None, None, None).
    """
    if not all([delta, gamma, theta, vega]):
        return None, None, None, None

    time_to_expiry = max(days_to_expiry / 365.0, 0.001)
    flag = option_type.lower()[0]  # 'c' or 'p'

    try:
        d = delta(flag, underlying_price, strike, time_to_expiry, RISK_FREE_RATE, implied_vol)
        g = gamma(flag, underlying_price, strike, time_to_expiry, RISK_FREE_RATE, implied_vol)
        t = theta(flag, underlying_price, strike, time_to_expiry, RISK_FREE_RATE, implied_vol) / 365.0
        v = vega(flag, underlying_price, strike, time_to_expiry, RISK_FREE_RATE, implied_vol) / 100.0
        return round(d, 3), round(g, 4), round(t, 3), round(v, 3)
    except:
        return None, None, None, None


def compute_put_call_ratio(ticker_symbol: str, expirations: list) -> float:
    """
    Aggregate open interest across multiple expirations
    to compute an overall put/call ratio. E.g. sum of put OI / sum of call OI.
    """
    ticker = yf.Ticker(ticker_symbol)
    total_call_oi = 0
    total_put_oi = 0
    for e in expirations:
        try:
            chain = ticker.option_chain(e)
            total_call_oi += chain.calls['openInterest'].sum()
            total_put_oi += chain.puts['openInterest'].sum()
        except:
            pass

    if total_call_oi == 0:
        return 0.0
    return round(total_put_oi / total_call_oi, 2)