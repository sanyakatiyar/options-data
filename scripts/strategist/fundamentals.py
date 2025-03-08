# fundamentals.py

import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np


def get_fundamental_data(ticker_symbol: str) -> dict:
    """
    Retrieve fundamental data for the given ticker, including:
      - Current price, market cap, beta, next earnings date
      - 52-week high/low, dividend yield
    """
    ticker = yf.Ticker(ticker_symbol)
    today = datetime.now()

    # Set default/fallback values
    current_price = 0
    market_cap = 0
    beta = 'N/A'
    next_earnings = 'Unknown'
    fifty_two_week_high = 'N/A'
    fifty_two_week_low = 'N/A'
    dividend_yield = 'None'

    try:
        stock_info = ticker.info
        current_price = stock_info.get('currentPrice', stock_info.get('regularMarketPrice', 0))
        market_cap = stock_info.get('marketCap', 0)
        beta = stock_info.get('beta', 'N/A')
        fifty_two_week_high = stock_info.get('fiftyTwoWeekHigh', 'N/A')
        fifty_two_week_low = stock_info.get('fiftyTwoWeekLow', 'N/A')

        dy_raw = stock_info.get('dividendYield', 0)
        if dy_raw:
            dividend_yield = f"{round(dy_raw * 100, 2)}%"
        else:
            dividend_yield = 'None'

        # Attempt to find next earnings
        try:
            earnings_dates = ticker.earnings_dates
            if not earnings_dates.empty:
                next_earnings = earnings_dates.index[0].strftime('%Y-%m-%d')
        except:
            pass
    except:
        # fallback if .info is incomplete or fails
        hist = ticker.history(period='1d')
        if not hist.empty:
            current_price = hist['Close'].iloc[-1]

    return {
        "ticker": ticker_symbol.upper(),
        "analysis_date": today.strftime('%Y-%m-%d'),
        "current_price": current_price,
        "market_cap": market_cap,
        "beta": beta,
        "fifty_two_week_high": fifty_two_week_high,
        "fifty_two_week_low": fifty_two_week_low,
        "dividend_yield": dividend_yield,
        "next_earnings": next_earnings
    }


def get_historical_data(ticker_symbol: str, period: str = "1y") -> pd.DataFrame:
    """
    Retrieve historical price data for the ticker over the specified period.
    Default = 1 year. This can be used for additional computations like
    historical volatility, ATR, RSI, etc.
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        return pd.DataFrame()
    return hist


def compute_historical_volatility(hist_df: pd.DataFrame) -> float:
    """
    Compute annualized historical volatility from daily log returns.
    HV = std(log_returns) * sqrt(252)
    """
    if hist_df.empty:
        return None

    hist_df['log_ret'] = (hist_df['Close'] / hist_df['Close'].shift(1)).apply(lambda x: 0 if x <= 0 else np.log(x))
    stdev = hist_df['log_ret'].std(skipna=True)
    annual_hv = stdev * (252 ** 0.5)
    return annual_hv