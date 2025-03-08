# json_packaging.py

import pandas as pd
from datetime import datetime

from fundamentals import (
    get_fundamental_data,
    get_historical_data,
    compute_historical_volatility
)
from options_chain import (
    get_option_expirations,
    fetch_option_chain,
    calculate_greeks,
    compute_put_call_ratio
)
from advanced_options import calculate_max_pain, compute_iv_skew


def build_compact_options_json(ticker_symbol: str,
                               expirations_limit: int = 3,
                               include_hv: bool = False) -> dict:
    """
    1. Fetch fundamentals for the ticker
    2. Choose up to `expirations_limit` option expirations
    3. Compute overall put/call ratio
    4. For each expiration:
       - fetch calls/puts
       - compute max pain
       - compute IV skew
       - compute Greeks for a few near-ATM strikes
    5. Optionally compute historical volatility if `include_hv` is True
    Returns a compact JSON structure for strategy analysis.
    """

    # 1. Fundamentals
    f_data = get_fundamental_data(ticker_symbol)
    current_price = f_data["current_price"]
    if not current_price:
        return {"error": f"Unable to retrieve price for {ticker_symbol}"}

    # 2. Option expirations
    all_exps = get_option_expirations(ticker_symbol)
    if not all_exps:
        return {"error": f"No options data available for {ticker_symbol}"}
    exps_used = all_exps[:expirations_limit]

    # 3. Overall put/call ratio
    pcr = compute_put_call_ratio(ticker_symbol, exps_used)

    # 4. Build JSON
    data = {
        "ticker": f_data["ticker"],
        "analysis_date": f_data["analysis_date"],
        "price": current_price,
        "market_cap": f_data["market_cap"],
        "beta": f_data["beta"],
        "fifty_two_week_high": f_data["fifty_two_week_high"],
        "fifty_two_week_low": f_data["fifty_two_week_low"],
        "dividend_yield": f_data["dividend_yield"],
        "next_earnings": f_data["next_earnings"],
        "put_call_ratio": pcr,
        "expirations": {}
    }

    for exp in exps_used:
        calls_df, puts_df = fetch_option_chain(ticker_symbol, exp)
        if calls_df.empty and puts_df.empty:
            continue

        # days to expiry
        dte = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.now().date()).days

        # max pain
        mp_strike = calculate_max_pain(calls_df, puts_df)

        # IV skew (calls)
        skew = compute_iv_skew(calls_df, current_price)

        # Subset calls/puts around ATM to keep size small
        calls_compact = _extract_near_atm(calls_df, 'c', current_price, dte)
        puts_compact = _extract_near_atm(puts_df, 'p', current_price, dte)

        data["expirations"][exp] = {
            "days_to_expiry": dte,
            "max_pain_strike": mp_strike,
            "iv_skew_calls": skew,
            "calls": calls_compact,
            "puts": puts_compact
        }

    # 5. Optionally compute & store historical volatility
    if include_hv:
        hist_df = get_historical_data(ticker_symbol, period="1y")
        hv = compute_historical_volatility(hist_df)
        if hv is not None:
            data["historical_volatility"] = round(hv * 100, 2)  # as a percentage
        else:
            data["historical_volatility"] = None

    return data


def _extract_near_atm(df: pd.DataFrame, option_type: str,
                      current_price: float, dte: int) -> list:
    """
    Internal helper that picks a handful of strikes near ATM and
    computes Greeks + classification (ITM/ATM/OTM).
    """
    if df.empty:
        return []

    df["distance"] = (df["strike"] - current_price).abs()
    atm_idx = df["distance"].idxmin()

    # Sort by strike, pick 2 below & 2 above
    df_sorted = df.sort_values("strike")
    if pd.isna(atm_idx):
        return []

    atm_strike = df.loc[atm_idx, "strike"]
    below = df_sorted[df_sorted["strike"] < atm_strike].iloc[::-1]  # descending
    above = df_sorted[df_sorted["strike"] >= atm_strike]

    # pick a subset
    subset = pd.concat([
        below.head(2).iloc[::-1],  # re-sort ascending
        above.head(3)  # ATM plus next 2
    ]).drop_duplicates()

    results = []
    for _, row in subset.iterrows():
        s = float(row["strike"])
        iv = float(row["impliedVolatility"])
        (dl, gm, th, vg) = calculate_greeks(option_type, current_price, s, dte, iv)

        # Moneyness
        if option_type == 'c':
            if s < current_price * 0.98:
                m = "ITM"
            elif s > current_price * 1.02:
                m = "OTM"
            else:
                m = "ATM"
        else:  # put
            if s > current_price * 1.02:
                m = "ITM"
            elif s < current_price * 0.98:
                m = "OTM"
            else:
                m = "ATM"

        results.append({
            "strike": s,
            "moneyness": m,
            "last": round(float(row["lastPrice"]), 2),
            "bid": round(float(row["bid"]), 2),
            "ask": round(float(row["ask"]), 2),
            "iv": round(iv * 100, 2),
            "delta": dl,
            "gamma": gm,
            "theta": th,
            "vega": vg,
            "open_interest": int(row["openInterest"]),
            "volume": int(row["volume"])
        })

    return results