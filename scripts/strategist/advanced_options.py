# advanced_options.py

import pandas as pd

def calculate_max_pain(calls_df: pd.DataFrame, puts_df: pd.DataFrame) -> float:
    """
    Simplistic 'Max Pain' calculation:
    1. Combine all strikes from calls & puts
    2. For each strike, estimate 'pain' if underlying expires exactly at that strike
       (i.e., in-the-money calls above strike, puts below strike)
    3. Return strike with minimal total payoff (lowest 'pain' for option buyers).
    """
    if calls_df.empty or puts_df.empty:
        return None

    all_strikes = sorted(set(calls_df['strike']) | set(puts_df['strike']))
    results = []

    for strike in all_strikes:
        # calls in-the-money if strike < underlying
        # for a naive approach, sum openInterest for calls with strike < this strike
        # Similarly for puts with strike > this strike
        # This is not the exact net payoff, but a typical approximation for demonstration
        in_money_calls = calls_df[calls_df['strike'] < strike]['openInterest'].sum()
        in_money_puts = puts_df[puts_df['strike'] > strike]['openInterest'].sum()

        total_pain = in_money_calls + in_money_puts
        results.append((strike, total_pain))

    if not results:
        return None

    results.sort(key=lambda x: x[1])  # sort by total_pain ascending
    return results[0][0]  # the strike with the minimal combined "pain"


def compute_iv_skew(calls_df: pd.DataFrame, current_price: float) -> dict:
    """
    Quick measure of calls' IV skew for ITM, ATM, OTM.
    For puts, you could do similarly or combine them as desired.
    """
    if calls_df.empty:
        return {}

    sorted_calls = calls_df.sort_values('strike')
    # ATM
    atm_idx = (sorted_calls['strike'] - current_price).abs().idxmin()
    atm_iv = round(sorted_calls.loc[atm_idx, 'impliedVolatility'] * 100, 2)

    # ITM: calls with strike < current_price
    itm_calls = sorted_calls[sorted_calls['strike'] < current_price]
    itm_iv = round(itm_calls['impliedVolatility'].iloc[-1] * 100, 2) if not itm_calls.empty else None

    # OTM: calls with strike > current_price
    otm_calls = sorted_calls[sorted_calls['strike'] > current_price]
    otm_iv = round(otm_calls['impliedVolatility'].iloc[0] * 100, 2) if not otm_calls.empty else None

    return {"itm": itm_iv, "atm": atm_iv, "otm": otm_iv}