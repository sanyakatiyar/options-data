# main.py

import argparse
import json

# If using an LLM
from openai import OpenAI
from dotenv import load_dotenv
import os

from json_packaging import build_compact_options_json

# Simple textual summary
def summarize_options_data(data: dict) -> str:
    if "error" in data:
        return data["error"]

    lines = []
    lines.append(f"Ticker: {data['ticker']}  Date: {data['analysis_date']}")
    lines.append(f"Price: ${data['price']}   Market Cap: {data['market_cap']}   Beta: {data['beta']}")
    lines.append(f"52W High: {data['fifty_two_week_high']}  52W Low: {data['fifty_two_week_low']}")
    lines.append(f"Dividend Yield: {data['dividend_yield']}  Next Earnings: {data['next_earnings']}")
    lines.append(f"Put/Call Ratio: {data['put_call_ratio']}")

    if "historical_volatility" in data:
        hv = data["historical_volatility"]
        if hv is not None:
            lines.append(f"Historical Volatility (1y): {hv:.2f}%")

    lines.append("")

    for exp, exp_data in data["expirations"].items():
        lines.append(f"EXPIRATION: {exp}  (DTE = {exp_data['days_to_expiry']})")
        lines.append(f"   Max Pain: {exp_data['max_pain_strike']}")
        if exp_data["iv_skew_calls"]:
            skew = exp_data["iv_skew_calls"]
            lines.append(f"   IV Skew (Calls) - ITM: {skew['itm']}%, ATM: {skew['atm']}%, OTM: {skew['otm']}%")

        lines.append("   CALLS near ATM:")
        for c in exp_data["calls"]:
            lines.append(f"     Strike={c['strike']} ({c['moneyness']}) IV={c['iv']}%, "
                         f"Delta={c['delta']}, Gamma={c['gamma']}, Theta={c['theta']}, Vega={c['vega']}, "
                         f"OI={c['open_interest']}, Vol={c['volume']}")
        lines.append("   PUTS near ATM:")
        for p in exp_data["puts"]:
            lines.append(f"     Strike={p['strike']} ({p['moneyness']}) IV={p['iv']}%, "
                         f"Delta={p['delta']}, Gamma={p['gamma']}, Theta={p['theta']}, Vega={p['vega']}, "
                         f"OI={p['open_interest']}, Vol={p['volume']}")
        lines.append("")

    return "\n".join(lines)


def ask_llm_about_options(summary_text: str, user_query: str,
                          max_tokens: int = 1000) -> str:
    """
    Demonstration of how you might integrate an LLM to answer user questions
    about the option data. 
    (Requires valid OpenRouter or OpenAI credentials in environment.)
    """
    system_prompt = (
        "You are an good options strategist. "
        f"Answer the user's question based on the following data, "
        f"and keep your response under {max_tokens} tokens.\n"
    )

    user_content = f"OPTION DATA:\n\n{summary_text}\n\nQUESTION: {user_query}"

    # For demonstration with openrouter.ai, or you can use openai.com 
    try:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")  # or OPENAI_API_KEY
        if not api_key:
            return "Error: No API key found. Please set OPENROUTER_API_KEY or OPENAI_API_KEY."

        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://options-analysis-tool.com",
                "X-Title": "OptionsAnalysisLLM",
            },
            model="deepseek/deepseek-chat:free",  # or another available model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=max_tokens
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"Error making LLM request: {str(e)}"


def main():
    parser = argparse.ArgumentParser(description="Modular Options Analysis with Additional Metrics")
    parser.add_argument("ticker", type=str, help="Stock ticker symbol (e.g. AAPL, MSFT)")
    parser.add_argument("--expirations", type=int, default=3, help="Number of expiration dates to include")
    parser.add_argument("--hv", action="store_true", help="Include historical volatility in the data")
    parser.add_argument("--tokens", type=int, default=1000, help="Max tokens for LLM responses")
    parser.add_argument("--interactive", action="store_true", help="Enter interactive LLM query mode")
    args = parser.parse_args()

    # Build the data
    data = build_compact_options_json(args.ticker, expirations_limit=args.expirations, include_hv=args.hv)
    summary = summarize_options_data(data)

    print("=== OPTIONS SUMMARY ===\n")
    print(summary)

    # Optional interactive LLM queries
    if args.interactive and "error" not in data:
        print("Enter your questions about the above data (type 'quit' to exit):\n")
        while True:
            user_q = input("> ")
            if user_q.lower() in ('quit', 'exit', 'q'):
                break
            if not user_q.strip():
                print("Please enter a question or type 'quit' to exit.")
                continue

            print("\nAnalyzing with LLM...")
            ans = ask_llm_about_options(summary, user_q, max_tokens=args.tokens)
            print("\n=== LLM RESPONSE ===\n")
            print(ans)
            print("\n---------------------\n")


if __name__ == "__main__":
    main()