import streamlit as st
import pandas as pd
import json
import os
import numpy as np
from dotenv import load_dotenv

# Import from your existing files
from main import build_compact_options_json, summarize_options_data, ask_llm_about_options

# Load environment variables for API keys
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Options Analysis Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        font-weight: 600;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 1.2rem;
        color: #9E9E9E;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 600;
        color: #FFFFFF;
    }
    .metric-value-green {
        color: #4CAF50;
    }
    .metric-value-red {
        color: #F44336;
    }
    .metric-value-blue {
        color: #2196F3;
    }
    .metric-value-amber {
        color: #FFC107;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2196F3;
        margin-top: 30px;
        margin-bottom: 20px;
    }
    .expiry-header {
        background-color: #232323;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 15px;
    }
    .analyst-section {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #333;
        margin-top: 30px;
    }
    .analyst-header {
        color: #2196F3;
        font-size: 1.8rem;
        margin-bottom: 15px;
    }
    .dataframe td {
        text-align: center;
    }
    .dataframe th {
        text-align: center;
        background-color: #232323;
    }
    .highlight-row {
        background-color: rgba(33, 150, 243, 0.1);
    }
    div[data-testid="stDataFrameResizable"] {
        padding: 10px;
        border-radius: 10px;
        background-color: #1E1E1E;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'options_data' not in st.session_state:
    st.session_state['options_data'] = None
if 'options_summary' not in st.session_state:
    st.session_state['options_summary'] = None

# Title and description
st.markdown('<div class="main-header">Options Analysis Dashboard</div>', unsafe_allow_html=True)
st.markdown("""
This dashboard provides comprehensive analysis of stock options, including key metrics like IV skew, Greeks, 
max pain levels, and put/call ratios. Explore the data across multiple expiration dates and get expert insights.
""")

# Sidebar for inputs
st.sidebar.header("Analysis Settings")

ticker = st.sidebar.text_input("Ticker Symbol", "AAPL").upper()
expirations_limit = st.sidebar.slider("Number of Expirations", min_value=1, max_value=10, value=3)
include_hv = st.sidebar.checkbox("Include Historical Volatility", value=True)

# Button to fetch options data
if st.sidebar.button("Analyze Options", key="analyze_button"):
    with st.spinner(f"Fetching options data for {ticker}..."):
        try:
            data = build_compact_options_json(ticker, expirations_limit, include_hv)
            summary = summarize_options_data(data)
            
            st.session_state['options_data'] = data
            st.session_state['options_summary'] = summary
            
            st.sidebar.success(f"Successfully retrieved options data for {ticker}")
        except Exception as e:
            st.sidebar.error(f"Error retrieving options data: {str(e)}")

# Function to format large numbers
def format_large_number(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)

# Display the options data if available
if st.session_state['options_data']:
    data = st.session_state['options_data']
    summary = st.session_state['options_summary']
    
    # Check for errors
    if "error" in data:
        st.error(data["error"])
    else:
        # Display fundamental data with custom formatting
        st.markdown(f'<div class="section-header">{data["ticker"]} Options Analysis ({data["analysis_date"]})</div>', unsafe_allow_html=True)
        
        # Main metrics in a 4-column grid with custom styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-label">Price</div>
                <div class="metric-value metric-value-green">${:,.2f}</div>
            </div>
            """.format(data['price']), unsafe_allow_html=True)
            
        with col2:
            # Format market cap properly
            market_cap = format_large_number(data['market_cap'])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Market Cap</div>
                <div class="metric-value">{market_cap}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Beta</div>
                <div class="metric-value metric-value-amber">{data['beta']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            pcr_color = "metric-value-green" if data['put_call_ratio'] < 1 else "metric-value-red"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Put/Call Ratio</div>
                <div class="metric-value {pcr_color}">{data['put_call_ratio']}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional metrics in a 3-column grid
        st.markdown('<div class="section-header">Additional Information</div>', unsafe_allow_html=True)
        
        fund_col1, fund_col2, fund_col3 = st.columns(3)
        
        with fund_col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">52-Week High</div>
                <div class="metric-value metric-value-green">{data['fifty_two_week_high']}</div>
            </div>
            <br>
            <div class="metric-card">
                <div class="metric-label">52-Week Low</div>
                <div class="metric-value metric-value-red">{data['fifty_two_week_low']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with fund_col2:
            dividend = data['dividend_yield']
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Dividend Yield</div>
                <div class="metric-value metric-value-green">{dividend}</div>
            </div>
            <br>
            <div class="metric-card">
                <div class="metric-label">Next Earnings</div>
                <div class="metric-value">{data['next_earnings']}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with fund_col3:
            if "historical_volatility" in data and data["historical_volatility"] is not None:
                hv = data["historical_volatility"]
                hv_color = "metric-value-amber" if hv > 20 else "metric-value-blue"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Historical Volatility (1y)</div>
                    <div class="metric-value {hv_color}">{hv}%</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Show options data for each expiration
        st.markdown('<div class="section-header">Options Expirations</div>', unsafe_allow_html=True)
        
        # Create tabs for each expiration date
        if data["expirations"]:
            expiration_dates = list(data["expirations"].keys())
            tabs = st.tabs(expiration_dates)
            
            for i, exp in enumerate(expiration_dates):
                exp_data = data["expirations"][exp]
                with tabs[i]:
                    # Expiration header with DTE
                    st.markdown(f"""
                    <div class="expiry-header">
                        <h3>Expiration: {exp} <span style="color: #9E9E9E;">(DTE: {exp_data['days_to_expiry']})</span></h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Key metrics
                    metrics_col1, metrics_col2 = st.columns(2)
                    
                    with metrics_col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Max Pain Strike</div>
                            <div class="metric-value metric-value-blue">${exp_data['max_pain_strike']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with metrics_col2:
                        if exp_data["iv_skew_calls"]:
                            skew = exp_data["iv_skew_calls"]
                            st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">IV Skew (Calls)</div>
                                <div style="display: flex; justify-content: space-between;">
                                    <div><span style="color: #9E9E9E;">ITM:</span> <span style="color: #4CAF50;">{skew['itm']}%</span></div>
                                    <div><span style="color: #9E9E9E;">ATM:</span> <span style="color: #FFC107;">{skew['atm']}%</span></div>
                                    <div><span style="color: #9E9E9E;">OTM:</span> <span style="color: #F44336;">{skew['otm']}%</span></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Calls and Puts tables
                    calls_col, puts_col = st.columns(2)
                    
                    with calls_col:
                        st.markdown('<h4 style="text-align: center; color: #4CAF50;">Calls (Near ATM)</h4>', unsafe_allow_html=True)
                        if exp_data["calls"]:
                            calls_df = pd.DataFrame(exp_data["calls"])
                            
                            # Format columns
                            calls_df['iv'] = calls_df['iv'].apply(lambda x: f"{x}%")
                            calls_df['delta'] = calls_df['delta'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                            calls_df['gamma'] = calls_df['gamma'].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
                            calls_df['theta'] = calls_df['theta'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                            calls_df['vega'] = calls_df['vega'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                            
                            # Color code rows by moneyness without using style.apply
                            # Set background colors manually for different moneyness levels
                            calls_df_display = calls_df.copy()
                            
                            # Display dataframe with better formatting
                            st.dataframe(
                                calls_df_display,
                                column_config={
                                    "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                                    "moneyness": st.column_config.TextColumn("Moneyness"),
                                    "last": st.column_config.NumberColumn("Last", format="$%.2f"),
                                    "bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                                    "ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                                    "iv": st.column_config.TextColumn("IV"),
                                    "delta": st.column_config.TextColumn("Delta"),
                                    "gamma": st.column_config.TextColumn("Gamma"),
                                    "theta": st.column_config.TextColumn("Theta"),
                                    "vega": st.column_config.TextColumn("Vega"),
                                    "open_interest": st.column_config.NumberColumn("Open Int"),
                                    "volume": st.column_config.NumberColumn("Volume")
                                },
                                use_container_width=True
                            )
                        else:
                            st.info("No call options data available")
                    
                    with puts_col:
                        st.markdown('<h4 style="text-align: center; color: #F44336;">Puts (Near ATM)</h4>', unsafe_allow_html=True)
                        if exp_data["puts"]:
                            puts_df = pd.DataFrame(exp_data["puts"])
                            
                            # Format columns
                            puts_df['iv'] = puts_df['iv'].apply(lambda x: f"{x}%")
                            puts_df['delta'] = puts_df['delta'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                            puts_df['gamma'] = puts_df['gamma'].apply(lambda x: f"{x:.4f}" if x is not None else "N/A")
                            puts_df['theta'] = puts_df['theta'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                            puts_df['vega'] = puts_df['vega'].apply(lambda x: f"{x:.3f}" if x is not None else "N/A")
                            
                            # Display dataframe with better formatting
                            st.dataframe(
                                puts_df,
                                column_config={
                                    "strike": st.column_config.NumberColumn("Strike", format="$%.2f"),
                                    "moneyness": st.column_config.TextColumn("Moneyness"),
                                    "last": st.column_config.NumberColumn("Last", format="$%.2f"),
                                    "bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
                                    "ask": st.column_config.NumberColumn("Ask", format="$%.2f"),
                                    "iv": st.column_config.TextColumn("IV"),
                                    "delta": st.column_config.TextColumn("Delta"),
                                    "gamma": st.column_config.TextColumn("Gamma"),
                                    "theta": st.column_config.TextColumn("Theta"),
                                    "vega": st.column_config.TextColumn("Vega"),
                                    "open_interest": st.column_config.NumberColumn("Open Int"),
                                    "volume": st.column_config.NumberColumn("Volume")
                                },
                                use_container_width=True
                            )
                        else:
                            st.info("No put options data available")
        
        # Options Strategy Advisor Section
        st.markdown('<div class="section-header">Options Strategy Advisor</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="analyst-section">
            <div class="analyst-header">Ask our AI Options Strategist</div>
            <p>Ask a question about the options data to get expert insights and potential strategies.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form(key="llm_form"):
            query = st.text_area(
                "Your question:",
                height=100,
                placeholder="Example: What's the implied volatility skew telling us? Is there unusual activity in any strikes?"
            )
            # Fixed token limit at 1000
            submit_button = st.form_submit_button("Let's make money!")
        
        if submit_button and query:
            with st.spinner("Analyzing options data..."):
                try:
                    llm_response = ask_llm_about_options(summary, query, max_tokens=1000)
                    st.markdown("""
                    <div class="analyst-section">
                        <div class="analyst-header">Analysis</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(llm_response)
                except Exception as e:
                    st.error(f"Error generating analysis: {str(e)}")
                    st.error("Make sure you have set your API key in the .env file (OPENROUTER_API_KEY or OPENAI_API_KEY)")
        
        # Option to download the raw JSON data
        if st.sidebar.button("Download Analysis Data"):
            json_str = json.dumps(data, indent=2)
            st.sidebar.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"{data['ticker']}_options_analysis.json",
                mime="application/json"
            )
else:
    st.info("Enter a ticker symbol and click 'Analyze Options' to get started.")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This dashboard uses real-time market data from yfinance to analyze options and provide strategic insights.
    
    """
)