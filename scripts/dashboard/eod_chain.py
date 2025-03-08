import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.tools as tls


############

ticker = st.text_input("Enter stock ticker:", value=None, placeholder='e.g. NVDA, AAPL, AMZN')

# ticker = 'nvda' #TODO: this is temp var for easier testing...remove later

col = st.columns((4,4), gap='small')

if ticker is not None:

    ticker = ticker.upper()

    # get option chain and proc
    yfticker = yf.Ticker(ticker)

    expiration_dates = yfticker.options
    opt = yfticker.option_chain(expiration_dates[0]) #current exp
    calls = opt.calls
    puts = opt.puts
    
    calls = calls.sort_values(by='strike')
    puts = puts.sort_values(by='strike')
    
    ###########
    # create widget
    with col[0]:
        tradingview_widget = """
                <div class="tradingview-widget-container">
                    <div id="tradingview_chart"></div>
                    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
                    <script type="text/javascript">
                        new TradingView.widget({
                            "width": "100%",
                            "height": 700,
                            "symbol": "NASDAQ:""" + ticker + """",
                            "interval": "1",
                            "timezone": "Etc/UTC",
                            "theme": "dark",
                            "style": "1",
                            "locale": "en",
                            "toolbar_bg": "#f1f3f6",
                            "enable_publishing": false,
                            "hide_top_toolbar": false,
                            "save_image": false,
                            "container_id": "tradingview_chart"
                        });
                    </script>
                </div>
                """
        st.components.v1.html(tradingview_widget, height=700)

    with col[1]:
        
        col_inner = st.columns((4,4), gap='small')
        ATM = opt.underlying['regularMarketPrice']
        span_end = int((calls.strike-ATM).abs().argmin())

        with col_inner[0]:
            ########################################
            col_name='openInterest'
            max_oi = np.maximum(calls.openInterest.values.max(), puts.openInterest.values.max())

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=puts['strike'],
                y=puts['openInterest'].values,
                name='Puts',
                orientation='v',
                marker_color='#D9534F'
            ))

            fig.add_trace(go.Bar(
                x=calls['strike'],
                y=calls['openInterest'],
                name='Calls',
                orientation='v',
                marker_color='#00C66B'#, marker_opacity=0.5
            ))
            # Add vertical span using layout shapes (highlight region)
            fig.add_shape(
                type="rect",
                x0=0, x1=ATM,
                y0=0, y1=max_oi,
                fillcolor="#B39DDB",
                opacity=0.15,
                line_width=0,
                name='ITM Level'
            )

            fig.update_layout(
                title="Open Interest by Strike ($)",
                xaxis_title="Strike Price ($)",
                yaxis_title="Open Interest",
                barmode="overlay",
                template="plotly_dark",
                bargap=0.01,  # Control the gap between bars (smaller value = thicker bars)
                bargroupgap=0.01, # Control the gap between groups of bars (if stacked or grouped),
            )
            st.plotly_chart(fig)

        with col_inner[1]:
            ########################################
            col_name='volume'
            max_vol = np.maximum(calls.volume.values.max(), puts.volume.values.max())

            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=puts['strike'],
                y=puts['volume'],
                name='Puts',
                orientation='v',
                marker_color='#D9534F'
            ))

            fig2.add_trace(go.Bar(
                x=calls['strike'],
                y=calls['volume'],
                name='Calls',
                orientation='v',
                marker_color='#00C66B'#, marker_opacity=0.5
            ))
            
            fig2.add_shape(
                type="rect",
                x0=0, x1=ATM,
                y0=0, y1=max_vol,
                fillcolor="#B39DDB",
                opacity=0.15,
                line_width=0,
                name='ITM Level'
            )

            fig2.update_layout(
                title="Volume by Strike ($)",
                xaxis_title="Strike Price ($)",
                yaxis_title="Volume",
                barmode="overlay",
                template="plotly_dark",
                bargap=0.01,  # Control the gap between bars (smaller value = thicker bars)
                bargroupgap=0.01, # Control the gap between groups of bars (if stacked or grouped),
            )
            st.plotly_chart(fig2)

        ################################################
        st.divider()

        # plot IV bar chart (overlap calls and puts)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=calls.strike, y=calls['impliedVolatility']*100,  mode='lines+markers', name='Call IV', line=dict(color='#00C66B')))
        fig.add_trace(go.Scatter(x=puts.strike, y=puts['impliedVolatility']*100, mode='lines+markers', name='Put IV',  line=dict(color='#D9534F')))
        fig.update_layout(title="Implied Volatility (%) by Strike ($)", xaxis_title="Strike Price ($)", yaxis_title="IV (%)")
        st.plotly_chart(fig)
