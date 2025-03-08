# Options Analysis Dashboard

An advanced options analysis tool that provides comprehensive insights and visualizations for stock options data. This dashboard uses real-time market data to analyze options metrics, including Greeks, IV skew, max pain levels, and put/call ratios, and provides strategic insights through an AI assistant.

![Dashboard Screenshot](https://via.placeholder.com/800x400?text=Options+Analysis+Dashboard)

## Features

- **Real-time Options Data Analysis**: Fetch and analyze current options data for any publicly traded stock
- **Comprehensive Metrics**: View key options metrics including:
  - Greeks (Delta, Gamma, Theta, Vega)
  - Implied Volatility Skew
  - Max Pain Analysis
  - Put/Call Ratio
  - Historical Volatility
- **Multiple Expiration Analysis**: Compare options data across different expiration dates
- **AI-Powered Strategy Insights**: Ask questions about the options data and receive strategic analysis
- **Beautiful Visualization**: Intuitive and visually appealing dashboard interface
- **Downloadable Data**: Export analysis data in JSON format for further analysis

## Installation

### Prerequisites

- Python 3.8+
- Pip package manager

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/options-analysis-dashboard.git
   cd options-analysis-dashboard
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Set up API keys for LLM integration:
   - Create a `.env` file in the project root directory:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   # or
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

### Running the Dashboard

Start the Streamlit app:
```bash
streamlit run app.py
```

This will open the dashboard in your default web browser at `http://localhost:8501`.

### Analyzing Options Data

1. Enter a ticker symbol (e.g., AAPL, MSFT, TSLA) in the sidebar
2. Select the number of expiration dates to analyze
3. Check/uncheck "Include Historical Volatility" as needed
4. Click "Analyze Options" to fetch and display the data

### Understanding the Dashboard

The dashboard is organized into several sections:

1. **Main Metrics**: At the top, you'll see the current price, market cap, beta, and put/call ratio
2. **Additional Information**: The next section displays 52-week high/low, dividend yield, next earnings date, and historical volatility
3. **Options Expirations**: Options data is organized by expiration date in tabs
   - For each expiration:
     - Max Pain Strike: The price at which option sellers would have minimal payout
     - IV Skew: The difference in implied volatility between ITM, ATM, and OTM options
     - Calls/Puts Tables: Detailed data for near-ATM options, including Greeks

### Using the AI Options Strategist

At the bottom of the dashboard, you'll find the "Options Strategy Advisor" section:

1. Type your question about the displayed options data
2. Click "Let's make money!" to receive an AI-generated analysis
3. Review the strategic insights provided

Example questions:
- "What's the implied volatility skew telling us about market sentiment?"
- "Is there unusual activity in any strikes?"
- "Would a iron condor strategy make sense with this IV profile?"
- "What option strategies would be appropriate given the current IV and put/call ratio?"

## Project Structure

- `app.py`: The main Streamlit application
- `main.py`: Core functions for options analysis and AI integration
- `options_chain.py`: Functions for fetching and analyzing options chains
- `advanced_options.py`: Advanced metrics calculations (max pain, IV skew)
- `fundamentals.py`: Functions for fetching fundamental stock data
- `json_packaging.py`: Functions for packaging analysis data in JSON format
- `requirements.txt`: Package dependencies

## Technical Details

### Data Sources

This dashboard uses [yfinance](https://github.com/ranaroussi/yfinance) to fetch real-time market data from Yahoo Finance, including:

- Stock price and fundamental data
- Options chains for various expirations
- Historical price data for volatility calculations

### Calculations

- **Max Pain**: Calculated by determining the strike price where option holders (both puts and calls) would lose the most money upon expiration
- **IV Skew**: Measures the difference in implied volatility across different strike prices
- **Greeks**: Calculated using the Black-Scholes model via py_vollib (if installed)
- **Historical Volatility**: Calculated from log returns over the past year, annualized

### AI Integration

The dashboard uses OpenRouter or OpenAI to provide strategic insights based on the options data. The AI model analyzes the compiled options data and responds to user questions with tailored strategic advice.

## Requirements

The following Python packages are required:

- streamlit
- pandas
- yfinance
- numpy
- python-dotenv
- openai
- py-vollib (optional, for Greeks calculations)

## Troubleshooting

### Common Issues

1. **No Options Data Available**: Some stocks may not have options, or the options data might be unavailable temporarily
2. **API Key Errors**: Ensure your OpenRouter or OpenAI API key is correctly set in the `.env` file
3. **Missing Greeks**: If py_vollib is not installed, Greek calculations will show as "N/A"

### Getting Help

If you encounter issues with the dashboard, please:
1. Check the console for error messages
2. Verify your API keys and internet connection
3. Ensure you have all required dependencies installed

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [yfinance](https://github.com/ranaroussi/yfinance) for providing easy access to Yahoo Finance data
- [Streamlit](https://streamlit.io/) for the excellent web app framework
- [py_vollib](https://github.com/vollib/py_vollib) for options Greeks calculations
- [OpenRouter](https://openrouter.ai/) and [OpenAI](https://openai.com/) for AI capabilities
