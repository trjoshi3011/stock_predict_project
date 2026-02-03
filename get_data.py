"""Download Apple and S&P 500 (SPY) stock data for the last 2 years and save to CSV."""

import yfinance as yf
from datetime import datetime, timedelta

# Date range: 2 years ago to today
end_date = datetime.now()
start_date = end_date - timedelta(days=5 * 365)

# Download Apple (AAPL) and save to CSV
aapl = yf.download("AAPL", start=start_date, end=end_date, progress=False)
aapl.to_csv("apple_stock_data.csv")
print(f"Downloaded {len(aapl)} days of Apple (AAPL) stock data to apple_stock_data.csv.")

# Download S&P 500 (SPY) and save to CSV
spy = yf.download("SPY", start=start_date, end=end_date, progress=False)
spy.to_csv("SPY_data.csv")
print(f"Downloaded {len(spy)} days of S&P 500 (SPY) data to SPY_data.csv.")
