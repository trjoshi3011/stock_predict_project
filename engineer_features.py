"""Load Apple stock CSV, add all features and target; save processed file."""

import pandas as pd
import pandas_ta as ta

# Load AAPL data (skip schema rows)
df = pd.read_csv("apple_stock_data.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
close = df["Close"].astype(float)
volume = df["Volume"].astype(float)

# --- Momentum ---
df["RSI_14 (Momentum)"] = ta.rsi(close, length=14)
df["EMA_50 (Trend)"] = ta.ema(close, length=50)
df["Daily_Return (Volatility)"] = close.pct_change() * 100

# --- Volume: On-Balance Volume (OBV) ---
df["OBV (Volume)"] = ta.obv(close=close, volume=volume)

# --- Volatility: Bollinger Bands (20, 2) ---
bb = ta.bbands(close=close, length=20, lower_std=2, upper_std=2)
df["BBU_20_2 (Volatility)"] = bb["BBU_20_2_2"]
df["BBP_20_2 (Volatility)"] = bb["BBP_20_2_2"]

# --- Market context: SPY daily returns (from SPY_data.csv) ---
spy_df = pd.read_csv("SPY_data.csv", skiprows=[1, 2], index_col=0, parse_dates=True)
spy_returns = spy_df["Close"].astype(float).pct_change() * 100
spy_returns.name = "SPY_Daily_Return (Market Context)"
df = df.join(spy_returns)

# --- Trend reversal: MACD ---
macd_df = ta.macd(close=close, fast=12, slow=26, signal=9)
df["MACD (Trend Reversal)"] = macd_df["MACD_12_26_9"]
df["MACD_hist (Trend Reversal)"] = macd_df["MACDh_12_26_9"]

# Target: 1 if price 10 days ahead > today, else 0 (predicting 2-week price increase)
future_close = close.shift(-10)
df["Target (Binary Classification)"] = (future_close > close).astype(int)

# Drop rows with NaN (warm-up, last 10 rows, SPY alignment)
df = df.dropna()

# Save processed data
output_file = "processed_data.csv"
df.to_csv(output_file)

print(f"Processed {len(df)} rows. Saved to {output_file}.")
