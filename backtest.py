"""Backtest the trained Random Forest model vs Buy & Hold."""

import joblib
import pandas as pd

# Load the trained model, threshold, and feature list
artifact = joblib.load("model.joblib")
model = artifact["model"]
decision_threshold = artifact.get("decision_threshold", 0.5)
feature_cols = artifact["feature_cols"]

# Load processed data
df = pd.read_csv("processed_data.csv", index_col=0, parse_dates=True)
X = df[feature_cols]
y = df["Target (Binary Classification)"]

# Use last-fold test set if saved (TimeSeriesSplit), else 80/20 random split
if "test_index" in artifact:
    test_index = artifact["test_index"]
    X_test = X.loc[test_index]
else:
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# Predict on test set (use probability threshold if saved)
proba = model.predict_proba(X_test)[:, 1]
predictions = (proba > decision_threshold).astype(int)
test_df = df.loc[X_test.index].copy()
test_df["pred"] = predictions
test_df = test_df.sort_index()  # chronological order for backtest

# --- Strategy backtest: buy on every "1", sell on "0" ---
INITIAL_BALANCE = 10_000
cash = INITIAL_BALANCE
shares = 0.0
equity_curve = []

for date, row in test_df.iterrows():
    close = row["Close"]
    pred = row["pred"]

    if pred == 1 and shares == 0:
        # Buy: spend all cash at today's close
        shares = cash / close
        cash = 0.0
    elif pred == 0 and shares > 0:
        # Sell: convert all shares to cash at today's close
        cash = shares * close
        shares = 0.0

    equity = cash + shares * close
    equity_curve.append(equity)

# Final balance (in case we end holding shares)
final_close = test_df["Close"].iloc[-1]
final_balance = cash + shares * final_close

# --- Buy & Hold: buy on first test day, hold to end ---
first_close = test_df["Close"].iloc[0]
last_close = test_df["Close"].iloc[-1]
buy_hold_final = INITIAL_BALANCE * (last_close / first_close)

# --- Maximum Drawdown (biggest drop from a peak) ---
equity_series = pd.Series(equity_curve)
peak = equity_series.cummax()
drawdown = (peak - equity_series) / peak
max_drawdown = drawdown.max()

# --- Results ---
strategy_return_pct = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
buy_hold_return_pct = (buy_hold_final - INITIAL_BALANCE) / INITIAL_BALANCE * 100

print("=== Backtest Results ===")
print(f"Initial balance:     ${INITIAL_BALANCE:,.2f}")
print(f"Final balance:       ${final_balance:,.2f}")
print(f"Strategy return:     {strategy_return_pct:+.2f}%")
print()
print("=== Buy & Hold (first day to last day) ===")
print(f"Buy & Hold final:    ${buy_hold_final:,.2f}")
print(f"Buy & Hold return:   {buy_hold_return_pct:+.2f}%")
print()
print("=== Risk ===")
print(f"Maximum Drawdown:    {max_drawdown * 100:.2f}%")
