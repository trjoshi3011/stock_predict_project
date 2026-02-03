# Apple Stock Predictor

### How to run:
1. **Pull Data:** Run `python get_data.py`
2. **Process Data:** Run `python engineer_features.py`
3. **Train/Evaluate:** Run `python train_model.py` (Outputs `model.joblib`)
4. **Backtest:** Run `python backtest.py`
5. **Visualize:** Run `streamlit dashboard.py`

### Current Performance:
- Window: 10 Days
- CV averages: Train set: Precision: 0.5486 Recall: 0.6514 F1: 0.5945 Balanced acc: 0.5539 Avg predicted positives per fold: 331.2 Test set: Precision: 0.5006 Recall: 0.7037 F1: 0.5794 Balanced acc: 0.4870 Avg predicted positives per fold: 125.2
- Feature Importance: RSI_14 (Momentum): 0.2345 EMA_50 (Trend): 0.1882 OBV (Volume): 0.1679 EMA_200 (Trend): 0.1076 BBU_20_2 (Volatility): 0.1037 MACD (Trend Reversal): 0.0991 MACD_hist (Trend Reversal): 0.0475 BBP_20_2 (Volatility): 0.0426 SPY_Daily_Return (Market Context): 0.0052 Daily_Return (Volatility): 0.0033 Above_EMA_200: 0.0004
- === Backtest Results === Initial balance: $10,000.00 Final balance: $13,365.63 Strategy return: +33.66% === Buy & Hold (first day to last day) === Buy & Hold final: $13,437.55 Buy & Hold return: +34.38%
