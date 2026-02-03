"""Train a Random Forest classifier on AAPL processed data and evaluate it."""

import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

# Load the processed data (all features and target are already in the CSV)
df = pd.read_csv("processed_data.csv", index_col=0, parse_dates=True)

feature_cols = [
    "RSI_14 (Momentum)",
    "EMA_50 (Trend)",
    "Daily_Return (Volatility)",
    "OBV (Volume)",
    "BBU_20_2 (Volatility)",
    "BBP_20_2 (Volatility)",
    "SPY_Daily_Return (Market Context)",
    "MACD (Trend Reversal)",
    "MACD_hist (Trend Reversal)",
]
X = df[feature_cols]
y = df["Target (Binary Classification)"]

# Decision threshold: probability > this counts as "1" (less conservative)
DECISION_THRESHOLD = 0.40

# Time-based cross-validation: 5 folds (expanding train set)
tscv = TimeSeriesSplit(n_splits=5)
scaler = StandardScaler()
model = RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=15,
    class_weight="balanced_subsample",
)

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    # Scale using train set only (no leakage)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)

# Last fold: scale and predict using probability threshold (not default 0.5)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
proba_train = model.predict_proba(X_train_scaled)[:, 1]
proba_test = model.predict_proba(X_test_scaled)[:, 1]
y_pred_train = (proba_train > DECISION_THRESHOLD).astype(int)
y_pred_test = (proba_test > DECISION_THRESHOLD).astype(int)
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test, zero_division=0)

print("Last fold (fold 4) — Train vs Test:")
print("  Train Accuracy Score:", round(train_accuracy, 4))
print("  Test Accuracy Score:", round(test_accuracy, 4))
print("  Precision Score (test):", round(test_precision, 4))
print("\nConfusion Matrix (test set):")
print(confusion_matrix(y_test, y_pred_test))

# Feature Importance
print("\nFeature Importance:")
importance = pd.DataFrame(
    {"feature": feature_cols, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
for _, row in importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")
print("\nMost helpful indicator:", importance.iloc[0]["feature"])

# Save model, scaler, threshold, feature list, and last-fold test index for backtesting/app
joblib.dump(
    {
        "model": model,
        "scaler": scaler,
        "decision_threshold": DECISION_THRESHOLD,
        "feature_cols": feature_cols,
        "test_index": X.index[test_idx].tolist(),
    },
    "model.joblib",
)
print("\nModel saved to model.joblib.")
