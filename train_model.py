import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# Load the processed data (all features and target are already in the CSV)
df = pd.read_csv("processed_data.csv", index_col=0, parse_dates=True)

feature_cols = [
    "RSI_14 (Momentum)",
    "EMA_50 (Trend)",
    "EMA_200 (Trend)",
    "Above_EMA_200",
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

# Per-fold threshold: min threshold with precision >= 0.40 and recall >= 0.08, else 0.5
MIN_PRECISION = 0.40
MIN_RECALL = 0.08
FALLBACK_THRESHOLD = 0.5

# Time-based cross-validation: 5 folds (expanding train set)
tscv = TimeSeriesSplit(n_splits=5)
model = RandomForestClassifier(
    random_state=42,
    n_estimators=300,
    max_depth=3,
    min_samples_leaf=30,
    min_samples_split=50,
    max_features=0.5,
    class_weight="balanced",
)

fold_summaries = []
chosen_threshold = FALLBACK_THRESHOLD
last_train_idx = None
last_test_idx = None

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    model.fit(X_train, y_train)

    proba_test = model.predict_proba(X_test)[:, 1]
    precision_curve, recall_curve, thresholds = precision_recall_curve(
        y_test, proba_test
    )
    valid = [
        t
        for i, t in enumerate(thresholds)
        if precision_curve[i] >= MIN_PRECISION and recall_curve[i] >= MIN_RECALL
    ]
    chosen_threshold = min(valid) if valid else FALLBACK_THRESHOLD

    y_pred_test = (proba_test > chosen_threshold).astype(int)
    if "Above_EMA_200" in X_test.columns:
        mask_below_ema = (X_test["Above_EMA_200"] == 0).to_numpy()
        n_suppressed = int(((y_pred_test == 1) & mask_below_ema).sum())
        y_pred_test[mask_below_ema] = 0
        print(f"Regime gating: suppressed {n_suppressed} predicted buy(s) (Below EMA 200).")
    n_positives_test = int((y_pred_test == 1).sum())

    # Training-set predictions (same threshold + regime gating)
    proba_train = model.predict_proba(X_train)[:, 1]
    y_pred_train = (proba_train > chosen_threshold).astype(int)
    if "Above_EMA_200" in X_train.columns:
        mask_below_ema_train = (X_train["Above_EMA_200"] == 0).to_numpy()
        y_pred_train[mask_below_ema_train] = 0
    n_positives_train = int((y_pred_train == 1).sum())

    # Test metrics
    prec_test = precision_score(y_test, y_pred_test, zero_division=0)
    rec_test = recall_score(y_test, y_pred_test, zero_division=0)
    f1_test = f1_score(y_test, y_pred_test, zero_division=0)
    bal_acc_test = balanced_accuracy_score(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)

    # Train metrics
    prec_train = precision_score(y_train, y_pred_train, zero_division=0)
    rec_train = recall_score(y_train, y_pred_train, zero_division=0)
    f1_train = f1_score(y_train, y_pred_train, zero_division=0)
    bal_acc_train = balanced_accuracy_score(y_train, y_pred_train)
    cm_train = confusion_matrix(y_train, y_pred_train)

    fold_summaries.append(
        {
            "precision_train": prec_train,
            "recall_train": rec_train,
            "f1_train": f1_train,
            "balanced_accuracy_train": bal_acc_train,
            "n_positives_train": n_positives_train,
            "precision": prec_test,
            "recall": rec_test,
            "f1": f1_test,
            "balanced_accuracy": bal_acc_test,
            "n_positives": n_positives_test,
        }
    )

    print(f"\n--- Fold {fold} (threshold = {chosen_threshold:.4f}) ---")
    print("  Train set:")
    print(f"    Confusion matrix:\n{cm_train}")
    print(f"    Precision: {prec_train:.4f}  Recall: {rec_train:.4f}  F1: {f1_train:.4f}  Balanced acc: {bal_acc_train:.4f}")
    print(f"    Predicted positives: {n_positives_train}")
    print("  Test set:")
    print(f"    Confusion matrix:\n{cm_test}")
    print(f"    Precision: {prec_test:.4f}  Recall: {rec_test:.4f}  F1: {f1_test:.4f}  Balanced acc: {bal_acc_test:.4f}")
    print(f"    Predicted positives: {n_positives_test}")

    last_train_idx, last_test_idx = train_idx, test_idx

# Averages across folds
print("\n" + "=" * 50)
print("CV averages:")
n_folds = len(fold_summaries)
print("  Train set:")
avg_prec_train = sum(s["precision_train"] for s in fold_summaries) / n_folds
avg_rec_train = sum(s["recall_train"] for s in fold_summaries) / n_folds
avg_f1_train = sum(s["f1_train"] for s in fold_summaries) / n_folds
avg_bal_train = sum(s["balanced_accuracy_train"] for s in fold_summaries) / n_folds
avg_pos_train = sum(s["n_positives_train"] for s in fold_summaries) / n_folds
print(f"    Precision: {avg_prec_train:.4f}  Recall: {avg_rec_train:.4f}  F1: {avg_f1_train:.4f}  Balanced acc: {avg_bal_train:.4f}")
print(f"    Avg predicted positives per fold: {avg_pos_train:.1f}")
print("  Test set:")
avg_prec = sum(s["precision"] for s in fold_summaries) / n_folds
avg_rec = sum(s["recall"] for s in fold_summaries) / n_folds
avg_f1 = sum(s["f1"] for s in fold_summaries) / n_folds
avg_bal = sum(s["balanced_accuracy"] for s in fold_summaries) / n_folds
avg_pos = sum(s["n_positives"] for s in fold_summaries) / n_folds
print(f"    Precision: {avg_prec:.4f}  Recall: {avg_rec:.4f}  F1: {avg_f1:.4f}  Balanced acc: {avg_bal:.4f}")
print(f"    Avg predicted positives per fold: {avg_pos:.1f}")

# Refit final model on last train window (same as last fold train), then save
X_train_final = X.iloc[last_train_idx]
y_train_final = y.iloc[last_train_idx]
model.fit(X_train_final, y_train_final)
test_idx = last_test_idx



#--------------------------------#  This is all good #--------------------------------#  
# Feature Importance
print("\nFeature Importance:")
importance = pd.DataFrame(
    {"feature": feature_cols, "importance": model.feature_importances_}
).sort_values("importance", ascending=False)
for _, row in importance.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")
print("\nMost helpful indicator:", importance.iloc[0]["feature"])

# Save model, last-fold threshold, feature list, and last-fold test index for backtesting/app
joblib.dump(
    {
        "model": model,
        "decision_threshold": chosen_threshold,
        "feature_cols": feature_cols,
        "test_index": X.index[test_idx].tolist(),
    },
    "model.joblib",
)
print("\nModel saved to model.joblib.")
