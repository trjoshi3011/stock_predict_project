# Apple Stock Predictor

### How to run:
1. **Update Data:** Run `python engineer_features.py`
2. **Train/Evaluate:** Run `python train_model.py` (Outputs `model.joblib`)
3. **Visualize:** (Next step) Run `streamlit run app.py`

### Current Performance:
- Window: 10 Days
- Train Acc: 73%
- Test Acc: 44% (Time-Series Split)