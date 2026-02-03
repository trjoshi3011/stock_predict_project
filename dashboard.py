"""Stock dashboard: price chart with model buy signals and metrics."""

import joblib
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Stock Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modern UI: custom CSS
st.markdown(
    """
    <style>
    /* Clean base */
    .stApp { background: linear-gradient(180deg, #0f1419 0%, #1a1f2e 100%); }
    h1, h2, h3, .stMarkdown { color: #e7e9ea !important; }
    /* Hero */
    .hero { padding: 1.5rem 0; margin-bottom: 1.25rem; }
    .ticker { font-size: 2rem; font-weight: 700; letter-spacing: -0.03em; color: #e7e9ea; }
    .price { font-size: 1.85rem; font-weight: 700; color: #00ba7c; }
    .price-down { color: #f4212e; }
    .sub { font-size: 0.85rem; color: #71767b; margin-top: 0.2rem; }
    /* Metric cards — uniform size, centered text */
    .card {
        background: rgba(29, 41, 54, 0.6);
        border: 1px solid #38444d;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        height: 7rem;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    .card-label { font-size: 0.75rem; color: #71767b; text-transform: uppercase; letter-spacing: 0.06em; }
    .card-value { font-size: 1.35rem; font-weight: 600; color: #e7e9ea; margin-top: 0.2rem; line-height: 1.3; overflow: hidden; text-overflow: ellipsis; }
    .card-value.green { color: #00ba7c; }
    .card-value.red { color: #f4212e; }
    .card .sub { min-height: 1.25rem; font-size: 0.85rem; color: #71767b; margin-top: 0.2rem; }
    .section { font-size: 0.9rem; font-weight: 600; color: #71767b; margin-bottom: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar: ticker selection
with st.sidebar:
    st.markdown("### Symbol")
    ticker = st.selectbox(
        "Ticker",
        options=["AAPL", "SPY", "MSFT", "GOOGL"],
        index=0,
        label_visibility="collapsed",
    )
    st.caption("Model buy signals available for AAPL only.")

has_model = ticker == "AAPL"

if has_model:
    try:
        artifact = joblib.load("model.joblib")
        model = artifact["model"]
        decision_threshold = artifact.get("decision_threshold", 0.5)
        feature_cols = artifact["feature_cols"]
        df = pd.read_csv("processed_data.csv", index_col=0, parse_dates=True)
    except FileNotFoundError as e:
        st.error(f"Run `python train_model.py` first. ({e})")
        st.stop()

    X = df[feature_cols]
    y = df["Target (Binary Classification)"]
    proba = model.predict_proba(X)[:, 1]
    predictions = (proba > decision_threshold).astype(int)
    df = df.copy()
    df["pred"] = predictions
    price_df = df[["Close", "pred"]].sort_index()
else:
    import yfinance as yf
    from datetime import datetime, timedelta
    end = datetime.now()
    start = end - timedelta(days=2 * 365)
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if data.empty or len(data) == 0:
        st.warning(f"No data for {ticker}.")
        st.stop()
    close_col = data["Close"].squeeze() if isinstance(data.columns, pd.MultiIndex) else data["Close"]
    price_df = pd.DataFrame({"Close": close_col, "pred": 0}).sort_index()

# Hero: ticker + price
latest = price_df["Close"].iloc[-1]
prev = price_df["Close"].iloc[-2] if len(price_df) > 1 else latest
chg_pct = (latest - prev) / prev * 100 if prev else 0
price_class = "price" if chg_pct >= 0 else "price-down"

st.markdown(
    f'<div class="hero">'
    f'<div class="ticker">{ticker}</div>'
    f'<div class="{price_class}">${latest:,.2f}</div>'
    f'<div class="sub">{chg_pct:+.2f}% from prior close</div>'
    f'</div>',
    unsafe_allow_html=True,
)

# Line chart + Buy markers (Plotly)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=price_df.index,
        y=price_df["Close"],
        mode="lines",
        name="Price",
        line=dict(color="#1d9bf0", width=2),
        fill="tozeroy",
        fillcolor="rgba(29, 155, 240, 0.1)",
    )
)

if has_model and price_df["pred"].sum() > 0:
    buy_mask = price_df["pred"] == 1
    buy_dates = price_df.index[buy_mask]
    buy_prices = price_df.loc[buy_mask, "Close"]
    fig.add_trace(
        go.Scatter(
            x=buy_dates,
            y=buy_prices,
            mode="markers",
            name="Buy",
            marker=dict(
                symbol="triangle-up",
                size=10,
                color="#00ba7c",
                line=dict(width=1.5, color="#0d9668"),
            ),
        )
    )

fig.update_layout(
    margin=dict(t=20, b=20, l=44, r=20),
    height=400,
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(
        showgrid=True,
        gridcolor="rgba(56, 68, 77, 0.5)",
        zeroline=False,
        showline=False,
        tickfont=dict(color="#71767b", size=10),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(56, 68, 77, 0.5)",
        zeroline=False,
        showline=False,
        tickfont=dict(color="#71767b", size=10),
        tickprefix="$",
    ),
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(color="#71767b", size=10),
        bgcolor="rgba(0,0,0,0)",
    ),
    showlegend=bool(has_model and price_df["pred"].sum() > 0),
)

st.plotly_chart(fig, width = "stretch")

# Metrics dashboard
st.markdown('<p class="section">Metrics Dashboard</p>', unsafe_allow_html=True)

if has_model:
    if "test_index" in artifact:
        test_index = artifact["test_index"]
        X_test = X.loc[test_index]
        y_test = y.loc[test_index]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    proba_test = model.predict_proba(X_test)[:, 1]
    y_pred = (proba_test > decision_threshold).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    # Backtest
    INITIAL_BALANCE = 10_000
    test_df = df.loc[X_test.index].copy()
    test_df["pred"] = y_pred
    test_df = test_df.sort_index()

    cash, shares = INITIAL_BALANCE, 0.0
    equity_curve = []
    for _, row in test_df.iterrows():
        close, pred = row["Close"], row["pred"]
        if pred == 1 and shares == 0:
            shares = cash / close
            cash = 0.0
        elif pred == 0 and shares > 0:
            cash = shares * close
            shares = 0.0
        equity_curve.append(cash + shares * close)

    final_balance = cash + shares * test_df["Close"].iloc[-1]
    first_close = test_df["Close"].iloc[0]
    last_close = test_df["Close"].iloc[-1]
    buy_hold_final = INITIAL_BALANCE * (last_close / first_close)
    strategy_return_pct = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    buy_hold_return_pct = (buy_hold_final - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    equity_series = pd.Series(equity_curve)
    peak = equity_series.cummax()
    drawdown = (peak - equity_series) / peak
    max_drawdown_pct = drawdown.max() * 100

    imp = pd.Series(model.feature_importances_, index=feature_cols)
    top_feature = imp.idxmax()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="card">'
            f'<div class="card-label">Model Accuracy</div>'
            f'<div class="card-value">{accuracy:.1%}</div>'
            f'<div class="sub">&nbsp;</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col2:
        strat_cls = "green" if strategy_return_pct >= 0 else "red"
        st.markdown(
            f'<div class="card">'
            f'<div class="card-label">Strategy (Backtest)</div>'
            f'<div class="card-value {strat_cls}">${final_balance:,.0f}</div>'
            f'<div class="sub">{strategy_return_pct:+.2f}% · Max DD {max_drawdown_pct:.1f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col3:
        bh_cls = "green" if buy_hold_return_pct >= 0 else "red"
        st.markdown(
            f'<div class="card">'
            f'<div class="card-label">Buy & Hold</div>'
            f'<div class="card-value {bh_cls}">${buy_hold_final:,.0f}</div>'
            f'<div class="sub">{buy_hold_return_pct:+.2f}%</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="card">'
            f'<div class="card-label">Most Helpful Indicator</div>'
            f'<div class="card-value" style="font-size: 0.9rem;">{top_feature}</div>'
            f'<div class="sub">&nbsp;</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("Choose **AAPL** in the sidebar to see model accuracy, backtest results, and top feature importance.")
