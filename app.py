"""
app.py
------
Main entry point for the BTC Forecasting Portal.

Run with:
    streamlit run app.py
"""

import sys
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# Make src/ importable when running from project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_utils    import parse_btc_csv, backtest_split, calc_metrics
from src.prophet_model import run_prophet
from src.arima_model   import run_arima
from src.charts        import build_forecast_chart


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="BTC Forecasting Portal",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
code, .stCode { font-family: 'Space Mono', monospace; }

.stApp { background: #0a0a0f; color: #e8e8f0; }
.block-container { padding-top: 2rem; }

[data-testid="stSidebar"] { background: #0f0f1a; border-right: 1px solid #f7931a33; }
[data-testid="stSidebar"] * { color: #e8e8f0 !important; }

div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #f7931a44;
    border-radius: 12px;
    padding: 1rem;
}
div[data-testid="metric-container"] label {
    color: #f7931a !important; font-size: 0.75rem;
    letter-spacing: 0.1em; text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important; font-family: 'Space Mono', monospace;
}

/* Better metric cards for backtest section */
.bt-card {
    background: linear-gradient(135deg, #12121f 0%, #1a1a2e 100%);
    border: 1px solid #f7931a44;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.5rem;
}
.bt-label {
    color: #f7931a;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.bt-value {
    color: #ffffff;
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
}
.bt-sub {
    color: #888;
    font-size: 0.75rem;
    margin-top: 0.2rem;
}

h1 { font-family: 'Syne', sans-serif !important; font-weight: 800; color: #f7931a !important; letter-spacing: -0.02em; }
h2, h3 { font-family: 'Syne', sans-serif !important; color: #e8e8f0 !important; }

.stButton > button {
    background: linear-gradient(135deg, #f7931a, #e8650a);
    color: #0a0a0f; border: none; border-radius: 8px;
    font-family: 'Syne', sans-serif; font-weight: 700;
    letter-spacing: 0.05em; padding: 0.6rem 2rem;
    width: 100%; transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 24px #f7931a44; }

hr { border-color: #f7931a33 !important; }
.stAlert { border-radius: 10px; }
[data-testid="stFileUploader"] { border: 2px dashed #f7931a44 !important; border-radius: 12px; background: #12121f; }
.js-plotly-plot, .plotly, .main-svg { touch-action: none !important; }
[data-testid="stPlotlyChart"] { overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ₿ Configuration")
    st.markdown("---")

    model_choice = st.selectbox(
        "Forecasting Model",
        ["Prophet", "ARIMA"],
        help="Prophet handles seasonality well; ARIMA is a classic time-series model.",
    )

    horizon = st.slider(
        "Forecast Horizon (days)", min_value=7, max_value=180, value=30, step=1
    )

    ci_pct = st.select_slider(
        "Confidence Interval", options=[80, 90, 95], value=95
    )
    ci = ci_pct / 100

    st.markdown("---")
    st.markdown("**Technical Indicators**")
    show_sma20 = st.toggle("SMA 20", value=True)
    show_sma50 = st.toggle("SMA 50", value=False)
    show_ema20 = st.toggle("EMA 20", value=False)

    st.markdown("---")
    st.markdown("**Display**")
    show_volume = st.toggle("Show Volume (if available)", value=True)

    st.markdown("---")
    st.caption("Bitcoin Price Forecasting Portal · v2.0")
    st.caption("Built with Streamlit · Prophet · ARIMA · Plotly")


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_backtest_chart(
    test: pd.Series,
    pred_bt: np.ndarray,
    model_name: str,
) -> go.Figure:
    """
    Render actual vs predicted prices over the backtest window.
    Both series are in USD — y-axis is labelled accordingly.
    """
    n = min(len(test), len(pred_bt))
    actual_vals = test.values[:n]
    pred_vals   = pred_bt[:n]
    dates        = test.index[:n]
    errors       = actual_vals - pred_vals          # signed USD error per day

    fig = go.Figure()

    # Actual
    fig.add_trace(go.Scatter(
        x=dates, y=actual_vals,
        name="Actual Price",
        line=dict(color="#f7931a", width=2),
        hovertemplate="Actual: $%{y:,.2f}<extra></extra>",
    ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=dates, y=pred_vals,
        name=f"{model_name} Prediction",
        line=dict(color="#00f2ff", width=2, dash="dot"),
        hovertemplate="Predicted: $%{y:,.2f}<extra></extra>",
    ))

    # Error band (shaded area between actual and predicted)
    fig.add_trace(go.Scatter(
        x=list(dates) + list(dates[::-1]),
        y=list(actual_vals) + list(pred_vals[::-1]),
        fill="toself",
        fillcolor="rgba(255,100,100,0.08)",
        line=dict(width=0),
        name="Error Band",
        hoverinfo="skip",
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="rgba(0,0,0,0)",
        height=340,
        margin=dict(l=10, r=10, t=36, b=10),
        hovermode="x unified",
        yaxis=dict(
            title="Price (USD)",
            gridcolor="#1e1e2e",
            tickprefix="$",
            tickformat=",.0f",
        ),
        xaxis=dict(gridcolor="#1e1e2e"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(
            text=f"Backtest Window — Actual vs {model_name} Predicted (USD)",
            font=dict(size=13, color="#e8e8f0"),
        ),
    )

    return fig


def calc_mape(actual: pd.Series, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%). Returns NaN if actuals contain zeros."""
    n = min(len(actual), len(predicted))
    a = actual.values[:n]
    p = np.array(predicted)[:n]
    mask = a != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("# ₿ Bitcoin Price Forecasting Portal")
st.markdown(
    "Upload a Kaggle-style BTC CSV, configure your model, and generate an interactive forecast."
)
st.markdown("---")

# ── File Upload ───────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload BTC Historical CSV",
    type=["csv"],
    help="Supports Kaggle BTC datasets (Date/Close columns)",
)

if uploaded is None:
    st.info(
        "👆 Upload a CSV file to get started. "
        "You can download BTC historical data from "
        "[Kaggle](https://www.kaggle.com/datasets/gamzegedik044/bitcoin-daily-price-20152025)."
    )
    st.stop()

# ── Parse ─────────────────────────────────────────────────────────────────────
try:
    raw_df = pd.read_csv(uploaded)
    df, price_cols = parse_btc_csv(raw_df)
except Exception as e:
    st.error(
        f"❌ Could not parse the file: {e}\n\n"
        "Please upload a CSV with Date and Close (or Open/High/Low) columns."
    )
    st.stop()

if len(df) < 60:
    st.warning("⚠️ Dataset is very short (< 60 rows). Forecast accuracy may be low.")

# ── Price column selector ─────────────────────────────────────────────────────
col_left, col_right = st.columns([2, 1])
with col_left:
    price_col = st.selectbox("Price Column to Forecast", price_cols)
with col_right:
    st.metric("Total Rows Loaded", f"{len(df):,}")

price_series = df.set_index("Date")[price_col].dropna()

# ── Data preview ──────────────────────────────────────────────────────────────
with st.expander("🔍 Preview Raw Data", expanded=False):
    st.dataframe(df.head(20), width='content')

st.markdown("---")

# ── Summary metrics ───────────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Price",  f"${price_series.iloc[-1]:,.2f}")
m2.metric("All-Time High", f"${price_series.max():,.2f}")
m3.metric("All-Time Low",  f"${price_series.min():,.2f}")
date_range = (price_series.index[-1] - price_series.index[0]).days
m4.metric("Date Range",    f"{date_range:,} days")

st.markdown("---")

# ── Generate button ───────────────────────────────────────────────────────────
generate = st.button("⚡ Generate Forecast")
if generate:
    with st.spinner(f"Training {model_choice} model and generating {horizon}-day forecast…"):
        try:
            train, test = backtest_split(price_series)

            if model_choice == "Prophet":
                fc_bt      = run_prophet(train, len(test), ci)
                pred_bt    = fc_bt["yhat"].values[-len(test):]
                
                fc_full    = run_prophet(price_series, horizon, ci)
                future_dates = pd.date_range(
                    price_series.index[-1] + timedelta(days=1), periods=horizon
                )
                fc_vals = fc_full["yhat"].values[-horizon:]
                fc_lo   = fc_full["yhat_lower"].values[-horizon:]
                fc_hi   = fc_full["yhat_upper"].values[-horizon:]

            else:  # ARIMA (Unified Static Approach)
                
                # 1. Static Backtest (Apples-to-Apples with Future)
                fc_bt_mean, _, _ = run_arima(train, horizon=len(test), ci=ci, is_backtest=True)
                pred_bt = fc_bt_mean.values
                
                # 2. Standard Forward Forecast
                future_dates = pd.date_range(
                    price_series.index[-1] + timedelta(days=1), periods=horizon
                )
                fc_series_f, lo_f, hi_f = run_arima(price_series, horizon, ci, is_backtest=False)
                
                fc_vals = fc_series_f.values
                fc_lo   = lo_f.values
                fc_hi   = hi_f.values

            # ── Compute all backtest metrics in USD ───────────────────────────
            mae, rmse = calc_metrics(test, pred_bt)
            mape      = calc_mape(test, pred_bt)

            # Mean actual price over the test window (used for % context)
            mean_actual = float(test.mean())

            # USD range context: what fraction of price is the error?
            mae_pct_of_price  = (mae  / mean_actual) * 100 if mean_actual else float("nan")
            rmse_pct_of_price = (rmse / mean_actual) * 100 if mean_actual else float("nan")

        except Exception as e:
            st.error(f"❌ Forecasting error: {e}")
            st.stop()
    # ═══════════════════════════════════════════════════════════════════════════
    #  BACKTEST PERFORMANCE SECTION
    # ═══════════════════════════════════════════════════════════════════════════

    st.markdown("### 📊 Backtest Performance")
    st.caption(
        f"Evaluated on the last **{len(test):,} days** of historical data "
        f"(≈ 10% hold-out). All error values are in **US Dollars (USD)**."
    )

    # ── Metric cards ──────────────────────────────────────────────────────────
    bt1, bt2, bt3, bt4 = st.columns(4)

    with bt1:
        st.metric(
            label="MAE — Mean Absolute Error (USD)",
            value=f"${mae:,.2f}",
            help=(
                "On average, the model's daily price prediction was off by this "
                "many US dollars during the backtest window."
            ),
        )
        st.caption(f"≈ {mae_pct_of_price:.1f}% of avg test price")

    with bt2:
        st.metric(
            label="RMSE — Root Mean Sq. Error (USD)",
            value=f"${rmse:,.2f}",
            help=(
                "RMSE penalises large individual errors more than MAE. "
                "Expressed in US dollars — same unit as the price."
            ),
        )
        st.caption(f"≈ {rmse_pct_of_price:.1f}% of avg test price")

    with bt3:
        mape_display = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
        st.metric(
            label="MAPE — Mean Abs. Pct. Error",
            value=mape_display,
            help=(
                "Percentage version of MAE — unit-free, useful for comparing "
                "across different price scales."
            ),
        )
        st.caption("lower is better")

    with bt4:
        st.metric(
            label="Backtest Window",
            value=f"{len(test):,} days",
            help="Number of days used for out-of-sample evaluation.",
        )
        st.caption(
            f"{test.index[0].strftime('%d %b %Y')} → "
            f"{test.index[-1].strftime('%d %b %Y')}"
        )

    # ── Backtest chart: actual vs predicted ───────────────────────────────────
    bt_fig = build_backtest_chart(test, pred_bt, model_choice)
    st.plotly_chart(
        bt_fig,
        use_container_width=True,                     # ← correct
        config={'scrollZoom': True, 'displayModeBar': True},
        key="main_forecast_chart",
    )
    # ── Metric explanation ────────────────────────────────────────────────────
    with st.expander("ℹ️ How to interpret these metrics", expanded=False):
        st.markdown(f"""
**MAE = ${mae:,.2f}**
> The model's predictions were off by **${mae:,.2f} on average** per day during 
> the test period. This is a straightforward USD error — if MAE is \\$500, the 
> model was typically \\$500 away from the real price each day.

**RMSE = ${rmse:,.2f}**
> RMSE is also in USD. Because it squares errors before averaging, a single 
> bad day has an outsized effect. An RMSE much larger than the MAE signals 
> that the model has occasional large misses even if it's usually accurate.

**MAPE = {mape_display}**
> A percentage view of the same error. A MAPE of 5% means predictions were 
> within 5% of the actual price on average.

**Rule of thumb:** for highly volatile assets like BTC, a MAPE under 5–10% 
is considered good. The USD figures (MAE / RMSE) let you judge whether the 
error is economically meaningful relative to the current price of 
**${price_series.iloc[-1]:,.2f}**.
        """)

    st.markdown("---")

    # ── Forecast end date banner ──────────────────────────────────────────────
    st.markdown("### 🔮 Forward Forecast")
    st.caption(
        f"Model: **{model_choice}** · Horizon: **{horizon} days** · "
        f"Confidence interval: **{ci_pct}%** · "
        f"Forecast ends: **{future_dates[-1].strftime('%d %b %Y')}**"
    )

    # ── Forecast chart ────────────────────────────────────────────────────────
    fig = build_forecast_chart(
        price_series=price_series,
        future_dates=future_dates,
        fc_vals=fc_vals,
        fc_lo=fc_lo,
        fc_hi=fc_hi,
        ci_pct=ci_pct,
        show_sma20=show_sma20,
        show_sma50=show_sma50,
        show_ema20=show_ema20,
        show_volume=show_volume,
        df=df,
    )
    st.plotly_chart(
        fig,                                          
        use_container_width=True,                  
        config={'scrollZoom': True, 'displayModeBar': True},
        key="backtest_result_chart",
    )
    # ── Forecast table ────────────────────────────────────────────────────────
    with st.expander("📋 Forecast Data Table", expanded=False):
        fc_df = pd.DataFrame({
            "Date":                  list(future_dates),
            "Forecast (USD)":        [f"${v:,.2f}" for v in fc_vals],
            f"Lower {ci_pct}% (USD)":[f"${v:,.2f}" for v in fc_lo],
            f"Upper {ci_pct}% (USD)":[f"${v:,.2f}" for v in fc_hi],
        })
        st.dataframe(fc_df, width="stretch", hide_index=True)