"""
src/charts.py
-------------
Plotly chart builders for the BTC Forecasting Portal.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_utils import compute_indicators, parse_volume


INDICATOR_COLORS = {
    "SMA_20": "#06d6a0",
    "SMA_50": "#118ab2",
    "EMA_20": "#ef476f",
}


def build_forecast_chart(
    price_series: pd.Series,
    future_dates,
    fc_vals,
    fc_lo,
    fc_hi,
    ci_pct: int,
    show_sma20: bool,
    show_sma50: bool,
    show_ema20: bool,
    show_volume: bool,
    df: pd.DataFrame,
) -> go.Figure:

    # -------------------------
    # Ensure clean datetime index
    # -------------------------
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").set_index("Date")

    price_series = price_series.sort_index()
    price_series = price_series[~price_series.index.duplicated()]

    # -------------------------
    # Indicators
    # -------------------------
    indicators = compute_indicators(price_series)

    # -------------------------
    # Volume handling
    # -------------------------
    vol_col = next((c for c in df.columns if "vol" in c.lower()), None)
    has_volume = show_volume and vol_col is not None

    # -------------------------
    # Subplots
    # -------------------------
    rows = 2 if has_volume else 1
    row_heights = [0.7, 0.3] if has_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
    )

    # -------------------------
    # 1. Historical price
    # -------------------------
    fig.add_trace(
        go.Scatter(
            x=price_series.index,
            y=price_series.values,
            name="Historical Price",
            line=dict(color="#f7931a", width=2),
            fill="tozeroy",
            fillcolor="rgba(247,147,26,0.05)",
            hovertemplate="Price: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # -------------------------
    # 2. Indicators
    # -------------------------
    show_map = {
        "SMA_20": show_sma20,
        "SMA_50": show_sma50,
        "EMA_20": show_ema20,
    }

    for key, vals in indicators.items():
        if show_map.get(key, False):
            fig.add_trace(
                go.Scatter(
                    x=vals.index,
                    y=vals.values,
                    name=key,
                    line=dict(color=INDICATOR_COLORS.get(key, "#aaa"), width=1.2),
                    opacity=0.8,
                    hovertemplate=f"{key}: $%{{y:,.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # -------------------------
    # Forecast data
    # -------------------------
    fc_dates = list(future_dates)
    fc_vals_list = list(fc_vals)
    fc_lo_list = list(fc_lo)
    fc_hi_list = list(fc_hi)

    # -------------------------
    # 3. Forecast region shading (background, added before lines so it sits behind)
    # -------------------------
    fig.add_vrect(
        x0=price_series.index[-1],
        x1=fc_dates[-1],
        fillcolor="rgba(0,242,255,0.03)",
        layer="below",
        line_width=0,
    )

    # -------------------------
    # 4. Confidence band (filled polygon)
    # -------------------------
    fig.add_trace(
        go.Scatter(
            x=fc_dates + fc_dates[::-1],
            y=fc_hi_list + fc_lo_list[::-1],
            fill="toself",
            fillcolor="rgba(247,147,26,0.12)",
            line=dict(width=0),
            name=f"{ci_pct}% Confidence Band",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )

    # Upper bound
    fig.add_trace(
        go.Scatter(
            x=fc_dates,
            y=fc_hi_list,
            name=f"Upper {ci_pct}%",
            mode="lines",
            line=dict(color="rgba(247,147,26,0.6)", width=1.2, dash="dot"),
            hovertemplate=f"Upper {ci_pct}%: $%{{y:,.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Lower bound
    fig.add_trace(
        go.Scatter(
            x=fc_dates,
            y=fc_lo_list,
            name=f"Lower {ci_pct}%",
            mode="lines",
            line=dict(color="rgba(247,147,26,0.6)", width=1.2, dash="dot"),
            hovertemplate=f"Lower {ci_pct}%: $%{{y:,.2f}}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Forecast mean line
    fig.add_trace(
        go.Scatter(
            x=fc_dates,
            y=fc_vals_list,
            name="AI Forecast",
            mode="lines",
            line=dict(color="#00f2ff", width=3, dash="dot"),
            hovertemplate="Forecast: $%{y:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # -------------------------
    # 5. Volume bars (row 2)
    # -------------------------
    if has_volume:
        vol_series = parse_volume(df[vol_col])
        fig.add_trace(
            go.Bar(
                x=vol_series.index,
                y=vol_series.values,
                name="Volume",
                marker_color="rgba(247,147,26,0.35)",
                hovertemplate="Volume: %{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor="#1e1e2e")

    # -------------------------
    # 6. Forecast-start divider line
    # -------------------------
    fig.add_shape(
        type="line",
        x0=price_series.index[-1],
        x1=price_series.index[-1],
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(color="#f7931a", width=2, dash="dash"),
    )

    # -------------------------
    # 7. "Forecast Start" annotation
    # -------------------------
    fig.add_annotation(
        x=price_series.index[-1],
        y=0.97,
        xref="x",
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        font=dict(color="#f7931a", size=11),
        bgcolor="rgba(10,10,15,0.75)",
        bordercolor="#f7931a",
        borderwidth=1,
        borderpad=4,
        xanchor="left",
    )

    # -------------------------
    # X-axis
    # -------------------------
    fig.update_xaxes(
        gridcolor="#1e1e2e",
        autorange=False,
        range=[price_series.index[0], fc_dates[-1]],
        constrain="domain",
        rangeslider=dict(
            visible=True,
            thickness=0.06,
            range=[price_series.index[0], fc_dates[-1]],
        ),
        rangeselector=dict(
            buttons=[
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=2, label="2y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            bgcolor="#12121f",
            activecolor="#f7931a",
        ),
    )

    # -------------------------
    # Layout
    # -------------------------
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="rgba(0,0,0,0)",
        height=750,
        margin=dict(l=10, r=10, t=50, b=10),
        autosize=True,
        xaxis_range=[price_series.index[0], fc_dates[-1]],
        hovermode="x",
    )

    return fig