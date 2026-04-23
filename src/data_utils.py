"""
src/data_utils.py
-----------------
Data loading, parsing, and indicator utilities for the BTC Forecasting Portal.
"""

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════════
#  CSV PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_btc_csv(df: pd.DataFrame):
    """
    Auto-detect and normalise common Kaggle BTC CSV formats.
    Automatically resamples minute- or hour-level data to daily OHLCV
    so forecasting models always receive a manageable daily series.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with a 'Date' column and price columns.
    price_candidates : list[str]
        Names of detected price columns.
    """
    df.columns = [c.strip() for c in df.columns]

    # Detect date column
    date_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["date", "time", "timestamp"])
    ]
    if not date_candidates:
        raise ValueError(
            "No date/time column found. Expected a column named Date, Timestamp, or similar."
        )
    date_col = next(
        (c for c in date_candidates if "date" in c.lower()), date_candidates[0]
    )

    # Detect price columns
    price_candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["close", "open", "high", "low", "price"])
    ]
    if not price_candidates:
        raise ValueError(
            "No price column found. Expected columns like Close, Open, High, or Low."
        )

    # Parse dates
    df[date_col] = pd.to_datetime(df[date_col], utc=False, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: "Date"})
    df = df.sort_values("Date").reset_index(drop=True)

    # Clean numeric price columns
    for col in price_candidates:
        if df[col].dtype == object:
            df[col] = (
                df[col].str.replace(",", "").str.replace("$", "").str.strip()
            )
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=price_candidates)

    # Auto-resample sub-daily data → daily OHLCV
    if len(df) > 1:
        median_gap = df["Date"].diff().dropna().abs().median()
        if median_gap < pd.Timedelta(hours=23):
            df = df.set_index("Date")
            agg = {}
            col_lower = {c.lower(): c for c in df.columns}
            if "open"  in col_lower: agg[col_lower["open"]]  = "first"
            if "high"  in col_lower: agg[col_lower["high"]]  = "max"
            if "low"   in col_lower: agg[col_lower["low"]]   = "min"
            if "close" in col_lower: agg[col_lower["close"]] = "last"
            for c in df.columns:
                if "vol" in c.lower() and c not in agg:
                    agg[c] = "sum"
            for c in price_candidates:
                if c not in agg:
                    agg[c] = "last"
            df = df.resample("D").agg(agg).dropna(how="all").reset_index()
            price_candidates = [c for c in price_candidates if c in df.columns]

    return df, price_candidates


# ═══════════════════════════════════════════════════════════════════════════════
#  VOLUME PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_volume(vol_series: pd.Series) -> pd.Series:
    """
    Safely parse volume strings that may contain K / M / B suffixes.
    """
    def _parse_val(v):
        if pd.isna(v):
            return np.nan
        s = str(v).strip().replace(",", "")
        multiplier = 1
        if s.endswith(("B", "b")):
            multiplier = 1_000_000_000
            s = s[:-1]
        elif s.endswith(("M", "m")):
            multiplier = 1_000_000
            s = s[:-1]
        elif s.endswith(("K", "k")):
            multiplier = 1_000
            s = s[:-1]
        try:
            return float(s) * multiplier
        except ValueError:
            return np.nan

    return vol_series.apply(_parse_val)


# ═══════════════════════════════════════════════════════════════════════════════
#  TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_indicators(
    series: pd.Series,
    sma_windows: tuple = (20, 50),
    ema_window: int = 20,
) -> dict:
    """
    Compute SMA and EMA technical indicators.

    Returns
    -------
    dict mapping indicator name → pd.Series aligned to `series.index`.
    """
    indicators = {}
    for w in sma_windows:
        if len(series) >= w:
            indicators[f"SMA_{w}"] = series.rolling(w).mean()
    if len(series) >= ema_window:
        indicators[f"EMA_{ema_window}"] = series.ewm(span=ema_window, adjust=False).mean()
    return indicators


# ═══════════════════════════════════════════════════════════════════════════════
#  TRAIN / TEST SPLIT & METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def backtest_split(series: pd.Series, test_ratio: float = 0.1):
    """Split a time series into train / test sets (no shuffle)."""
    n = len(series)
    split = int(n * (1 - test_ratio))
    return series.iloc[:split], series.iloc[split:]


def calc_metrics(actual: pd.Series, predicted) -> tuple:
    """Return (MAE, RMSE) between actual and predicted arrays."""
    n = min(len(actual), len(predicted))
    a = actual.values[:n]
    p = np.array(predicted)[:n]
    mae  = np.mean(np.abs(a - p))
    rmse = np.sqrt(np.mean((a - p) ** 2))
    return mae, rmse
