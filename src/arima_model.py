import numpy as np
import pandas as pd
from datetime import timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsforecast import StatsForecast
from statsforecast.models import MSTL

def _mstl_decompose(series: pd.Series):
    """Decompose series using MSTL with lengths identified in the notebook."""
    sf_df = pd.DataFrame({
        "ds": series.index,
        "y": series.values,
        "unique_id": "btc",
    })
    # Weekly (7), Monthly (30), and Yearly (365)
    mstl = StatsForecast(
        models=[MSTL(season_length=[7, 30, 365])],
        freq="D",
    )
    mstl.fit(sf_df)
    decompose = mstl.fitted_[0, 0].model_.copy()
    decompose["ds"] = series.index
    return decompose

def run_arima(train: pd.Series, horizon: int, ci: float, is_backtest: bool = False):
    """Static Multi-step Forecast returning (mean, lower, upper)."""
    arima_order = (1, 0, 5) 
    decompose = _mstl_decompose(train)
    
    remainder = decompose["remainder"].fillna(0)
    model = SARIMAX(remainder, order=arima_order, enforce_stationarity=False)
    fitted = model.fit(disp=False)

    forecast_result = fitted.get_forecast(steps=horizon)
    alpha = 1 - ci
    conf_int = forecast_result.conf_int(alpha=alpha)

    # Recompose components
    last_trend = decompose["trend"].iloc[-1]
    seasonal_cols = [c for c in decompose.columns if "seasonal" in c.lower()]
    
    tile_window = horizon if is_backtest else 365
    last_cycle = decompose[seasonal_cols].iloc[-tile_window:].values
    repeat_n = int(np.ceil(horizon / len(last_cycle))) + 1
    tiled_seasonal = np.tile(last_cycle, (repeat_n, 1))[:horizon].sum(axis=1)

    mean_fc = last_trend + tiled_seasonal + forecast_result.predicted_mean.values
    lo = last_trend + tiled_seasonal + conf_int.iloc[:, 0].values
    hi = last_trend + tiled_seasonal + conf_int.iloc[:, 1].values
    
    future_dates = pd.date_range(train.index[-1] + timedelta(days=1), periods=horizon)
    
    # Return 3 values as expected by app.py
    return (
        pd.Series(mean_fc, index=future_dates),
        pd.Series(lo, index=future_dates),
        pd.Series(hi, index=future_dates)
    )

def run_walk_forward_backtest(train: pd.Series, test: pd.Series, ci: float = 0.95):
    """Dynamic 1-step ahead evaluation for Backtest UI."""
    history = list(train.values)
    predictions, lowers, uppers = [], [], []
    alpha = 1 - ci
    
    for t in range(len(test)):
        model = SARIMAX(history, order=(1, 0, 5), enforce_stationarity=False)
        res = model.fit(disp=False)
        
        fc = res.get_forecast(steps=1)
        predictions.append(fc.predicted_mean[0])
        
        conf = fc.conf_int(alpha=alpha)
        lowers.append(conf.iloc[0, 0])
        uppers.append(conf.iloc[0, 1])
        
        history.append(test.iloc[t])
        
    return (
        pd.Series(predictions, index=test.index),
        pd.Series(lowers, index=test.index),
        pd.Series(uppers, index=test.index)
    )