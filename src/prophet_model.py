import numpy as np
import pandas as pd
if not hasattr(np, "float_"):
    np.float_ = np.float64
from prophet import Prophet

def run_prophet(train: pd.Series, horizon: int, ci: float, is_backtest: bool = False):
    """
    Final Prophet implementation matching notebook Section 8.
    """
    df_p = train.reset_index()
    df_p.columns = ["ds", "y"]
    
    # 1. Scaling Logic (Critical for Convergence)
    if is_backtest:
        df_p["y"] = np.log1p(df_p["y"]) # Handling smaller slices
    else:
        df_p["y"] = np.log(df_p["y"].clip(lower=1e-9)) # Handling full historical scale

    # 2. Model Configuration
    model = Prophet(
        interval_width=ci,
        seasonality_mode="multiplicative", # Matches 'fixed' notebook best performance
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    
    # Add the custom monthly cycle tuned in the notebook
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    
    model.fit(df_p)

    # 3. Forecast & Inverse Transform
    future = model.make_future_dataframe(periods=horizon)
    forecast = model.predict(future)
    
    # Convert back from Log space to USD
    for col in ['yhat', 'yhat_lower', 'yhat_upper']:
        forecast[col] = np.exp(forecast[col])
        
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]