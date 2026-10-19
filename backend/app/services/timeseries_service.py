"""
Topic 8: Forecast monthly claim volume (needs statsmodels).
Data note: data_synthetic.csv has no per-claim event date, so monthly volume =
sum(Claim History) grouped by month of Policy Renewal Date -- a disclosed proxy.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller


def build_monthly_series(df: pd.DataFrame) -> pd.Series:
    m = df.groupby(df["Policy Renewal Date"].dt.to_period("M"))["Claim History"].sum()
    m.index = m.index.to_timestamp()
    return m.sort_index().asfreq("MS", fill_value=0)  # explicit freq required by statsmodels


def stationarity_check(series: pd.Series) -> dict:
    r = adfuller(series.dropna())
    return {"adf_statistic": float(r[0]), "p_value": float(r[1]),
            "is_stationary_at_5pct": bool(r[1] < 0.05),
            "critical_values": {k: float(v) for k, v in r[4].items()}}


def decompose(series: pd.Series, period: int = 12) -> dict:
    if len(series) < 2 * period:
        return {"warning": f"Need >= {2 * period} points for seasonal decomposition; have {len(series)}."}
    r = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")
    return {"dates": [d.strftime("%Y-%m-%d") for d in series.index],
            "trend": r.trend.tolist(), "seasonal": r.seasonal.tolist(), "residual": r.resid.tolist()}


def forecast(series: pd.Series, horizon: int = 6, holdout: int = 6) -> dict:
    """Holt-Winters with a holdout backtest, then a full-sample forecast with ~95% bands."""
    sp = 12 if len(series) >= 24 else None
    seasonal = "add" if sp else None
    mape = rmse = None
    if len(series) > holdout + 6:
        train, test = series[:-holdout], series[-holdout:]
        pred = ExponentialSmoothing(train, trend="add", seasonal=seasonal, seasonal_periods=sp).fit().forecast(holdout)
        mape = float(np.mean(np.abs((test.to_numpy() - pred.to_numpy()) / test.to_numpy())) * 100)
        rmse = float(np.sqrt(np.mean((test.to_numpy() - pred.to_numpy()) ** 2)))
    model = ExponentialSmoothing(series, trend="add", seasonal=seasonal, seasonal_periods=sp).fit()
    fut = model.forecast(horizon)
    s = float(np.std(model.resid))
    return {"backtest_mape_pct": mape, "backtest_rmse": rmse,
            "forecast_dates": [d.strftime("%Y-%m-%d") for d in fut.index],
            "forecast_values": fut.tolist(),
            "ci_lower_95": (fut - 1.96 * s).tolist(), "ci_upper_95": (fut + 1.96 * s).tolist(),
            "model": f"Holt-Winters (trend=add, seasonal={seasonal}, seasonal_periods={sp})"}
