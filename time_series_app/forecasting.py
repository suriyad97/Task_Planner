"""Forecasting utilities that power the Streamlit application."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.holtwinters import ExponentialSmoothing


@dataclass
class ForecastConfig:
    """Configuration for Holt-Winters forecasting."""

    seasonal_periods: int = 7
    trend: Optional[str] = "add"  # can be "add", "mul" or None
    seasonal: Optional[str] = "add"  # can be "add", "mul" or None
    damped_trend: bool = False
    alpha: float = 0.05


@dataclass
class ForecastResult:
    """Container for forecast output."""

    fitted_values: pd.Series
    forecast: pd.Series
    conf_int: pd.DataFrame
    model_summary: str


def holt_winters_forecast(train: pd.Series, horizon: int, config: ForecastConfig) -> ForecastResult:
    """Fit a Holt-Winters model and produce a forecast."""

    model = ExponentialSmoothing(
        train,
        trend=config.trend,
        seasonal=config.seasonal,
        seasonal_periods=config.seasonal_periods,
        damped_trend=config.damped_trend,
    )
    fitted_model = model.fit(optimized=True, use_brute=True, remove_bias=True)

    fitted_values = fitted_model.fittedvalues
    forecast = fitted_model.forecast(horizon)
    conf_int = _confidence_intervals(forecast, fitted_model.resid, alpha=config.alpha)
    summary = fitted_model.summary().as_text()

    return ForecastResult(fitted_values=fitted_values, forecast=forecast, conf_int=conf_int, model_summary=summary)


def _confidence_intervals(
    forecast: pd.Series,
    residuals: pd.Series,
    alpha: float,
) -> pd.DataFrame:
    """Approximate confidence intervals assuming normally distributed residuals."""

    std = np.nanstd(residuals, ddof=1)
    if np.isnan(std) or np.isclose(std, 0):
        lower = forecast.copy()
        upper = forecast.copy()
    else:
        z_score = norm.ppf(1 - alpha / 2)
        lower = forecast - z_score * std
        upper = forecast + z_score * std
    return pd.DataFrame({"lower": lower, "upper": upper})


__all__ = ["ForecastConfig", "ForecastResult", "holt_winters_forecast"]
