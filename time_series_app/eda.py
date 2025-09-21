"""Exploratory data analysis helpers for the time series Streamlit application."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose


def compute_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return key descriptive statistics for the time series."""

    series = df["value"]
    summary = series.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame(name="value")
    summary.loc["variance"] = series.var()
    summary.loc["skewness"] = series.skew()
    summary.loc["kurtosis"] = series.kurtosis()
    summary.loc["missing_ratio"] = series.isna().mean()
    return summary


def detect_anomalies(df: pd.DataFrame, window: int = 14, z_threshold: float = 3.0) -> pd.DataFrame:
    """Detect simple anomalies based on rolling z-scores."""

    series = df["value"].copy()
    rolling_mean = series.rolling(window=window, min_periods=1).mean()
    rolling_std = series.rolling(window=window, min_periods=1).std(ddof=0)
    z_scores = (series - rolling_mean) / rolling_std.replace(0, np.nan)
    anomalies = df.copy()
    anomalies["z_score"] = z_scores
    anomalies["is_anomaly"] = z_scores.abs() > z_threshold
    return anomalies.dropna(subset=["value"])


def compute_decomposition(df: pd.DataFrame, period: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Perform a classical seasonal decomposition of the series."""

    result = seasonal_decompose(df["value"], model="additive", period=period, extrapolate_trend="freq")
    return result.trend, result.seasonal, result.resid


def autocorrelation(df: pd.DataFrame, max_lag: int = 30) -> pd.Series:
    """Compute autocorrelation values up to the requested lag."""

    series = df["value"]
    return pd.Series(
        [series.autocorr(lag=i) for i in range(1, max_lag + 1)],
        index=range(1, max_lag + 1),
        name="autocorrelation",
    )


def seasonal_strength(df: pd.DataFrame, period: int) -> Dict[str, float]:
    """Compute strength of trend and seasonality as described by Hyndman & Athanasopoulos."""

    trend, seasonal, resid = compute_decomposition(df, period=period)
    resid_var = np.nanvar(resid)
    trend_strength = max(0.0, 1 - resid_var / np.nanvar(trend + resid))
    seasonal_strength = max(0.0, 1 - resid_var / np.nanvar(seasonal + resid))
    return {"trend_strength": trend_strength, "seasonal_strength": seasonal_strength}


__all__ = [
    "compute_summary_statistics",
    "detect_anomalies",
    "compute_decomposition",
    "autocorrelation",
    "seasonal_strength",
]
