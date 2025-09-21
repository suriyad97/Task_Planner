"""Feature engineering utilities for time series forecasting."""
from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


def add_time_based_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features derived from the datetime index."""

    enriched = df.copy()
    index = enriched.index.to_period("D").to_timestamp()
    enriched["year"] = index.year
    enriched["month"] = index.month
    enriched["day"] = index.day
    enriched["dayofweek"] = index.dayofweek
    enriched["weekofyear"] = index.isocalendar().week.astype(int)
    enriched["is_weekend"] = (enriched["dayofweek"] >= 5).astype(int)
    return enriched


def add_lag_features(df: pd.DataFrame, lags: Iterable[int]) -> pd.DataFrame:
    """Create lagged versions of the target variable."""

    engineered = df.copy()
    for lag in lags:
        engineered[f"lag_{lag}"] = engineered["value"].shift(lag)
    return engineered


def add_rolling_features(df: pd.DataFrame, windows: Iterable[int]) -> pd.DataFrame:
    """Add rolling statistics for the target variable."""

    engineered = df.copy()
    for window in windows:
        engineered[f"roll_mean_{window}"] = engineered["value"].rolling(window=window).mean()
        engineered[f"roll_std_{window}"] = engineered["value"].rolling(window=window).std()
    return engineered


def build_supervised_dataset(
    df: pd.DataFrame,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> pd.DataFrame:
    """Combine all engineered features into a single modelling dataframe."""

    engineered = add_time_based_features(df)
    engineered = add_lag_features(engineered, lags)
    engineered = add_rolling_features(engineered, rolling_windows)
    engineered = engineered.dropna()
    return engineered


def train_test_split_time_series(
    df: pd.DataFrame,
    test_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the time series preserving the chronological order."""

    if test_size <= 0 or test_size >= len(df):
        raise ValueError("test_size must be between 1 and the length of the dataframe - 1")

    train = df.iloc[:-test_size]
    test = df.iloc[-test_size:]
    return train, test


__all__ = [
    "add_time_based_features",
    "add_lag_features",
    "add_rolling_features",
    "build_supervised_dataset",
    "train_test_split_time_series",
]
