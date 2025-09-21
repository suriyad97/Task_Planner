"""Metrics used to evaluate time series forecasts."""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute a set of regression metrics."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    y_true_safe = y_true.replace(0, np.nan)
    mape = np.nanmean(np.abs((y_true - y_pred) / y_true_safe)) * 100
    denominator = np.abs(y_true) + np.abs(y_pred)
    smape = 100 * np.nanmean(2 * np.abs(y_pred - y_true) / denominator.replace(0, np.nan))
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "SMAPE": smape}


def coverage_rate(y_true: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Compute the proportion of actuals that fall within the prediction interval."""

    within = (y_true >= lower) & (y_true <= upper)
    return within.mean()


__all__ = ["regression_metrics", "coverage_rate"]
