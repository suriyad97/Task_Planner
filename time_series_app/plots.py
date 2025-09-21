"""Plotting utilities leveraging Plotly for rich visualisations."""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


COLOR_PRIMARY = "#1f77b4"
COLOR_FORECAST = "#ff7f0e"
COLOR_INTERVAL = "rgba(255, 127, 14, 0.2)"
COLOR_ANOMALY = "#d62728"


def time_series_overview(df: pd.DataFrame) -> go.Figure:
    """Plot the raw time series."""

    fig = px.line(df.reset_index(), x="date", y="value", title="Time Series Overview")
    fig.update_traces(line=dict(color=COLOR_PRIMARY, width=2))
    fig.update_layout(template="plotly_white")
    return fig


def decomposition_plot(trend: pd.Series, seasonal: pd.Series, resid: pd.Series) -> go.Figure:
    """Visualise the components of a seasonal decomposition."""

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("Trend", "Seasonal", "Residual"))
    fig.add_trace(go.Scatter(x=trend.index, y=trend.values, mode="lines", name="Trend", line=dict(color=COLOR_PRIMARY)), row=1, col=1)
    fig.add_trace(go.Scatter(x=seasonal.index, y=seasonal.values, mode="lines", name="Seasonal", line=dict(color="#2ca02c")), row=2, col=1)
    fig.add_trace(go.Scatter(x=resid.index, y=resid.values, mode="lines", name="Residual", line=dict(color="#9467bd")), row=3, col=1)
    fig.update_layout(height=600, template="plotly_white", title="Seasonal Decomposition")
    return fig


def anomaly_plot(anomalies: pd.DataFrame) -> go.Figure:
    """Plot anomalies on top of the time series."""

    base = px.line(anomalies.reset_index(), x="date", y="value", title="Anomaly Detection", labels={"value": "Value"})
    base.update_traces(line=dict(color=COLOR_PRIMARY, width=2))

    anomaly_points = anomalies[anomalies["is_anomaly"]]
    base.add_trace(
        go.Scatter(
            x=anomaly_points.index,
            y=anomaly_points["value"],
            mode="markers",
            name="Anomaly",
            marker=dict(color=COLOR_ANOMALY, size=10, symbol="diamond"),
        )
    )
    base.update_layout(template="plotly_white")
    return base


def autocorrelation_plot(ac_series: pd.Series) -> go.Figure:
    """Plot autocorrelation coefficients."""

    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=list(ac_series.index), y=ac_series.values, marker_color=COLOR_PRIMARY, name="ACF")
    )
    fig.update_layout(
        title="Autocorrelation",
        xaxis_title="Lag",
        yaxis_title="Correlation",
        template="plotly_white",
    )
    return fig


def forecast_plot(train: pd.Series, test: pd.Series, forecast: pd.Series, conf_int: pd.DataFrame) -> go.Figure:
    """Plot training data, actuals, forecast and confidence intervals."""

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train.index, y=train.values, mode="lines", name="Train", line=dict(color=COLOR_PRIMARY)))
    fig.add_trace(go.Scatter(x=test.index, y=test.values, mode="lines", name="Actual", line=dict(color="#2ca02c")))
    fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode="lines", name="Forecast", line=dict(color=COLOR_FORECAST)))

    fig.add_trace(
        go.Scatter(
            x=conf_int.index.tolist() + conf_int.index[::-1].tolist(),
            y=conf_int["upper"].tolist() + conf_int["lower"][::-1].tolist(),
            fill="toself",
            fillcolor=COLOR_INTERVAL,
            line=dict(color="rgba(255,255,255,0)"),
            showlegend=True,
            name="Confidence Interval",
        )
    )

    fig.update_layout(template="plotly_white", title="Forecast vs Actuals", xaxis_title="Date", yaxis_title="Value")
    return fig


def feature_importance_heatmap(df: pd.DataFrame) -> go.Figure:
    """Plot a correlation heatmap of engineered features."""

    corr = df.corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Feature Correlation Heatmap", color_continuous_scale="Blues")
    fig.update_layout(template="plotly_white")
    return fig


__all__ = [
    "time_series_overview",
    "decomposition_plot",
    "anomaly_plot",
    "autocorrelation_plot",
    "forecast_plot",
    "feature_importance_heatmap",
]
