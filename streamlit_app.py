"""Streamlit application orchestrating the time series workflow."""
from __future__ import annotations

import io
from typing import Any, Dict

import pandas as pd
import streamlit as st

from time_series_app.data_ingestion import DataIngestionConfig, load_data, validate_minimum_length
from time_series_app import eda
from time_series_app import feature_engineering as fe
from time_series_app import forecasting
from time_series_app import metrics
from time_series_app import plots


st.set_page_config(page_title="Time Series Forecasting Studio", layout="wide")
st.title("📈 Time Series Forecasting Studio")
st.caption(
    "Experiment with ingestion, exploration, feature engineering and forecasting inside a single, polished interface."
)


def _load_dataset(uploaded_file: Any, config: DataIngestionConfig) -> pd.DataFrame:
    if uploaded_file is None:
        return load_data(None, config)

    buffer = io.BytesIO(uploaded_file.getvalue())
    return load_data(buffer, config)


with st.sidebar:
    st.header("Data Ingestion")
    uploaded_file = st.file_uploader("Upload a CSV", type=["csv"])
    datetime_col = st.text_input("Datetime column", value="")
    value_col = st.text_input("Value column", value="")
    frequency = st.selectbox("Resample frequency", options=["None", "D", "W", "M"], index=0)
    fill_method = st.selectbox("Gap filling method", options=["interpolate", "ffill", "bfill"], index=0)

config = DataIngestionConfig(
    datetime_column=datetime_col or None,
    value_column=value_col or None,
    frequency=None if frequency == "None" else frequency,
    fill_method=fill_method,
)

try:
    df = _load_dataset(uploaded_file, config)
    validate_minimum_length(df)
except Exception as exc:  # noqa: BLE001 - display friendly error in UI
    st.error(f"Unable to load dataset: {exc}")
    st.stop()

st.success(f"Dataset loaded with {len(df):,} records spanning {df.index.min().date()} to {df.index.max().date()}.")

# Cache EDA artefacts to keep the application responsive
@st.cache_data(show_spinner=False)
def cached_eda(data: pd.DataFrame, decomposition_period: int, acf_lag: int) -> Dict[str, object]:
    summary = eda.compute_summary_statistics(data)
    anomalies = eda.detect_anomalies(data)
    trend, seasonal, resid = eda.compute_decomposition(data, period=decomposition_period)
    acf_series = eda.autocorrelation(data, max_lag=acf_lag)
    strength = eda.seasonal_strength(data, period=decomposition_period)
    return {
        "summary": summary,
        "anomalies": anomalies,
        "trend": trend,
        "seasonal": seasonal,
        "resid": resid,
        "acf": acf_series,
        "strength": strength,
    }


tab_overview, tab_eda, tab_features, tab_forecast, tab_metrics = st.tabs(
    ["Data Overview", "Exploration", "Feature Engineering", "Forecasting", "Metrics"]
)

with tab_overview:
    st.subheader("Raw Series")
    st.plotly_chart(plots.time_series_overview(df), use_container_width=True)
    st.dataframe(df.head(20))

with tab_eda:
    st.subheader("Exploratory Analysis")
    period = st.select_slider("Seasonal period", options=[7, 30, 90, 180, 365], value=365)
    max_lag = st.slider("Autocorrelation lag", min_value=10, max_value=60, value=30)

    artefacts = cached_eda(df, decomposition_period=period, acf_lag=max_lag)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Summary statistics")
        st.dataframe(artefacts["summary"])
        st.markdown("#### Trend & Seasonality strength")
        strength = artefacts["strength"]
        st.metric("Trend strength", f"{strength['trend_strength']:.2f}")
        st.metric("Seasonality strength", f"{strength['seasonal_strength']:.2f}")
    with col2:
        st.markdown("#### Anomalies")
        st.plotly_chart(plots.anomaly_plot(artefacts["anomalies"]), use_container_width=True)

    st.markdown("#### Seasonal decomposition")
    st.plotly_chart(
        plots.decomposition_plot(artefacts["trend"], artefacts["seasonal"], artefacts["resid"]),
        use_container_width=True,
    )

    st.markdown("#### Autocorrelation")
    st.plotly_chart(plots.autocorrelation_plot(artefacts["acf"]), use_container_width=True)

with tab_features:
    st.subheader("Feature engineering")
    st.caption("Craft powerful features to boost model performance.")
    lags = st.multiselect("Lag features", options=list(range(1, 31)), default=[1, 7, 14])
    windows = st.multiselect("Rolling windows", options=[3, 7, 14, 30, 60], default=[7, 30])

    engineered = fe.build_supervised_dataset(df, lags=lags, rolling_windows=windows)

    st.markdown("#### Preview")
    st.dataframe(engineered.head(20))

    if not engineered.empty:
        st.markdown("#### Feature correlation heatmap")
        st.plotly_chart(plots.feature_importance_heatmap(engineered), use_container_width=True)

    st.session_state["engineered_features"] = engineered

with tab_forecast:
    st.subheader("Model training & forecasting")
    st.caption("Configure Holt-Winters smoothing parameters and evaluate hold-out performance.")

    col_left, col_right = st.columns(2)
    with col_left:
        max_test_size = min(max(30, len(df) // 3), len(df) - 1)
        default_test = min(90, max_test_size)
        test_size = st.slider(
            "Test size (observations)",
            min_value=30,
            max_value=max_test_size,
            value=default_test,
        )
        max_horizon = max(14, min(180, len(df)))
        default_horizon = min(90, max_horizon)
        if default_horizon > 14:
            default_horizon -= (default_horizon - 14) % 7
        horizon = st.slider(
            "Forecast horizon",
            min_value=14,
            max_value=max_horizon,
            value=default_horizon,
            step=7,
        )
    with col_right:
        seasonal_periods = st.selectbox("Seasonal periods", options=[7, 30, 90, 180, 365], index=0)
        trend = st.selectbox("Trend component", options=["add", "mul", None], index=0)
        seasonal = st.selectbox("Seasonal component", options=["add", "mul", None], index=0)
        damped = st.checkbox("Use damped trend", value=False)

    train, test = fe.train_test_split_time_series(df, test_size=test_size)
    config_forecast = forecasting.ForecastConfig(
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped,
    )

    forecast_result = forecasting.holt_winters_forecast(train["value"], horizon=horizon, config=config_forecast)
    forecast_series = forecast_result.forecast

    st.plotly_chart(
        plots.forecast_plot(train["value"], test["value"], forecast_series, forecast_result.conf_int),
        use_container_width=True,
    )

    st.markdown("#### Model summary")
    with st.expander("Show statsmodels summary"):
        st.text(forecast_result.model_summary)

    st.session_state["forecast_output"] = {
        "train": train,
        "test": test,
        "forecast": forecast_series,
        "conf_int": forecast_result.conf_int,
    }

with tab_metrics:
    st.subheader("Evaluation metrics")
    results = st.session_state.get("forecast_output")
    if not results:
        st.info("Train a model in the Forecasting tab to populate evaluation metrics.")
    else:
        actuals = results["test"]["value"].iloc[: len(results["forecast"])]
        forecast_values = results["forecast"].iloc[: len(actuals)]
        conf_int = results["conf_int"].iloc[: len(actuals)]

        performance = metrics.regression_metrics(actuals, forecast_values)
        coverage = metrics.coverage_rate(actuals, conf_int["lower"], conf_int["upper"])

        metric_cols = st.columns(len(performance) + 1)
        for (name, value), col in zip(performance.items(), metric_cols):
            col.metric(name, f"{value:.2f}")
        metric_cols[-1].metric("PI Coverage", f"{coverage:.2%}")

        residuals = actuals - forecast_values
        st.markdown("#### Residual diagnostics")
        st.line_chart(residuals)

st.sidebar.info(
    "No dataset? The app ships with a synthetic energy-demand example spanning five years. "
    "Upload your own CSV to replace it."
)
