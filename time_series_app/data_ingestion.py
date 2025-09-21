"""Data ingestion utilities for the time series Streamlit application."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample_energy_demand.csv"


@dataclass
class DataIngestionConfig:
    """Configuration for loading a time series dataset."""

    datetime_column: Optional[str] = None
    value_column: Optional[str] = None
    frequency: Optional[str] = None  # e.g. "D" for daily
    fill_method: str = "interpolate"


def _standardise_columns(df: pd.DataFrame, config: DataIngestionConfig) -> pd.DataFrame:
    """Ensure that the loaded dataframe has the expected column names."""

    datetime_col = config.datetime_column or df.columns[0]
    value_col = config.value_column or df.columns[1]

    cleaned = (
        df[[datetime_col, value_col]]
        .rename(columns={datetime_col: "date", value_col: "value"})
        .copy()
    )
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["date"])
    cleaned = cleaned.sort_values("date").set_index("date")
    cleaned["value"] = pd.to_numeric(cleaned["value"], errors="coerce")
    cleaned = cleaned.dropna(subset=["value"])
    return cleaned


def load_sample_data() -> pd.DataFrame:
    """Load the bundled demonstration dataset."""

    if not SAMPLE_DATA_PATH.exists():
        raise FileNotFoundError(
            "Sample dataset is missing. Expected to find it at " f"{SAMPLE_DATA_PATH}."
        )

    df = pd.read_csv(SAMPLE_DATA_PATH)
    df = _standardise_columns(df, DataIngestionConfig(datetime_column="date", value_column="demand"))
    return df


def load_data(
    source: Optional[IO[str]] = None,
    config: Optional[DataIngestionConfig] = None,
) -> pd.DataFrame:
    """Load a dataset either from the provided source or the bundled sample."""

    config = config or DataIngestionConfig()

    if source is None:
        df = load_sample_data()
    else:
        df = pd.read_csv(source)
        df = _standardise_columns(df, config)

    if config.frequency:
        df = _resample_series(df, frequency=config.frequency, fill_method=config.fill_method)

    return df


def _resample_series(df: pd.DataFrame, frequency: str, fill_method: str = "interpolate") -> pd.DataFrame:
    """Resample the series to a uniform frequency and fill gaps if required."""

    series = df["value"].asfreq(frequency)

    if fill_method == "ffill":
        series = series.ffill()
    elif fill_method == "bfill":
        series = series.bfill()
    elif fill_method == "interpolate":
        series = series.interpolate(limit_direction="both")

    return series.to_frame(name="value")


def validate_minimum_length(df: pd.DataFrame, minimum_points: int = 30) -> None:
    """Raise a ValueError when the dataset does not contain enough observations."""

    if len(df.index) < minimum_points:
        raise ValueError(
            "The dataset is too short for modelling. "
            f"Received {len(df.index)} records but require at least {minimum_points}."
        )


__all__ = [
    "DataIngestionConfig",
    "load_sample_data",
    "load_data",
    "validate_minimum_length",
]
