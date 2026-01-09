"""Forecast-derived features for tabular model.

Assembles time series forecast outputs into tabular features.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np


# Feature set configurations
FEATURE_SET_CONFIGS = {
    "small": {
        "horizons": [1, 4, 13, 26],
        "features": ["mu", "unc"],
        "description": "Minimal forecast features: mean and uncertainty at key horizons",
    },
    "medium": {
        "horizons": [1, 2, 4, 13, 26],
        "features": ["mu", "unc", "skew"],
        "description": "Medium feature set: adds skew and more horizons",
    },
    "large": {
        "horizons": [1, 2, 4, 8, 13, 20, 26],
        "features": ["mu", "unc", "skew", "q10", "q90"],
        "description": "Large feature set: all features and horizons",
    },
}


def get_forecast_feature_names(
    feature_set: str = "small",
    prefix: str = "",
) -> List[str]:
    """
    Get list of forecast feature names for a feature set.

    Args:
        feature_set: One of "small", "medium", "large".
        prefix: Optional prefix for feature names (e.g., "spy_").

    Returns:
        List of feature column names.
    """
    config = FEATURE_SET_CONFIGS.get(feature_set, FEATURE_SET_CONFIGS["small"])
    horizons = config["horizons"]
    features = config["features"]

    names = []
    for feat in features:
        for h in horizons:
            name = f"{prefix}{feat}_{h}" if prefix else f"{feat}_{h}"
            names.append(name)

    return names


def pivot_forecasts_to_features(
    forecasts_df: pd.DataFrame,
    feature_set: str = "small",
    prefix: str = "",
) -> pd.DataFrame:
    """
    Pivot forecast DataFrame into feature columns.

    Args:
        forecasts_df: DataFrame with forecast features (from rolling_predict).
            Expected columns: timestamp, symbol, mu_1, mu_4, ..., unc_1, etc.
        feature_set: Feature set configuration to use.
        prefix: Optional prefix for feature names.

    Returns:
        DataFrame with (symbol, timestamp) index and feature columns.
    """
    config = FEATURE_SET_CONFIGS.get(feature_set, FEATURE_SET_CONFIGS["small"])
    horizons = config["horizons"]
    features = config["features"]

    df = forecasts_df.copy()

    # Select relevant columns
    base_cols = ["timestamp", "symbol"]
    feature_cols = []

    for feat in features:
        for h in horizons:
            col = f"{feat}_{h}"
            if col in df.columns:
                feature_cols.append(col)

    # Keep only needed columns
    available_cols = base_cols + [c for c in feature_cols if c in df.columns]
    df = df[available_cols].copy()

    # Rename with prefix if provided
    if prefix:
        rename_map = {col: f"{prefix}{col}" for col in feature_cols if col in df.columns}
        df = df.rename(columns=rename_map)

    return df


def merge_forecast_features(
    tabular_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    feature_set: str = "small",
    prefix: str = "fc_",
) -> pd.DataFrame:
    """
    Merge forecast features into tabular dataset.

    Args:
        tabular_df: Base tabular dataset with (symbol, timestamp) or timestamp index.
        forecasts_df: Forecast features DataFrame.
        feature_set: Feature set to use.
        prefix: Prefix for forecast feature names.

    Returns:
        Merged DataFrame with forecast features added.
    """
    # Pivot forecasts
    fc_features = pivot_forecasts_to_features(forecasts_df, feature_set, prefix)

    # Prepare for merge
    df = tabular_df.copy()

    # Reset index if needed
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()
    elif df.index.name == "timestamp":
        df = df.reset_index()

    # Ensure timestamp columns are datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fc_features["timestamp"] = pd.to_datetime(fc_features["timestamp"])

    # Merge on symbol and timestamp
    if "symbol" in df.columns and "symbol" in fc_features.columns:
        merged = df.merge(
            fc_features,
            on=["symbol", "timestamp"],
            how="left",
        )
    else:
        # Single symbol case
        merged = df.merge(
            fc_features.drop(columns=["symbol"], errors="ignore"),
            on=["timestamp"],
            how="left",
        )

    return merged


def build_forecast_feature_dataset(
    base_df: pd.DataFrame,
    forecast_cache_dir: Path,
    ts_model_run_id: str,
    symbols: List[str],
    feature_set: str = "small",
    prefix: str = "fc_",
) -> pd.DataFrame:
    """
    Build complete dataset with forecast features from cached forecasts.

    Args:
        base_df: Base tabular dataset.
        forecast_cache_dir: Directory containing cached forecasts.
        ts_model_run_id: Run ID for the TS model.
        symbols: Symbols to include.
        feature_set: Feature set configuration.
        prefix: Prefix for feature names.

    Returns:
        DataFrame with base features + forecast features.
    """
    all_forecasts = []

    for symbol in symbols:
        cache_path = forecast_cache_dir / "forecasts" / ts_model_run_id / f"{symbol}.parquet"

        if not cache_path.exists():
            print(f"Warning: No cached forecasts for {symbol} at {cache_path}")
            continue

        fc_df = pd.read_parquet(cache_path)
        all_forecasts.append(fc_df)

    if not all_forecasts:
        print("Warning: No forecast data found")
        return base_df

    forecasts_combined = pd.concat(all_forecasts, ignore_index=True)

    # Merge with base dataset
    result = merge_forecast_features(
        base_df, forecasts_combined, feature_set, prefix
    )

    return result


def count_forecast_features(feature_set: str = "small", prefix: str = "") -> int:
    """Count number of forecast features for a feature set."""
    return len(get_forecast_feature_names(feature_set, prefix))


def get_feature_set_info(feature_set: str = "small") -> Dict[str, Any]:
    """Get information about a feature set configuration."""
    config = FEATURE_SET_CONFIGS.get(feature_set, FEATURE_SET_CONFIGS["small"])
    return {
        "name": feature_set,
        "horizons": config["horizons"],
        "features": config["features"],
        "description": config["description"],
        "n_features": count_forecast_features(feature_set),
    }
