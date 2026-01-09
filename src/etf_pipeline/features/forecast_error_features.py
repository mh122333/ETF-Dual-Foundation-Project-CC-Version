"""Forecast error and coverage features.

Computes rolling forecast accuracy metrics as features to capture
model health and regime drift.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np


def compute_forecast_errors(
    realized_returns: pd.Series,
    forecast_mu: pd.Series,
    forecast_q10: Optional[pd.Series] = None,
    forecast_q90: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Compute forecast errors and coverage at each timestamp.

    The forecast at t-1 predicts the return at t.
    Error at t = realized_return_t - forecast_mu_{t-1, h=1}

    Args:
        realized_returns: Series of realized 1-bar returns.
        forecast_mu: Series of 1-step ahead forecast means (aligned with decision time).
        forecast_q10: Series of 1-step ahead q10 forecasts.
        forecast_q90: Series of 1-step ahead q90 forecasts.

    Returns:
        DataFrame with error, abs_error, and coverage columns.
    """
    # Shift forecasts forward by 1 to align with realized returns
    # forecast at t-1 predicts return at t
    fc_mu_shifted = forecast_mu.shift(1)

    # Compute errors
    errors = realized_returns - fc_mu_shifted
    abs_errors = errors.abs()

    result = pd.DataFrame({
        "error": errors,
        "abs_error": abs_errors,
    }, index=realized_returns.index)

    # Compute coverage if quantiles available
    if forecast_q10 is not None and forecast_q90 is not None:
        q10_shifted = forecast_q10.shift(1)
        q90_shifted = forecast_q90.shift(1)

        # Coverage: 1 if realized is within [q10, q90]
        coverage = (
            (realized_returns >= q10_shifted) &
            (realized_returns <= q90_shifted)
        ).astype(float)

        result["coverage"] = coverage

    return result


def compute_rolling_error_features(
    error_df: pd.DataFrame,
    mae_window: int = 20,
    bias_window: int = 20,
    coverage_window: int = 50,
) -> pd.DataFrame:
    """
    Compute rolling aggregate error features.

    Args:
        error_df: DataFrame with error, abs_error, coverage columns.
        mae_window: Window for MAE computation.
        bias_window: Window for bias computation.
        coverage_window: Window for coverage computation.

    Returns:
        DataFrame with rolling error features.
    """
    result = pd.DataFrame(index=error_df.index)

    # Rolling MAE
    if "abs_error" in error_df.columns:
        result[f"mae_{mae_window}"] = (
            error_df["abs_error"]
            .rolling(window=mae_window, min_periods=mae_window)
            .mean()
        )

    # Rolling bias
    if "error" in error_df.columns:
        result[f"bias_{bias_window}"] = (
            error_df["error"]
            .rolling(window=bias_window, min_periods=bias_window)
            .mean()
        )

    # Rolling coverage
    if "coverage" in error_df.columns:
        result[f"coverage_{coverage_window}"] = (
            error_df["coverage"]
            .rolling(window=coverage_window, min_periods=coverage_window)
            .mean()
        )

    return result


def build_forecast_error_features(
    bars_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    symbol: str,
    mae_window: int = 20,
    bias_window: int = 20,
    coverage_window: int = 50,
    return_type: str = "log",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Build forecast error features for a single symbol.

    Args:
        bars_df: Bars DataFrame with price data.
        forecasts_df: Forecast DataFrame with mu_1, q10_1, q90_1 columns.
        symbol: Symbol to process.
        mae_window: Window for MAE.
        bias_window: Window for bias.
        coverage_window: Window for coverage.
        return_type: Type of returns.
        price_col: Price column.

    Returns:
        DataFrame with forecast error features.
    """
    from .baseline import compute_baseline_features
    from ..timeseries.dataset import build_returns_series

    # Extract symbol data
    if isinstance(bars_df.index, pd.MultiIndex):
        symbol_bars = bars_df.loc[symbol].copy()
    else:
        symbol_bars = bars_df[bars_df["symbol"] == symbol].copy()
        if "timestamp" in symbol_bars.columns:
            symbol_bars = symbol_bars.set_index("timestamp")

    symbol_bars = symbol_bars.sort_index()

    # Compute realized returns
    realized_returns = build_returns_series(symbol_bars, return_type, price_col)
    realized_returns.name = "realized_return"

    # Get forecast columns
    fc_df = forecasts_df[forecasts_df["symbol"] == symbol].copy()
    fc_df["timestamp"] = pd.to_datetime(fc_df["timestamp"])
    fc_df = fc_df.set_index("timestamp").sort_index()

    # Align forecasts with realized returns
    aligned = pd.DataFrame(index=symbol_bars.index)
    aligned["realized_return"] = realized_returns

    # Merge forecasts
    if "mu_1" in fc_df.columns:
        aligned = aligned.join(fc_df[["mu_1"]], how="left")
    if "q10_1" in fc_df.columns:
        aligned = aligned.join(fc_df[["q10_1"]], how="left")
    if "q90_1" in fc_df.columns:
        aligned = aligned.join(fc_df[["q90_1"]], how="left")

    # Compute errors
    error_df = compute_forecast_errors(
        aligned["realized_return"],
        aligned.get("mu_1", pd.Series(dtype=float)),
        aligned.get("q10_1"),
        aligned.get("q90_1"),
    )

    # Compute rolling features
    rolling_features = compute_rolling_error_features(
        error_df, mae_window, bias_window, coverage_window
    )

    # Add timestamp and symbol
    rolling_features["timestamp"] = rolling_features.index
    rolling_features["symbol"] = symbol

    return rolling_features.reset_index(drop=True)


def add_forecast_error_features(
    df: pd.DataFrame,
    bars_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    symbols: List[str],
    prefix: str = "err_",
    mae_window: int = 20,
    bias_window: int = 20,
    coverage_window: int = 50,
) -> pd.DataFrame:
    """
    Add forecast error features to tabular dataset.

    Args:
        df: Base tabular dataset.
        bars_df: Bars data for computing realized returns.
        forecasts_df: Forecast data.
        symbols: Symbols to process.
        prefix: Prefix for error feature columns.
        mae_window: Window for MAE.
        bias_window: Window for bias.
        coverage_window: Window for coverage.

    Returns:
        DataFrame with error features added.
    """
    result = df.copy()

    # Ensure timestamp is in columns
    if "timestamp" not in result.columns:
        result = result.reset_index()
    result["timestamp"] = pd.to_datetime(result["timestamp"])

    all_error_features = []

    for symbol in symbols:
        try:
            err_features = build_forecast_error_features(
                bars_df,
                forecasts_df,
                symbol,
                mae_window,
                bias_window,
                coverage_window,
            )
            all_error_features.append(err_features)
        except Exception as e:
            print(f"Warning: Failed to compute error features for {symbol}: {e}")

    if not all_error_features:
        return result

    error_df = pd.concat(all_error_features, ignore_index=True)
    error_df["timestamp"] = pd.to_datetime(error_df["timestamp"])

    # Rename columns with prefix
    err_cols = [c for c in error_df.columns if c not in ["timestamp", "symbol"]]
    rename_map = {c: f"{prefix}{c}" for c in err_cols}
    error_df = error_df.rename(columns=rename_map)

    # Merge with base data
    result = result.merge(
        error_df,
        on=["symbol", "timestamp"],
        how="left",
    )

    return result


def get_error_feature_names(
    prefix: str = "err_",
    mae_window: int = 20,
    bias_window: int = 20,
    coverage_window: int = 50,
) -> List[str]:
    """Get list of error feature column names."""
    return [
        f"{prefix}mae_{mae_window}",
        f"{prefix}bias_{bias_window}",
        f"{prefix}coverage_{coverage_window}",
    ]


def count_error_features() -> int:
    """Count number of error features."""
    return 3  # mae, bias, coverage
