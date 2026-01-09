"""Time series dataset preparation for AutoGluon TimeSeriesPredictor.

Builds return series from OHLCV bars and formats for TimeSeriesDataFrame.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np
import pandas as pd


def build_returns_series(
    df: pd.DataFrame,
    return_type: str = "log",
    price_col: str = "close",
) -> pd.Series:
    """
    Build return series from price data.

    Args:
        df: DataFrame with price column and DatetimeIndex.
        return_type: Type of return - "log" or "simple".
        price_col: Column to use for price.

    Returns:
        Series of returns with same index as input.
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found")

    prices = df[price_col]

    if return_type == "log":
        returns = np.log(prices / prices.shift(1))
    elif return_type == "simple":
        returns = prices.pct_change(1)
    else:
        raise ValueError(f"Unknown return_type: {return_type}")

    return returns


def add_time_covariates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based covariates to DataFrame.

    Args:
        df: DataFrame with DatetimeIndex.

    Returns:
        DataFrame with added covariate columns.
    """
    df = df.copy()

    # Extract time features from index
    if hasattr(df.index, 'hour'):
        df["hour_of_day"] = df.index.hour
    elif "timestamp" in df.columns:
        df["hour_of_day"] = pd.to_datetime(df["timestamp"]).dt.hour
    else:
        df["hour_of_day"] = 0

    if hasattr(df.index, 'dayofweek'):
        df["day_of_week"] = df.index.dayofweek
    elif "timestamp" in df.columns:
        df["day_of_week"] = pd.to_datetime(df["timestamp"]).dt.dayofweek
    else:
        df["day_of_week"] = 0

    return df


def create_timeseries_dataframe(
    bars_df: pd.DataFrame,
    symbols: List[str],
    target_col: str = "target",
    return_type: str = "log",
    price_col: str = "close",
    include_covariates: bool = True,
) -> pd.DataFrame:
    """
    Create a TimeSeriesDataFrame-compatible DataFrame from bars.

    Args:
        bars_df: DataFrame with MultiIndex (symbol, timestamp), symbol column,
                 or single-symbol DataFrame (DatetimeIndex only).
        symbols: List of symbols to include. If bars_df is single-symbol,
                 the first symbol in this list is used as item_id.
        target_col: Name for the target column.
        return_type: Type of return calculation.
        price_col: Price column to use.
        include_covariates: Whether to add time covariates.

    Returns:
        DataFrame formatted for TimeSeriesDataFrame with columns:
        item_id, timestamp, target, [covariates]
    """
    results = []

    # Check if bars_df is already a single-symbol DataFrame
    is_multi_index = isinstance(bars_df.index, pd.MultiIndex)
    has_symbol_col = "symbol" in bars_df.columns

    # If it's a simple DataFrame with DatetimeIndex (single symbol already extracted)
    if not is_multi_index and not has_symbol_col:
        # Use the first symbol in the list as the item_id
        symbol = symbols[0] if symbols else "UNKNOWN"
        symbol_df = bars_df.copy()

        # Ensure it's sorted by time
        symbol_df = symbol_df.sort_index()

        # Compute returns
        returns = build_returns_series(symbol_df, return_type, price_col)

        # Build result DataFrame
        result = pd.DataFrame(index=symbol_df.index)
        result["item_id"] = symbol
        result["timestamp"] = result.index
        result[target_col] = returns

        # Add covariates
        if include_covariates:
            result = add_time_covariates(result)

        # Drop first row (NaN return)
        result = result.dropna(subset=[target_col])

        return result.reset_index(drop=True)

    # Handle multi-symbol case
    for symbol in symbols:
        # Extract symbol data
        if is_multi_index:
            if symbol not in bars_df.index.get_level_values("symbol"):
                continue
            symbol_df = bars_df.loc[symbol].copy()
        else:
            symbol_df = bars_df[bars_df["symbol"] == symbol].copy()
            if "timestamp" in symbol_df.columns:
                symbol_df = symbol_df.set_index("timestamp")

        # Sort by time
        symbol_df = symbol_df.sort_index()

        # Compute returns
        returns = build_returns_series(symbol_df, return_type, price_col)

        # Build result DataFrame
        result = pd.DataFrame(index=symbol_df.index)
        result["item_id"] = symbol
        result["timestamp"] = result.index
        result[target_col] = returns

        # Add covariates
        if include_covariates:
            result = add_time_covariates(result)

        # Drop first row (NaN return)
        result = result.dropna(subset=[target_col])

        results.append(result)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return combined


def prepare_ts_training_data(
    bars_df: pd.DataFrame,
    symbols: List[str],
    train_end_timestamp: datetime,
    return_type: str = "log",
    include_covariates: bool = True,
    min_history_bars: int = 500,
    lookback_years: Optional[float] = None,
) -> pd.DataFrame:
    """
    Prepare time series training data ending strictly before a cutoff.

    Args:
        bars_df: Full bars DataFrame.
        symbols: Symbols to include.
        train_end_timestamp: Exclusive end timestamp (data < this time).
        return_type: Type of return calculation.
        include_covariates: Whether to add time covariates.
        min_history_bars: Minimum bars per symbol required.
        lookback_years: Maximum years of history to use (None = all available).

    Returns:
        TimeSeriesDataFrame-compatible DataFrame for training.
    """
    # Create full TS dataframe
    ts_df = create_timeseries_dataframe(
        bars_df,
        symbols,
        return_type=return_type,
        include_covariates=include_covariates,
    )

    if ts_df.empty:
        return ts_df

    # Filter to training window
    ts_df = ts_df[ts_df["timestamp"] < train_end_timestamp]

    # Apply lookback limit if specified
    if lookback_years is not None:
        lookback_days = int(lookback_years * 365.25)
        lookback_start = train_end_timestamp - pd.Timedelta(days=lookback_days)
        ts_df = ts_df[ts_df["timestamp"] >= lookback_start]

    # Validate minimum history
    valid_symbols = []
    for symbol in symbols:
        symbol_count = (ts_df["item_id"] == symbol).sum()
        if symbol_count >= min_history_bars:
            valid_symbols.append(symbol)
        else:
            print(f"Warning: {symbol} has only {symbol_count} bars, skipping")

    ts_df = ts_df[ts_df["item_id"].isin(valid_symbols)]

    return ts_df


def get_ts_data_info(ts_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary information about time series data.

    Args:
        ts_df: TimeSeriesDataFrame-compatible DataFrame.

    Returns:
        Dictionary with data statistics.
    """
    if ts_df.empty:
        return {"empty": True}

    info = {
        "total_rows": len(ts_df),
        "symbols": ts_df["item_id"].unique().tolist(),
        "n_symbols": ts_df["item_id"].nunique(),
    }

    # Per-symbol stats
    per_symbol = {}
    for symbol in info["symbols"]:
        symbol_data = ts_df[ts_df["item_id"] == symbol]
        per_symbol[symbol] = {
            "n_rows": len(symbol_data),
            "start": str(symbol_data["timestamp"].min()),
            "end": str(symbol_data["timestamp"].max()),
        }
    info["per_symbol"] = per_symbol

    # Date range
    info["start_date"] = str(ts_df["timestamp"].min())
    info["end_date"] = str(ts_df["timestamp"].max())

    return info
