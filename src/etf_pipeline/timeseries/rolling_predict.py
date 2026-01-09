"""Rolling forecast generation for time series features.

Generates out-of-sample forecasts for each decision timestamp using only
historical data up to that point. The frozen TS predictor trained on older
data is used to produce forecasts on the tabular window.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime

import numpy as np
import pandas as pd


def generate_single_forecast(
    predictor: Any,
    history_df: pd.DataFrame,
    symbol: str,
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Generate a single forecast from historical data.

    Args:
        predictor: Trained TimeSeriesPredictor.
        history_df: Historical data up to decision point.
        symbol: Symbol being forecast.
        target_col: Target column name.

    Returns:
        DataFrame with forecast for each horizon step.
    """
    try:
        from autogluon.timeseries import TimeSeriesDataFrame
    except ImportError:
        raise ImportError("AutoGluon timeseries not installed")

    # Prepare history in TimeSeriesDataFrame format
    hist_data = history_df.copy()
    if "item_id" not in hist_data.columns:
        hist_data["item_id"] = symbol
    if "timestamp" not in hist_data.columns:
        hist_data["timestamp"] = hist_data.index

    ts_data = TimeSeriesDataFrame.from_data_frame(
        hist_data[["item_id", "timestamp", target_col]],
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # Generate forecast
    forecast = predictor.predict(ts_data)

    return forecast


def extract_forecast_features_from_prediction(
    forecast_df: pd.DataFrame,
    symbol: str,
    decision_timestamp: datetime,
    horizons: List[int] = None,
    selected_models: List[str] = None,
) -> Dict[str, Any]:
    """
    Extract forecast features from a prediction DataFrame.

    For log returns, cumulative h-step forecast is sum of 1-step forecasts.

    Args:
        forecast_df: Forecast from predictor.predict().
        symbol: Symbol being forecast.
        decision_timestamp: Timestamp of decision point.
        horizons: List of horizons to extract (e.g., [1, 4, 13, 26]).
        selected_models: Models to extract features for.

    Returns:
        Dictionary of features keyed by feature name.
    """
    if horizons is None:
        horizons = [1, 4, 13, 26]

    features = {
        "timestamp": decision_timestamp,
        "symbol": symbol,
    }

    # Get available columns
    cols = forecast_df.columns.tolist()

    # Identify quantile columns
    # AutoGluon uses format like "0.1", "0.5", "0.9" or "mean"
    q10_col = None
    q50_col = None
    q90_col = None
    mean_col = None

    for col in cols:
        col_str = str(col)
        if "0.1" in col_str:
            q10_col = col
        elif "0.5" in col_str:
            q50_col = col
        elif "0.9" in col_str:
            q90_col = col
        elif "mean" in col_str.lower():
            mean_col = col

    # Get values as arrays (one per forecast step)
    n_steps = len(forecast_df)

    # Extract mean/median forecast
    if mean_col and mean_col in forecast_df.columns:
        mean_vals = forecast_df[mean_col].values
    elif q50_col and q50_col in forecast_df.columns:
        mean_vals = forecast_df[q50_col].values
    else:
        # Try first numeric column
        numeric_cols = forecast_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            mean_vals = forecast_df[numeric_cols[0]].values
        else:
            mean_vals = np.zeros(n_steps)

    # Extract quantiles if available
    q10_vals = forecast_df[q10_col].values if q10_col and q10_col in forecast_df.columns else None
    q90_vals = forecast_df[q90_col].values if q90_col and q90_col in forecast_df.columns else None

    # Compute features for each horizon
    for h in horizons:
        if h > n_steps:
            # Not enough forecast steps
            features[f"mu_{h}"] = np.nan
            features[f"unc_{h}"] = np.nan
            features[f"q10_{h}"] = np.nan
            features[f"q90_{h}"] = np.nan
            continue

        # Cumulative mean forecast (sum of log returns for h steps)
        # For h-step ahead cumulative return: sum of steps 1..h
        mu_h = np.sum(mean_vals[:h])
        features[f"mu_{h}"] = mu_h

        # Quantiles - for cumulative, we use the h-step values directly
        # This is an approximation; proper would require distributional assumptions
        if q10_vals is not None:
            features[f"q10_{h}"] = np.sum(q10_vals[:h])
        else:
            features[f"q10_{h}"] = np.nan

        if q90_vals is not None:
            features[f"q90_{h}"] = np.sum(q90_vals[:h])
        else:
            features[f"q90_{h}"] = np.nan

        # Uncertainty: q90 - q10 width
        if q10_vals is not None and q90_vals is not None:
            # Sum of per-step widths as approximation
            unc_h = np.sum(q90_vals[:h] - q10_vals[:h])
            features[f"unc_{h}"] = unc_h
        else:
            features[f"unc_{h}"] = np.nan

        # Skew approximation: (q90 - q50) / (q50 - q10)
        if q10_vals is not None and q90_vals is not None:
            q50_h = np.sum(mean_vals[:h])  # Use mean as median proxy
            q10_h = features[f"q10_{h}"]
            q90_h = features[f"q90_{h}"]
            denom = q50_h - q10_h
            if abs(denom) > 1e-10:
                features[f"skew_{h}"] = (q90_h - q50_h) / denom
            else:
                features[f"skew_{h}"] = 0.0
        else:
            features[f"skew_{h}"] = np.nan

    return features


def generate_rolling_forecasts(
    predictor: Any,
    bars_df: pd.DataFrame,
    symbol: str,
    decision_timestamps: List[datetime],
    horizons: List[int] = None,
    min_history_bars: int = 100,
    target_col: str = "target",
    return_type: str = "log",
    price_col: str = "close",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Generate rolling forecasts for all decision timestamps.

    For each decision time t, uses only history up to and including t
    to generate forecasts for t+1, t+2, ..., t+H.

    Args:
        predictor: Trained TimeSeriesPredictor.
        bars_df: Full bars DataFrame for the symbol.
        symbol: Symbol to forecast.
        decision_timestamps: List of decision timestamps.
        horizons: Horizons to extract features for.
        min_history_bars: Minimum history required for forecast.
        target_col: Target column name.
        return_type: Return type ("log" or "simple").
        price_col: Price column for returns.
        verbose: Print progress.

    Returns:
        DataFrame with forecast features for each decision timestamp.
    """
    from .dataset import build_returns_series

    if horizons is None:
        horizons = [1, 4, 13, 26]

    # Extract symbol data
    if isinstance(bars_df.index, pd.MultiIndex):
        symbol_df = bars_df.loc[symbol].copy()
    else:
        symbol_df = bars_df[bars_df["symbol"] == symbol].copy()
        if "timestamp" in symbol_df.columns:
            symbol_df = symbol_df.set_index("timestamp")

    symbol_df = symbol_df.sort_index()

    # Compute returns
    returns = build_returns_series(symbol_df, return_type, price_col)
    symbol_df[target_col] = returns

    # Generate forecasts for each decision timestamp
    all_features = []
    n_total = len(decision_timestamps)

    for i, dt in enumerate(decision_timestamps):
        if verbose and (i % 100 == 0 or i == n_total - 1):
            print(f"  Generating forecast {i+1}/{n_total} for {symbol} at {dt}")

        # Get history up to and including dt
        history = symbol_df[symbol_df.index <= dt].copy()

        if len(history) < min_history_bars:
            # Not enough history, create NaN features
            features = {"timestamp": dt, "symbol": symbol}
            for h in horizons:
                features[f"mu_{h}"] = np.nan
                features[f"unc_{h}"] = np.nan
                features[f"q10_{h}"] = np.nan
                features[f"q90_{h}"] = np.nan
                features[f"skew_{h}"] = np.nan
            all_features.append(features)
            continue

        # Prepare history for prediction
        history_for_pred = history[[target_col]].copy()
        history_for_pred = history_for_pred.dropna()

        try:
            # Generate forecast
            forecast = generate_single_forecast(
                predictor, history_for_pred, symbol, target_col
            )

            # Extract features
            features = extract_forecast_features_from_prediction(
                forecast, symbol, dt, horizons
            )

        except Exception as e:
            if verbose:
                print(f"    Warning: Forecast failed at {dt}: {e}")
            features = {"timestamp": dt, "symbol": symbol}
            for h in horizons:
                features[f"mu_{h}"] = np.nan
                features[f"unc_{h}"] = np.nan
                features[f"q10_{h}"] = np.nan
                features[f"q90_{h}"] = np.nan
                features[f"skew_{h}"] = np.nan

        all_features.append(features)

    return pd.DataFrame(all_features)


def get_forecast_cache_path(
    base_path: Path,
    ts_model_run_id: str,
    symbol: str,
) -> Path:
    """Get cache path for forecast parquet."""
    return base_path / "data" / "processed" / "forecasts" / ts_model_run_id / f"{symbol}.parquet"


def load_or_generate_forecasts(
    predictor: Any,
    bars_df: pd.DataFrame,
    symbol: str,
    decision_timestamps: List[datetime],
    cache_path: Path,
    force_regenerate: bool = False,
    horizons: List[int] = None,
    min_history_bars: int = 100,
    target_col: str = "target",
    return_type: str = "log",
    price_col: str = "close",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load cached forecasts or generate new ones.

    Args:
        predictor: Trained TimeSeriesPredictor.
        bars_df: Full bars DataFrame.
        symbol: Symbol to forecast.
        decision_timestamps: Decision timestamps needed.
        cache_path: Path to cache parquet file.
        force_regenerate: If True, regenerate even if cache exists.
        ... (other args same as generate_rolling_forecasts)

    Returns:
        DataFrame with forecast features.
    """
    cache_path = Path(cache_path)

    # Try to load from cache
    if cache_path.exists() and not force_regenerate:
        print(f"Loading cached forecasts from {cache_path}")
        cached = pd.read_parquet(cache_path)

        # Check if cache covers needed timestamps
        cached_ts = set(pd.to_datetime(cached["timestamp"]))
        needed_ts = set(pd.to_datetime(decision_timestamps))

        if needed_ts.issubset(cached_ts):
            print(f"Cache covers all {len(needed_ts)} needed timestamps")
            # Filter to needed timestamps
            cached["timestamp"] = pd.to_datetime(cached["timestamp"])
            result = cached[cached["timestamp"].isin(needed_ts)]
            return result
        else:
            missing = len(needed_ts - cached_ts)
            print(f"Cache missing {missing} timestamps, regenerating...")

    # Generate forecasts
    print(f"Generating rolling forecasts for {symbol}...")
    forecasts = generate_rolling_forecasts(
        predictor=predictor,
        bars_df=bars_df,
        symbol=symbol,
        decision_timestamps=decision_timestamps,
        horizons=horizons,
        min_history_bars=min_history_bars,
        target_col=target_col,
        return_type=return_type,
        price_col=price_col,
        verbose=verbose,
    )

    # Save to cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(cache_path)
    print(f"Cached forecasts to {cache_path}")

    return forecasts
