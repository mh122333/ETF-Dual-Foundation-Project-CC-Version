"""Baseline feature engineering.

All features are computed causally: features at time t use only data <= t.
"""

import numpy as np
import pandas as pd


def compute_baseline_features(
    df: pd.DataFrame,
    vol_window: int = 20,
    vol_zscore_window: int = 50,
    include_volume_zscore: bool = True,
) -> pd.DataFrame:
    """
    Compute baseline features for a single symbol.

    All features are causal: computed using data up to and including time t.

    Features computed:
    - ret_1: 1-bar close-to-close return
    - ret_2: 2-bar close-to-close return
    - rolling_vol: Rolling volatility (std of returns) over vol_window bars
    - atr: ATR value (should already be in df from labeling)
    - atr_over_close: ATR normalized by close price
    - volume_zscore: Volume z-score over vol_zscore_window (optional)

    Args:
        df: DataFrame with columns: open, high, low, close, volume, atr.
            Must have DatetimeIndex sorted ascending.
        vol_window: Window for rolling volatility (default 20).
        vol_zscore_window: Window for volume z-score (default 50).
        include_volume_zscore: Whether to include volume z-score feature.

    Returns:
        DataFrame with added feature columns.
    """
    df = df.copy()

    # Ensure sorted by time
    df = df.sort_index()

    # 1-bar return: (close_t - close_{t-1}) / close_{t-1}
    df["ret_1"] = df["close"].pct_change(1)

    # 2-bar return: (close_t - close_{t-2}) / close_{t-2}
    df["ret_2"] = df["close"].pct_change(2)

    # Rolling volatility: std of 1-bar returns
    df["rolling_vol"] = df["ret_1"].rolling(window=vol_window, min_periods=vol_window).std()

    # ATR over close (normalized ATR)
    if "atr" in df.columns:
        df["atr_over_close"] = df["atr"] / df["close"]
    else:
        # Compute ATR if not present
        from ..labels.triple_barrier import compute_atr

        df["atr"] = compute_atr(df["high"], df["low"], df["close"], window=14)
        df["atr_over_close"] = df["atr"] / df["close"]

    # Volume z-score (optional)
    if include_volume_zscore and "volume" in df.columns:
        vol_mean = df["volume"].rolling(window=vol_zscore_window, min_periods=vol_zscore_window).mean()
        vol_std = df["volume"].rolling(window=vol_zscore_window, min_periods=vol_zscore_window).std()
        df["volume_zscore"] = (df["volume"] - vol_mean) / vol_std
        # Handle division by zero
        df["volume_zscore"] = df["volume_zscore"].replace([np.inf, -np.inf], np.nan)

    return df


def compute_baseline_features_multi(
    df: pd.DataFrame,
    vol_window: int = 20,
    vol_zscore_window: int = 50,
    include_volume_zscore: bool = True,
) -> pd.DataFrame:
    """
    Compute baseline features for multiple symbols.

    Args:
        df: DataFrame with MultiIndex (symbol, timestamp) and bar columns.
        vol_window: Window for rolling volatility.
        vol_zscore_window: Window for volume z-score.
        include_volume_zscore: Whether to include volume z-score feature.

    Returns:
        DataFrame with features computed per symbol.
    """
    results = []

    for symbol in df.index.get_level_values("symbol").unique():
        symbol_df = df.loc[symbol].copy()
        featured = compute_baseline_features(
            symbol_df,
            vol_window=vol_window,
            vol_zscore_window=vol_zscore_window,
            include_volume_zscore=include_volume_zscore,
        )
        featured["symbol"] = symbol
        results.append(featured)

    combined = pd.concat(results, axis=0)
    combined = combined.reset_index()
    combined = combined.rename(columns={"index": "timestamp"})
    combined = combined.set_index(["symbol", "timestamp"]).sort_index()

    return combined


def get_feature_columns(include_volume_zscore: bool = True) -> list:
    """
    Get list of baseline feature column names.

    Args:
        include_volume_zscore: Whether volume z-score is included.

    Returns:
        List of feature column names.
    """
    cols = ["ret_1", "ret_2", "rolling_vol", "atr", "atr_over_close"]
    if include_volume_zscore:
        cols.append("volume_zscore")
    return cols
