"""Context asset forecast features.

Adds market/sector context by including forecast features from reference
assets (SPY, QQQ, sector ETFs) and computing relative features.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path

import pandas as pd
import numpy as np

from .forecast_features import (
    FEATURE_SET_CONFIGS,
    pivot_forecasts_to_features,
    get_forecast_feature_names,
)


# Default context configuration
DEFAULT_CONTEXT_SYMBOLS = ["SPY", "QQQ"]

# Sector mapping for individual stocks
DEFAULT_SECTOR_MAP = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "GOOGL": "XLC",
    "AMZN": "XLY",
    "META": "XLC",
    "NVDA": "XLK",
    "TSLA": "XLY",
}


def get_context_symbols(
    target_symbol: str,
    context_symbols: List[str] = None,
    sector_map: Dict[str, str] = None,
    include_sector: bool = True,
) -> List[str]:
    """
    Get context symbols for a target asset.

    Args:
        target_symbol: The target symbol being predicted.
        context_symbols: Base context symbols (default: SPY, QQQ).
        sector_map: Mapping of symbols to sector ETFs.
        include_sector: Whether to include sector ETF.

    Returns:
        List of context symbols to use.
    """
    if context_symbols is None:
        context_symbols = DEFAULT_CONTEXT_SYMBOLS.copy()
    else:
        context_symbols = list(context_symbols)

    if include_sector and sector_map:
        sector = sector_map.get(target_symbol)
        if sector and sector not in context_symbols:
            context_symbols.append(sector)

    # Don't include target as its own context
    context_symbols = [s for s in context_symbols if s != target_symbol]

    return context_symbols


def compute_relative_features(
    asset_features: pd.DataFrame,
    context_features: pd.DataFrame,
    horizons: List[int],
    asset_prefix: str = "fc_",
    context_prefix: str = "",
    eps: float = 1e-8,
) -> pd.DataFrame:
    """
    Compute relative features between asset and context.

    Args:
        asset_features: Asset forecast features.
        context_features: Context asset forecast features.
        horizons: Horizons for which to compute relative features.
        asset_prefix: Prefix for asset feature columns.
        context_prefix: Prefix for context feature columns.
        eps: Small value to avoid division by zero.

    Returns:
        DataFrame with relative feature columns.
    """
    result = pd.DataFrame(index=asset_features.index)

    for h in horizons:
        # Relative mean: asset_mu - context_mu
        asset_mu_col = f"{asset_prefix}mu_{h}"
        ctx_mu_col = f"{context_prefix}mu_{h}"

        if asset_mu_col in asset_features.columns and ctx_mu_col in context_features.columns:
            result[f"rel_mu_{h}"] = (
                asset_features[asset_mu_col].values -
                context_features[ctx_mu_col].values
            )

        # Relative uncertainty: asset_unc / context_unc
        asset_unc_col = f"{asset_prefix}unc_{h}"
        ctx_unc_col = f"{context_prefix}unc_{h}"

        if asset_unc_col in asset_features.columns and ctx_unc_col in context_features.columns:
            ctx_unc = context_features[ctx_unc_col].values
            asset_unc = asset_features[asset_unc_col].values
            result[f"rel_unc_{h}"] = asset_unc / (np.abs(ctx_unc) + eps)

    return result


def add_context_features(
    df: pd.DataFrame,
    context_forecasts: Dict[str, pd.DataFrame],
    target_symbol: str,
    context_symbols: List[str],
    feature_set: str = "small",
    include_relative: bool = True,
) -> pd.DataFrame:
    """
    Add context asset forecast features to dataset.

    Args:
        df: Base dataset with asset forecast features.
        context_forecasts: Dict mapping context symbol to forecast DataFrame.
        target_symbol: Target symbol.
        context_symbols: Context symbols to include.
        feature_set: Feature set configuration.
        include_relative: Whether to add relative features.

    Returns:
        DataFrame with context features added.
    """
    result = df.copy()

    # Ensure timestamp is datetime
    if "timestamp" not in result.columns:
        result = result.reset_index()
    result["timestamp"] = pd.to_datetime(result["timestamp"])

    config = FEATURE_SET_CONFIGS.get(feature_set, FEATURE_SET_CONFIGS["small"])
    horizons = config["horizons"]

    for ctx_symbol in context_symbols:
        if ctx_symbol not in context_forecasts:
            print(f"Warning: No forecast data for context symbol {ctx_symbol}")
            continue

        ctx_fc = context_forecasts[ctx_symbol].copy()
        ctx_fc["timestamp"] = pd.to_datetime(ctx_fc["timestamp"])

        # Pivot context forecasts with symbol-specific prefix
        ctx_prefix = f"ctx_{ctx_symbol.lower()}_"
        ctx_features = pivot_forecasts_to_features(ctx_fc, feature_set, ctx_prefix)

        # Merge on timestamp
        ctx_features = ctx_features.drop(columns=["symbol"], errors="ignore")
        result = result.merge(ctx_features, on="timestamp", how="left")

        # Add relative features if requested
        if include_relative:
            # Get asset feature columns
            asset_cols = [c for c in result.columns if c.startswith("fc_")]
            if asset_cols:
                asset_features = result[asset_cols]
                ctx_cols = [c for c in result.columns if c.startswith(ctx_prefix)]
                ctx_features_df = result[ctx_cols].copy()

                # Rename context columns to remove prefix for relative computation
                ctx_rename = {c: c.replace(ctx_prefix, "") for c in ctx_cols}
                ctx_features_df = ctx_features_df.rename(columns=ctx_rename)

                rel_features = compute_relative_features(
                    asset_features, ctx_features_df, horizons,
                    asset_prefix="fc_", context_prefix="", eps=1e-8
                )

                # Add relative features with context-specific prefix
                rel_rename = {c: f"rel_{ctx_symbol.lower()}_{c.replace('rel_', '')}" for c in rel_features.columns}
                rel_features = rel_features.rename(columns=rel_rename)

                for col in rel_features.columns:
                    result[col] = rel_features[col].values

    return result


def build_context_feature_dataset(
    base_df: pd.DataFrame,
    forecast_cache_dir: Path,
    ts_model_run_id: str,
    target_symbols: List[str],
    context_symbols: List[str] = None,
    sector_map: Dict[str, str] = None,
    feature_set: str = "small",
    include_relative: bool = True,
) -> pd.DataFrame:
    """
    Build dataset with context features for multiple target symbols.

    Args:
        base_df: Base dataset with asset forecast features.
        forecast_cache_dir: Directory with cached forecasts.
        ts_model_run_id: TS model run ID.
        target_symbols: Target symbols.
        context_symbols: Context symbols.
        sector_map: Sector mapping.
        feature_set: Feature set configuration.
        include_relative: Whether to include relative features.

    Returns:
        DataFrame with context features added.
    """
    if context_symbols is None:
        context_symbols = DEFAULT_CONTEXT_SYMBOLS

    # Load all context forecasts
    all_context_symbols = set(context_symbols)
    if sector_map:
        all_context_symbols.update(sector_map.values())

    context_forecasts = {}
    for ctx_sym in all_context_symbols:
        cache_path = forecast_cache_dir / "forecasts" / ts_model_run_id / f"{ctx_sym}.parquet"
        if cache_path.exists():
            context_forecasts[ctx_sym] = pd.read_parquet(cache_path)
        else:
            print(f"Warning: No cached forecasts for context symbol {ctx_sym}")

    # Process each target symbol
    result_dfs = []

    for target in target_symbols:
        # Get data for this target
        if "symbol" in base_df.columns:
            target_df = base_df[base_df["symbol"] == target].copy()
        elif isinstance(base_df.index, pd.MultiIndex):
            target_df = base_df.loc[target].copy()
            target_df["symbol"] = target
        else:
            target_df = base_df.copy()
            target_df["symbol"] = target

        # Get context symbols for this target
        ctx_for_target = get_context_symbols(
            target, context_symbols, sector_map, include_sector=True
        )

        # Add context features
        target_df = add_context_features(
            target_df,
            context_forecasts,
            target,
            ctx_for_target,
            feature_set,
            include_relative,
        )

        result_dfs.append(target_df)

    if not result_dfs:
        return base_df

    return pd.concat(result_dfs, ignore_index=True)


def count_context_features(
    n_context_symbols: int = 1,
    feature_set: str = "small",
    include_relative: bool = True,
) -> int:
    """
    Count number of context features.

    Args:
        n_context_symbols: Number of context symbols.
        feature_set: Feature set configuration.
        include_relative: Whether relative features are included.

    Returns:
        Total number of context features.
    """
    # Features per context symbol
    n_fc_features = len(get_forecast_feature_names(feature_set))

    # Relative features per context (mu and unc for each horizon)
    config = FEATURE_SET_CONFIGS.get(feature_set, FEATURE_SET_CONFIGS["small"])
    n_rel_features = len(config["horizons"]) * 2 if include_relative else 0

    return n_context_symbols * (n_fc_features + n_rel_features)
