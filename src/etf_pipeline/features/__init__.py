"""Feature engineering modules."""

from .baseline import compute_baseline_features, get_feature_columns
from .forecast_features import (
    FEATURE_SET_CONFIGS,
    get_forecast_feature_names,
    pivot_forecasts_to_features,
    merge_forecast_features,
    build_forecast_feature_dataset,
    count_forecast_features,
    get_feature_set_info,
)
from .context_features import (
    DEFAULT_CONTEXT_SYMBOLS,
    DEFAULT_SECTOR_MAP,
    get_context_symbols,
    compute_relative_features,
    add_context_features,
    build_context_feature_dataset,
    count_context_features,
)
from .forecast_error_features import (
    compute_forecast_errors,
    compute_rolling_error_features,
    build_forecast_error_features,
    add_forecast_error_features,
    get_error_feature_names,
    count_error_features,
)
from .registry import FeatureRegistry, get_default_registry

__all__ = [
    # Baseline
    "compute_baseline_features",
    "get_feature_columns",
    # Forecast features
    "FEATURE_SET_CONFIGS",
    "get_forecast_feature_names",
    "pivot_forecasts_to_features",
    "merge_forecast_features",
    "build_forecast_feature_dataset",
    "count_forecast_features",
    "get_feature_set_info",
    # Context features
    "DEFAULT_CONTEXT_SYMBOLS",
    "DEFAULT_SECTOR_MAP",
    "get_context_symbols",
    "compute_relative_features",
    "add_context_features",
    "build_context_feature_dataset",
    "count_context_features",
    # Error features
    "compute_forecast_errors",
    "compute_rolling_error_features",
    "build_forecast_error_features",
    "add_forecast_error_features",
    "get_error_feature_names",
    "count_error_features",
    # Registry
    "FeatureRegistry",
    "get_default_registry",
]
