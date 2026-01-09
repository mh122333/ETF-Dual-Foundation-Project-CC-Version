"""Feature registry for managing feature sets across experiments.

Provides a central place to define which features are used in each experiment
and ensure consistent feature selection.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

from .baseline import get_feature_columns
from .forecast_features import get_forecast_feature_names, FEATURE_SET_CONFIGS
from .context_features import count_context_features, DEFAULT_CONTEXT_SYMBOLS
from .forecast_error_features import get_error_feature_names


# Columns that must NEVER be used as features (label leakage)
FORBIDDEN_COLUMNS = [
    "label",
    "first_hit",
    "bars_to_hit",
    "realized_pnl_at_hit",
    "entry",
    "tp",
    "sl",
]

# Metadata columns that should not be features
METADATA_COLUMNS = [
    "symbol",
    "timestamp",
    "index",
]


@dataclass
class FeatureConfig:
    """Configuration for a feature set."""

    # Baseline features
    include_baseline: bool = True
    include_volume_zscore: bool = True

    # Forecast features
    include_forecast: bool = False
    forecast_feature_set: str = "small"  # small, medium, large
    forecast_prefix: str = "fc_"

    # Context features
    include_context: bool = False
    context_symbols: List[str] = field(default_factory=lambda: ["SPY"])
    include_relative: bool = True

    # Error features
    include_error: bool = False
    error_prefix: str = "err_"
    mae_window: int = 20
    bias_window: int = 20
    coverage_window: int = 50

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "include_baseline": self.include_baseline,
            "include_volume_zscore": self.include_volume_zscore,
            "include_forecast": self.include_forecast,
            "forecast_feature_set": self.forecast_feature_set,
            "forecast_prefix": self.forecast_prefix,
            "include_context": self.include_context,
            "context_symbols": self.context_symbols,
            "include_relative": self.include_relative,
            "include_error": self.include_error,
            "error_prefix": self.error_prefix,
            "mae_window": self.mae_window,
            "bias_window": self.bias_window,
            "coverage_window": self.coverage_window,
        }


class FeatureRegistry:
    """
    Registry for managing feature sets.

    Provides methods to get feature lists, validate features,
    and ensure no label leakage.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize registry with configuration.

        Args:
            config: Feature configuration. Uses defaults if None.
        """
        self.config = config or FeatureConfig()

    def get_baseline_features(self) -> List[str]:
        """Get baseline feature column names."""
        if not self.config.include_baseline:
            return []
        return get_feature_columns(self.config.include_volume_zscore)

    def get_forecast_features(self) -> List[str]:
        """Get forecast feature column names."""
        if not self.config.include_forecast:
            return []
        return get_forecast_feature_names(
            self.config.forecast_feature_set,
            self.config.forecast_prefix,
        )

    def get_context_features(self) -> List[str]:
        """Get context feature column names."""
        if not self.config.include_context:
            return []

        features = []
        fc_config = FEATURE_SET_CONFIGS.get(
            self.config.forecast_feature_set,
            FEATURE_SET_CONFIGS["small"]
        )
        horizons = fc_config["horizons"]
        fc_features = fc_config["features"]

        for ctx_sym in self.config.context_symbols:
            prefix = f"ctx_{ctx_sym.lower()}_"
            for feat in fc_features:
                for h in horizons:
                    features.append(f"{prefix}{feat}_{h}")

            if self.config.include_relative:
                for h in horizons:
                    features.append(f"rel_{ctx_sym.lower()}_mu_{h}")
                    features.append(f"rel_{ctx_sym.lower()}_unc_{h}")

        return features

    def get_error_features(self) -> List[str]:
        """Get error feature column names."""
        if not self.config.include_error:
            return []
        return get_error_feature_names(
            self.config.error_prefix,
            self.config.mae_window,
            self.config.bias_window,
            self.config.coverage_window,
        )

    def get_all_features(self) -> List[str]:
        """Get all feature column names based on configuration."""
        features = []
        features.extend(self.get_baseline_features())
        features.extend(self.get_forecast_features())
        features.extend(self.get_context_features())
        features.extend(self.get_error_features())
        return features

    def get_feature_count(self) -> int:
        """Get total number of features."""
        return len(self.get_all_features())

    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a DataFrame has expected features and no forbidden columns.

        Args:
            df: DataFrame to validate.

        Returns:
            Dictionary with validation results.
        """
        expected = set(self.get_all_features())
        available = set(df.columns)

        # Check for forbidden columns
        forbidden_present = [c for c in FORBIDDEN_COLUMNS if c in available]

        # Check for missing features
        missing = expected - available

        # Check for extra features (not forbidden, not metadata)
        non_feature_cols = set(FORBIDDEN_COLUMNS + METADATA_COLUMNS)
        extra = available - expected - non_feature_cols

        return {
            "valid": len(forbidden_present) == 0,
            "forbidden_present": forbidden_present,
            "missing_features": list(missing),
            "extra_columns": list(extra),
            "expected_count": len(expected),
            "available_count": len(available - non_feature_cols),
        }

    def filter_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter DataFrame to only include configured features.

        Args:
            df: DataFrame with all columns.

        Returns:
            DataFrame with only feature columns.
        """
        expected = self.get_all_features()
        available = [c for c in expected if c in df.columns]
        return df[available].copy()

    def get_info(self) -> Dict[str, Any]:
        """Get information about the feature configuration."""
        return {
            "config": self.config.to_dict(),
            "baseline_features": self.get_baseline_features(),
            "forecast_features": self.get_forecast_features(),
            "context_features": self.get_context_features(),
            "error_features": self.get_error_features(),
            "total_features": self.get_feature_count(),
        }


# Predefined configurations for each experiment
EXPERIMENT_CONFIGS = {
    "exp1": FeatureConfig(
        include_baseline=True,
        include_forecast=False,
        include_context=False,
        include_error=False,
    ),
    "exp2": FeatureConfig(
        include_baseline=True,
        include_forecast=True,
        forecast_feature_set="small",
        include_context=False,
        include_error=False,
    ),
    "exp3": FeatureConfig(
        include_baseline=True,
        include_forecast=True,
        forecast_feature_set="small",
        include_context=True,
        context_symbols=["SPY"],
        include_relative=True,
        include_error=False,
    ),
    "exp4": FeatureConfig(
        include_baseline=True,
        include_forecast=True,
        forecast_feature_set="small",
        include_context=True,
        context_symbols=["SPY"],
        include_relative=True,
        include_error=True,
    ),
    "exp5_small": FeatureConfig(
        include_baseline=True,
        include_forecast=True,
        forecast_feature_set="small",
        include_context=True,
        context_symbols=["SPY"],
        include_relative=True,
        include_error=True,
    ),
    "exp5_medium": FeatureConfig(
        include_baseline=True,
        include_forecast=True,
        forecast_feature_set="medium",
        include_context=True,
        context_symbols=["SPY", "QQQ"],
        include_relative=True,
        include_error=True,
    ),
    "exp5_large": FeatureConfig(
        include_baseline=True,
        include_forecast=True,
        forecast_feature_set="large",
        include_context=True,
        context_symbols=["SPY", "QQQ"],
        include_relative=True,
        include_error=True,
    ),
}


def get_default_registry(experiment: str = "exp1") -> FeatureRegistry:
    """
    Get a feature registry with default configuration for an experiment.

    Args:
        experiment: Experiment name (exp1, exp2, exp3, exp4, exp5_small, etc.)

    Returns:
        FeatureRegistry configured for the experiment.
    """
    config = EXPERIMENT_CONFIGS.get(experiment, EXPERIMENT_CONFIGS["exp1"])
    return FeatureRegistry(config)
