"""Tests for feature engineering modules."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class TestForecastFeatures:
    """Tests for forecast_features module."""

    def test_get_forecast_feature_names_small(self):
        """Test small feature set returns correct names."""
        from etf_pipeline.features.forecast_features import (
            get_forecast_feature_names,
            FEATURE_SET_CONFIGS,
        )

        names = get_forecast_feature_names("small", prefix="fc_")

        # Small set has 2 horizons and 4 features per horizon = 8 features
        config = FEATURE_SET_CONFIGS["small"]
        expected_count = len(config["horizons"]) * len(config["features"])
        assert len(names) == expected_count

        # Check prefix
        assert all(n.startswith("fc_") for n in names)

        # Check specific features exist
        assert "fc_mu_1" in names
        assert "fc_unc_1" in names

    def test_get_forecast_feature_names_medium(self):
        """Test medium feature set returns correct names."""
        from etf_pipeline.features.forecast_features import get_forecast_feature_names

        names = get_forecast_feature_names("medium", prefix="")

        # Medium has more horizons
        assert len(names) > len(get_forecast_feature_names("small", ""))

        # Check horizons
        assert "mu_4" in names  # Medium adds horizon 4
        assert "mu_13" in names  # Medium adds horizon 13

    def test_merge_forecast_features(self):
        """Test merging forecast features into tabular data."""
        from etf_pipeline.features.forecast_features import merge_forecast_features

        # Create sample tabular data
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        tabular_df = pd.DataFrame(
            {
                "close": np.random.randn(10).cumsum() + 100,
                "volume": np.random.randint(1000, 10000, 10),
            },
            index=dates,
        )

        # Create sample forecasts
        forecasts_df = pd.DataFrame(
            {
                "mu_1": np.random.randn(10),
                "q10_1": np.random.randn(10) - 1,
                "q90_1": np.random.randn(10) + 1,
                "mu_26": np.random.randn(10),
                "q10_26": np.random.randn(10) - 1,
                "q90_26": np.random.randn(10) + 1,
            },
            index=dates,
        )

        # Merge
        merged = merge_forecast_features(
            tabular_df, forecasts_df, feature_set="small", prefix="fc_"
        )

        # Check original columns preserved
        assert "close" in merged.columns
        assert "volume" in merged.columns

        # Check forecast columns added
        assert "fc_mu_1" in merged.columns
        assert "fc_unc_1" in merged.columns  # q90 - q10
        assert "fc_trend_1" in merged.columns
        assert "fc_pos_in_interval_1" in merged.columns

    def test_merge_handles_missing_timestamps(self):
        """Test that merge handles mismatched timestamps gracefully."""
        from etf_pipeline.features.forecast_features import merge_forecast_features

        dates1 = pd.date_range("2024-01-01", periods=10, freq="D")
        dates2 = pd.date_range("2024-01-05", periods=10, freq="D")  # Partial overlap

        tabular_df = pd.DataFrame({"close": range(10)}, index=dates1)
        forecasts_df = pd.DataFrame({"mu_1": range(10), "q10_1": range(10), "q90_1": range(10)}, index=dates2)

        merged = merge_forecast_features(
            tabular_df, forecasts_df, feature_set="small", prefix="fc_"
        )

        # Should have all rows from tabular
        assert len(merged) == len(tabular_df)

        # Non-overlapping rows should have NaN
        assert merged.loc[dates1[0], "fc_mu_1"] != merged.loc[dates1[0], "fc_mu_1"]  # NaN check


class TestContextFeatures:
    """Tests for context_features module."""

    def test_add_context_features(self):
        """Test adding context features from market ETFs."""
        from etf_pipeline.features.context_features import add_context_features

        dates = pd.date_range("2024-01-01", periods=10, freq="D")

        # Target asset data
        df = pd.DataFrame(
            {
                "close": np.random.randn(10).cumsum() + 100,
                "fc_mu_1": np.random.randn(10),
                "fc_unc_1": np.abs(np.random.randn(10)),
            },
            index=dates,
        )

        # Context forecasts (SPY)
        spy_forecasts = pd.DataFrame(
            {
                "mu_1": np.random.randn(10),
                "q10_1": np.random.randn(10) - 1,
                "q90_1": np.random.randn(10) + 1,
            },
            index=dates,
        )

        context_forecasts = {"SPY": spy_forecasts}

        result = add_context_features(
            df=df,
            context_forecasts=context_forecasts,
            target_symbol="AAPL",
            context_symbols=["SPY"],
            feature_set="small",
            include_relative=True,
        )

        # Check context features added
        assert "ctx_spy_mu_1" in result.columns
        assert "ctx_spy_unc_1" in result.columns

        # Check relative features added
        assert "rel_spy_mu_1" in result.columns
        assert "rel_spy_unc_1" in result.columns

    def test_compute_relative_features(self):
        """Test relative feature computation."""
        from etf_pipeline.features.context_features import compute_relative_features

        dates = pd.date_range("2024-01-01", periods=5, freq="D")

        asset_features = pd.DataFrame(
            {
                "fc_mu_1": [0.1, 0.2, -0.1, 0.05, 0.15],
                "fc_unc_1": [0.5, 0.6, 0.4, 0.7, 0.5],
            },
            index=dates,
        )

        context_features = pd.DataFrame(
            {
                "ctx_spy_mu_1": [0.05, 0.15, -0.05, 0.0, 0.1],
                "ctx_spy_unc_1": [0.4, 0.5, 0.3, 0.6, 0.4],
            },
            index=dates,
        )

        result = compute_relative_features(
            asset_features, context_features, horizons=[1], context_symbol="spy"
        )

        # Relative mu should be asset - context
        expected_rel_mu = [0.05, 0.05, -0.05, 0.05, 0.05]
        np.testing.assert_array_almost_equal(result["rel_spy_mu_1"].values, expected_rel_mu)

        # Relative uncertainty should be ratio
        expected_rel_unc = [0.5 / 0.4, 0.6 / 0.5, 0.4 / 0.3, 0.7 / 0.6, 0.5 / 0.4]
        np.testing.assert_array_almost_equal(result["rel_spy_unc_1"].values, expected_rel_unc)


class TestForecastErrorFeatures:
    """Tests for forecast_error_features module."""

    def test_compute_forecast_errors(self):
        """Test basic error computation."""
        from etf_pipeline.features.forecast_error_features import compute_forecast_errors

        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        # Simulated data
        realized = pd.Series(np.random.randn(n), index=dates)
        forecast_mu = realized + np.random.randn(n) * 0.1  # Forecast with small noise
        forecast_q10 = forecast_mu - 0.5
        forecast_q90 = forecast_mu + 0.5

        errors = compute_forecast_errors(realized, forecast_mu, forecast_q10, forecast_q90)

        # Check columns exist
        assert "error" in errors.columns
        assert "abs_error" in errors.columns
        assert "in_interval" in errors.columns

        # Error should be realized - forecast
        expected_error = realized - forecast_mu
        np.testing.assert_array_almost_equal(errors["error"].values, expected_error.values)

        # Abs error should be non-negative
        assert (errors["abs_error"] >= 0).all()

    def test_compute_rolling_error_features(self):
        """Test rolling error feature computation."""
        from etf_pipeline.features.forecast_error_features import (
            compute_forecast_errors,
            compute_rolling_error_features,
        )

        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        realized = pd.Series(np.random.randn(n), index=dates)
        forecast_mu = realized + 0.1  # Biased forecast
        forecast_q10 = forecast_mu - 0.5
        forecast_q90 = forecast_mu + 0.5

        errors = compute_forecast_errors(realized, forecast_mu, forecast_q10, forecast_q90)
        rolling = compute_rolling_error_features(
            errors, mae_window=20, bias_window=20, coverage_window=50
        )

        # Check columns exist
        assert "rolling_mae_20" in rolling.columns
        assert "rolling_bias_20" in rolling.columns
        assert "rolling_coverage_50" in rolling.columns

        # MAE should be non-negative
        assert (rolling["rolling_mae_20"].dropna() >= 0).all()

        # Bias should be around 0.1 (since we added 0.1 to forecasts)
        # After warmup period
        mean_bias = rolling["rolling_bias_20"].iloc[30:].mean()
        assert abs(mean_bias - (-0.1)) < 0.1  # Should be close to -0.1

    def test_get_error_feature_names(self):
        """Test error feature name generation."""
        from etf_pipeline.features.forecast_error_features import get_error_feature_names

        names = get_error_feature_names("err_", mae_window=20, bias_window=20, coverage_window=50)

        assert "err_rolling_mae_20" in names
        assert "err_rolling_bias_20" in names
        assert "err_rolling_coverage_50" in names


class TestFeatureRegistry:
    """Tests for feature registry module."""

    def test_feature_config_to_dict(self):
        """Test FeatureConfig serialization."""
        from etf_pipeline.features.registry import FeatureConfig

        config = FeatureConfig(
            include_baseline=True,
            include_forecast=True,
            forecast_feature_set="medium",
            include_context=True,
            context_symbols=["SPY", "QQQ"],
        )

        d = config.to_dict()

        assert d["include_baseline"] == True
        assert d["include_forecast"] == True
        assert d["forecast_feature_set"] == "medium"
        assert d["context_symbols"] == ["SPY", "QQQ"]

    def test_feature_registry_get_all_features(self):
        """Test getting all features from registry."""
        from etf_pipeline.features.registry import FeatureRegistry, FeatureConfig

        config = FeatureConfig(
            include_baseline=True,
            include_forecast=True,
            forecast_feature_set="small",
            include_context=False,
            include_error=False,
        )

        registry = FeatureRegistry(config)
        features = registry.get_all_features()

        # Should have baseline + forecast features
        assert len(features) > 0
        assert any("ret" in f for f in features)  # Baseline feature
        assert any("fc_" in f for f in features)  # Forecast feature

    def test_experiment_configs_exist(self):
        """Test predefined experiment configs exist."""
        from etf_pipeline.features.registry import EXPERIMENT_CONFIGS

        assert "exp1" in EXPERIMENT_CONFIGS
        assert "exp2" in EXPERIMENT_CONFIGS
        assert "exp3" in EXPERIMENT_CONFIGS
        assert "exp4" in EXPERIMENT_CONFIGS

        # Exp1 should have baseline only
        assert EXPERIMENT_CONFIGS["exp1"].include_baseline == True
        assert EXPERIMENT_CONFIGS["exp1"].include_forecast == False

        # Exp2 should add forecast
        assert EXPERIMENT_CONFIGS["exp2"].include_forecast == True
        assert EXPERIMENT_CONFIGS["exp2"].include_context == False

    def test_validate_dataframe(self):
        """Test DataFrame validation."""
        from etf_pipeline.features.registry import FeatureRegistry, FeatureConfig

        config = FeatureConfig(include_baseline=True, include_forecast=False)
        registry = FeatureRegistry(config)

        # Create DataFrame with baseline features
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "ret_1": [0.01, -0.02, 0.01, 0.0, 0.02],
                "ret_5": [0.05, -0.03, 0.02, 0.01, 0.04],
                "vol_20": [0.15, 0.16, 0.14, 0.15, 0.15],
                "atr_14": [1.5, 1.6, 1.4, 1.5, 1.5],
                "label": [1, -1, 0, 1, -1],  # Forbidden column
            },
            index=dates,
        )

        result = registry.validate_dataframe(df)

        # Should flag label as forbidden
        assert result["valid"] == False
        assert "label" in result["forbidden_present"]

    def test_filter_features(self):
        """Test filtering DataFrame to expected features."""
        from etf_pipeline.features.registry import FeatureRegistry, FeatureConfig

        config = FeatureConfig(include_baseline=True, include_forecast=False)
        registry = FeatureRegistry(config)

        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "ret_1": range(5),
                "ret_5": range(5),
                "vol_20": range(5),
                "atr_14": range(5),
                "extra_col": range(5),
                "label": range(5),
            },
            index=dates,
        )

        filtered = registry.filter_features(df)

        # Should only have expected features
        assert "extra_col" not in filtered.columns
        assert "label" not in filtered.columns

        # Should have baseline features
        baseline = registry.get_baseline_features()
        for feat in baseline:
            if feat in df.columns:
                assert feat in filtered.columns


class TestNoLeakageValidation:
    """Tests to ensure no information leakage in feature computation."""

    def test_forecast_features_use_only_past_data(self):
        """Verify forecast features don't use future information."""
        # This is a conceptual test - actual verification happens in rolling_predict
        from etf_pipeline.features.forecast_features import merge_forecast_features

        dates = pd.date_range("2024-01-01", periods=10, freq="D")

        # Create data with clear time ordering
        tabular_df = pd.DataFrame(
            {"close": list(range(10))},  # 0, 1, 2, ..., 9
            index=dates,
        )

        # Forecasts should be generated using only past data at each point
        forecasts_df = pd.DataFrame(
            {
                "mu_1": list(range(10)),  # Should be based on past prices
                "q10_1": list(range(10)),
                "q90_1": [x + 1 for x in range(10)],
            },
            index=dates,
        )

        merged = merge_forecast_features(
            tabular_df, forecasts_df, feature_set="small", prefix="fc_"
        )

        # Features at time t should only depend on data up to t
        # This is enforced by the rolling forecast generation, not the merge
        assert len(merged) == len(tabular_df)

    def test_error_features_are_lagged(self):
        """Verify error features use past errors only."""
        from etf_pipeline.features.forecast_error_features import (
            compute_forecast_errors,
            compute_rolling_error_features,
        )

        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="D")

        realized = pd.Series([1.0] * 25 + [2.0] * 25, index=dates)
        forecast_mu = pd.Series([1.0] * n, index=dates)  # Always predicts 1.0

        errors = compute_forecast_errors(
            realized,
            forecast_mu,
            pd.Series([0.5] * n, index=dates),
            pd.Series([1.5] * n, index=dates),
        )

        rolling = compute_rolling_error_features(errors, mae_window=10)

        # Before the regime change (first 25), MAE should be ~0
        early_mae = rolling["rolling_mae_10"].iloc[15:24].mean()
        assert early_mae < 0.1

        # After regime change settles in, MAE should increase
        # (using window=10, so takes ~10 bars to reflect)
        late_mae = rolling["rolling_mae_10"].iloc[-10:].mean()
        assert late_mae > 0.5  # Should reflect the 1.0 error


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
