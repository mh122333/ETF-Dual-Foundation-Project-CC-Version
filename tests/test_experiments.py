"""Tests for experiment infrastructure modules."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile
import json


class TestExperimentConfig:
    """Tests for ExperimentConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        from etf_pipeline.experiments.runner import ExperimentConfig

        config = ExperimentConfig()

        assert config.experiment_name == "exp1"
        assert config.symbols == ["SPY"]
        assert config.vertical_barrier_bars == 26
        assert config.embargo_bars == 26
        assert config.random_seed == 42

    def test_auto_run_id_generation(self):
        """Test that run_id is auto-generated if not provided."""
        from etf_pipeline.experiments.runner import ExperimentConfig

        config = ExperimentConfig(experiment_name="test_exp")

        assert config.run_id is not None
        assert config.run_id.startswith("test_exp_")

    def test_explicit_run_id(self):
        """Test that explicit run_id is preserved."""
        from etf_pipeline.experiments.runner import ExperimentConfig

        config = ExperimentConfig(run_id="my_custom_run")

        assert config.run_id == "my_custom_run"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        from etf_pipeline.experiments.runner import ExperimentConfig

        config = ExperimentConfig(
            experiment_name="exp2",
            symbols=["AAPL", "MSFT"],
            feature_set="medium",
        )

        d = config.to_dict()

        assert d["experiment_name"] == "exp2"
        assert d["symbols"] == ["AAPL", "MSFT"]
        assert d["feature_set"] == "medium"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        from etf_pipeline.experiments.runner import ExperimentConfig

        d = {
            "experiment_name": "exp3",
            "symbols": ["QQQ"],
            "time_limit_sec": 600,
        }

        config = ExperimentConfig.from_dict(d)

        assert config.experiment_name == "exp3"
        assert config.symbols == ["QQQ"]
        assert config.time_limit_sec == 600

    def test_save_and_load(self):
        """Test saving and loading config to/from YAML."""
        from etf_pipeline.experiments.runner import ExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(
                experiment_name="test",
                symbols=["SPY"],
                run_id="test_run_001",
            )

            path = Path(tmpdir) / "config.yaml"
            config.save(path)

            assert path.exists()

            loaded = ExperimentConfig.load(path)

            assert loaded.experiment_name == config.experiment_name
            assert loaded.symbols == config.symbols


class TestExperimentRunner:
    """Tests for ExperimentRunner class."""

    def test_initialization(self):
        """Test runner initialization."""
        from etf_pipeline.experiments.runner import ExperimentRunner, ExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(experiment_name="test")
            runner = ExperimentRunner(config, drive_base=Path(tmpdir))

            assert runner.config == config
            assert "raw" in runner.paths
            assert "processed" in runner.paths
            assert "runs" in runner.paths
            assert "models" in runner.paths

    def test_ensure_directories(self):
        """Test directory creation."""
        from etf_pipeline.experiments.runner import ExperimentRunner, ExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(experiment_name="test")
            runner = ExperimentRunner(config, drive_base=Path(tmpdir))
            runner.ensure_directories()

            for path in runner.paths.values():
                assert path.exists()
                assert path.is_dir()

    def test_get_run_dir(self):
        """Test run directory path generation."""
        from etf_pipeline.experiments.runner import ExperimentRunner, ExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(run_id="test_run_123")
            runner = ExperimentRunner(config, drive_base=Path(tmpdir))

            run_dir = runner.get_run_dir()

            assert "test_run_123" in str(run_dir)

    def test_save_predictions(self):
        """Test saving predictions to parquet."""
        from etf_pipeline.experiments.runner import ExperimentRunner, ExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(run_id="test_run")
            runner = ExperimentRunner(config, drive_base=Path(tmpdir))

            predictions = pd.DataFrame(
                {
                    "predicted_label": [1, -1, 0, 1],
                    "prob_-1": [0.1, 0.7, 0.2, 0.1],
                    "prob_0": [0.2, 0.2, 0.6, 0.2],
                    "prob_1": [0.7, 0.1, 0.2, 0.7],
                }
            )

            path = runner.save_predictions(predictions, "SPY")

            assert path.exists()
            loaded = pd.read_parquet(path)
            assert len(loaded) == 4
            assert "predicted_label" in loaded.columns

    def test_save_metrics(self):
        """Test saving metrics to JSON."""
        from etf_pipeline.experiments.runner import ExperimentRunner, ExperimentConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            config = ExperimentConfig(run_id="test_run")
            runner = ExperimentRunner(config, drive_base=Path(tmpdir))

            metrics = {
                "accuracy": 0.65,
                "balanced_accuracy": 0.62,
                "macro_f1": 0.58,
            }

            path = runner.save_metrics(metrics, "SPY")

            assert path.exists()
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["accuracy"] == 0.65


class TestResultsModule:
    """Tests for results aggregation module."""

    def test_load_metrics(self):
        """Test loading metrics from run directory."""
        from etf_pipeline.experiments.results import load_metrics

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_001"
            run_dir.mkdir()

            metrics = {"accuracy": 0.7, "balanced_accuracy": 0.65}
            metrics_path = run_dir / "metrics_SPY.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f)

            loaded = load_metrics(run_dir, "SPY")

            assert loaded["accuracy"] == 0.7
            assert loaded["balanced_accuracy"] == 0.65

    def test_load_predictions(self):
        """Test loading predictions from run directory."""
        from etf_pipeline.experiments.results import load_predictions

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run_001"
            run_dir.mkdir()

            predictions = pd.DataFrame({"predicted_label": [1, -1, 0]})
            pred_path = run_dir / "predictions_SPY.parquet"
            predictions.to_parquet(pred_path)

            loaded = load_predictions(run_dir, "SPY")

            assert len(loaded) == 3
            assert list(loaded["predicted_label"]) == [1, -1, 0]

    def test_collect_experiment_results(self):
        """Test collecting results from multiple runs."""
        from etf_pipeline.experiments.results import collect_experiment_results

        with tempfile.TemporaryDirectory() as tmpdir:
            runs_dir = Path(tmpdir)

            # Create two runs
            for i, exp_name in enumerate(["exp1", "exp2"]):
                run_dir = runs_dir / f"{exp_name}_run_{i}"
                run_dir.mkdir()

                # Save config
                import yaml
                config = {"experiment_name": exp_name}
                with open(run_dir / "config.yaml", "w") as f:
                    yaml.dump(config, f)

                # Save metrics
                metrics = {
                    "accuracy": 0.6 + i * 0.1,
                    "balanced_accuracy": 0.55 + i * 0.1,
                    "macro_f1": 0.5 + i * 0.1,
                }
                with open(run_dir / "metrics_SPY.json", "w") as f:
                    json.dump(metrics, f)

            results = collect_experiment_results(runs_dir)

            assert len(results) == 2
            assert "exp1" in results["experiment_name"].values
            assert "exp2" in results["experiment_name"].values

    def test_compare_experiments(self):
        """Test comparing experiments by metric."""
        from etf_pipeline.experiments.results import compare_experiments

        results_df = pd.DataFrame(
            {
                "experiment_name": ["exp1", "exp1", "exp2", "exp2"],
                "symbol": ["SPY", "QQQ", "SPY", "QQQ"],
                "balanced_accuracy": [0.55, 0.52, 0.62, 0.58],
            }
        )

        comparison = compare_experiments(results_df, metric="balanced_accuracy")

        assert len(comparison) == 2
        # exp2 should be first (higher mean)
        assert comparison.iloc[0]["experiment_name"] == "exp2"

    def test_create_results_summary(self):
        """Test creating results summary."""
        from etf_pipeline.experiments.results import create_results_summary

        results_df = pd.DataFrame(
            {
                "experiment_name": ["exp1", "exp2", "exp3"],
                "symbol": ["SPY", "SPY", "SPY"],
                "run_id": ["run1", "run2", "run3"],
                "balanced_accuracy": [0.5, 0.6, 0.7],
                "macro_f1": [0.45, 0.55, 0.65],
            }
        )

        summary = create_results_summary(results_df)

        assert summary["n_experiments"] == 3
        assert summary["n_runs"] == 3
        assert "balanced_accuracy" in summary["metrics_summary"]
        assert summary["metrics_summary"]["balanced_accuracy"]["mean"] == 0.6

    def test_compute_feature_ablation_impact(self):
        """Test computing feature ablation impact."""
        from etf_pipeline.experiments.results import compute_feature_ablation_impact

        results_df = pd.DataFrame(
            {
                "experiment_name": ["exp1", "exp1", "exp2", "exp2"],
                "symbol": ["SPY", "QQQ", "SPY", "QQQ"],
                "balanced_accuracy": [0.5, 0.5, 0.6, 0.55],
            }
        )

        impact = compute_feature_ablation_impact(
            results_df, baseline_exp="exp1", metric="balanced_accuracy"
        )

        assert len(impact) == 2  # One row per symbol for exp2
        assert "absolute_change" in impact.columns
        assert "relative_change_pct" in impact.columns

        # SPY should show +0.1 absolute change
        spy_impact = impact[impact["symbol"] == "SPY"].iloc[0]
        assert spy_impact["absolute_change"] == pytest.approx(0.1)


class TestTimeSeriesDataset:
    """Tests for time series dataset module."""

    def test_build_returns_series(self):
        """Test building returns series from price data."""
        from etf_pipeline.timeseries.dataset import build_returns_series

        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 110]
        df = pd.DataFrame({"close": prices}, index=dates)

        # Simple returns
        simple_ret = build_returns_series(df, return_type="simple")
        assert len(simple_ret) == 10
        expected_first = (102 - 100) / 100
        assert simple_ret.iloc[1] == pytest.approx(expected_first)

        # Log returns
        log_ret = build_returns_series(df, return_type="log")
        expected_log = np.log(102 / 100)
        assert log_ret.iloc[1] == pytest.approx(expected_log)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
