"""Experiment runner for standardized evaluation.

Provides infrastructure to run experiments with consistent configuration,
data splits, and artifact saving.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import yaml

import pandas as pd
import numpy as np


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""

    # Experiment identification
    experiment_name: str = "exp1"
    run_id: Optional[str] = None  # Auto-generated if None

    # Symbols
    symbols: List[str] = field(default_factory=lambda: ["SPY"])
    context_symbols: List[str] = field(default_factory=lambda: ["SPY", "QQQ"])

    # Data windows
    tabular_max_rows: int = 6500
    ts_train_lookback_years: float = 5.0

    # Label parameters
    vertical_barrier_bars: int = 26
    k_up: float = 2.0
    k_dn: float = 1.0

    # Split parameters
    embargo_bars: int = 26
    tune_window: int = 260
    test_window: int = 520
    min_train_size: int = 2000

    # Feature configuration
    feature_set: str = "small"  # small, medium, large
    include_context: bool = False
    include_error: bool = False

    # Time series model
    ts_presets: str = "chronos_ensemble"
    ts_prediction_length: int = 26
    cross_learning: bool = False
    include_chronos_zero_shot: bool = True
    include_chronos_fine_tuned: bool = True

    # Tabular model
    tabular_presets: str = "best_quality"
    time_limit_sec: int = 1200

    # Reproducibility
    random_seed: int = 42

    # Force options
    force_ts_retrain: bool = False
    force_forecast_regenerate: bool = False
    force_data_refresh: bool = False

    def __post_init__(self):
        if self.run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{self.experiment_name}_{timestamp}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "symbols": self.symbols,
            "context_symbols": self.context_symbols,
            "tabular_max_rows": self.tabular_max_rows,
            "ts_train_lookback_years": self.ts_train_lookback_years,
            "vertical_barrier_bars": self.vertical_barrier_bars,
            "k_up": self.k_up,
            "k_dn": self.k_dn,
            "embargo_bars": self.embargo_bars,
            "tune_window": self.tune_window,
            "test_window": self.test_window,
            "min_train_size": self.min_train_size,
            "feature_set": self.feature_set,
            "include_context": self.include_context,
            "include_error": self.include_error,
            "ts_presets": self.ts_presets,
            "ts_prediction_length": self.ts_prediction_length,
            "cross_learning": self.cross_learning,
            "tabular_presets": self.tabular_presets,
            "time_limit_sec": self.time_limit_sec,
            "random_seed": self.random_seed,
        }

    def save(self, path: Path):
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)


class ExperimentRunner:
    """
    Runner for executing experiments with standardized evaluation.

    Handles data loading, feature engineering, model training, and
    artifact saving.
    """

    def __init__(
        self,
        config: ExperimentConfig,
        drive_base: Path = None,
    ):
        """
        Initialize experiment runner.

        Args:
            config: Experiment configuration.
            drive_base: Base path for Drive artifacts.
        """
        self.config = config
        self.drive_base = Path(drive_base) if drive_base else Path(
            "/content/drive/MyDrive/ETF Duel Foundation Model Project/claude_build"
        )

        # Initialize paths
        self.paths = {
            "raw": self.drive_base / "data" / "raw",
            "processed": self.drive_base / "data" / "processed",
            "runs": self.drive_base / "runs",
            "models": self.drive_base / "models",
        }

    def ensure_directories(self):
        """Create all necessary directories."""
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def get_run_dir(self) -> Path:
        """Get directory for this run's artifacts."""
        return self.paths["runs"] / self.config.run_id

    def get_model_dir(self, model_type: str, symbol: str = None) -> Path:
        """Get directory for model artifacts."""
        if symbol:
            return self.paths["models"] / model_type / self.config.experiment_name / symbol / self.config.run_id
        return self.paths["models"] / model_type / self.config.run_id

    def get_ts_model_dir(self) -> Path:
        """Get directory for time series model."""
        return self.paths["models"] / "ts" / self.config.run_id

    def get_forecast_cache_dir(self) -> Path:
        """Get directory for cached forecasts."""
        return self.paths["processed"] / "forecasts" / self.config.run_id

    def save_predictions(
        self,
        predictions: pd.DataFrame,
        symbol: str,
    ) -> Path:
        """Save predictions to parquet."""
        run_dir = self.get_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"predictions_{symbol}.parquet"
        predictions.to_parquet(path)
        return path

    def save_metrics(
        self,
        metrics: Dict[str, Any],
        symbol: str,
    ) -> Path:
        """Save metrics to JSON."""
        run_dir = self.get_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"metrics_{symbol}.json"
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        return path

    def save_config(self) -> Path:
        """Save experiment configuration."""
        run_dir = self.get_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "config.yaml"
        self.config.save(path)
        return path


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load experiment configuration from file."""
    return ExperimentConfig.load(path)


def run_experiment(config: ExperimentConfig, drive_base: Path = None) -> Dict[str, Any]:
    """
    Run a single experiment.

    This is a high-level function that orchestrates the experiment.
    The actual implementation is in the notebooks.

    Args:
        config: Experiment configuration.
        drive_base: Base path for artifacts.

    Returns:
        Dictionary with run results summary.
    """
    runner = ExperimentRunner(config, drive_base)
    runner.ensure_directories()

    # Save config
    config_path = runner.save_config()

    return {
        "run_id": config.run_id,
        "config_path": str(config_path),
        "run_dir": str(runner.get_run_dir()),
        "status": "initialized",
    }
