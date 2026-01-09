"""Time series model training with AutoGluon TimeSeriesPredictor.

Trains forecasting models on historical data for generating forecast features.
"""

from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd


def get_chronos_hyperparameters(
    include_zero_shot: bool = True,
    include_fine_tuned: bool = True,
    cross_learning: bool = False,
) -> Dict[str, Any]:
    """
    Get hyperparameters for Chronos2 models.

    Args:
        include_zero_shot: Include zero-shot Chronos2.
        include_fine_tuned: Include fine-tuned Chronos2.
        cross_learning: Enable cross-learning across series.

    Returns:
        Hyperparameters dict for TimeSeriesPredictor.
    """
    hyperparameters = {}

    if include_zero_shot:
        hyperparameters["Chronos"] = [
            {
                "model_path": "amazon/chronos-t5-small",
                "fine_tune": False,
                "cross_learning": cross_learning,
            }
        ]

    if include_fine_tuned:
        # Add fine-tuned variant
        ft_config = {
            "model_path": "amazon/chronos-t5-small",
            "fine_tune": True,
            "cross_learning": cross_learning,
        }
        if "Chronos" in hyperparameters:
            hyperparameters["Chronos"].append(ft_config)
        else:
            hyperparameters["Chronos"] = [ft_config]

    return hyperparameters


def train_timeseries_predictor(
    train_data: pd.DataFrame,
    model_path: Path,
    prediction_length: int = 26,
    presets: str = "chronos_ensemble",
    time_limit: Optional[int] = None,
    include_chronos_zero_shot: bool = True,
    include_chronos_fine_tuned: bool = True,
    cross_learning: bool = False,
    target_col: str = "target",
    quantile_levels: List[float] = None,
    verbosity: int = 2,
) -> Any:
    """
    Train an AutoGluon TimeSeriesPredictor.

    Args:
        train_data: Training data in TimeSeriesDataFrame format.
        model_path: Path to save the predictor.
        prediction_length: Forecast horizon in bars.
        presets: AutoGluon presets - "chronos_ensemble", "fast_training", "best_quality".
        time_limit: Training time limit in seconds.
        include_chronos_zero_shot: Include zero-shot Chronos2.
        include_chronos_fine_tuned: Include fine-tuned Chronos2.
        cross_learning: Enable cross-learning for Chronos models.
        target_col: Name of target column.
        quantile_levels: Quantile levels for probabilistic forecasts.
        verbosity: Verbosity level.

    Returns:
        Trained TimeSeriesPredictor.
    """
    try:
        from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame
    except ImportError:
        raise ImportError(
            "AutoGluon timeseries not installed. Run: pip install autogluon.timeseries"
        )

    if quantile_levels is None:
        quantile_levels = [0.1, 0.5, 0.9]

    # Convert to TimeSeriesDataFrame
    ts_data = TimeSeriesDataFrame.from_data_frame(
        train_data,
        id_column="item_id",
        timestamp_column="timestamp",
    )

    # Configure predictor
    predictor = TimeSeriesPredictor(
        path=str(model_path),
        target=target_col,
        prediction_length=prediction_length,
        quantile_levels=quantile_levels,
        verbosity=verbosity,
    )

    # Build hyperparameters if using custom Chronos config
    hyperparameters = None
    if presets == "best_quality" or presets == "high_quality":
        # Add explicit Chronos configuration for best_quality
        hyperparameters = get_chronos_hyperparameters(
            include_zero_shot=include_chronos_zero_shot,
            include_fine_tuned=include_chronos_fine_tuned,
            cross_learning=cross_learning,
        )

    # Train
    fit_kwargs = {
        "train_data": ts_data,
        "presets": presets,
    }

    if time_limit is not None:
        fit_kwargs["time_limit"] = time_limit

    if hyperparameters:
        fit_kwargs["hyperparameters"] = hyperparameters

    predictor.fit(**fit_kwargs)

    return predictor


def load_or_train_timeseries_predictor(
    train_data: pd.DataFrame,
    model_path: Path,
    force_retrain: bool = False,
    prediction_length: int = 26,
    presets: str = "chronos_ensemble",
    time_limit: Optional[int] = None,
    include_chronos_zero_shot: bool = True,
    include_chronos_fine_tuned: bool = True,
    cross_learning: bool = False,
    target_col: str = "target",
    quantile_levels: List[float] = None,
    verbosity: int = 2,
) -> Any:
    """
    Load existing predictor or train new one.

    Args:
        train_data: Training data.
        model_path: Path to model directory.
        force_retrain: If True, retrain even if model exists.
        ... (other args same as train_timeseries_predictor)

    Returns:
        TimeSeriesPredictor (loaded or newly trained).
    """
    try:
        from autogluon.timeseries import TimeSeriesPredictor
    except ImportError:
        raise ImportError(
            "AutoGluon timeseries not installed. Run: pip install autogluon.timeseries"
        )

    model_path = Path(model_path)

    # Check if model exists
    if model_path.exists() and not force_retrain:
        print(f"Loading existing TimeSeriesPredictor from {model_path}")
        try:
            predictor = TimeSeriesPredictor.load(str(model_path))
            print("Successfully loaded existing model")
            return predictor
        except Exception as e:
            print(f"Failed to load model: {e}. Will retrain.")

    # Train new model
    print(f"Training new TimeSeriesPredictor, saving to {model_path}")
    predictor = train_timeseries_predictor(
        train_data=train_data,
        model_path=model_path,
        prediction_length=prediction_length,
        presets=presets,
        time_limit=time_limit,
        include_chronos_zero_shot=include_chronos_zero_shot,
        include_chronos_fine_tuned=include_chronos_fine_tuned,
        cross_learning=cross_learning,
        target_col=target_col,
        quantile_levels=quantile_levels,
        verbosity=verbosity,
    )

    return predictor


def get_predictor_info(predictor: Any) -> Dict[str, Any]:
    """
    Get information about a trained predictor.

    Args:
        predictor: Trained TimeSeriesPredictor.

    Returns:
        Dictionary with predictor information.
    """
    info = {
        "prediction_length": predictor.prediction_length,
        "quantile_levels": predictor.quantile_levels,
        "target": predictor.target,
    }

    # Try to get leaderboard
    try:
        leaderboard = predictor.leaderboard()
        info["models"] = leaderboard["model"].tolist()
        info["n_models"] = len(leaderboard)
    except Exception:
        info["models"] = []
        info["n_models"] = 0

    return info
