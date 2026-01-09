"""AutoGluon TabularPredictor baseline model training.

Trains a tabular classifier on baseline causal features to predict
triple-barrier labels {-1, 0, +1}.
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np


# Columns that must be excluded from features to prevent leakage
LABEL_LEAK_COLUMNS = [
    "label",
    "first_hit",
    "bars_to_hit",
    "realized_pnl_at_hit",
    "entry",
    "tp",
    "sl",
]

# OHLCV columns that shouldn't be used as features directly
RAW_PRICE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
]


def get_feature_columns_for_training(
    df: pd.DataFrame,
    include_volume_zscore: bool = True,
) -> List[str]:
    """
    Get the list of feature columns safe for training.

    Excludes label columns, raw price columns, and metadata columns.

    Args:
        df: DataFrame with all columns.
        include_volume_zscore: Whether volume_zscore is expected.

    Returns:
        List of feature column names.
    """
    exclude = set(LABEL_LEAK_COLUMNS + RAW_PRICE_COLUMNS)
    exclude.add("symbol")  # Metadata
    exclude.add("timestamp")  # Metadata

    # Base features from baseline.py
    base_features = ["ret_1", "ret_2", "rolling_vol", "atr", "atr_over_close"]
    if include_volume_zscore:
        base_features.append("volume_zscore")

    # Use intersection of base features and what's in the dataframe
    available = [c for c in base_features if c in df.columns and c not in exclude]

    return available


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and label vector y for training.

    Args:
        df: DataFrame with features and labels.
        feature_cols: List of feature column names.
        label_col: Name of the label column.

    Returns:
        Tuple of (X, y) where X is the feature DataFrame and y is the label Series.
    """
    # Validate columns exist
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found")

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    return X, y


def train_tabular_baseline(
    train_df: pd.DataFrame,
    tune_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    model_path: Path = None,
    time_limit: int = 1200,
    presets: str = "best_quality",
    random_seed: int = 42,
    verbosity: int = 2,
) -> Any:
    """
    Train an AutoGluon TabularPredictor on the training data.

    Args:
        train_df: Training DataFrame.
        tune_df: Tuning DataFrame (used as tuning_data for AutoGluon).
        feature_cols: List of feature column names.
        label_col: Name of the label column.
        model_path: Path to save the model. If None, uses temp directory.
        time_limit: Training time limit in seconds (default 1200 = 20 min).
        presets: AutoGluon presets (default "best_quality", can use "high_quality", etc.).
        random_seed: Random seed for reproducibility.
        verbosity: Verbosity level (0-4).

    Returns:
        Trained TabularPredictor instance.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        raise ImportError(
            "AutoGluon not installed. Run: pip install autogluon.tabular[all]"
        )

    # Prepare training data
    X_train, y_train = prepare_training_data(train_df, feature_cols, label_col)
    train_data = X_train.copy()
    train_data[label_col] = y_train

    # Prepare tuning data
    X_tune, y_tune = prepare_training_data(tune_df, feature_cols, label_col)
    tune_data = X_tune.copy()
    tune_data[label_col] = y_tune

    # Set random seed
    np.random.seed(random_seed)

    # Configure predictor
    predictor = TabularPredictor(
        label=label_col,
        path=str(model_path) if model_path else None,
        problem_type="multiclass",
        eval_metric="balanced_accuracy",
        verbosity=verbosity,
    )

    # Train
    predictor.fit(
        train_data=train_data,
        tuning_data=tune_data,
        time_limit=time_limit,
        presets=presets,
        random_seed=random_seed,
    )

    return predictor


def predict_tabular(
    predictor: Any,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Generate predictions and probabilities on test data.

    Args:
        predictor: Trained TabularPredictor.
        test_df: Test DataFrame with features.
        feature_cols: List of feature column names.

    Returns:
        DataFrame with predictions and probabilities.
    """
    X_test = test_df[feature_cols].copy()

    # Get predictions
    predictions = predictor.predict(X_test)
    probabilities = predictor.predict_proba(X_test)

    # Build results DataFrame
    results = pd.DataFrame(index=test_df.index)
    results["predicted_label"] = predictions.values

    # Add probabilities for each class
    for col in probabilities.columns:
        results[f"prob_class_{col}"] = probabilities[col].values

    return results


def run_leakage_smoke_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "label",
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run a quick leakage smoke test using logistic regression.

    Compares accuracy with correct features vs features shifted by +1 bar.
    If there's no leakage, shifted features should perform worse.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        feature_cols: Feature column names.
        label_col: Label column name.
        random_seed: Random seed.

    Returns:
        Dictionary with test results.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    # Prepare data
    X_train, y_train = prepare_training_data(train_df, feature_cols, label_col)
    X_test, y_test = prepare_training_data(test_df, feature_cols, label_col)

    # Drop NaN
    train_mask = ~(X_train.isna().any(axis=1) | y_train.isna())
    test_mask = ~(X_test.isna().any(axis=1) | y_test.isna())

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    if len(X_train) < 100 or len(X_test) < 50:
        return {
            "error": "Not enough data for leakage test",
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train with correct features
    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_seed,
        multi_class="multinomial",
    )
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc_original = accuracy_score(y_test, y_pred)
    bal_acc_original = balanced_accuracy_score(y_test, y_pred)

    # Create shifted features (shift by +1, simulating look-ahead)
    X_train_shifted = X_train.shift(1).dropna()
    y_train_shifted = y_train.iloc[1:].reset_index(drop=True)
    X_train_shifted = X_train_shifted.reset_index(drop=True)

    X_test_shifted = X_test.shift(1).dropna()
    y_test_shifted = y_test.iloc[1:].reset_index(drop=True)
    X_test_shifted = X_test_shifted.reset_index(drop=True)

    # Scale shifted features
    scaler_shifted = StandardScaler()
    X_train_shifted_scaled = scaler_shifted.fit_transform(X_train_shifted)
    X_test_shifted_scaled = scaler_shifted.transform(X_test_shifted)

    # Train with shifted features
    clf_shifted = LogisticRegression(
        max_iter=1000,
        random_state=random_seed,
        multi_class="multinomial",
    )
    clf_shifted.fit(X_train_shifted_scaled, y_train_shifted)
    y_pred_shifted = clf_shifted.predict(X_test_shifted_scaled)
    acc_shifted = accuracy_score(y_test_shifted, y_pred_shifted)
    bal_acc_shifted = balanced_accuracy_score(y_test_shifted, y_pred_shifted)

    degradation = acc_original - acc_shifted
    passed = degradation >= -0.02  # Allow small variance

    return {
        "original_accuracy": round(acc_original, 4),
        "original_balanced_accuracy": round(bal_acc_original, 4),
        "shifted_accuracy": round(acc_shifted, 4),
        "shifted_balanced_accuracy": round(bal_acc_shifted, 4),
        "degradation": round(degradation, 4),
        "passed": passed,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "interpretation": (
            "PASS: Shifted features perform worse or similar (expected)"
            if passed
            else "WARNING: Shifted features perform better - possible leakage"
        ),
    }
