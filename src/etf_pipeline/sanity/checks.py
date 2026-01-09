"""Sanity checks and leakage smoke tests."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from ..features.baseline import get_feature_columns
from ..utils.paths import get_summary_path


def compute_label_distribution(
    df: pd.DataFrame,
    label_col: str = "label",
) -> Dict[str, Any]:
    """
    Compute label distribution overall and per symbol.

    Args:
        df: DataFrame with labels and symbol column.
        label_col: Name of the label column.

    Returns:
        Dictionary with label distribution statistics.
    """
    result = {}

    # Overall distribution
    overall_counts = df[label_col].value_counts().to_dict()
    overall_pcts = (df[label_col].value_counts(normalize=True) * 100).to_dict()
    result["overall"] = {
        "counts": {int(k): int(v) for k, v in overall_counts.items()},
        "percentages": {int(k): round(v, 2) for k, v in overall_pcts.items()},
        "total": int(len(df)),
    }

    # Per symbol distribution
    if "symbol" in df.index.names:
        df_reset = df.reset_index()
    elif "symbol" in df.columns:
        df_reset = df
    else:
        return result

    per_symbol = {}
    for symbol in df_reset["symbol"].unique():
        symbol_df = df_reset[df_reset["symbol"] == symbol]
        counts = symbol_df[label_col].value_counts().to_dict()
        pcts = (symbol_df[label_col].value_counts(normalize=True) * 100).to_dict()
        per_symbol[symbol] = {
            "counts": {int(k): int(v) for k, v in counts.items()},
            "percentages": {int(k): round(v, 2) for k, v in pcts.items()},
            "total": int(len(symbol_df)),
        }

    result["per_symbol"] = per_symbol

    return result


def check_nan_values(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "label",
) -> Dict[str, Any]:
    """
    Check for NaN values in features and labels.

    Args:
        df: DataFrame to check.
        feature_cols: List of feature column names to check.
        label_col: Name of the label column.

    Returns:
        Dictionary with NaN statistics.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    result = {}

    # Check features
    feature_nans = {}
    for col in feature_cols:
        if col in df.columns:
            nan_count = int(df[col].isna().sum())
            nan_pct = round(df[col].isna().mean() * 100, 2)
            feature_nans[col] = {"count": nan_count, "percentage": nan_pct}

    result["features"] = feature_nans

    # Check labels
    if label_col in df.columns:
        nan_count = int(df[label_col].isna().sum())
        nan_pct = round(df[label_col].isna().mean() * 100, 2)
        result["labels"] = {"count": nan_count, "percentage": nan_pct}

    # Overall
    result["total_rows"] = int(len(df))

    return result


def leakage_smoke_test(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "label",
    train_ratio: float = 0.7,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run leakage smoke test using time-based split.

    Trains a simple classifier on:
    1. Original features (proper time split)
    2. Shifted features (features shifted by +1, simulating look-ahead)

    If there's no leakage, shifted features should perform worse because
    they introduce misalignment between features and labels.

    Args:
        df: DataFrame with features and labels.
        feature_cols: List of feature columns to use.
        label_col: Name of the label column.
        train_ratio: Fraction of data for training (time-based split).
        random_state: Random seed for classifier.

    Returns:
        Dictionary with test results.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns()

    # Prepare data - ensure sorted by time
    if isinstance(df.index, pd.MultiIndex):
        df_work = df.reset_index()
        df_work = df_work.sort_values(["timestamp", "symbol"])
    else:
        df_work = df.copy()
        df_work = df_work.sort_index()

    # Drop rows with NaN in features or labels
    cols_to_check = [c for c in feature_cols if c in df_work.columns] + [label_col]
    df_clean = df_work.dropna(subset=cols_to_check)

    if len(df_clean) < 100:
        return {
            "error": "Not enough data for leakage test",
            "rows_available": int(len(df_clean)),
        }

    # Available features
    available_features = [c for c in feature_cols if c in df_clean.columns]

    # Time-based split
    split_idx = int(len(df_clean) * train_ratio)
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]

    X_train = train[available_features].values
    y_train = train[label_col].values
    X_test = test[available_features].values
    y_test = test[label_col].values

    # Train classifier on original features
    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    y_pred_orig = clf.predict(X_test)
    acc_original = accuracy_score(y_test, y_pred_orig)

    # Create shifted features (shift by +1 to simulate look-ahead)
    df_shifted = df_clean.copy()
    for col in available_features:
        df_shifted[col] = df_shifted[col].shift(1)

    # Drop NaN rows introduced by shift
    df_shifted = df_shifted.dropna(subset=available_features + [label_col])

    # Re-split with same ratio
    split_idx_shifted = int(len(df_shifted) * train_ratio)
    train_shifted = df_shifted.iloc[:split_idx_shifted]
    test_shifted = df_shifted.iloc[split_idx_shifted:]

    X_train_shifted = train_shifted[available_features].values
    y_train_shifted = train_shifted[label_col].values
    X_test_shifted = test_shifted[available_features].values
    y_test_shifted = test_shifted[label_col].values

    # Train classifier on shifted features
    clf_shifted = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=random_state,
        n_jobs=-1,
    )
    clf_shifted.fit(X_train_shifted, y_train_shifted)
    y_pred_shifted = clf_shifted.predict(X_test_shifted)
    acc_shifted = accuracy_score(y_test_shifted, y_pred_shifted)

    # Analyze results
    degradation = acc_original - acc_shifted
    passed = degradation >= 0  # Shifted should not be better

    return {
        "original_accuracy": round(acc_original, 4),
        "shifted_accuracy": round(acc_shifted, 4),
        "degradation": round(degradation, 4),
        "passed": passed,
        "train_samples": int(len(train)),
        "test_samples": int(len(test)),
        "features_used": available_features,
        "interpretation": (
            "PASS: Shifted features perform worse or equal (expected behavior)"
            if passed
            else "WARNING: Shifted features perform better - possible leakage concern"
        ),
    }


def run_sanity_checks(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    label_col: str = "label",
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run all sanity checks and compile results.

    Args:
        df: DataFrame with features and labels.
        feature_cols: List of feature columns.
        label_col: Name of the label column.
        config: Configuration dict to include in summary.

    Returns:
        Complete sanity check results dictionary.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config or {},
    }

    # Label distribution
    print("Computing label distribution...")
    results["label_distribution"] = compute_label_distribution(df, label_col)

    # NaN checks
    print("Checking for NaN values...")
    results["nan_check"] = check_nan_values(df, feature_cols, label_col)

    # Leakage smoke test
    print("Running leakage smoke test...")
    results["leakage_test"] = leakage_smoke_test(df, feature_cols, label_col)

    # Data summary
    results["data_summary"] = {
        "total_rows": int(len(df)),
        "columns": list(df.columns),
    }

    if isinstance(df.index, pd.MultiIndex):
        results["data_summary"]["symbols"] = list(
            df.index.get_level_values("symbol").unique()
        )
    elif "symbol" in df.columns:
        results["data_summary"]["symbols"] = list(df["symbol"].unique())

    return results


def save_summary(
    results: Dict[str, Any],
    path: Optional[Path] = None,
) -> Path:
    """
    Save sanity check results to JSON file.

    Args:
        results: Results dictionary from run_sanity_checks.
        path: Path to save JSON file. Uses default if None.

    Returns:
        Path where the file was saved.
    """
    if path is None:
        path = get_summary_path()

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Summary saved to: {path}")
    return path
