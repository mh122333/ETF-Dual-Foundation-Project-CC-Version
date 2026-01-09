"""Classification metrics for triple-barrier label prediction."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def compute_label_distribution(
    y: pd.Series,
    name: str = "data",
) -> Dict[str, Any]:
    """
    Compute label distribution statistics.

    Args:
        y: Series of labels.
        name: Name for this distribution (e.g., "train", "test").

    Returns:
        Dictionary with counts and percentages per class.
    """
    counts = y.value_counts().sort_index()
    percentages = (y.value_counts(normalize=True) * 100).sort_index()

    return {
        "name": name,
        "total": int(len(y)),
        "counts": {int(k): int(v) for k, v in counts.items()},
        "percentages": {int(k): round(v, 2) for k, v in percentages.items()},
    }


def compute_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    class_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_labels: List of class labels (default [-1, 0, 1]).

    Returns:
        Dictionary with all metrics.
    """
    if class_labels is None:
        class_labels = [-1, 0, 1]

    # Overall metrics
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Per-class metrics
    precision_per_class = precision_score(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )
    recall_per_class = recall_score(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )
    f1_per_class = f1_score(
        y_true, y_pred, labels=class_labels, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)

    # Build per-class dict
    per_class = {}
    for i, label in enumerate(class_labels):
        label_name = {-1: "SL", 0: "timeout", 1: "TP"}.get(label, str(label))
        per_class[label_name] = {
            "label": int(label),
            "precision": round(float(precision_per_class[i]), 4),
            "recall": round(float(recall_per_class[i]), 4),
            "f1": round(float(f1_per_class[i]), 4),
            "support": int((y_true == label).sum()),
        }

    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(weighted_f1), 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": class_labels,
        "n_samples": int(len(y_true)),
    }


def compute_baseline_metrics(
    y_true: pd.Series,
    class_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute metrics for simple baseline predictors.

    Baselines:
    - Always predict majority class
    - Always predict class 0 (timeout)

    Args:
        y_true: True labels.
        class_labels: List of class labels.

    Returns:
        Dictionary with baseline metrics.
    """
    if class_labels is None:
        class_labels = [-1, 0, 1]

    n = len(y_true)

    # Majority class baseline
    majority_class = y_true.mode().iloc[0]
    y_pred_majority = pd.Series([majority_class] * n)
    acc_majority = accuracy_score(y_true, y_pred_majority)
    bal_acc_majority = balanced_accuracy_score(y_true, y_pred_majority)

    # Always predict 0 baseline
    y_pred_zero = pd.Series([0] * n)
    acc_zero = accuracy_score(y_true, y_pred_zero)
    bal_acc_zero = balanced_accuracy_score(y_true, y_pred_zero)

    # Random baseline (expected values)
    class_probs = y_true.value_counts(normalize=True)
    expected_random_acc = (class_probs ** 2).sum()  # Sum of p_i^2

    return {
        "majority_class": {
            "predicted_class": int(majority_class),
            "accuracy": round(float(acc_majority), 4),
            "balanced_accuracy": round(float(bal_acc_majority), 4),
        },
        "always_zero": {
            "accuracy": round(float(acc_zero), 4),
            "balanced_accuracy": round(float(bal_acc_zero), 4),
        },
        "random_expected": {
            "accuracy": round(float(expected_random_acc), 4),
            "balanced_accuracy": round(1.0 / len(class_labels), 4),
        },
    }


def compute_all_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_train: Optional[pd.Series] = None,
    y_tune: Optional[pd.Series] = None,
    class_labels: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Compute all classification metrics including baselines and distributions.

    Args:
        y_true: True test labels.
        y_pred: Predicted labels.
        y_train: Training labels (for distribution).
        y_tune: Tuning labels (for distribution).
        class_labels: List of class labels.

    Returns:
        Comprehensive metrics dictionary.
    """
    if class_labels is None:
        class_labels = [-1, 0, 1]

    metrics = {
        "classification": compute_classification_metrics(y_true, y_pred, class_labels),
        "baselines": compute_baseline_metrics(y_true, class_labels),
        "label_distributions": {
            "test": compute_label_distribution(y_true, "test"),
        },
    }

    if y_train is not None:
        metrics["label_distributions"]["train"] = compute_label_distribution(y_train, "train")

    if y_tune is not None:
        metrics["label_distributions"]["tune"] = compute_label_distribution(y_tune, "tune")

    # Add improvement over baselines
    model_acc = metrics["classification"]["accuracy"]
    model_bal_acc = metrics["classification"]["balanced_accuracy"]
    majority_acc = metrics["baselines"]["majority_class"]["accuracy"]
    majority_bal_acc = metrics["baselines"]["majority_class"]["balanced_accuracy"]

    metrics["improvement_over_baseline"] = {
        "accuracy_vs_majority": round(model_acc - majority_acc, 4),
        "balanced_accuracy_vs_majority": round(model_bal_acc - majority_bal_acc, 4),
    }

    return metrics


def save_metrics(
    metrics: Dict[str, Any],
    path: Path,
    indent: int = 2,
) -> Path:
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics: Metrics dictionary.
        path: Path to save JSON file.
        indent: JSON indentation level.

    Returns:
        Path where the file was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=indent, default=str)

    return path


def print_metrics_summary(metrics: Dict[str, Any]) -> None:
    """
    Print a concise summary of classification metrics.

    Args:
        metrics: Metrics dictionary from compute_all_metrics.
    """
    clf = metrics.get("classification", {})
    baselines = metrics.get("baselines", {})
    improvement = metrics.get("improvement_over_baseline", {})

    print("\n" + "=" * 50)
    print("CLASSIFICATION METRICS")
    print("=" * 50)

    print(f"\nAccuracy:          {clf.get('accuracy', 'N/A'):.4f}")
    print(f"Balanced Accuracy: {clf.get('balanced_accuracy', 'N/A'):.4f}")
    print(f"Macro F1:          {clf.get('macro_f1', 'N/A'):.4f}")
    print(f"Weighted F1:       {clf.get('weighted_f1', 'N/A'):.4f}")

    print("\nPer-class metrics:")
    per_class = clf.get("per_class", {})
    for class_name, class_metrics in per_class.items():
        print(f"  {class_name}: P={class_metrics['precision']:.3f}, "
              f"R={class_metrics['recall']:.3f}, F1={class_metrics['f1']:.3f}, "
              f"support={class_metrics['support']}")

    print("\nConfusion Matrix:")
    cm = clf.get("confusion_matrix", [])
    labels = clf.get("confusion_matrix_labels", [-1, 0, 1])
    if cm:
        print(f"  Predicted ->  {labels}")
        for i, row in enumerate(cm):
            print(f"  True {labels[i]:2d}: {row}")

    print("\nBaselines:")
    maj = baselines.get("majority_class", {})
    print(f"  Majority class ({maj.get('predicted_class', '?')}): "
          f"acc={maj.get('accuracy', 'N/A'):.4f}, "
          f"bal_acc={maj.get('balanced_accuracy', 'N/A'):.4f}")

    az = baselines.get("always_zero", {})
    print(f"  Always predict 0: acc={az.get('accuracy', 'N/A'):.4f}, "
          f"bal_acc={az.get('balanced_accuracy', 'N/A'):.4f}")

    print("\nImprovement over majority baseline:")
    print(f"  Accuracy:          {improvement.get('accuracy_vs_majority', 'N/A'):+.4f}")
    print(f"  Balanced Accuracy: {improvement.get('balanced_accuracy_vs_majority', 'N/A'):+.4f}")
