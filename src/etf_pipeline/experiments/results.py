"""Results aggregation and comparison for experiments.

Provides utilities to load, compare, and visualize results across
multiple experiment runs.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json

import pandas as pd
import numpy as np


def load_metrics(run_dir: Path, symbol: str) -> Dict[str, Any]:
    """
    Load metrics from a run directory.

    Args:
        run_dir: Path to run directory.
        symbol: Symbol to load metrics for.

    Returns:
        Dictionary of metrics.
    """
    path = Path(run_dir) / f"metrics_{symbol}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_predictions(run_dir: Path, symbol: str) -> pd.DataFrame:
    """
    Load predictions from a run directory.

    Args:
        run_dir: Path to run directory.
        symbol: Symbol to load predictions for.

    Returns:
        DataFrame of predictions.
    """
    path = Path(run_dir) / f"predictions_{symbol}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def load_config(run_dir: Path) -> Dict[str, Any]:
    """
    Load configuration from a run directory.

    Args:
        run_dir: Path to run directory.

    Returns:
        Dictionary of configuration.
    """
    import yaml
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def collect_experiment_results(
    runs_dir: Path,
    experiment_names: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Collect results from multiple experiment runs.

    Args:
        runs_dir: Base directory containing run directories.
        experiment_names: Filter to specific experiments. None = all.
        symbols: Filter to specific symbols. None = all found.

    Returns:
        DataFrame with columns:
            - run_id
            - experiment_name
            - symbol
            - accuracy
            - balanced_accuracy
            - macro_f1
            - cohen_kappa
            - class_*_precision, class_*_recall, class_*_f1
    """
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return pd.DataFrame()

    rows = []
    for run_path in sorted(runs_dir.iterdir()):
        if not run_path.is_dir():
            continue

        config = load_config(run_path)
        exp_name = config.get("experiment_name", run_path.name.split("_")[0])

        if experiment_names and exp_name not in experiment_names:
            continue

        # Find all metric files
        metric_files = list(run_path.glob("metrics_*.json"))
        for mf in metric_files:
            symbol = mf.stem.replace("metrics_", "")
            if symbols and symbol not in symbols:
                continue

            metrics = load_metrics(run_path, symbol)
            if not metrics:
                continue

            row = {
                "run_id": run_path.name,
                "experiment_name": exp_name,
                "symbol": symbol,
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "cohen_kappa": metrics.get("cohen_kappa"),
                "mcc": metrics.get("mcc"),
            }

            # Add per-class metrics
            per_class = metrics.get("per_class", {})
            for label, class_metrics in per_class.items():
                row[f"class_{label}_precision"] = class_metrics.get("precision")
                row[f"class_{label}_recall"] = class_metrics.get("recall")
                row[f"class_{label}_f1"] = class_metrics.get("f1")

            rows.append(row)

    return pd.DataFrame(rows)


def compare_experiments(
    results_df: pd.DataFrame,
    metric: str = "balanced_accuracy",
    group_by: str = "experiment_name",
) -> pd.DataFrame:
    """
    Compare experiments by a metric.

    Args:
        results_df: DataFrame from collect_experiment_results.
        metric: Metric column to compare.
        group_by: Column to group by.

    Returns:
        DataFrame with comparison statistics.
    """
    if results_df.empty or metric not in results_df.columns:
        return pd.DataFrame()

    grouped = results_df.groupby(group_by)[metric].agg(
        ["mean", "std", "min", "max", "count"]
    ).reset_index()

    grouped.columns = [group_by, f"{metric}_mean", f"{metric}_std",
                       f"{metric}_min", f"{metric}_max", "n_runs"]

    return grouped.sort_values(f"{metric}_mean", ascending=False)


def create_results_summary(
    results_df: pd.DataFrame,
    metrics: List[str] = None,
) -> Dict[str, Any]:
    """
    Create a summary of experiment results.

    Args:
        results_df: DataFrame from collect_experiment_results.
        metrics: List of metrics to include. None = default set.

    Returns:
        Dictionary with summary statistics.
    """
    if metrics is None:
        metrics = ["balanced_accuracy", "macro_f1", "cohen_kappa"]

    summary = {
        "n_experiments": results_df["experiment_name"].nunique(),
        "n_runs": len(results_df),
        "n_symbols": results_df["symbol"].nunique(),
        "experiments": sorted(results_df["experiment_name"].unique().tolist()),
        "symbols": sorted(results_df["symbol"].unique().tolist()),
        "metrics_summary": {},
    }

    for metric in metrics:
        if metric in results_df.columns:
            summary["metrics_summary"][metric] = {
                "mean": results_df[metric].mean(),
                "std": results_df[metric].std(),
                "min": results_df[metric].min(),
                "max": results_df[metric].max(),
            }

    # Best run per experiment
    summary["best_runs"] = {}
    for exp in results_df["experiment_name"].unique():
        exp_df = results_df[results_df["experiment_name"] == exp]
        if not exp_df.empty and "balanced_accuracy" in exp_df.columns:
            best_idx = exp_df["balanced_accuracy"].idxmax()
            best_row = exp_df.loc[best_idx]
            summary["best_runs"][exp] = {
                "run_id": best_row["run_id"],
                "symbol": best_row["symbol"],
                "balanced_accuracy": best_row["balanced_accuracy"],
            }

    return summary


def format_results_table(
    results_df: pd.DataFrame,
    metrics: List[str] = None,
    sort_by: str = "balanced_accuracy",
) -> str:
    """
    Format results as a markdown table.

    Args:
        results_df: DataFrame from collect_experiment_results.
        metrics: Metrics to include. None = default set.
        sort_by: Column to sort by.

    Returns:
        Markdown-formatted table string.
    """
    if results_df.empty:
        return "No results found."

    if metrics is None:
        metrics = ["balanced_accuracy", "macro_f1", "cohen_kappa"]

    cols = ["experiment_name", "symbol"] + [m for m in metrics if m in results_df.columns]
    df = results_df[cols].copy()

    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    # Format numeric columns
    for col in metrics:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    return df.to_markdown(index=False)


def compute_feature_ablation_impact(
    results_df: pd.DataFrame,
    baseline_exp: str = "exp1",
    metric: str = "balanced_accuracy",
) -> pd.DataFrame:
    """
    Compute the impact of feature ablations relative to baseline.

    Args:
        results_df: DataFrame from collect_experiment_results.
        baseline_exp: Name of baseline experiment.
        metric: Metric to compare.

    Returns:
        DataFrame with impact analysis.
    """
    if results_df.empty:
        return pd.DataFrame()

    baseline = results_df[results_df["experiment_name"] == baseline_exp]
    if baseline.empty:
        return pd.DataFrame()

    baseline_by_symbol = baseline.groupby("symbol")[metric].mean().to_dict()

    rows = []
    for exp in results_df["experiment_name"].unique():
        if exp == baseline_exp:
            continue

        exp_df = results_df[results_df["experiment_name"] == exp]
        for symbol in exp_df["symbol"].unique():
            symbol_df = exp_df[exp_df["symbol"] == symbol]
            exp_value = symbol_df[metric].mean()
            base_value = baseline_by_symbol.get(symbol)

            if base_value is not None and pd.notna(exp_value):
                rows.append({
                    "experiment": exp,
                    "symbol": symbol,
                    f"baseline_{metric}": base_value,
                    f"experiment_{metric}": exp_value,
                    "absolute_change": exp_value - base_value,
                    "relative_change_pct": 100 * (exp_value - base_value) / base_value if base_value != 0 else np.nan,
                })

    return pd.DataFrame(rows)
