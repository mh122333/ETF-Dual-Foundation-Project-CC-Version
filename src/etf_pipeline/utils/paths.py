"""Path utilities for Google Drive storage."""

from pathlib import Path
from typing import Dict


# Base path for all artifacts on Google Drive
DRIVE_BASE = Path("/content/drive/MyDrive/ETF Duel Foundation Model Project/claude_build")


def get_drive_paths() -> Dict[str, Path]:
    """
    Get dictionary of standard paths for project artifacts.

    Returns:
        Dict with keys: 'base', 'raw', 'processed', 'runs', 'models'
    """
    return {
        "base": DRIVE_BASE,
        "raw": DRIVE_BASE / "data" / "raw",
        "processed": DRIVE_BASE / "data" / "processed",
        "runs": DRIVE_BASE / "runs",
        "models": DRIVE_BASE / "models",
    }


def ensure_dirs() -> Dict[str, Path]:
    """
    Create all standard directories if they don't exist.

    Returns:
        Dict of paths that were ensured to exist.
    """
    paths = get_drive_paths()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_bars_cache_path(symbol: str) -> Path:
    """
    Get the cache path for raw bars parquet file.

    Args:
        symbol: Stock ticker symbol (e.g., 'SPY')

    Returns:
        Path to the parquet file for this symbol's bars.
    """
    return get_drive_paths()["raw"] / f"{symbol}_30min_bars.parquet"


def get_labeled_dataset_path() -> Path:
    """
    Get the path for the labeled dataset parquet file.

    Returns:
        Path to the labeled dataset file.
    """
    return get_drive_paths()["processed"] / "labeled_dataset.parquet"


def get_summary_path() -> Path:
    """
    Get the path for the summary JSON file.

    Returns:
        Path to the summary.json file.
    """
    return get_drive_paths()["base"] / "summary.json"
