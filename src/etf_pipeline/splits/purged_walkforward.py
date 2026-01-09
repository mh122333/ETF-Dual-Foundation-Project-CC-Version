"""Purged walk-forward time-series split with embargo.

Implements time-series splits that prevent leakage from overlapping label horizons.
For triple-barrier labels where decision at time t uses bars t+1..t+N, we must
ensure TRAIN labels do not look into TUNE/TEST periods, and TUNE labels do not
look into TEST period.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class SplitIndices:
    """Container for train/tune/test index ranges."""
    train_start: int
    train_end: int
    tune_start: int
    tune_end: int
    test_start: int
    test_end: int

    @property
    def train_slice(self) -> slice:
        return slice(self.train_start, self.train_end)

    @property
    def tune_slice(self) -> slice:
        return slice(self.tune_start, self.tune_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.test_start, self.test_end)

    def get_train_mask(self, n: int) -> np.ndarray:
        """Get boolean mask for training indices."""
        mask = np.zeros(n, dtype=bool)
        mask[self.train_start:self.train_end] = True
        return mask

    def get_tune_mask(self, n: int) -> np.ndarray:
        """Get boolean mask for tuning indices."""
        mask = np.zeros(n, dtype=bool)
        mask[self.tune_start:self.tune_end] = True
        return mask

    def get_test_mask(self, n: int) -> np.ndarray:
        """Get boolean mask for test indices."""
        mask = np.zeros(n, dtype=bool)
        mask[self.test_start:self.test_end] = True
        return mask

    def to_dict(self) -> Dict[str, int]:
        """Convert to dictionary."""
        return {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "tune_start": self.tune_start,
            "tune_end": self.tune_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_size": self.train_end - self.train_start,
            "tune_size": self.tune_end - self.tune_start,
            "test_size": self.test_end - self.test_start,
        }


def create_single_split(
    n_samples: int,
    vertical_barrier_bars: int = 26,
    embargo_bars: int = 26,
    tune_window: int = 260,
    test_window: int = 520,
    min_train_size: int = 2000,
) -> SplitIndices:
    """
    Create a single train/tune/test split with purging and embargo.

    Layout:
    [ TRAIN ]  gap  [ TUNE ]  gap  [ TEST ]

    Where gap = purge_gap + embargo_gap.

    The purge gap ensures that the last TRAIN label's horizon (which extends
    N bars into the future) does not overlap with TUNE data. Similarly for
    TUNE vs TEST.

    Args:
        n_samples: Total number of samples (rows) in the dataset.
        vertical_barrier_bars: N - the label horizon in bars (default 26).
            Labels at time t scan bars t+1..t+N.
        embargo_bars: Additional embargo period after purge (default 26).
        tune_window: Number of bars for tuning set (default 260 = ~1 month 30-min bars).
        test_window: Number of bars for test set (default 520 = ~2 months 30-min bars).
        min_train_size: Minimum training set size (default 2000).

    Returns:
        SplitIndices with train/tune/test ranges.

    Raises:
        ValueError: If there isn't enough data for the requested split sizes.
    """
    # Total gap between sets = purge + embargo
    # Purge gap ensures label horizon doesn't leak
    purge_gap = vertical_barrier_bars
    total_gap = purge_gap + embargo_bars

    # Calculate required space
    # Layout: [train] [gap] [tune] [gap] [test]
    required_non_train = total_gap + tune_window + total_gap + test_window
    available_for_train = n_samples - required_non_train

    if available_for_train < min_train_size:
        # Try to shrink windows proportionally
        scale = (n_samples - 2 * total_gap - min_train_size) / (tune_window + test_window)
        if scale < 0.3:
            raise ValueError(
                f"Not enough data for split. Have {n_samples} samples, "
                f"need at least {min_train_size + 2 * total_gap + int(0.3 * (tune_window + test_window))}. "
                f"Consider reducing tune_window, test_window, or min_train_size."
            )
        tune_window = max(50, int(tune_window * scale))
        test_window = max(100, int(test_window * scale))
        available_for_train = n_samples - 2 * total_gap - tune_window - test_window

    # Compute boundaries
    # TRAIN: [0, train_end)
    train_end = available_for_train

    # Gap after train
    # TUNE: [tune_start, tune_end)
    tune_start = train_end + total_gap
    tune_end = tune_start + tune_window

    # Gap after tune
    # TEST: [test_start, test_end)
    test_start = tune_end + total_gap
    test_end = test_start + test_window

    # Ensure we don't exceed data
    if test_end > n_samples:
        # Adjust test_end to fit
        test_end = n_samples
        test_window = test_end - test_start
        if test_window < 50:
            raise ValueError(
                f"Test window too small ({test_window}). Need more data."
            )

    return SplitIndices(
        train_start=0,
        train_end=train_end,
        tune_start=tune_start,
        tune_end=tune_end,
        test_start=test_start,
        test_end=test_end,
    )


class PurgedWalkForwardSplit:
    """
    Walk-forward cross-validation with purging and embargo.

    Generates multiple folds where each fold has:
    - TRAIN: expanding or sliding window
    - TUNE: fixed window for hyperparameter tuning
    - TEST: fixed window for final evaluation

    Gaps between sets prevent label leakage from overlapping horizons.
    """

    def __init__(
        self,
        n_splits: int = 1,
        vertical_barrier_bars: int = 26,
        embargo_bars: int = 26,
        tune_window: int = 260,
        test_window: int = 520,
        min_train_size: int = 2000,
        expanding: bool = True,
    ):
        """
        Initialize the splitter.

        Args:
            n_splits: Number of walk-forward folds (default 1).
            vertical_barrier_bars: Label horizon N (default 26).
            embargo_bars: Additional embargo after purge (default 26).
            tune_window: Bars for tuning set per fold (default 260).
            test_window: Bars for test set per fold (default 520).
            min_train_size: Minimum training size (default 2000).
            expanding: If True, train window expands; if False, slides (default True).
        """
        self.n_splits = n_splits
        self.vertical_barrier_bars = vertical_barrier_bars
        self.embargo_bars = embargo_bars
        self.tune_window = tune_window
        self.test_window = test_window
        self.min_train_size = min_train_size
        self.expanding = expanding

    def split(self, n_samples: int) -> List[SplitIndices]:
        """
        Generate train/tune/test splits.

        Args:
            n_samples: Total number of samples.

        Returns:
            List of SplitIndices, one per fold.
        """
        if self.n_splits == 1:
            return [create_single_split(
                n_samples=n_samples,
                vertical_barrier_bars=self.vertical_barrier_bars,
                embargo_bars=self.embargo_bars,
                tune_window=self.tune_window,
                test_window=self.test_window,
                min_train_size=self.min_train_size,
            )]

        # Multi-fold walk-forward
        total_gap = self.vertical_barrier_bars + self.embargo_bars
        fold_size = self.tune_window + total_gap + self.test_window

        # Reserve space for all folds at the end
        reserved = self.n_splits * fold_size
        if n_samples - reserved < self.min_train_size + total_gap:
            raise ValueError(
                f"Not enough data for {self.n_splits} folds. "
                f"Have {n_samples} samples, reserved {reserved} for folds."
            )

        splits = []
        for fold_idx in range(self.n_splits):
            # Calculate fold boundaries working backwards from end
            fold_offset = (self.n_splits - 1 - fold_idx) * fold_size

            test_end = n_samples - fold_offset
            test_start = test_end - self.test_window
            tune_end = test_start - total_gap
            tune_start = tune_end - self.tune_window
            train_end = tune_start - total_gap

            if self.expanding:
                train_start = 0
            else:
                # Sliding window - keep train size consistent
                train_start = max(0, train_end - self.min_train_size)

            if train_end - train_start < self.min_train_size:
                raise ValueError(
                    f"Fold {fold_idx}: train size {train_end - train_start} "
                    f"< min_train_size {self.min_train_size}"
                )

            splits.append(SplitIndices(
                train_start=train_start,
                train_end=train_end,
                tune_start=tune_start,
                tune_end=tune_end,
                test_start=test_start,
                test_end=test_end,
            ))

        return splits

    def get_split_info(self, n_samples: int) -> Dict[str, Any]:
        """
        Get information about the splits without generating them.

        Args:
            n_samples: Total number of samples.

        Returns:
            Dictionary with split configuration and sizes.
        """
        splits = self.split(n_samples)
        return {
            "n_splits": len(splits),
            "vertical_barrier_bars": self.vertical_barrier_bars,
            "embargo_bars": self.embargo_bars,
            "total_gap": self.vertical_barrier_bars + self.embargo_bars,
            "tune_window": self.tune_window,
            "test_window": self.test_window,
            "expanding": self.expanding,
            "splits": [s.to_dict() for s in splits],
        }


def apply_split_to_dataframe(
    df: pd.DataFrame,
    split: SplitIndices,
    reset_index: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply a split to a DataFrame.

    Args:
        df: DataFrame sorted by time (index should be sequential).
        split: SplitIndices specifying the ranges.
        reset_index: Whether to reset index on resulting DataFrames.

    Returns:
        Tuple of (train_df, tune_df, test_df).
    """
    train_df = df.iloc[split.train_slice].copy()
    tune_df = df.iloc[split.tune_slice].copy()
    test_df = df.iloc[split.test_slice].copy()

    if reset_index:
        train_df = train_df.reset_index(drop=True)
        tune_df = tune_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

    return train_df, tune_df, test_df


def validate_split_no_leakage(
    split: SplitIndices,
    vertical_barrier_bars: int,
) -> bool:
    """
    Validate that a split has no label leakage.

    For labels at decision time t that use bars t+1..t+N:
    - Last TRAIN decision time's horizon must not overlap with first TUNE time
    - Last TUNE decision time's horizon must not overlap with first TEST time

    Args:
        split: The split to validate.
        vertical_barrier_bars: N - the label horizon.

    Returns:
        True if valid (no leakage), False otherwise.
    """
    # Last train index
    last_train_idx = split.train_end - 1
    # Its label horizon extends to last_train_idx + vertical_barrier_bars
    train_horizon_end = last_train_idx + vertical_barrier_bars

    # First tune index
    first_tune_idx = split.tune_start

    # Check: train horizon must be strictly < first tune index
    if train_horizon_end >= first_tune_idx:
        return False

    # Similarly for tune -> test
    last_tune_idx = split.tune_end - 1
    tune_horizon_end = last_tune_idx + vertical_barrier_bars
    first_test_idx = split.test_start

    if tune_horizon_end >= first_test_idx:
        return False

    return True
