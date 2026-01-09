"""Tests for purged walk-forward time-series split."""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "src")

from etf_pipeline.splits.purged_walkforward import (
    create_single_split,
    PurgedWalkForwardSplit,
    SplitIndices,
    validate_split_no_leakage,
    apply_split_to_dataframe,
)


class TestSplitIndices:
    """Tests for SplitIndices dataclass."""

    def test_slices(self):
        """Test slice properties."""
        split = SplitIndices(
            train_start=0,
            train_end=100,
            tune_start=150,
            tune_end=200,
            test_start=250,
            test_end=350,
        )

        assert split.train_slice == slice(0, 100)
        assert split.tune_slice == slice(150, 200)
        assert split.test_slice == slice(250, 350)

    def test_masks(self):
        """Test boolean mask generation."""
        split = SplitIndices(
            train_start=0,
            train_end=10,
            tune_start=15,
            tune_end=20,
            test_start=25,
            test_end=30,
        )

        n = 30
        train_mask = split.get_train_mask(n)
        tune_mask = split.get_tune_mask(n)
        test_mask = split.get_test_mask(n)

        assert train_mask.sum() == 10
        assert tune_mask.sum() == 5
        assert test_mask.sum() == 5

        # No overlap
        assert not (train_mask & tune_mask).any()
        assert not (train_mask & test_mask).any()
        assert not (tune_mask & test_mask).any()

    def test_to_dict(self):
        """Test dictionary conversion."""
        split = SplitIndices(
            train_start=0,
            train_end=100,
            tune_start=150,
            tune_end=200,
            test_start=250,
            test_end=350,
        )

        d = split.to_dict()
        assert d["train_size"] == 100
        assert d["tune_size"] == 50
        assert d["test_size"] == 100


class TestCreateSingleSplit:
    """Tests for create_single_split function."""

    def test_basic_split(self):
        """Test basic split creation with sufficient data."""
        n_samples = 5000
        split = create_single_split(
            n_samples=n_samples,
            vertical_barrier_bars=26,
            embargo_bars=26,
            tune_window=260,
            test_window=520,
            min_train_size=2000,
        )

        # Check boundaries
        assert split.train_start == 0
        assert split.train_end > 0
        assert split.tune_start > split.train_end
        assert split.tune_end > split.tune_start
        assert split.test_start > split.tune_end
        assert split.test_end <= n_samples

        # Check sizes
        assert split.train_end - split.train_start >= 2000  # min_train_size
        assert split.test_end - split.test_start > 0

    def test_gap_sizes(self):
        """Test that gaps are correctly sized."""
        n_samples = 5000
        vertical_barrier = 26
        embargo = 26
        total_gap = vertical_barrier + embargo

        split = create_single_split(
            n_samples=n_samples,
            vertical_barrier_bars=vertical_barrier,
            embargo_bars=embargo,
            tune_window=260,
            test_window=520,
            min_train_size=2000,
        )

        # Gap between train and tune
        gap1 = split.tune_start - split.train_end
        assert gap1 == total_gap

        # Gap between tune and test
        gap2 = split.test_start - split.tune_end
        assert gap2 == total_gap

    def test_insufficient_data_raises(self):
        """Test that insufficient data raises ValueError."""
        with pytest.raises(ValueError, match="Not enough data"):
            create_single_split(
                n_samples=500,  # Too small
                vertical_barrier_bars=26,
                embargo_bars=26,
                tune_window=260,
                test_window=520,
                min_train_size=2000,
            )

    def test_no_leakage_with_default_params(self):
        """Test that default split has no label leakage."""
        split = create_single_split(
            n_samples=5000,
            vertical_barrier_bars=26,
            embargo_bars=26,
            tune_window=260,
            test_window=520,
            min_train_size=2000,
        )

        assert validate_split_no_leakage(split, vertical_barrier_bars=26)


class TestValidateSplitNoLeakage:
    """Tests for validate_split_no_leakage function."""

    def test_valid_split(self):
        """Test that a properly gapped split passes validation."""
        # With N=26, gap must be >= 26
        split = SplitIndices(
            train_start=0,
            train_end=1000,
            tune_start=1052,  # Gap of 52 (>= 26)
            tune_end=1312,
            test_start=1364,  # Gap of 52 (>= 26)
            test_end=1884,
        )

        assert validate_split_no_leakage(split, vertical_barrier_bars=26)

    def test_invalid_split_train_tune_overlap(self):
        """Test that insufficient train-tune gap fails validation."""
        # Last train index = 999, horizon extends to 999 + 26 = 1025
        # First tune index = 1020, which is < 1025 (leakage!)
        split = SplitIndices(
            train_start=0,
            train_end=1000,
            tune_start=1020,  # Gap of only 20 (< 26)
            tune_end=1280,
            test_start=1332,
            test_end=1852,
        )

        assert not validate_split_no_leakage(split, vertical_barrier_bars=26)

    def test_invalid_split_tune_test_overlap(self):
        """Test that insufficient tune-test gap fails validation."""
        split = SplitIndices(
            train_start=0,
            train_end=1000,
            tune_start=1052,
            tune_end=1312,
            test_start=1330,  # Gap of only 18 (< 26)
            test_end=1850,
        )

        assert not validate_split_no_leakage(split, vertical_barrier_bars=26)

    def test_edge_case_exactly_sufficient_gap(self):
        """Test edge case where gap is exactly sufficient."""
        # Last train index = 999
        # Horizon end = 999 + 26 = 1025
        # First tune must be >= 1026 for no leakage
        split = SplitIndices(
            train_start=0,
            train_end=1000,
            tune_start=1026,  # Gap of 26, horizon_end=1025, first_tune=1026
            tune_end=1286,
            test_start=1312,  # Similarly for tune-test
            test_end=1832,
        )

        assert validate_split_no_leakage(split, vertical_barrier_bars=26)


class TestPurgedWalkForwardSplit:
    """Tests for PurgedWalkForwardSplit class."""

    def test_single_fold(self):
        """Test single fold split."""
        splitter = PurgedWalkForwardSplit(
            n_splits=1,
            vertical_barrier_bars=26,
            embargo_bars=26,
            tune_window=260,
            test_window=520,
            min_train_size=2000,
        )

        splits = splitter.split(5000)
        assert len(splits) == 1

        split = splits[0]
        assert validate_split_no_leakage(split, vertical_barrier_bars=26)

    def test_multi_fold(self):
        """Test multiple fold splits."""
        splitter = PurgedWalkForwardSplit(
            n_splits=3,
            vertical_barrier_bars=26,
            embargo_bars=26,
            tune_window=200,
            test_window=400,
            min_train_size=1000,
            expanding=True,
        )

        # Need enough data for 3 folds
        n_samples = 6000
        splits = splitter.split(n_samples)

        assert len(splits) == 3

        # All splits should be valid
        for split in splits:
            assert validate_split_no_leakage(split, vertical_barrier_bars=26)

        # Later folds should have more training data (expanding)
        if splitter.expanding:
            train_sizes = [s.train_end - s.train_start for s in splits]
            assert train_sizes[0] <= train_sizes[1] <= train_sizes[2]

    def test_get_split_info(self):
        """Test split info retrieval."""
        splitter = PurgedWalkForwardSplit(n_splits=1)
        info = splitter.get_split_info(5000)

        assert "n_splits" in info
        assert "splits" in info
        assert len(info["splits"]) == 1


class TestApplySplitToDataFrame:
    """Tests for apply_split_to_dataframe function."""

    def test_basic_application(self):
        """Test applying split to a DataFrame."""
        # Create test DataFrame
        n = 1000
        df = pd.DataFrame({
            "feature": np.random.randn(n),
            "label": np.random.randint(-1, 2, n),
        })

        split = SplitIndices(
            train_start=0,
            train_end=600,
            tune_start=650,
            tune_end=750,
            test_start=800,
            test_end=1000,
        )

        train_df, tune_df, test_df = apply_split_to_dataframe(df, split)

        assert len(train_df) == 600
        assert len(tune_df) == 100
        assert len(test_df) == 200

    def test_no_data_overlap(self):
        """Test that split DataFrames don't overlap."""
        n = 1000
        df = pd.DataFrame({
            "idx": np.arange(n),
            "feature": np.random.randn(n),
        })

        split = SplitIndices(
            train_start=0,
            train_end=500,
            tune_start=550,
            tune_end=650,
            test_start=700,
            test_end=900,
        )

        train_df, tune_df, test_df = apply_split_to_dataframe(df, split)

        train_idx = set(train_df["idx"])
        tune_idx = set(tune_df["idx"])
        test_idx = set(test_df["idx"])

        # No overlap
        assert len(train_idx & tune_idx) == 0
        assert len(train_idx & test_idx) == 0
        assert len(tune_idx & test_idx) == 0


class TestLeakageValidationWithRealScenario:
    """Integration tests for leakage validation in realistic scenarios."""

    def test_label_horizon_does_not_leak(self):
        """
        Verify that label at decision time t with horizon N
        does not look into tune/test data.

        For triple-barrier: label at t uses bars t+1..t+N
        """
        n_samples = 5000
        vertical_barrier = 26

        split = create_single_split(
            n_samples=n_samples,
            vertical_barrier_bars=vertical_barrier,
            embargo_bars=vertical_barrier,
            tune_window=260,
            test_window=520,
            min_train_size=2000,
        )

        # The last decision time in TRAIN is train_end - 1
        last_train_decision = split.train_end - 1

        # Its label uses bars up to last_train_decision + vertical_barrier
        label_horizon_end = last_train_decision + vertical_barrier

        # First decision time in TUNE
        first_tune_decision = split.tune_start

        # For no leakage: label_horizon_end < first_tune_decision
        assert label_horizon_end < first_tune_decision, (
            f"Train label horizon ({label_horizon_end}) overlaps with "
            f"tune start ({first_tune_decision})"
        )

        # Similarly for tune -> test
        last_tune_decision = split.tune_end - 1
        tune_horizon_end = last_tune_decision + vertical_barrier
        first_test_decision = split.test_start

        assert tune_horizon_end < first_test_decision, (
            f"Tune label horizon ({tune_horizon_end}) overlaps with "
            f"test start ({first_test_decision})"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
