"""Unit tests for temporal split loading and validation.

These tests ensure that:
1. Split files can be loaded correctly
2. Temporal ordering is enforced
3. Data leakage is detected
4. Random/stratified methods are forbidden
"""

import json
from pathlib import Path

import numpy as np
import pytest

from moola.data.splits import (
    assert_no_random,
    assert_temporal,
    create_stratified_splits,
    get_default_split,
    load_split,
)


class TestLoadSplit:
    """Tests for load_split function."""

    def test_load_split_valid(self, tmp_path):
        """Test loading a valid split file."""
        # Create a valid split file
        split_file = tmp_path / "test_split.json"
        split_data = {
            "name": "test_split",
            "fold": 0,
            "train_indices": [0, 1, 2, 3, 4],
            "val_indices": [5, 6, 7],
            "test_indices": [8, 9],
        }

        with open(split_file, "w") as f:
            json.dump(split_data, f)

        # Load and verify
        loaded = load_split(str(split_file))
        assert loaded["name"] == "test_split"
        assert loaded["train_indices"] == [0, 1, 2, 3, 4]
        assert loaded["val_indices"] == [5, 6, 7]
        assert loaded["test_indices"] == [8, 9]

    def test_load_split_legacy_field_names(self, tmp_path):
        """Test loading split with legacy field names (train_idx, val_idx)."""
        split_file = tmp_path / "legacy_split.json"
        split_data = {
            "fold": 0,
            "train_idx": [0, 1, 2],
            "val_idx": [3, 4],
        }

        with open(split_file, "w") as f:
            json.dump(split_data, f)

        # Should map old names to new names
        loaded = load_split(str(split_file))
        assert "train_indices" in loaded
        assert "val_indices" in loaded
        assert loaded["train_indices"] == [0, 1, 2]
        assert loaded["val_indices"] == [3, 4]

    def test_load_split_missing_file(self):
        """Test loading a non-existent split file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Split file not found"):
            load_split("nonexistent.json")

    def test_load_split_missing_required_fields(self, tmp_path):
        """Test loading split with missing required fields raises ValueError."""
        split_file = tmp_path / "incomplete_split.json"
        split_data = {"name": "incomplete", "train_indices": [0, 1, 2]}
        # Missing val_indices

        with open(split_file, "w") as f:
            json.dump(split_data, f)

        with pytest.raises(ValueError, match="missing required fields"):
            load_split(str(split_file))

    def test_load_split_auto_name(self, tmp_path):
        """Test that split name is auto-generated from filename if missing."""
        split_file = tmp_path / "my_split.json"
        split_data = {"train_indices": [0, 1], "val_indices": [2, 3]}

        with open(split_file, "w") as f:
            json.dump(split_data, f)

        loaded = load_split(str(split_file))
        assert loaded["name"] == "my_split"

    def test_load_split_optional_test_indices(self, tmp_path):
        """Test that test_indices is optional (defaults to empty list)."""
        split_file = tmp_path / "two_way_split.json"
        split_data = {"train_indices": [0, 1, 2], "val_indices": [3, 4]}

        with open(split_file, "w") as f:
            json.dump(split_data, f)

        loaded = load_split(str(split_file))
        assert loaded["test_indices"] == []


class TestAssertTemporal:
    """Tests for assert_temporal function."""

    def test_assert_temporal_valid_split(self):
        """Test that valid temporal split passes validation."""
        split = {
            "name": "valid_temporal",
            "train_indices": [0, 1, 2, 3, 4],
            "val_indices": [5, 6, 7, 8],
            "test_indices": [9, 10, 11],
        }

        # Should not raise
        assert_temporal(split)

    def test_assert_temporal_non_monotonic_train(self):
        """Test that non-monotonic train indices are rejected."""
        split = {
            "name": "bad_train",
            "train_indices": [0, 2, 1, 3],  # Not sorted!
            "val_indices": [4, 5],
            "test_indices": [],
        }

        with pytest.raises(AssertionError, match="not monotonic"):
            assert_temporal(split)

    def test_assert_temporal_non_monotonic_val(self):
        """Test that non-monotonic val indices are rejected."""
        split = {
            "name": "bad_val",
            "train_indices": [0, 1, 2],
            "val_indices": [5, 4, 3],  # Not sorted!
            "test_indices": [],
        }

        with pytest.raises(AssertionError, match="not monotonic"):
            assert_temporal(split)

    def test_assert_temporal_non_monotonic_test(self):
        """Test that non-monotonic test indices are rejected."""
        split = {
            "name": "bad_test",
            "train_indices": [0, 1],
            "val_indices": [2, 3],
            "test_indices": [6, 5, 4],  # Not sorted!
        }

        with pytest.raises(AssertionError, match="not monotonic"):
            assert_temporal(split)

    def test_assert_temporal_train_val_leakage(self):
        """Test that train/val overlap is detected."""
        split = {
            "name": "train_val_leak",
            "train_indices": [0, 1, 2, 3],
            "val_indices": [2, 4, 5],  # 2 overlaps!
            "test_indices": [],
        }

        with pytest.raises(AssertionError, match="Train/Val leakage"):
            assert_temporal(split)

    def test_assert_temporal_train_test_leakage(self):
        """Test that train/test overlap is detected."""
        split = {
            "name": "train_test_leak",
            "train_indices": [0, 1, 2],
            "val_indices": [3, 4],
            "test_indices": [2, 5, 6],  # 2 overlaps with train!
        }

        with pytest.raises(AssertionError, match="Train/Test leakage"):
            assert_temporal(split)

    def test_assert_temporal_val_test_leakage(self):
        """Test that val/test overlap is detected."""
        split = {
            "name": "val_test_leak",
            "train_indices": [0, 1],
            "val_indices": [2, 3, 4],
            "test_indices": [4, 5, 6],  # 4 overlaps with val!
        }

        with pytest.raises(AssertionError, match="Val/Test leakage"):
            assert_temporal(split)

    def test_assert_temporal_empty_splits(self):
        """Test that empty splits are handled correctly."""
        split = {
            "name": "empty_test",
            "train_indices": [0, 1, 2],
            "val_indices": [3, 4],
            "test_indices": [],  # Empty is OK
        }

        # Should not raise
        assert_temporal(split)


class TestAssertNoRandom:
    """Tests for assert_no_random function."""

    def test_assert_no_random_valid_config(self):
        """Test that valid temporal config passes."""
        config = {"split_impl": "temporal", "split_strategy": "forward_chaining"}

        # Should not raise
        assert_no_random(config)

    def test_assert_no_random_detects_train_test_split(self):
        """Test that train_test_split is forbidden."""
        config = {"split_impl": "train_test_split"}

        with pytest.raises(AssertionError, match="FORBIDDEN"):
            assert_no_random(config)

    def test_assert_no_random_detects_kfold(self):
        """Test that KFold is forbidden."""
        config = {"split_impl": "KFold"}

        with pytest.raises(AssertionError, match="FORBIDDEN"):
            assert_no_random(config)

    def test_assert_no_random_detects_stratified_kfold(self):
        """Test that StratifiedKFold is forbidden."""
        config = {"split_impl": "StratifiedKFold"}

        with pytest.raises(AssertionError, match="FORBIDDEN"):
            assert_no_random(config)

    def test_assert_no_random_detects_shuffle(self):
        """Test that shuffle is forbidden."""
        config = {"split_impl": "shuffle_split"}

        with pytest.raises(AssertionError, match="FORBIDDEN"):
            assert_no_random(config)

    def test_assert_no_random_detects_random_strategy(self):
        """Test that random strategy is forbidden."""
        config = {"split_strategy": "random"}

        with pytest.raises(AssertionError, match="FORBIDDEN"):
            assert_no_random(config)

    def test_assert_no_random_detects_stratified_strategy(self):
        """Test that stratified strategy is forbidden."""
        config = {"split_strategy": "stratified"}

        with pytest.raises(AssertionError, match="FORBIDDEN"):
            assert_no_random(config)


class TestDeprecatedFunctions:
    """Tests for deprecated split creation functions."""

    def test_create_stratified_splits_raises(self):
        """Test that create_stratified_splits always raises RuntimeError."""
        with pytest.raises(RuntimeError, match="DEPRECATED"):
            create_stratified_splits()

        # Even with arguments
        with pytest.raises(RuntimeError, match="DEPRECATED"):
            create_stratified_splits(np.random.rand(100, 10), np.random.randint(0, 2, 100))


class TestGetDefaultSplit:
    """Tests for get_default_split function."""

    def test_get_default_split_finds_existing(self):
        """Test that get_default_split can find an existing split."""
        # This test only runs if the actual split file exists
        try:
            split = get_default_split()
            assert "train_indices" in split
            assert "val_indices" in split
            assert len(split["train_indices"]) > 0
        except FileNotFoundError:
            pytest.skip("Default split file not found - run in moola project root")

    def test_get_default_split_missing_raises(self, monkeypatch):
        """Test that get_default_split raises if no files exist."""

        # Mock Path.exists to always return False
        def mock_exists(self):
            return False

        monkeypatch.setattr(Path, "exists", mock_exists)

        with pytest.raises(FileNotFoundError, match="Default split not found"):
            get_default_split()


class TestRealWorldScenarios:
    """Integration tests with realistic split scenarios."""

    def test_forward_chaining_split(self, tmp_path):
        """Test a realistic forward-chaining split scenario."""
        # Simulate a time series with 100 samples
        # Forward chaining: train on [0:60], val on [60:80], test on [80:100]
        split_file = tmp_path / "forward_chain.json"
        split_data = {
            "name": "forward_chain_fold_0",
            "train_indices": list(range(0, 60)),
            "val_indices": list(range(60, 80)),
            "test_indices": list(range(80, 100)),
        }

        with open(split_file, "w") as f:
            json.dump(split_data, f)

        loaded = load_split(str(split_file))
        assert_temporal(loaded)

        # Verify no overlap
        train_set = set(loaded["train_indices"])
        val_set = set(loaded["val_indices"])
        test_set = set(loaded["test_indices"])

        assert train_set.isdisjoint(val_set)
        assert train_set.isdisjoint(test_set)
        assert val_set.isdisjoint(test_set)

    def test_expanding_window_split(self, tmp_path):
        """Test an expanding window split (common in time series)."""
        # Fold 0: train [0:40], val [40:50]
        # Fold 1: train [0:50], val [50:60]
        # Fold 2: train [0:60], val [60:70]

        for fold in range(3):
            train_end = 40 + fold * 10
            val_start = train_end
            val_end = val_start + 10

            split_file = tmp_path / f"expanding_fold_{fold}.json"
            split_data = {
                "name": f"expanding_fold_{fold}",
                "train_indices": list(range(0, train_end)),
                "val_indices": list(range(val_start, val_end)),
                "test_indices": [],
            }

            with open(split_file, "w") as f:
                json.dump(split_data, f)

            loaded = load_split(str(split_file))
            assert_temporal(loaded)

    def test_sliding_window_split(self, tmp_path):
        """Test a sliding window split."""
        # Window size: 40 train + 10 val
        # Fold 0: train [0:40], val [40:50]
        # Fold 1: train [10:50], val [50:60]
        # Fold 2: train [20:60], val [60:70]

        for fold in range(3):
            train_start = fold * 10
            train_end = train_start + 40
            val_end = train_end + 10

            split_file = tmp_path / f"sliding_fold_{fold}.json"
            split_data = {
                "name": f"sliding_fold_{fold}",
                "train_indices": list(range(train_start, train_end)),
                "val_indices": list(range(train_end, val_end)),
                "test_indices": [],
            }

            with open(split_file, "w") as f:
                json.dump(split_data, f)

            loaded = load_split(str(split_file))
            assert_temporal(loaded)
