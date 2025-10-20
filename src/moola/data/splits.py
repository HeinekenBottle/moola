"""Temporal split loading and validation for financial time series.

This module enforces forward-chaining (temporal) splits and forbids random/stratified
splitting which creates look-ahead bias in time series data.

CRITICAL: Financial time series MUST use temporal ordering to prevent data leakage.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np


def load_split(split_path: str) -> dict[str, Any]:
    """Load split definition from JSON file.

    Args:
        split_path: Path to split JSON (e.g., "data/artifacts/splits/v1/fold_0.json")

    Returns:
        Dictionary with train/val/test indices per fold

    Raises:
        FileNotFoundError: If split file doesn't exist
        ValueError: If split format is invalid
    """
    path = Path(split_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_path}\n"
            f"Available splits should be in: data/artifacts/splits/v1/fold_*.json"
        )

    with open(path) as f:
        split_data = json.load(f)

    # Map old field names to new standard names
    field_mapping = {
        "train_idx": "train_indices",
        "val_idx": "val_indices",
        "test_idx": "test_indices",
    }

    for old_name, new_name in field_mapping.items():
        if old_name in split_data and new_name not in split_data:
            split_data[new_name] = split_data[old_name]

    # Add name if missing
    if "name" not in split_data:
        split_data["name"] = Path(split_path).stem

    # Validate required fields
    required = ["train_indices", "val_indices"]
    missing = [f for f in required if f not in split_data]
    if missing:
        raise ValueError(
            f"Split file missing required fields: {missing}\n"
            f"Expected: {required}\n"
            f"Found: {list(split_data.keys())}"
        )

    # Test indices are optional for 2-way splits
    if "test_indices" not in split_data:
        split_data["test_indices"] = []

    return split_data


def assert_temporal(split_data: dict[str, Any]) -> None:
    """Validate that split uses forward-chaining (temporal ordering).

    Args:
        split_data: Split dictionary from load_split()

    Raises:
        AssertionError: If indices are not monotonic or have leakage
    """
    train_idx = np.array(split_data["train_indices"])
    val_idx = np.array(split_data["val_indices"])
    test_idx = np.array(split_data.get("test_indices", []))

    # Check monotonic ordering within each split
    if len(train_idx) > 0:
        if not np.all(np.diff(train_idx) >= 0):
            raise AssertionError(
                f"Train indices not monotonic (time-ordered)!\n"
                f"First 10 indices: {train_idx[:10]}\n"
                f"This indicates random/shuffled splitting which creates look-ahead bias."
            )

    if len(val_idx) > 0:
        if not np.all(np.diff(val_idx) >= 0):
            raise AssertionError(
                f"Val indices not monotonic (time-ordered)!\n"
                f"First 10 indices: {val_idx[:10]}\n"
                f"This indicates random/shuffled splitting which creates look-ahead bias."
            )

    if len(test_idx) > 0:
        if not np.all(np.diff(test_idx) >= 0):
            raise AssertionError(
                f"Test indices not monotonic (time-ordered)!\n"
                f"First 10 indices: {test_idx[:10]}\n"
                f"This indicates random/shuffled splitting which creates look-ahead bias."
            )

    # Check for leakage (no overlap)
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    if not train_set.isdisjoint(val_set):
        overlap = train_set & val_set
        raise AssertionError(
            f"Train/Val leakage detected!\n"
            f"Overlapping indices: {sorted(list(overlap))[:10]}...\n"
            f"Total overlap: {len(overlap)} samples"
        )

    if not train_set.isdisjoint(test_set):
        overlap = train_set & test_set
        raise AssertionError(
            f"Train/Test leakage detected!\n"
            f"Overlapping indices: {sorted(list(overlap))[:10]}...\n"
            f"Total overlap: {len(overlap)} samples"
        )

    if not val_set.isdisjoint(test_set):
        overlap = val_set & test_set
        raise AssertionError(
            f"Val/Test leakage detected!\n"
            f"Overlapping indices: {sorted(list(overlap))[:10]}...\n"
            f"Total overlap: {len(overlap)} samples"
        )

    print(f"✓ Split validation passed: {split_data['name']}")
    print(f"  Train: {len(train_idx)} samples")
    print(f"  Val: {len(val_idx)} samples")
    if len(test_idx) > 0:
        print(f"  Test: {len(test_idx)} samples")


def assert_no_random(config: dict[str, Any]) -> None:
    """Forbid random or stratified split strategies.

    Args:
        config: Configuration dictionary

    Raises:
        AssertionError: If random split methods detected
    """
    banned_methods = {
        "train_test_split",
        "KFold",
        "StratifiedKFold",
        "ShuffleSplit",
        "random",
        "shuffle",
    }

    # Check split implementation
    split_impl = str(config.get("split_impl", "")).lower()
    for banned in banned_methods:
        if banned.lower() in split_impl:
            raise AssertionError(
                f"❌ Random/stratified split is FORBIDDEN in financial time series!\n"
                f"Detected banned method: {banned}\n"
                f"Use forward-chaining splits only: data/artifacts/splits/v1/fold_*.json"
            )

    # Check split strategy
    strategy = str(config.get("split_strategy", "")).lower()
    if "random" in strategy or "stratified" in strategy:
        raise AssertionError(
            f"❌ Random/stratified strategy is FORBIDDEN: {strategy}\n"
            f"Use 'forward_chaining' or 'temporal' only"
        )

    print("✓ No random split methods detected")


def create_forward_chaining_split(
    n_samples: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    name: str = "forward_chain",
    output_path: str = None,
) -> dict[str, Any]:
    """Create a forward-chaining split for time series.

    This creates a true temporal split where:
    - Train data: earliest samples
    - Val data: middle samples (after train)
    - Test data: latest samples (after val)

    Args:
        n_samples: Total number of samples
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.2)
        test_ratio: Fraction for testing (default: 0.0)
        name: Split name
        output_path: Optional path to save JSON file

    Returns:
        Split dictionary with train/val/test indices

    Raises:
        ValueError: If ratios don't sum to 1.0

    Example:
        >>> # Create 80/20 train/val split
        >>> split = create_forward_chaining_split(
        ...     n_samples=100,
        ...     train_ratio=0.8,
        ...     val_ratio=0.2,
        ...     output_path="data/splits/my_split.json"
        ... )
        >>> print(split['train_indices'])  # [0, 1, ..., 79]
        >>> print(split['val_indices'])    # [80, 81, ..., 99]
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(
            f"Ratios must sum to 1.0, got {total_ratio} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})"
        )

    # Calculate split points
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    # Create split indices
    split_data = {
        "name": name,
        "method": "forward_chaining",
        "n_samples": n_samples,
        "train_indices": list(range(0, train_end)),
        "val_indices": list(range(train_end, val_end)),
        "test_indices": list(range(val_end, n_samples)) if test_ratio > 0 else [],
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

    # Validate the split
    assert_temporal(split_data)

    # Save if path provided
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            import json

            json.dump(split_data, f, indent=2)
        print(f"Saved forward-chaining split to {output_path}")

    return split_data


def get_default_split() -> dict[str, Any]:
    """Load the default forward-chaining split.

    Returns:
        Split data for first fold
    """
    # Try multiple locations
    possible_paths = [
        "data/artifacts/splits/v1/fold_0.json",
        "/Users/jack/projects/moola/data/artifacts/splits/v1/fold_0.json",
    ]

    for path in possible_paths:
        if Path(path).exists():
            return load_split(path)

    raise FileNotFoundError(
        "Default split not found. Tried:\n"
        + "\n".join(f"  - {p}" for p in possible_paths)
        + "\n\nSplits should exist in data/artifacts/splits/v1/"
    )


# DEPRECATED - DO NOT USE
def create_stratified_splits(*args, **kwargs):
    """DEPRECATED: Stratified splits are forbidden for time series.

    Raises:
        RuntimeError: Always - this function should never be called
    """
    raise RuntimeError(
        "❌ create_stratified_splits() is DEPRECATED and FORBIDDEN!\n"
        "\n"
        "Stratified splits create look-ahead bias in time series by shuffling data.\n"
        "Financial time series MUST preserve temporal ordering.\n"
        "\n"
        "Use temporal splits instead:\n"
        "  from moola.data.splits import load_split, assert_temporal\n"
        "  split_data = load_split('data/artifacts/splits/v1/fold_0.json')\n"
        "  assert_temporal(split_data)\n"
        "\n"
        "This ensures train data comes before validation data in time."
    )
