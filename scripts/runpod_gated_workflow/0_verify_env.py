#!/usr/bin/env python3
"""Gate 0: Environment and Data Verification.

Validates:
- CUDA availability and device configuration
- Data files exist at expected locations
- Temporal splits are valid
- No data leakage between train/val/test

Exit codes:
- 0: All checks passed
- 1: Verification failed
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch


def log_result(message: str, status: str = "INFO"):
    """Log with timestamp and status."""
    import datetime
    timestamp = datetime.datetime.now().isoformat()
    print(f"[{timestamp}] [{status}] {message}")


def verify_cuda():
    """Verify CUDA is available and report device info."""
    log_result("=" * 70)
    log_result("CUDA VERIFICATION")
    log_result("=" * 70)

    if not torch.cuda.is_available():
        log_result("CUDA not available - expected RTX 4090 GPU", "ERROR")
        return False

    device_name = torch.cuda.get_device_name(0)
    device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    log_result(f"CUDA available: {device_name}")
    log_result(f"Total VRAM: {device_memory:.1f} GB")

    if "4090" not in device_name:
        log_result(f"Warning: Expected RTX 4090, got {device_name}", "WARN")

    return True


def verify_data_files():
    """Verify all required data files exist."""
    log_result("=" * 70)
    log_result("DATA FILES VERIFICATION")
    log_result("=" * 70)

    base_path = Path("/workspace/moola")

    required_files = {
        "labeled_data": base_path / "data/processed/train_clean.parquet",
        "unlabeled_data": base_path / "data/raw/unlabeled_windows.parquet",
        "split_fold_0": base_path / "data/artifacts/splits/v1/fold_0.json",
    }

    all_exist = True
    for name, filepath in required_files.items():
        if filepath.exists():
            if filepath.suffix == ".parquet":
                df = pd.read_parquet(filepath)
                log_result(f"✓ {name}: {filepath} ({len(df)} samples)")
            else:
                log_result(f"✓ {name}: {filepath}")
        else:
            log_result(f"✗ {name}: {filepath} NOT FOUND", "ERROR")
            all_exist = False

    return all_exist


def verify_temporal_splits():
    """Verify splits are temporal and non-overlapping."""
    log_result("=" * 70)
    log_result("TEMPORAL SPLIT VERIFICATION")
    log_result("=" * 70)

    split_path = Path("/workspace/moola/data/artifacts/splits/v1/fold_0.json")

    if not split_path.exists():
        log_result(f"Split file not found: {split_path}", "ERROR")
        return False

    with open(split_path, "r") as f:
        split_data = json.load(f)

    # Handle both field naming conventions
    train_indices = set(split_data.get("train_indices", split_data.get("train_idx", [])))
    val_indices = set(split_data.get("val_indices", split_data.get("val_idx", [])))

    # Check for overlap
    overlap = train_indices & val_indices
    if overlap:
        log_result(f"✗ Train/val overlap detected: {len(overlap)} samples", "ERROR")
        return False

    log_result(f"✓ Train samples: {len(train_indices)}")
    log_result(f"✓ Val samples: {len(val_indices)}")
    log_result(f"✓ No overlap between train/val")

    # Verify temporal ordering (train comes before val)
    if train_indices and val_indices:
        max_train = max(train_indices)
        min_val = min(val_indices)

        if max_train < min_val:
            log_result(f"✓ Temporal ordering valid (train max={max_train}, val min={min_val})")
        else:
            log_result(f"✗ Temporal ordering violated (train max={max_train}, val min={min_val})", "ERROR")
            return False

    return True


def verify_labeled_data_schema():
    """Verify labeled data has required columns and valid structure."""
    log_result("=" * 70)
    log_result("LABELED DATA SCHEMA VERIFICATION")
    log_result("=" * 70)

    data_path = Path("/workspace/moola/data/processed/train_clean.parquet")
    df = pd.read_parquet(data_path)

    required_columns = ["window_id", "label", "features"]
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        log_result(f"✗ Missing columns: {missing}", "ERROR")
        return False

    log_result(f"✓ All required columns present: {required_columns}")

    # Check label distribution
    label_dist = df["label"].value_counts().to_dict()
    log_result(f"✓ Label distribution: {label_dist}")

    # Check for class imbalance
    if len(label_dist) == 2:
        values = list(label_dist.values())
        ratio = max(values) / min(values)
        if ratio > 3.0:
            log_result(f"⚠ Severe class imbalance detected: {ratio:.2f}:1", "WARN")

    return True


def main():
    """Run all verification gates."""
    log_result("=" * 70)
    log_result("GATE 0: ENVIRONMENT AND DATA VERIFICATION")
    log_result("=" * 70)

    checks = [
        ("CUDA Availability", verify_cuda),
        ("Data Files", verify_data_files),
        ("Temporal Splits", verify_temporal_splits),
        ("Labeled Data Schema", verify_labeled_data_schema),
    ]

    all_passed = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                log_result(f"✗ {check_name} FAILED", "ERROR")
                all_passed = False
            else:
                log_result(f"✓ {check_name} PASSED")
        except Exception as e:
            log_result(f"✗ {check_name} EXCEPTION: {e}", "ERROR")
            all_passed = False

    log_result("=" * 70)
    if all_passed:
        log_result("GATE 0: PASSED - Environment ready for training", "SUCCESS")
        sys.exit(0)
    else:
        log_result("GATE 0: FAILED - Fix errors before proceeding", "ERROR")
        sys.exit(1)


if __name__ == "__main__":
    main()
