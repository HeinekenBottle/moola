#!/usr/bin/env python3
"""
Validate data integrity using checksums.

Usage:
    python3 scripts/validate_data.py
"""

import subprocess
import sys
from pathlib import Path


def validate_checksums():
    """Validate data files against checksums."""
    data_dir = Path("data")
    checksum_file = data_dir / "checksums.sha256"

    if not checksum_file.exists():
        print("❌ Checksum file not found")
        return False

    # Run shasum to verify
    result = subprocess.run(
        ["shasum", "-a", "256", "-c", str(checksum_file)], capture_output=True, text=True
    )

    if result.returncode == 0:
        print("✅ All data checksums verified")
        return True
    else:
        print("❌ Data checksum validation failed")
        print(result.stdout)
        print(result.stderr)
        return False


def validate_data_structure():
    """Validate that required data files exist."""
    required_files = [
        "data/raw/nq_5year.parquet",  # To be downloaded
        "data/processed/train_174.parquet",
        "data/splits/temporal_split.json",
    ]

    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)

    if missing:
        print(f"❌ Missing required data files: {missing}")
        return False
    else:
        print("✅ All required data files present")
        return True


if __name__ == "__main__":
    print("Validating Moola data integrity...")

    structure_ok = validate_data_structure()
    checksums_ok = validate_checksums()

    if structure_ok and checksums_ok:
        print("🎉 Data validation passed")
        sys.exit(0)
    else:
        print("❌ Data validation failed")
        sys.exit(1)
