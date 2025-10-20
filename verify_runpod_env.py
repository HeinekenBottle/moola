#!/usr/bin/env python3
"""
RunPod Environment Verification Script
Run this FIRST on RunPod to verify everything is ready for training.

Usage:
    python3 verify_runpod_env.py
"""

import sys
from pathlib import Path


def check_python_version():
    """Check Python version >= 3.10"""
    print("=" * 70)
    print("1. Checking Python Version")
    print("=" * 70)
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 10:
        print("✅ Python version OK (>= 3.10)")
        return True
    else:
        print("❌ Python version too old (need >= 3.10)")
        return False


def check_pytorch_cuda():
    """Check PyTorch and CUDA availability"""
    print("\n" + "=" * 70)
    print("2. Checking PyTorch and CUDA")
    print("=" * 70)
    try:
        import torch
        print(f"✅ PyTorch installed: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print(f"✅ CUDA available: {torch.version.cuda}")
            print(f"✅ GPU count: {torch.cuda.device_count()}")
            print(f"✅ GPU name: {torch.cuda.get_device_name(0)}")
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU memory: {memory_gb:.2f} GB")

            if memory_gb < 20:
                print("⚠️  Warning: GPU memory < 20 GB. May need smaller batch sizes.")

            return True
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False


def check_required_packages():
    """Check required packages"""
    print("\n" + "=" * 70)
    print("3. Checking Required Packages")
    print("=" * 70)

    packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "sklearn": "scikit-learn",
    }

    all_ok = True
    for module, pkg_name in packages.items():
        try:
            mod = __import__(module)
            version = getattr(mod, "__version__", "unknown")
            print(f"✅ {pkg_name}: {version}")
        except ImportError:
            print(f"❌ {pkg_name} not installed")
            print(f"   Install: pip3 install {pkg_name}")
            all_ok = False

    return all_ok


def check_project_structure():
    """Check project structure"""
    print("\n" + "=" * 70)
    print("4. Checking Project Structure")
    print("=" * 70)

    required_paths = [
        "src/moola",
        "data/processed/labeled/train_latest_relative.parquet",
        "data/splits/fwd_chain_v3.json",
    ]

    all_ok = True
    for path in required_paths:
        p = Path(path)
        if p.exists():
            if p.is_file():
                size_kb = p.stat().st_size / 1024
                print(f"✅ {path} ({size_kb:.1f} KB)")
            else:
                print(f"✅ {path}/")
        else:
            print(f"❌ {path} not found")
            all_ok = False

    return all_ok


def check_output_directories():
    """Check/create output directories"""
    print("\n" + "=" * 70)
    print("5. Checking Output Directories")
    print("=" * 70)

    required_dirs = [
        "artifacts/encoders/pretrained",
        "artifacts/runs",
        "data/logs",
    ]

    for dir_path in required_dirs:
        p = Path(dir_path)
        if p.exists():
            print(f"✅ {dir_path}/ exists")
        else:
            p.mkdir(parents=True, exist_ok=True)
            print(f"✅ {dir_path}/ created")

    return True


def verify_dataset():
    """Verify 11D dataset structure"""
    print("\n" + "=" * 70)
    print("6. Verifying 11D Dataset")
    print("=" * 70)

    try:
        import pandas as pd
        import numpy as np

        df = pd.read_parquet("data/processed/labeled/train_latest_relative.parquet")
        print(f"✅ Dataset loaded: {len(df)} samples")

        # Check columns
        required_cols = ["window_id", "label", "features"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return False
        print(f"✅ Required columns present")

        # Check feature shape
        first_feat = df["features"].iloc[0]
        stacked = np.array([f for f in first_feat])
        print(f"✅ Feature shape: {stacked.shape}")

        if stacked.shape != (105, 11):
            print(f"❌ Expected shape (105, 11), got {stacked.shape}")
            return False

        # Check labels
        label_counts = df["label"].value_counts().to_dict()
        print(f"✅ Label distribution: {label_counts}")

        return True
    except Exception as e:
        print(f"❌ Error verifying dataset: {e}")
        return False


def check_pythonpath():
    """Check PYTHONPATH"""
    print("\n" + "=" * 70)
    print("7. Checking PYTHONPATH")
    print("=" * 70)

    import os
    pythonpath = os.environ.get("PYTHONPATH", "")

    if "/workspace/moola/src" in pythonpath or str(Path.cwd() / "src") in pythonpath:
        print(f"✅ PYTHONPATH configured: {pythonpath}")
        return True
    else:
        print(f"⚠️  PYTHONPATH not set correctly")
        print(f"   Current: {pythonpath}")
        print(f"   Run: export PYTHONPATH=/workspace/moola/src:$PYTHONPATH")
        return False


def main():
    """Run all checks"""
    print("\n" + "=" * 70)
    print("RUNPOD ENVIRONMENT VERIFICATION")
    print("=" * 70)
    print()

    checks = [
        ("Python Version", check_python_version),
        ("PyTorch and CUDA", check_pytorch_cuda),
        ("Required Packages", check_required_packages),
        ("Project Structure", check_project_structure),
        ("Output Directories", check_output_directories),
        ("11D Dataset", verify_dataset),
        ("PYTHONPATH", check_pythonpath),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            results[name] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(results.values())

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL CHECKS PASSED - READY FOR TRAINING!")
    else:
        print("❌ SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
    print("=" * 70)
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
