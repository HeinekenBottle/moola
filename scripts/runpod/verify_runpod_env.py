#!/usr/bin/env python3
"""Verify RunPod environment dependencies and compatibility.

This script checks that all required packages are installed with compatible versions
for GPU training on RunPod. It verifies:
- Critical ML packages (torch, numpy, sklearn, xgboost, etc.)
- NumPy version compatibility with PyTorch
- CUDA availability and GPU detection
- Version constraints from requirements-runpod.txt

Usage:
    python scripts/verify_runpod_env.py

Exit codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import sys
from typing import List, Tuple


def check_package(name: str, import_name: str = None) -> Tuple[str, str, str]:
    """Check if a package is installed and return (name, version, status).

    Args:
        name: Display name of the package
        import_name: Import name if different from display name

    Returns:
        Tuple of (name, version_or_error, status_emoji)
    """
    if import_name is None:
        import_name = name.replace("-", "_").lower()

    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return (name, version, "✅")
    except ImportError as e:
        return (name, str(e), "❌")


def check_numpy_compatibility(numpy_version: str, torch_version: str) -> Tuple[str, str]:
    """Check NumPy-PyTorch compatibility.

    Args:
        numpy_version: NumPy version string
        torch_version: PyTorch version string

    Returns:
        Tuple of (status_emoji, message)
    """
    try:
        np_major = int(numpy_version.split('.')[0])
        torch_major = int(torch_version.split('.')[0])
        torch_minor = int(torch_version.split('.')[1])

        # PyTorch 2.0-2.2 requires NumPy < 2.0
        if torch_major == 2 and torch_minor < 3:
            if np_major >= 2:
                return ("❌", f"PyTorch {torch_version} incompatible with NumPy {numpy_version} (need <2.0)")
            else:
                return ("✅", f"NumPy {numpy_version} compatible with PyTorch {torch_version}")

        # PyTorch 2.3+ supports NumPy 2.0+
        elif torch_major == 2 and torch_minor >= 3:
            return ("✅", f"NumPy {numpy_version} compatible with PyTorch {torch_version}")

        else:
            return ("⚠️", f"Unknown compatibility: PyTorch {torch_version}, NumPy {numpy_version}")

    except (ValueError, IndexError):
        return ("⚠️", "Could not parse version numbers")


def check_cuda() -> List[Tuple[str, str, str]]:
    """Check CUDA availability and GPU detection.

    Returns:
        List of (name, value, status) tuples
    """
    results = []

    try:
        import torch

        if torch.cuda.is_available():
            results.append(("CUDA Available", "Yes", "✅"))
            results.append(("CUDA Version", torch.version.cuda, "✅"))

            device_count = torch.cuda.device_count()
            results.append(("GPU Count", str(device_count), "✅"))

            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                results.append((f"GPU {i}", gpu_name, "✅"))

            # Check VRAM
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024**3)
            status = "✅" if vram_gb >= 16 else "⚠️"
            results.append(("GPU VRAM", f"{vram_gb:.1f} GB", status))

        else:
            results.append(("CUDA Available", "No", "❌"))

    except Exception as e:
        results.append(("CUDA Check", str(e), "❌"))

    return results


def main():
    """Run all environment checks and print results."""
    print("\n" + "="*70)
    print(" "*15 + "RunPod Environment Verification")
    print("="*70 + "\n")

    # Check core packages
    print("📦 Core ML Packages")
    print("-"*70)

    checks = [
        check_package("torch"),
        check_package("numpy"),
        check_package("pandas"),
        check_package("scipy"),
        check_package("scikit-learn", "sklearn"),
        check_package("xgboost"),
        check_package("imbalanced-learn", "imblearn"),
        check_package("pytorch-lightning", "pytorch_lightning"),
    ]

    for name, version, status in checks:
        print(f"{status} {name:25s} {version}")

    # Check additional packages
    print("\n📋 Additional Packages")
    print("-"*70)

    additional_checks = [
        check_package("mlflow"),
        check_package("loguru"),
        check_package("click"),
        check_package("typer"),
        check_package("hydra-core", "hydra"),
        check_package("pydantic"),
        check_package("pyarrow"),
        check_package("pandera"),
        check_package("rich"),
    ]

    for name, version, status in additional_checks:
        print(f"{status} {name:25s} {version}")

    # NumPy-PyTorch compatibility check
    print("\n🔗 Version Compatibility")
    print("-"*70)

    numpy_version = None
    torch_version = None

    for name, version, status in checks:
        if name == "numpy" and status == "✅":
            numpy_version = version
        elif name == "torch" and status == "✅":
            torch_version = version

    if numpy_version and torch_version:
        compat_status, compat_msg = check_numpy_compatibility(numpy_version, torch_version)
        print(f"{compat_status} {compat_msg}")
    else:
        print("⚠️  Cannot check NumPy-PyTorch compatibility (packages missing)")

    # CUDA check
    print("\n🚀 GPU & CUDA")
    print("-"*70)

    cuda_checks = check_cuda()
    for name, value, status in cuda_checks:
        print(f"{status} {name:25s} {value}")

    # Summary
    print("\n" + "="*70)

    all_checks = checks + additional_checks
    failed = [c for c in all_checks if c[2] == "❌"]
    warnings = [c for c in all_checks if c[2] == "⚠️"]

    if failed:
        print(f"❌ {len(failed)} CRITICAL FAILURES")
        print("\nFailed packages:")
        for name, error, _ in failed:
            print(f"  - {name}: {error}")
        print("\nRun: pip install --no-cache-dir -r requirements-runpod.txt")
        return 1

    elif warnings:
        print(f"⚠️  {len(warnings)} WARNINGS (check manually)")
        for name, msg, _ in warnings:
            print(f"  - {name}: {msg}")
        return 0

    else:
        print("✅ ALL CHECKS PASSED - Environment ready for training!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
