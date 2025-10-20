#!/usr/bin/env python3
"""Setup and Validate MLOps Experiment Environment.

Checks:
- Data availability
- MLflow installation
- CUDA availability
- Directory structure
- Experiment config integrity

Usage:
    python scripts/setup_mlops_experiments.py
    python scripts/setup_mlops_experiments.py --fix  # Auto-create missing directories
"""

import argparse
import sys
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

from experiment_configs import (
    PHASE_1_EXPERIMENTS,
    get_phase2_experiments,
    get_phase3_experiments,
    get_all_experiments_sequential
)


class EnvironmentValidator:
    """Validates MLOps experiment environment."""

    def __init__(self, fix: bool = False):
        """Initialize validator.

        Args:
            fix: If True, auto-create missing directories
        """
        self.fix = fix
        self.errors = []
        self.warnings = []
        self.passed = []

    def check_dependencies(self):
        """Check required Python packages."""
        print("\n" + "="*70)
        print("CHECKING DEPENDENCIES")
        print("="*70)

        # Required packages
        if TORCH_AVAILABLE:
            import torch
            print(f"✓ PyTorch: {torch.__version__}")
            self.passed.append("PyTorch installed")
        else:
            print("✗ PyTorch: NOT FOUND")
            self.errors.append("PyTorch not installed. Install with: pip install torch")

        if NUMPY_AVAILABLE:
            import numpy as np
            print(f"✓ NumPy: {np.__version__}")
            self.passed.append("NumPy installed")
        else:
            print("✗ NumPy: NOT FOUND")
            self.errors.append("NumPy not installed. Install with: pip install numpy")

        if PANDAS_AVAILABLE:
            import pandas as pd
            print(f"✓ Pandas: {pd.__version__}")
            self.passed.append("Pandas installed")
        else:
            print("✗ Pandas: NOT FOUND")
            self.errors.append("Pandas not installed. Install with: pip install pandas")

        # Optional packages
        if MLFLOW_AVAILABLE:
            import mlflow
            print(f"✓ MLflow: {mlflow.__version__}")
            self.passed.append("MLflow installed")
        else:
            print("⚠ MLflow: NOT FOUND (optional)")
            self.warnings.append("MLflow not installed. Experiments will run without tracking. "
                               "Install with: pip install mlflow")

        print()

    def check_cuda(self):
        """Check CUDA availability."""
        print("="*70)
        print("CHECKING CUDA")
        print("="*70)

        if not TORCH_AVAILABLE:
            print("✗ PyTorch not available, skipping CUDA check")
            return

        import torch

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Device count: {torch.cuda.device_count()}")

            # Memory info
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / (1024**3)
                print(f"  GPU {i}: {props.name}, {total_memory:.1f} GB")

            self.passed.append("CUDA available")

            # Check if sufficient memory
            if total_memory < 20:
                self.warnings.append(
                    f"GPU memory ({total_memory:.1f} GB) may be insufficient for pre-training. "
                    f"Recommend 24GB+ (RTX 4090 or better)"
                )
        else:
            print("⚠ CUDA not available - experiments will run on CPU (very slow)")
            self.warnings.append(
                "CUDA not available. Experiments will run on CPU (~10x slower). "
                "Consider using cloud GPU."
            )

        print()

    def check_data(self):
        """Check data availability."""
        print("="*70)
        print("CHECKING DATA")
        print("="*70)

        data_dir = Path("data/processed")

        required_files = [
            "train.npz",
            "test.npz",
            "unlabeled_augmented.npz"
        ]

        all_present = True
        for filename in required_files:
            filepath = data_dir / filename
            if filepath.exists():
                # Load and check shape
                data = np.load(filepath)
                if 'X' in data:
                    print(f"✓ {filename}: shape {data['X'].shape}")
                    self.passed.append(f"{filename} present")
                else:
                    print(f"⚠ {filename}: exists but missing 'X' key")
                    self.warnings.append(f"{filename} has unexpected format")
            else:
                print(f"✗ {filename}: NOT FOUND")
                self.errors.append(f"{filename} not found in {data_dir}")
                all_present = False

        if not all_present:
            print("\n  Run data pipeline first:")
            print("  python scripts/generate_augmented_data.py  # Or equivalent")

        print()

    def check_directories(self):
        """Check and optionally create required directories."""
        print("="*70)
        print("CHECKING DIRECTORIES")
        print("="*70)

        required_dirs = [
            Path("data/processed"),
            Path("data/artifacts"),
            Path("data/artifacts/pretrained"),
            Path("mlruns"),
        ]

        for dir_path in required_dirs:
            if dir_path.exists():
                print(f"✓ {dir_path}/")
                self.passed.append(f"{dir_path} exists")
            else:
                if self.fix:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"✓ {dir_path}/ (CREATED)")
                    self.passed.append(f"{dir_path} created")
                else:
                    print(f"✗ {dir_path}/ (MISSING)")
                    self.errors.append(f"{dir_path} not found. Run with --fix to create.")

        print()

    def check_experiment_configs(self):
        """Validate experiment configurations."""
        print("="*70)
        print("CHECKING EXPERIMENT CONFIGS")
        print("="*70)

        try:
            # Load all experiments
            all_exps = get_all_experiments_sequential()
            print(f"✓ Loaded {len(all_exps)} experiment configs")

            # Check uniqueness
            ids = [exp.experiment_id for exp in all_exps]
            if len(ids) == len(set(ids)):
                print(f"✓ All experiment IDs are unique")
                self.passed.append("Experiment configs valid")
            else:
                print(f"✗ Duplicate experiment IDs found")
                duplicates = [id for id in ids if ids.count(id) > 1]
                self.errors.append(f"Duplicate experiment IDs: {set(duplicates)}")

            # Check parameter ranges
            for exp in all_exps:
                if not (0 < exp.time_warp_sigma < 1):
                    self.errors.append(
                        f"{exp.experiment_id}: invalid time_warp_sigma={exp.time_warp_sigma}"
                    )
                if not (0 < exp.expected_accuracy_min <= exp.expected_accuracy_max < 1):
                    self.errors.append(
                        f"{exp.experiment_id}: invalid accuracy range"
                    )

            # Print phase breakdown
            print(f"  Phase 1: {len(PHASE_1_EXPERIMENTS)} experiments")
            print(f"  Phase 2: {len(get_phase2_experiments())} experiments")
            print(f"  Phase 3: {len(get_phase3_experiments())} experiments")

        except Exception as e:
            print(f"✗ Failed to load experiment configs: {e}")
            self.errors.append(f"Experiment config error: {e}")

        print()

    def check_scripts(self):
        """Check required scripts exist."""
        print("="*70)
        print("CHECKING SCRIPTS")
        print("="*70)

        required_scripts = [
            "scripts/experiment_configs.py",
            "scripts/run_lstm_experiment.py",
            "scripts/orchestrate_phases.py",
            "scripts/aggregate_results.py",
        ]

        for script in required_scripts:
            if Path(script).exists():
                print(f"✓ {script}")
                self.passed.append(f"{script} exists")
            else:
                print(f"✗ {script}: NOT FOUND")
                self.errors.append(f"{script} not found")

        print()

    def print_summary(self):
        """Print validation summary."""
        print("="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        print(f"✓ Passed: {len(self.passed)}")
        print(f"⚠ Warnings: {len(self.warnings)}")
        print(f"✗ Errors: {len(self.errors)}")
        print()

        if self.warnings:
            print("WARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
            print()

        if self.errors:
            print("ERRORS:")
            for error in self.errors:
                print(f"  ✗ {error}")
            print()

        if self.errors:
            print("❌ VALIDATION FAILED - Fix errors before running experiments")
            return False
        elif self.warnings:
            print("⚠️  VALIDATION PASSED WITH WARNINGS - Experiments can run but may have issues")
            return True
        else:
            print("✅ VALIDATION PASSED - Ready to run experiments!")
            return True

    def run_all_checks(self) -> bool:
        """Run all validation checks.

        Returns:
            True if validation passed, False otherwise
        """
        print("\n" + "#"*70)
        print("# MLOPS EXPERIMENT ENVIRONMENT VALIDATION")
        print("#"*70)

        self.check_dependencies()
        self.check_cuda()
        self.check_data()
        self.check_directories()
        self.check_experiment_configs()
        self.check_scripts()

        return self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="Setup and validate MLOps experiment environment"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-create missing directories"
    )

    args = parser.parse_args()

    validator = EnvironmentValidator(fix=args.fix)
    success = validator.run_all_checks()

    if success:
        print("\nNEXT STEPS:")
        print("1. Preview experiments: python scripts/preview_experiments.py")
        print("2. Run Phase 1: python scripts/orchestrate_phases.py --mode sequential --phase 1")
        print("3. Run all phases: python scripts/orchestrate_phases.py --mode sequential")
        print("4. Monitor MLflow: mlflow ui")
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
