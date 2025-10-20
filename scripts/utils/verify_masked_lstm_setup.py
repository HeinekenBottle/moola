#!/usr/bin/env python3
"""Verify masked LSTM pre-training setup before RunPod deployment.

Checks:
- Required files exist
- Dependencies installed
- Data format correct
- Model can be instantiated
- Pre-training runs on CPU (smoke test)

Usage:
    python scripts/verify_masked_lstm_setup.py
    python scripts/verify_masked_lstm_setup.py --smoke-test  # Run quick CPU pre-training test
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def check_file_exists(path: Path, description: str) -> bool:
    """Check if file exists and report status.

    Args:
        path: Path to check
        description: Human-readable description

    Returns:
        True if file exists
    """
    exists = path.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {path}")
    return exists


def check_import(module_name: str, package_name: str = None) -> bool:
    """Check if Python module can be imported.

    Args:
        module_name: Module name to import
        package_name: Package name for error messages (default: module_name)

    Returns:
        True if import successful
    """
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError as e:
        print(f"  ✗ {package_name}: {e}")
        return False


def verify_dependencies() -> bool:
    """Verify all required dependencies are installed.

    Returns:
        True if all dependencies available
    """
    print("\n[DEPENDENCIES]")

    deps = [
        ("torch", "PyTorch"),
        ("numpy", "NumPy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("tqdm", "tqdm"),
    ]

    results = []
    for module, name in deps:
        results.append(check_import(module, name))

    # Check moola imports
    try:
        from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
        from moola.models import SimpleLSTMModel
        print(f"  ✓ moola.pretraining.masked_lstm_pretrain")
        print(f"  ✓ moola.models.SimpleLSTMModel")
        results.extend([True, True])
    except ImportError as e:
        print(f"  ✗ moola: {e}")
        results.extend([False, False])

    return all(results)


def verify_files() -> bool:
    """Verify all required files exist.

    Returns:
        True if all files exist
    """
    print("\n[FILES]")

    files = [
        (Path("data/processed/train_pivot_134.parquet"), "Training data"),
        (Path("scripts/generate_unlabeled_data.py"), "Unlabeled data generator"),
        (Path(".runpod/deploy_pretrain_masked_lstm.sh"), "Pre-training deployment script"),
        (Path(".runpod/full_pipeline_masked_lstm.sh"), "Full pipeline script"),
        (Path("src/moola/pretraining/masked_lstm_pretrain.py"), "Pre-training module"),
        (Path("src/moola/models/simple_lstm.py"), "SimpleLSTM model"),
    ]

    results = []
    for path, desc in files:
        results.append(check_file_exists(path, desc))

    return all(results)


def verify_data_format() -> bool:
    """Verify training data has correct format.

    Returns:
        True if data format is valid
    """
    print("\n[DATA FORMAT]")

    data_path = Path("data/processed/train_pivot_134.parquet")
    if not data_path.exists():
        print(f"  ✗ Data file not found: {data_path}")
        return False

    try:
        df = pd.read_parquet(data_path)
        print(f"  ✓ Loaded parquet: {len(df)} samples")

        # Check required columns
        required_cols = ["ohlc_sequence", "label"]
        for col in required_cols:
            if col in df.columns:
                print(f"  ✓ Column '{col}' present")
            else:
                print(f"  ✗ Column '{col}' missing")
                return False

        # Check sequence format
        sample_seq = df.iloc[0]["ohlc_sequence"]
        if isinstance(sample_seq, np.ndarray):
            print(f"  ✓ Sequence type: np.ndarray")
            print(f"  ✓ Sequence shape: {sample_seq.shape}")

            # Validate shape
            if sample_seq.ndim == 2:
                seq_len, features = sample_seq.shape
                print(f"  ✓ Sequence dimensions: 2D ({seq_len} timesteps, {features} features)")

                if seq_len == 105 and features == 4:
                    print(f"  ✓ Expected shape: (105, 4)")
                else:
                    print(f"  ⚠ Unexpected shape: ({seq_len}, {features}) (expected: (105, 4))")
                    return False
            else:
                print(f"  ✗ Sequence dimensions: {sample_seq.ndim}D (expected: 2D)")
                return False
        else:
            print(f"  ✗ Sequence type: {type(sample_seq)} (expected: np.ndarray)")
            return False

        return True

    except Exception as e:
        print(f"  ✗ Data validation failed: {e}")
        return False


def verify_model_instantiation() -> bool:
    """Verify models can be instantiated.

    Returns:
        True if models can be instantiated
    """
    print("\n[MODEL INSTANTIATION]")

    try:
        from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
        from moola.models import SimpleLSTMModel

        # Pre-trainer
        pretrainer = MaskedLSTMPretrainer(device="cpu", seed=1337)
        print(f"  ✓ MaskedLSTMPretrainer instantiated")
        print(f"    - Hidden dim: {pretrainer.hidden_dim}")
        print(f"    - Num layers: {pretrainer.num_layers}")
        print(f"    - Mask strategy: {pretrainer.mask_strategy}")

        # SimpleLSTM
        model = SimpleLSTMModel(device="cpu", seed=1337)
        print(f"  ✓ SimpleLSTMModel instantiated")
        print(f"    - Hidden size: {model.hidden_size}")
        print(f"    - Num layers: {model.num_layers}")
        print(f"    - Num heads: {model.num_heads}")

        return True

    except Exception as e:
        print(f"  ✗ Model instantiation failed: {e}")
        return False


def run_smoke_test() -> bool:
    """Run quick CPU pre-training test to verify functionality.

    Returns:
        True if smoke test passed
    """
    print("\n[SMOKE TEST] Running quick CPU pre-training test...")

    try:
        import torch
        from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer

        # Generate synthetic data
        np.random.seed(1337)
        torch.manual_seed(1337)

        n_samples = 50
        seq_len = 105
        features = 4

        X_synthetic = np.random.randn(n_samples, seq_len, features).astype(np.float32)
        print(f"  Generated synthetic data: {X_synthetic.shape}")

        # Initialize pre-trainer
        pretrainer = MaskedLSTMPretrainer(
            device="cpu",
            batch_size=8,
            learning_rate=1e-3,
            seed=1337
        )
        print(f"  Initialized pre-trainer")

        # Run mini pre-training (1 epoch)
        print(f"  Running 1 epoch (this may take 30-60 seconds on CPU)...")
        history = pretrainer.pretrain(
            X_unlabeled=X_synthetic,
            n_epochs=1,
            val_split=0.2,
            patience=10,
            save_path=None,  # Don't save
            verbose=False
        )

        # Check results
        if len(history['train_loss']) > 0 and len(history['val_loss']) > 0:
            print(f"  ✓ Training completed")
            print(f"    - Train loss: {history['train_loss'][-1]:.4f}")
            print(f"    - Val loss: {history['val_loss'][-1]:.4f}")
            return True
        else:
            print(f"  ✗ No training history recorded")
            return False

    except Exception as e:
        print(f"  ✗ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify masked LSTM pre-training setup"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick CPU pre-training test (30-60 seconds)",
    )

    args = parser.parse_args()

    print("="*80)
    print("MASKED LSTM PRE-TRAINING SETUP VERIFICATION")
    print("="*80)

    # Run checks
    checks = {
        "Dependencies": verify_dependencies(),
        "Files": verify_files(),
        "Data Format": verify_data_format(),
        "Model Instantiation": verify_model_instantiation(),
    }

    if args.smoke_test:
        checks["Smoke Test"] = run_smoke_test()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")

    print()

    if all(checks.values()):
        print("✅ All checks passed! Ready for RunPod deployment.")
        print()
        print("Next steps:")
        print("  1. Start RunPod pod with PyTorch 2.4 template")
        print("  2. Run: .runpod/full_pipeline_masked_lstm.sh <HOST> <PORT>")
        print("  3. Monitor: python scripts/monitor_pretraining.py --host <HOST> --port <PORT> --watch")
        print()
        return 0
    else:
        print("❌ Some checks failed. Please fix issues before deployment.")
        print()
        failed_checks = [name for name, passed in checks.items() if not passed]
        print("Failed checks:")
        for check in failed_checks:
            print(f"  - {check}")
        print()
        return 1


if __name__ == "__main__":
    exit(main())
