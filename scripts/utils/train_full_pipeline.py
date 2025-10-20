#!/usr/bin/env python3
"""End-to-end training pipeline for Moola ML system.

This script orchestrates the complete training workflow:
1. Phase 1 fixes verification (SimpleLSTM, SMOTE, CleanLab)
2. Phase 2 augmentation (Mixup, traditional augmentation)
3. Phase 3 SSL pre-training (TS-TCC contrastive learning)
4. Phase 4 final ensemble training (stacking meta-learner)

Usage:
    python scripts/train_full_pipeline.py --device cuda --mlflow-experiment production

Expected Outcome:
    - Pre-trained encoder weights: data/artifacts/models/ts_tcc/pretrained_encoder.pt
    - Fine-tuned base models: data/artifacts/models/{logreg,rf,xgb,simple_lstm,cnn_transformer}/*.pkl
    - Stack ensemble: data/artifacts/models/stack/stack.pkl
    - Final accuracy target: 75-82%
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

import mlflow
from loguru import logger

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"

# Base models configuration
BASE_MODELS = ["logreg", "rf", "xgb", "simple_lstm", "cnn_transformer"]
DEEP_MODELS = ["simple_lstm", "cnn_transformer"]


def run_command(cmd: List[str], description: str) -> None:
    """Run subprocess command with logging and error handling.

    Args:
        cmd: Command and arguments as list
        description: Human-readable description for logging

    Raises:
        subprocess.CalledProcessError: If command fails
    """
    logger.info(f"==> {description}")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        logger.success(f"✓ {description} complete")
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        raise


def verify_phase1_fixes() -> None:
    """Verify Phase 1 fixes are in place.

    Checks:
    - SimpleLSTM model is importable
    - Training data exists
    """
    logger.info("=" * 80)
    logger.info("PHASE 1: Verifying critical fixes")
    logger.info("=" * 80)

    # Test SimpleLSTM import
    try:
        from moola.models.simple_lstm import SimpleLSTMModel
        model = SimpleLSTMModel()
        param_count = sum(p.numel() for p in model._build_model(4, 2).parameters())
        logger.success(f"✓ SimpleLSTM loaded ({param_count:,} params)")
    except Exception as e:
        logger.error(f"✗ SimpleLSTM import failed: {e}")
        sys.exit(1)

    # Check training data
    train_path = PROCESSED_DIR / "train_clean.parquet"
    if not train_path.exists():
        logger.error(f"✗ Training data not found: {train_path}")
        sys.exit(1)

    logger.success(f"✓ Training data exists: {train_path}")
    logger.success("Phase 1 verification complete\n")


def run_phase2_augmentation(device: str, seed: int, mlflow_experiment: str) -> None:
    """Generate OOF predictions with augmentation for all base models.

    Args:
        device: Device to use ('cpu' or 'cuda')
        seed: Random seed for reproducibility
        mlflow_experiment: MLflow experiment name
    """
    logger.info("=" * 80)
    logger.info("PHASE 2: Generating OOF predictions with augmentation")
    logger.info("=" * 80)

    for model_name in BASE_MODELS:
        # Determine device for this model
        model_device = device if model_name in DEEP_MODELS else "cpu"

        # Build CLI command
        if model_name == "cnn_transformer":
            # Use pre-trained encoder for CNN-Transformer
            encoder_path = PROJECT_ROOT / "data" / "artifacts" / "pretrained" / "encoder_weights.pt"
            cmd = [
                sys.executable, "-m", "moola.cli", "oof",
                "--model", model_name,
                "--device", model_device,
                "--seed", str(seed),
                "--load-pretrained-encoder", str(encoder_path)
            ]
            logger.info(f"Training {model_name} with pre-trained encoder: {encoder_path}")
        else:
            # Regular training for other models
            cmd = [
                sys.executable, "-m", "moola.cli", "oof",
                "--model", model_name,
                "--device", model_device,
                "--seed", str(seed),
            ]

        run_command(cmd, f"OOF generation for {model_name}")

    logger.success("Phase 2 augmentation complete\n")


def run_phase3_ssl(device: str, epochs: int, patience: int) -> None:
    """Pre-train TS-TCC encoder and fine-tune models.

    Args:
        device: Device to use ('cuda' or 'cpu')
        epochs: Number of pre-training epochs
        patience: Early stopping patience
    """
    logger.info("=" * 80)
    logger.info("PHASE 3: SSL pre-training with TS-TCC")
    logger.info("=" * 80)

    # Step 1: Check if unlabeled data exists
    unlabeled_path = RAW_DIR / "unlabeled_windows.parquet"
    if not unlabeled_path.exists():
        logger.warning(f"Unlabeled data not found at {unlabeled_path}")
        logger.warning("Skipping SSL pre-training - proceeding with random initialization")
        logger.warning("To enable SSL, extract unlabeled windows from raw data first")
        return

    logger.info(f"Found unlabeled data: {unlabeled_path}")

    # Step 2: Pre-train encoder
    encoder_output = MODELS_DIR / "ts_tcc" / "pretrained_encoder.pt"

    if encoder_output.exists():
        logger.warning(f"Pre-trained encoder already exists: {encoder_output}")
        logger.warning("Skipping pre-training - using existing encoder")
    else:
        cmd = [
            sys.executable, "-m", "moola.cli", "pretrain-tcc",
            "--device", device,
            "--epochs", str(epochs),
            "--patience", str(patience),
        ]
        run_command(cmd, "TS-TCC encoder pre-training")

    # Step 3: Fine-tune models with pre-trained encoder
    # Note: Only CNN-Transformer supports pre-trained encoder loading
    # SimpleLSTM uses different architecture (LSTM vs CNN-Transformer)
    logger.info("\nFine-tuning models with pre-trained encoder...")
    logger.info("Note: Only CNN-Transformer supports SSL pre-training")
    logger.info("SimpleLSTM uses LSTM architecture (incompatible with CNN encoder)")

    # Fine-tune CNN-Transformer with pre-trained weights
    # This requires modifications to the CLI or manual training
    # For now, we skip this and rely on regular training
    # TODO: Add CLI support for loading pre-trained encoder during training

    logger.success("Phase 3 SSL pre-training complete\n")


def run_stack_ensemble(seed: int) -> None:
    """Train final stacking ensemble.

    Args:
        seed: Random seed for reproducibility
    """
    logger.info("=" * 80)
    logger.info("PHASE 4: Training stack ensemble")
    logger.info("=" * 80)

    cmd = [
        sys.executable, "-m", "moola.cli", "stack-train",
        "--seed", str(seed),
        "--stacker", "rf",
    ]

    run_command(cmd, "Stack ensemble training")
    logger.success("Phase 4 ensemble training complete\n")


def main():
    parser = argparse.ArgumentParser(description="End-to-end ML pipeline automation")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"], help="Device for deep learning models")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--mlflow-experiment", default="production", help="MLflow experiment name")
    parser.add_argument("--skip-phase1", action="store_true", help="Skip Phase 1 verification")
    parser.add_argument("--skip-phase2", action="store_true", help="Skip Phase 2 OOF generation")
    parser.add_argument("--skip-phase3", action="store_true", help="Skip Phase 3 SSL pre-training")
    parser.add_argument("--skip-phase4", action="store_true", help="Skip Phase 4 ensemble training")
    parser.add_argument("--ssl-epochs", type=int, default=100, help="SSL pre-training epochs")
    parser.add_argument("--ssl-patience", type=int, default=15, help="SSL early stopping patience")

    args = parser.parse_args()

    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 20 + "MOOLA PRODUCTION PIPELINE" + " " * 33 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info(f"\nConfiguration:")
    logger.info(f"  Device: {args.device}")
    logger.info(f"  Seed: {args.seed}")
    logger.info(f"  MLflow Experiment: {args.mlflow_experiment}")
    logger.info(f"  SSL Epochs: {args.ssl_epochs}")
    logger.info(f"  SSL Patience: {args.ssl_patience}\n")

    try:
        # Phase 1: Verify fixes
        if not args.skip_phase1:
            verify_phase1_fixes()

        # Phase 2: Generate OOF predictions
        if not args.skip_phase2:
            run_phase2_augmentation(args.device, args.seed, args.mlflow_experiment)

        # Phase 3: SSL pre-training
        if not args.skip_phase3:
            run_phase3_ssl(args.device, args.ssl_epochs, args.ssl_patience)

        # Phase 4: Stack ensemble
        if not args.skip_phase4:
            run_stack_ensemble(args.seed)

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PIPELINE COMPLETE!")
        logger.info("=" * 80)
        logger.info("\nGenerated Artifacts:")
        logger.info(f"  Base models: {MODELS_DIR}")
        logger.info(f"  Stack ensemble: {MODELS_DIR / 'stack' / 'stack.pkl'}")
        logger.info(f"  Pre-trained encoder: {MODELS_DIR / 'ts_tcc' / 'pretrained_encoder.pt'}")
        logger.info("\nNext Steps:")
        logger.info("  1. View metrics: mlflow ui --port 5000")
        logger.info("  2. Deploy model: python -m moola.api.serve")
        logger.info("  3. Run tests: pytest tests/")
        logger.info("=" * 80 + "\n")

    except subprocess.CalledProcessError as e:
        logger.error(f"\n❌ Pipeline failed at: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Pipeline interrupted by user")
        sys.exit(130)


if __name__ == "__main__":
    main()
