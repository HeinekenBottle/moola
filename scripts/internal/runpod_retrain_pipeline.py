"""
Complete retraining pipeline for RunPod with all fixes applied.

This script orchestrates end-to-end model retraining:
1. Classical models (CPU): logreg, rf, xgb
2. SimpleLSTM (GPU): Baseline deep learning
3. CNN-Transformer (GPU): With pre-trained encoder + fixes
4. Stack ensemble: Meta-learner on OOF predictions

Features:
- Automatic error detection and recovery
- Class collapse monitoring
- Result validation
- Artifact collection
"""

import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from moola.runpod import RunPodOrchestrator


def validate_model_results(
    model: str,
    orch: RunPodOrchestrator,
    output_dir: Path,
) -> Dict[str, float]:
    """
    Validate OOF predictions for a trained model.

    Checks:
    - OOF predictions exist
    - No class collapse
    - Reasonable accuracy per class

    Args:
        model: Model name
        orch: RunPod orchestrator
        output_dir: Local directory with downloaded results

    Returns:
        Dictionary with validation metrics
    """
    print(f"\n[VALIDATE] Checking {model} results...")

    # Check if predictions exist
    pred_files = list(output_dir.glob("seed_*.npy"))

    if not pred_files:
        print(f"  ✗ No prediction files found")
        return {"status": "missing", "accuracy": 0.0}

    # Load predictions
    pred_file = pred_files[0]
    predictions = np.load(pred_file)

    # Basic validation
    metrics = {
        "status": "ok",
        "num_predictions": len(predictions),
        "unique_classes": len(np.unique(predictions)),
    }

    # Check for class collapse
    if metrics["unique_classes"] < 2:
        print(f"  ✗ CLASS COLLAPSE DETECTED - only {metrics['unique_classes']} unique classes")
        metrics["status"] = "collapsed"
    else:
        print(f"  ✓ No class collapse - {metrics['unique_classes']} classes predicted")

    # Load labels for accuracy if available
    labels_path = output_dir.parent.parent.parent / "data" / "processed" / "train.parquet"
    if labels_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(labels_path)
            labels = df['label'].values

            if len(labels) == len(predictions):
                accuracy = (predictions == labels).mean()
                metrics["accuracy"] = accuracy
                print(f"  ✓ Overall accuracy: {accuracy:.1%}")

                # Per-class accuracy
                for cls in np.unique(labels):
                    mask = labels == cls
                    cls_acc = (predictions[mask] == labels[mask]).mean()
                    metrics[f"class_{cls}_accuracy"] = cls_acc
                    print(f"    Class {cls}: {cls_acc:.1%}")
            else:
                print(f"  ⚠ Label/prediction mismatch: {len(labels)} labels, {len(predictions)} predictions")

        except Exception as e:
            print(f"  ⚠ Could not load labels: {e}")

    return metrics


def check_class_collapse(model: str, output_dir: Path) -> bool:
    """
    Check if model exhibits class collapse.

    Args:
        model: Model name
        output_dir: Local directory with results

    Returns:
        True if class collapse detected, False otherwise
    """
    pred_files = list(output_dir.glob("seed_*.npy"))

    if not pred_files:
        print(f"[WARNING] No predictions found for {model}")
        return False

    predictions = np.load(pred_files[0])
    unique_classes = len(np.unique(predictions))

    return unique_classes < 2


def debug_class_collapse(model: str, orch: RunPodOrchestrator) -> None:
    """
    Debug class collapse issue by inspecting model internals.

    Downloads:
    - Model checkpoint
    - Training logs
    - Loss curves

    Checks:
    - Encoder frozen status
    - Gradient flow
    - Loss weights
    """
    print(f"\n[DEBUG] Investigating class collapse for {model}...")

    # Download logs
    log_dir = Path(f"/tmp/debug_{model}_logs/")
    orch.download_logs(log_dir)

    # Check for common issues
    print("\n[DEBUG] Common causes:")
    print("  1. Encoder not frozen → Check load_pretrained_encoder() call")
    print("  2. Loss weight mismatch → Check alpha/beta values")
    print("  3. Learning rate too high → Check optimizer settings")
    print("  4. Class imbalance → Check Focal Loss configuration")

    # Check encoder freezing in remote model
    orch.execute_command(
        f"cd {orch.workspace} && "
        f"python -c '"
        f"import torch; "
        f"from moola.models import CnnTransformerModel; "
        f"model = CnnTransformerModel(seed=1337); "
        f"model._build_model(4, 2); "
        f"model.load_pretrained_encoder(\"{orch.workspace}/artifacts/pretrained/encoder_weights.pt\"); "
        f"model.freeze_encoder(); "
        f"frozen = sum(1 for p in model.model.parameters() if not p.requires_grad); "
        f"trainable = sum(1 for p in model.model.parameters() if p.requires_grad); "
        f"print(f\"Frozen: {{frozen}}, Trainable: {{trainable}}\"); "
        f"'",
        timeout=60,
    )


def deploy_encoder_fixes(orch: RunPodOrchestrator) -> bool:
    """
    Deploy fixes for encoder freezing and class collapse.

    Uploads:
    - Fixed CNN-Transformer model (with freeze_encoder method)
    - Updated training config (with SSL parameters)
    - Validation utilities

    Args:
        orch: RunPod orchestrator

    Returns:
        True if all fixes deployed successfully
    """
    print("\n[FIX] Deploying encoder fixes...")

    fix_files = [
        PROJECT_ROOT / "src/moola/models/cnn_transformer.py",  # freeze_encoder method
        PROJECT_ROOT / "src/moola/config/training_config.py",  # SSL hyperparams
        PROJECT_ROOT / "src/moola/pipelines/oof.py",  # Updated OOF pipeline
    ]

    return orch.deploy_fixes(fix_files)


def retrain_all_models(
    orch: RunPodOrchestrator,
    fix_encoder: bool = True,
    models: Optional[list] = None,
) -> Dict[str, Dict]:
    """
    Retrain all models with fixes.

    Order:
    1. Classical models (CPU): logreg, rf, xgb
    2. SimpleLSTM (GPU): Baseline deep learning
    3. CNN-Transformer (GPU): With pre-trained encoder + fixes
    4. Stack ensemble: Meta-learner on OOF predictions

    Args:
        orch: RunPod orchestrator
        fix_encoder: Apply encoder freezing fixes (default: True)
        models: List of models to train (default: all)

    Returns:
        Dictionary of model -> validation metrics
    """
    if models is None:
        models = ["logreg", "rf", "xgb", "simple_lstm", "cnn_transformer"]

    results = {}

    # Phase 1: Classical Models (Fast, CPU)
    classical_models = [m for m in ["logreg", "rf", "xgb"] if m in models]

    for model in classical_models:
        print(f"\n{'='*60}")
        print(f"TRAINING: {model.upper()}")
        print(f"{'='*60}")

        exit_code = orch.run_training(model, device="cpu", timeout=1800)

        if exit_code != 0:
            print(f"[ERROR] {model} training failed")
            results[model] = {"status": "failed", "exit_code": exit_code}
            continue

        # Download and validate results
        output_dir = Path(f"/tmp/results/{model}/")
        orch.download_results(model, output_dir)

        metrics = validate_model_results(model, orch, output_dir)
        results[model] = metrics

        print(f"\n✓ {model}: {metrics.get('accuracy', 0.0):.1%} accuracy")

    # Phase 2: SimpleLSTM (GPU, no pre-training yet)
    if "simple_lstm" in models:
        print(f"\n{'='*60}")
        print(f"TRAINING: SIMPLE_LSTM (Baseline)")
        print(f"{'='*60}")

        exit_code = orch.run_training("simple_lstm", device="cuda", timeout=3600)

        if exit_code == 0:
            output_dir = Path("/tmp/results/simple_lstm/")
            orch.download_results("simple_lstm", output_dir)
            results["simple_lstm"] = validate_model_results("simple_lstm", orch, output_dir)
        else:
            results["simple_lstm"] = {"status": "failed", "exit_code": exit_code}

    # Phase 3: CNN-Transformer with Pre-trained Encoder + Fixes
    if "cnn_transformer" in models:
        print(f"\n{'='*60}")
        print(f"TRAINING: CNN-TRANSFORMER (Pre-trained + FIXES)")
        print(f"{'='*60}")

        # Deploy fixes first
        if fix_encoder:
            print("\n[FIX] Deploying encoder freezing fixes...")
            deploy_encoder_fixes(orch)

        # Train with pre-trained encoder
        encoder_path = f"{orch.workspace}/artifacts/pretrained/encoder_weights.pt"

        exit_code = orch.run_training(
            "cnn_transformer",
            device="cuda",
            encoder_path=encoder_path,
            timeout=7200,  # 2 hours for full training
        )

        if exit_code == 0:
            # Download and validate
            output_dir = Path("/tmp/results/cnn_transformer/")
            orch.download_results("cnn_transformer", output_dir)

            metrics = validate_model_results("cnn_transformer", orch, output_dir)
            results["cnn_transformer"] = metrics

            # Check for class collapse fix
            if check_class_collapse("cnn_transformer", output_dir):
                print("\n⚠️ CLASS COLLAPSE STILL PRESENT - investigating...")
                debug_class_collapse("cnn_transformer", orch)
            else:
                print("\n✓ Class collapse FIXED!")
        else:
            results["cnn_transformer"] = {"status": "failed", "exit_code": exit_code}

    # Phase 4: Stack Ensemble
    if "stack" in models:
        print(f"\n{'='*60}")
        print(f"TRAINING: STACK ENSEMBLE")
        print(f"{'='*60}")

        exit_code = orch.execute_command(
            f"cd {orch.workspace} && "
            f"source /tmp/moola-venv/bin/activate && "
            f"export PYTHONPATH=\"{orch.workspace}/src:$PYTHONPATH\" && "
            f"python -m moola.cli stack-train --stacker rf --seed 1337",
            timeout=1800,
        )

        if exit_code == 0:
            output_dir = Path("/tmp/results/stack/")
            orch.download_results("stack_rf", output_dir)
            results["stack_ensemble"] = validate_model_results("stack_rf", orch, output_dir)
        else:
            results["stack_ensemble"] = {"status": "failed", "exit_code": exit_code}

    # Summary Report
    print(f"\n{'='*60}")
    print(f"RETRAINING COMPLETE - RESULTS SUMMARY")
    print(f"{'='*60}")

    for model, metrics in results.items():
        status = metrics.get("status", "unknown")
        accuracy = metrics.get("accuracy", 0.0)

        if status == "ok":
            print(f"✓ {model:20s}: {accuracy:.1%} accuracy")
        elif status == "collapsed":
            print(f"✗ {model:20s}: CLASS COLLAPSE")
        elif status == "failed":
            print(f"✗ {model:20s}: TRAINING FAILED (exit code {metrics.get('exit_code', -1)})")
        else:
            print(f"? {model:20s}: {status}")

    return results


def main():
    """
    Main entry point for retraining pipeline.

    Usage:
        python runpod_retrain_pipeline.py [--host HOST] [--port PORT] [--models MODEL1,MODEL2]
    """
    import argparse

    parser = argparse.ArgumentParser(description="RunPod retraining pipeline")
    parser.add_argument("--host", default="213.173.98.6", help="RunPod host IP")
    parser.add_argument("--port", type=int, default=14385, help="SSH port")
    parser.add_argument("--key", default="~/.ssh/id_ed25519", help="SSH key path")
    parser.add_argument("--workspace", default="/workspace/moola", help="Remote workspace")
    parser.add_argument(
        "--models",
        default="logreg,rf,xgb,simple_lstm,cnn_transformer",
        help="Comma-separated model list",
    )
    parser.add_argument("--no-fix-encoder", action="store_true", help="Skip encoder fixes")
    parser.add_argument("--verify-only", action="store_true", help="Only verify environment")

    args = parser.parse_args()

    # Initialize orchestrator
    orch = RunPodOrchestrator(
        host=args.host,
        port=args.port,
        key_path=args.key,
        workspace=args.workspace,
        verbose=True,
    )

    # Verify environment
    env_status = orch.verify_environment()

    if not all(env_status.values()):
        print("\n[ERROR] Environment checks failed - fix issues before retraining")
        return 1

    if args.verify_only:
        print("\n[VERIFY] Environment OK - ready for training")
        return 0

    # Parse models
    models_to_train = args.models.split(",")

    # Run retraining pipeline
    results = retrain_all_models(
        orch,
        fix_encoder=not args.no_fix_encoder,
        models=models_to_train,
    )

    # Exit code based on success
    failed_models = [m for m, metrics in results.items() if metrics.get("status") != "ok"]

    if failed_models:
        print(f"\n[ERROR] {len(failed_models)} models failed: {', '.join(failed_models)}")
        return 1
    else:
        print(f"\n[SUCCESS] All models trained successfully")
        return 0


if __name__ == "__main__":
    sys.exit(main())
