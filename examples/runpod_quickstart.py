"""
RunPod Orchestrator Quick Start Example

This script demonstrates the complete workflow for using the RunPod orchestrator
to train models with iterative debugging.

Usage:
    python examples/runpod_quickstart.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from moola.runpod import RunPodOrchestrator
from moola.validation import monitor_training_with_error_detection


def example_1_basic_connection():
    """Example 1: Connect and verify environment."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Basic Connection and Verification")
    print("=" * 60)

    # Initialize orchestrator
    orch = RunPodOrchestrator(
        host="213.173.98.6",
        port=14385,
        key_path="~/.ssh/id_ed25519",
        workspace="/workspace/moola",
    )

    # Verify environment
    env_status = orch.verify_environment()

    if all(env_status.values()):
        print("\n✓ Environment ready for training")
    else:
        print("\n✗ Fix environment issues first")
        return False

    return True


def example_2_deploy_fixes():
    """Example 2: Deploy specific fixes to RunPod."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Deploy Encoder Fixes")
    print("=" * 60)

    orch = RunPodOrchestrator(
        host="213.173.98.6",
        port=14385,
        key_path="~/.ssh/id_ed25519",
    )

    # Deploy encoder freezing fixes
    fix_files = [
        PROJECT_ROOT / "src/moola/models/cnn_transformer.py",
        PROJECT_ROOT / "src/moola/config/training_config.py",
    ]

    success = orch.deploy_fixes(fix_files)

    if success:
        print("\n✓ Fixes deployed successfully")

        # Verify deployment
        print("\n[VERIFY] Checking deployed files...")
        orch.execute_command(
            "cd /workspace/moola && "
            "grep -n 'def freeze_encoder' src/moola/models/cnn_transformer.py | head -1",
            stream_output=True,
        )
    else:
        print("\n✗ Deployment failed")

    return success


def example_3_train_with_monitoring():
    """Example 3: Train model with real-time monitoring."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Train CNN-Transformer with Monitoring")
    print("=" * 60)

    orch = RunPodOrchestrator(
        host="213.173.98.6",
        port=14385,
        key_path="~/.ssh/id_ed25519",
    )

    # Train with automatic error detection
    print("\n[TRAINING] Starting CNN-Transformer with monitoring...")

    exit_code, errors, metrics = monitor_training_with_error_detection(
        orch,
        model="cnn_transformer",
        device="cuda",
        encoder_path="/workspace/artifacts/pretrained/encoder_weights.pt",
    )

    # Check results
    if exit_code == 0:
        print("\n✓ Training completed successfully")

        if errors:
            print(f"\n⚠ {len(errors)} warnings detected:")
            for error in errors:
                print(f"  - {error.error_type}: {error.suggestion}")
    else:
        print(f"\n✗ Training failed with exit code {exit_code}")

        if errors:
            print(f"\n❌ {len(errors)} errors detected:")
            for error in errors:
                print(f"  - {error.error_type}: {error.message[:100]}...")
                print(f"    Fix: {error.suggestion}")

    return exit_code == 0


def example_4_download_and_validate():
    """Example 4: Download results and validate."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Download and Validate Results")
    print("=" * 60)

    import numpy as np

    orch = RunPodOrchestrator(
        host="213.173.98.6",
        port=14385,
        key_path="~/.ssh/id_ed25519",
    )

    # Download OOF predictions
    output_dir = Path("/tmp/cnn_results/")
    success = orch.download_results("cnn_transformer", output_dir)

    if not success:
        print("✗ Failed to download results")
        return False

    # Validate results
    pred_files = list(output_dir.glob("seed_*.npy"))

    if not pred_files:
        print("✗ No prediction files found")
        return False

    predictions = np.load(pred_files[0])

    # Check for class collapse
    unique_classes = len(np.unique(predictions))
    print(f"\n[VALIDATE] Predictions shape: {predictions.shape}")
    print(f"[VALIDATE] Unique classes: {unique_classes}")

    if unique_classes < 2:
        print("⚠️ CLASS COLLAPSE DETECTED")
        print("   Model is predicting only one class")
        print("   Investigate:")
        print("   1. Encoder freezing not working")
        print("   2. Loss function misconfigured")
        print("   3. Learning rate too high")
        return False
    else:
        print(f"✓ No class collapse - {unique_classes} classes predicted")

        # Per-class distribution
        for cls in np.unique(predictions):
            count = (predictions == cls).sum()
            pct = count / len(predictions) * 100
            print(f"   Class {cls}: {count} samples ({pct:.1f}%)")

        return True


def example_5_incremental_debugging():
    """Example 5: Incremental debugging workflow."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Incremental Debugging Workflow")
    print("=" * 60)

    orch = RunPodOrchestrator(
        host="213.173.98.6",
        port=14385,
        key_path="~/.ssh/id_ed25519",
    )

    # Simulate debugging workflow
    print("\n[STEP 1] Run training")
    exit_code = orch.run_training("cnn_transformer", device="cuda", timeout=300)

    if exit_code != 0:
        print("\n[STEP 2] Training failed - download logs")
        orch.download_logs("/tmp/debug_logs/")

        print("\n[STEP 3] Analyze logs (simulated)")
        print("   Found issue: CUDA out of memory")
        print("   Fix: Reduce batch size from 512 to 256")

        print("\n[STEP 4] Deploy fix")
        # In real scenario, AI would edit config and redeploy
        print("   Editing training_config.py locally...")
        print("   Deploying updated config...")

        orch.upload_file(
            PROJECT_ROOT / "src/moola/config/training_config.py",
            "/workspace/moola/src/moola/config/training_config.py",
        )

        print("\n[STEP 5] Retry training with fix")
        exit_code = orch.run_training("cnn_transformer", device="cuda", timeout=300)

        if exit_code == 0:
            print("\n✓ Fix worked! Training succeeded")
        else:
            print("\n✗ Still failing - need more investigation")

    return exit_code == 0


def main():
    """Run all examples."""
    print("=" * 60)
    print("RUNPOD ORCHESTRATOR QUICK START")
    print("=" * 60)
    print("\nThis script demonstrates common RunPod orchestrator workflows.")
    print("Press Ctrl+C at any time to abort.\n")

    examples = [
        ("Basic Connection", example_1_basic_connection),
        ("Deploy Fixes", example_2_deploy_fixes),
        ("Train with Monitoring", example_3_train_with_monitoring),
        ("Download and Validate", example_4_download_and_validate),
        ("Incremental Debugging", example_5_incremental_debugging),
    ]

    # Let user choose which examples to run
    print("Available examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nOptions:")
    print("  - Press Enter to run all examples")
    print("  - Enter example numbers (e.g., 1,3,4) to run specific examples")
    print("  - Enter 'q' to quit")

    choice = input("\nYour choice: ").strip()

    if choice.lower() == 'q':
        print("Exiting...")
        return 0

    # Determine which examples to run
    if choice == "":
        examples_to_run = examples
    else:
        try:
            indices = [int(x.strip()) - 1 for x in choice.split(",")]
            examples_to_run = [examples[i] for i in indices if 0 <= i < len(examples)]
        except (ValueError, IndexError):
            print("Invalid choice. Running all examples.")
            examples_to_run = examples

    # Run selected examples
    results = {}

    for name, func in examples_to_run:
        try:
            results[name] = func()
        except KeyboardInterrupt:
            print("\n\n[ABORT] User interrupted")
            return 1
        except Exception as e:
            print(f"\n✗ {name} raised exception: {e}")
            results[name] = False

        input("\nPress Enter to continue to next example...")

    # Summary
    print("\n" + "=" * 60)
    print("QUICK START SUMMARY")
    print("=" * 60)

    for name, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {name}")

    print("\nNext steps:")
    print("  1. Read docs/runpod_orchestrator_runbook.md for full documentation")
    print("  2. Run test_orchestrator.py to verify your setup")
    print("  3. Use deploy_fixes.py for incremental deployments")
    print("  4. Use runpod_retrain_pipeline.py for full retraining")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nAborted by user")
        sys.exit(1)
