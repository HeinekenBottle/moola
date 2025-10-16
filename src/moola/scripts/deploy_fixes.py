"""
Incremental fix deployment to RunPod.

This script allows precise, file-by-file deployment of fixes without
re-uploading the entire codebase. Perfect for iterative debugging.

Usage:
    python deploy_fixes.py --files src/moola/models/cnn_transformer.py
    python deploy_fixes.py --preset encoder_fixes
    python deploy_fixes.py --all-models
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from moola.runpod import RunPodOrchestrator


# Predefined fix presets for common issues
FIX_PRESETS = {
    "encoder_fixes": [
        "src/moola/models/cnn_transformer.py",  # freeze_encoder() method
        "src/moola/config/training_config.py",  # SSL hyperparameters
    ],
    "augmentation_fixes": [
        "src/moola/utils/augmentation.py",
        "src/moola/utils/temporal_augmentation.py",
    ],
    "loss_fixes": [
        "src/moola/utils/losses.py",
        "src/moola/utils/focal_loss.py",
    ],
    "oof_pipeline": [
        "src/moola/pipelines/oof.py",
    ],
    "all_models": [
        "src/moola/models/cnn_transformer.py",
        "src/moola/models/simple_lstm.py",
        "src/moola/models/logreg.py",
        "src/moola/models/rf.py",
        "src/moola/models/xgb.py",
        "src/moola/models/base.py",
    ],
    "all_pipelines": [
        "src/moola/pipelines/oof.py",
        "src/moola/pipelines/stack_train.py",
        "src/moola/pipelines/ssl_pretrain.py",
    ],
}


def deploy_files(
    orch: RunPodOrchestrator,
    files: List[Path],
    verify: bool = True,
) -> bool:
    """
    Deploy specific files to RunPod.

    Args:
        orch: RunPod orchestrator
        files: List of file paths to deploy
        verify: Verify files after deployment

    Returns:
        True if all files deployed successfully
    """
    print(f"\n[DEPLOY] Deploying {len(files)} files...")

    # Deploy files
    success = orch.deploy_fixes(files)

    if not success:
        print("[ERROR] Deployment failed")
        return False

    # Verify deployment
    if verify:
        print("\n[VERIFY] Checking deployed files...")

        for file_path in files:
            # Get remote path
            if "src/moola" in str(file_path):
                relative_path = str(file_path).split("src/moola/")[-1]
                remote_path = f"{orch.workspace}/src/moola/{relative_path}"
            else:
                remote_path = f"{orch.workspace}/{file_path.name}"

            # Check file exists
            exit_code = orch.execute_command(
                f"ls -lh {remote_path}",
                stream_output=False,
                timeout=10,
            )

            if exit_code == 0:
                print(f"  ✓ {file_path.name}")
            else:
                print(f"  ✗ {file_path.name} (not found)")

    print("\n[DEPLOY] Complete")
    return success


def test_imports(orch: RunPodOrchestrator) -> bool:
    """
    Test that deployed modules can be imported.

    Args:
        orch: RunPod orchestrator

    Returns:
        True if all imports succeed
    """
    print("\n[TEST] Verifying imports...")

    import_tests = [
        "from moola.models import CnnTransformerModel",
        "from moola.models import SimpleLstmModel",
        "from moola.config import training_config",
        "from moola.utils.augmentation import mixup_cutmix",
        "from moola.pipelines.oof import run_oof_pipeline",
    ]

    all_passed = True

    for import_stmt in import_tests:
        exit_code = orch.execute_command(
            f"cd {orch.workspace} && "
            f"source /tmp/moola-venv/bin/activate && "
            f"python -c '{import_stmt}' && echo 'OK'",
            stream_output=False,
            timeout=30,
        )

        module = import_stmt.split()[-1]
        if exit_code == 0:
            print(f"  ✓ {module}")
        else:
            print(f"  ✗ {module} (import failed)")
            all_passed = False

    return all_passed


def main():
    """
    Main entry point for deployment script.

    Usage:
        python deploy_fixes.py --files src/moola/models/cnn_transformer.py
        python deploy_fixes.py --preset encoder_fixes
        python deploy_fixes.py --all-models
    """
    import argparse

    parser = argparse.ArgumentParser(description="Deploy fixes to RunPod")
    parser.add_argument("--host", default="213.173.98.6", help="RunPod host IP")
    parser.add_argument("--port", type=int, default=14385, help="SSH port")
    parser.add_argument("--key", default="~/.ssh/id_ed25519", help="SSH key path")
    parser.add_argument("--workspace", default="/workspace/moola", help="Remote workspace")

    # File selection
    parser.add_argument(
        "--files",
        nargs="+",
        help="Specific files to deploy (relative to project root)",
    )
    parser.add_argument(
        "--preset",
        choices=list(FIX_PRESETS.keys()),
        help="Deploy predefined fix preset",
    )

    # Verification
    parser.add_argument("--no-verify", action="store_true", help="Skip verification")
    parser.add_argument("--test-imports", action="store_true", help="Test imports after deployment")

    args = parser.parse_args()

    # Determine files to deploy
    files_to_deploy = []

    if args.files:
        files_to_deploy = [PROJECT_ROOT / f for f in args.files]
    elif args.preset:
        files_to_deploy = [PROJECT_ROOT / f for f in FIX_PRESETS[args.preset]]
    else:
        print("[ERROR] Must specify --files or --preset")
        return 1

    # Validate files exist
    missing_files = [f for f in files_to_deploy if not f.exists()]
    if missing_files:
        print("[ERROR] Files not found:")
        for f in missing_files:
            print(f"  - {f}")
        return 1

    # Initialize orchestrator
    orch = RunPodOrchestrator(
        host=args.host,
        port=args.port,
        key_path=args.key,
        workspace=args.workspace,
        verbose=True,
    )

    # Deploy files
    success = deploy_files(
        orch,
        files_to_deploy,
        verify=not args.no_verify,
    )

    if not success:
        return 1

    # Test imports if requested
    if args.test_imports:
        if not test_imports(orch):
            print("\n[ERROR] Import tests failed")
            return 1

    print("\n[SUCCESS] Deployment complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
