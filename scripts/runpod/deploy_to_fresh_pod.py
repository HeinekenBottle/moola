#!/usr/bin/env python3
"""
Moola RunPod Deployment Script
Consolidated SSH/SCP deployment to fresh RunPod instances.
Handles: workspace setup → code deployment → training with monitoring

Usage:
    python deploy_to_fresh_pod.py --host HOST --port PORT --key KEY_PATH
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, 'src')

from moola.runpod.scp_orchestrator import RunPodOrchestrator

def parse_args():
    """Parse command line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(description="Deploy Moola to RunPod via SSH/SCP")
    parser.add_argument("--host", default="213.173.110.220", help="RunPod host IP")
    parser.add_argument("--port", type=int, default=36832, help="RunPod SSH port")
    parser.add_argument("--key", default="/Users/jack/.ssh/id_ed25519", help="SSH key path")
    parser.add_argument("--workspace", default="/workspace/moola", help="Remote workspace path")
    parser.add_argument("--model", default="cnn_transformer", help="Model to train")
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()

    print("=" * 70)
    print("🚀 MOOLA DEPLOYMENT TO RUNPOD")
    print("=" * 70)
    print(f"Pod: {args.host}:{args.port}")
    print(f"Workspace: {args.workspace}")
    print(f"Model: {args.model}")
    print()

    # Initialize orchestrator
    orch = RunPodOrchestrator(host=args.host, port=args.port, key_path=args.key, workspace=args.workspace)

    # Phase 1: Setup workspace
    print("\n[PHASE 1] Setting up workspace...")
    orch.execute_command("mkdir -p /workspace/moola /workspace/data/processed /workspace/artifacts/pretrained")
    orch.execute_command("cd /workspace && git clone https://github.com/HeinekenBottle/moola.git || cd /workspace/moola && git pull origin main")
    print("✓ Workspace ready")

    # Phase 2: Deploy core code (using components that actually exist)
    print("\n[PHASE 2] Deploying core code...")

    core_files = [
        "src/moola/models/simple_lstm.py",
        "src/moola/config/training_config.py", 
        "src/molla/cli.py",
        "src/moola/runpod/scp_orchestrator.py",
    ]

    for file_path in core_files:
        local_file = Path(file_path)
        remote_file = f"{args.workspace}/{file_path}"

        if local_file.exists():
            orch.upload_file(local_file, remote_file)
            print(f"  ✓ {file_path}")
        else:
            print(f"  ⚠ {file_path} not found locally")

    print("✓ Code deployed")

    # Phase 3: Install dependencies 
    print("\n[PHASE 3] Setting up environment...")
    orch.execute_command("""
    python3 -m venv /tmp/moola-venv --system-site-packages
    source /tmp/moola-venv/bin/activate
    pip install --no-cache-dir -q xgboost imbalanced-learn pytorch-lightning pyarrow pandera click typer hydra-core pydantic pydantic-settings python-dotenv loguru rich joblib
    cd /workspace/moola && pip install --no-cache-dir -q -e . --no-deps
    """)
    print("✓ Environment ready")

    # Phase 4: Upload data files
    print("\n[PHASE 4] Uploading data...")

    data_files = [
        ("data/processed/train.parquet", "/workspace/data/processed/"),
        ("data/artifacts/pretrained/encoder_weights.pt", "/workspace/artifacts/pretrained/"),
    ]

    for data_file, remote_dir in data_files:
        local_file = Path(data_file)
        if local_file.exists():
            orch.upload_file(local_file, f"{remote_dir}{local_file.name}")
            print(f"  ✓ {data_file} ({local_file.stat().st_size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  ⚠ {data_file} not found locally")

    print("✓ Data uploaded")

    # Phase 5: Run training
    print("\n[PHASE 5] Starting training...")
    print("-" * 70)

    train_cmd = f"""
    cd {args.workspace}
    source /tmp/moola-venv/bin/activate
    export PYTHONPATH="{args.workspace}/src:$PYTHONPATH"
    export MOOLA_DATA_DIR="/workspace/data"
    export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

    echo "🚀 Training {args.model} (seed: {args.seed})"
    echo "=========================================="
    python3 -m moola.cli train \\
        --model {args.model} \\
        --device {args.device} \\
        --seed {args.seed}
    """

    exit_code = orch.execute_command(train_cmd, timeout=600)

    if exit_code == 0:
        print("-" * 70)
        print("✅ Training completed successfully!")
    else:
        print("-" * 70)
        print("❌ Training failed - check logs above")
        return 1

    # Phase 6: Download results
    print("\n[PHASE 6] Downloading results...")
    
    results_dir = Path("/tmp/moola_results")
    results_dir.mkdir(exist_ok=True)

    # Try to download typical result files
    result_files = [
        f"/workspace/artifacts/oof/{args.model}/v1/seed_{args.seed}.npy",
        "/workspace/logs/training.log",
    ]

    for result_file in result_files:
        local_name = Path(result_file).name
        try:
            orch.download_file(result_file, results_dir / local_name)
            print(f"  ✓ Downloaded {local_name}")
        except:
            print(f"  ⚠ Could not download {local_name}")

    print(f"✓ Results saved to {results_dir}")

    # Phase 7: Summary
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults location: {results_dir}")
    print(f"Model: {args.model}")
    print(f"Seed: {args.seed}")
    print("\nManual access:")
    print(f"  SSH: ssh root@{args.host} -p {args.port} -i {args.key}")
    print(f"  SCP artifacts: scp -P {args.port} -i {args.key} -r root@{args.host}:/workspace/artifacts ~/local_path/")

    return 0

if __name__ == "__main__":
    sys.exit(main())
