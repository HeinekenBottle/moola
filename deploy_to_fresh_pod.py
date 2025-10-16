#!/usr/bin/env python3
"""
Complete deployment to fresh RunPod instance.
Handles: workspace setup → code deployment → training with monitoring
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, 'src')

from moola.runpod.scp_orchestrator import RunPodOrchestrator

def main():
    # New pod credentials
    HOST = "213.173.110.220"
    PORT = 36832
    KEY = "/Users/jack/.ssh/id_ed25519"
    WORKSPACE = "/workspace/moola"

    print("=" * 70)
    print("🚀 COMPLETE DEPLOYMENT TO FRESH RUNPOD")
    print("=" * 70)
    print(f"Pod: {HOST}:{PORT}")
    print(f"Workspace: {WORKSPACE}")
    print()

    # Initialize orchestrator
    orch = RunPodOrchestrator(host=HOST, port=PORT, key_path=KEY, workspace=WORKSPACE)

    # Phase 1: Setup workspace
    print("\n[PHASE 1] Setting up workspace...")
    orch.execute_command("mkdir -p /workspace/moola /workspace/data/processed /workspace/artifacts/pretrained")
    orch.execute_command("cd /workspace && git clone https://github.com/HeinekenBottle/moola.git || cd /workspace/moola && git pull origin main")
    print("✓ Workspace ready")

    # Phase 2: Deploy fixed code
    print("\n[PHASE 2] Deploying fixed code...")

    fixes_to_deploy = [
        "src/moola/models/cnn_transformer.py",
        "src/moola/config/training_config.py",
        "src/moola/validation/training_validator.py",
        "src/moola/runpod/scp_orchestrator.py",
    ]

    for file_path in fixes_to_deploy:
        local_file = Path(file_path)
        remote_file = f"{WORKSPACE}/{file_path}"

        if local_file.exists():
            orch.upload_file(local_file, remote_file)
            print(f"  ✓ {file_path}")
        else:
            print(f"  ⚠ {file_path} not found locally")

    print("✓ Code deployed")

    # Phase 3: Upload data and encoder
    print("\n[PHASE 3] Uploading data and pre-trained encoder...")

    data_file = Path("data/processed/train_pivot_134.parquet")
    encoder_file = Path("data/artifacts/pretrained/encoder_weights.pt")

    if data_file.exists():
        orch.upload_file(data_file, f"/workspace/data/processed/train_pivot_134.parquet")
        print(f"  ✓ Training data (size: {data_file.stat().st_size / 1024 / 1024:.1f} MB)")

    if encoder_file.exists():
        orch.upload_file(encoder_file, f"/workspace/artifacts/pretrained/encoder_weights.pt")
        print(f"  ✓ Pre-trained encoder (size: {encoder_file.stat().st_size / 1024 / 1024:.1f} MB)")

    print("✓ Data uploaded")

    # Phase 4: Create symlink and verify
    print("\n[PHASE 4] Verifying setup...")
    orch.execute_command("cd /workspace/data/processed && ln -sf train_pivot_134.parquet train.parquet || true")
    orch.execute_command("""
    cd /workspace/moola
    python3 -c "
import torch
import pandas as pd
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ CUDA: {torch.cuda.is_available()}')
df = pd.read_parquet('/workspace/data/processed/train.parquet')
print(f'✓ Data: {df.shape[0]} samples')
import os
if os.path.exists('/workspace/artifacts/pretrained/encoder_weights.pt'):
    print(f'✓ Encoder: ready')
    "
    """)

    # Phase 5: Create venv and install dependencies
    print("\n[PHASE 5] Setting up environment...")
    orch.execute_command("""
    python3 -m venv /tmp/moola-venv --system-site-packages
    source /tmp/moola-venv/bin/activate
    pip install --no-cache-dir -q xgboost imbalanced-learn pytorch-lightning pyarrow pandera click typer hydra-core pydantic pydantic-settings python-dotenv loguru rich mlflow joblib
    cd /workspace/moola && pip install --no-cache-dir -q -e . --no-deps
    """)
    print("✓ Environment ready")

    # Phase 6: Run training
    print("\n[PHASE 6] Starting training...")
    print("-" * 70)

    train_cmd = """
    cd /workspace/moola
    source /tmp/moola-venv/bin/activate
    export PYTHONPATH="/workspace/moola/src:$PYTHONPATH"
    export MOOLA_DATA_DIR="/workspace/data"
    export MOOLA_ARTIFACTS_DIR="/workspace/artifacts"

    echo "🚀 CNN-Transformer with Pre-trained Encoder (FIXES APPLIED)"
    echo "==========================================================="
    python3 -m moola.cli oof \\
        --model cnn_transformer \\
        --device cuda \\
        --seed 1337 \\
        --load-pretrained-encoder /workspace/artifacts/pretrained/encoder_weights.pt
    """

    exit_code = orch.execute_command(train_cmd, timeout=600)

    if exit_code == 0:
        print("-" * 70)
        print("✅ Training completed successfully!")
    else:
        print("-" * 70)
        print("❌ Training failed - check logs above")
        return 1

    # Phase 7: Download results
    print("\n[PHASE 7] Downloading results...")

    results_dir = Path("/tmp/cnn_pretrained_results_new")
    results_dir.mkdir(exist_ok=True)

    orch.download_file(
        "/workspace/artifacts/oof/cnn_transformer/v1/seed_1337.npy",
        results_dir / "seed_1337.npy"
    )

    print(f"✓ Results saved to {results_dir}")

    # Phase 8: Summary
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT COMPLETE")
    print("=" * 70)
    print(f"\nResults location: {results_dir}")
    print(f"OOF predictions: {results_dir / 'seed_1337.npy'}")
    print("\nTo download all artifacts:")
    print(f"  scp -P {PORT} -i {KEY} -r root@{HOST}:/workspace/artifacts ~/local_path/")
    print("\nTo retrain with changes:")
    print(f"  ssh root@{HOST} -p {PORT} -i {KEY}")
    print("  cd /workspace/moola && bash scripts/fast-train.sh")

    return 0

if __name__ == "__main__":
    sys.exit(main())
