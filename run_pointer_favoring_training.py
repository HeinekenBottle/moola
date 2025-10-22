#!/usr/bin/env python3
"""
Pointer-Favoring Patch Training Script for RunPod
Implements Kendall bias with hit_at_3 optimization for Jade/Sapphire/Opal models
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run command and handle errors."""
    print(f"\n🚀 {description}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ Success: {description}")
        if result.stdout:
            print("Output:", result.stdout[:500])
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {description}")
        print("Error:", e.stderr[:500])
        return False

def main():
    """Execute pointer-favoring training pipeline."""
    print("🎯 Pointer-Favoring Patch Training Pipeline")
    print("=" * 60)
    
    # Change to workspace directory
    os.chdir("/workspace/moola")
    
    # 1. Verify environment
    if not run_command("python3 scripts/runpod/verify_runpod_env.py", "Environment Verification"):
        return False
    
    # 2. Train pretrained encoder with specified config
    encoder_cmd = """python3 -m moola.cli pretrain-bilstm \\
        --input data/raw/unlabeled_windows.parquet \\
        --output artifacts/encoders/pretrained/stones_encoder_mae.pt \\
        --device cuda \\
        --epochs 100 \\
        --batch-size 64 \\
        --mask-ratio 0.4 \\
        --input-dim 11 \\
        --mask-strategy patch \\
        --seed 1337"""
    
    if not run_command(encoder_cmd, "Pretrained Encoder Training"):
        return False
    
    # 3. Train Jade model with pointer-favoring patch
    jade_cmd = """python3 -m moola.cli train \\
        --cfg-dir configs \\
        --model jade \\
        --data data/processed/train.parquet \\
        --split artifacts/splits/v1/fold_0_temporal.json \\
        --device cuda \\
        --epochs 60 \\
        --batch-size 29 \\
        --predict-pointers \\
        --save-run"""
    
    if not run_command(jade_cmd, "Jade Model Training"):
        return False
    
    # 4. Train Sapphire model with pretrained encoder
    sapphire_cmd = """python3 -m moola.cli train \\
        --cfg-dir configs \\
        --model sapphire \\
        --data data/processed/train.parquet \\
        --split artifacts/splits/v1/fold_0_temporal.json \\
        --device cuda \\
        --epochs 40 \\
        --batch-size 29 \\
        --predict-pointers \\
        --pretrained-encoder artifacts/encoders/pretrained/stones_encoder_mae.pt \\
        --freeze-encoder \\
        --save-run"""
    
    if not run_command(sapphire_cmd, "Sapphire Model Training"):
        return False
    
    # 5. Train Opal model with adaptive fine-tuning
    opal_cmd = """python3 -m moola.cli train \\
        --cfg-dir configs \\
        --model opal \\
        --data data/processed/train.parquet \\
        --split artifacts/splits/v1/fold_0_temporal.json \\
        --device cuda \\
        --epochs 40 \\
        --batch-size 29 \\
        --predict-pointers \\
        --pretrained-encoder artifacts/encoders/pretrained/stones_encoder_mae.pt \\
        --no-freeze-encoder \\
        --save-run"""
    
    if not run_command(opal_cmd, "Opal Model Training"):
        return False
    
    print("\n🎉 All training completed successfully!")
    print("Check artifacts/runs/ for model checkpoints and logs")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)