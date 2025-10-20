#!/usr/bin/env python3
"""RunPod BiLSTM Pre-training Script.

Standalone script for executing masked BiLSTM pre-training on RunPod GPU.
Designed for RTX 4090 (24GB VRAM) with optimized settings.

Usage:
    python runpod_pretrain_bilstm.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add moola package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
from moola.utils.seeds import print_gpu_info, set_seed


def main():
    """Execute BiLSTM pre-training on RunPod."""
    print("=" * 80)
    print("RUNPOD BiLSTM PRE-TRAINING")
    print("=" * 80)

    # Configuration
    config = {
        "input_path": Path("/workspace/data/raw/unlabeled_windows.parquet"),
        "output_path": Path("/workspace/artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt"),
        "device": "cuda",
        "epochs": 75,
        "patience": 15,
        "mask_ratio": 0.15,
        "mask_strategy": "patch",
        "patch_size": 7,
        "hidden_dim": 128,
        "num_layers": 1,  # Changed from 2 to 1 as per requirements
        "batch_size": 64,  # RTX 4090 optimized
        "learning_rate": 0.001,
        "seed": 1337,
    }

    # Print configuration
    print("\n[CONFIGURATION]")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    # GPU verification
    if config["device"] == "cuda":
        if not torch.cuda.is_available():
            print("ERROR: CUDA not available!")
            sys.exit(1)
        print_gpu_info()
        print()

    # Set seed for reproducibility
    set_seed(config["seed"])

    # Load unlabeled data
    print("[DATA LOADING]")
    print(f"  Reading: {config['input_path']}")

    if not config["input_path"].exists():
        print(f"ERROR: Input file not found: {config['input_path']}")
        sys.exit(1)

    df = pd.read_parquet(config["input_path"])
    print(f"  Loaded dataframe: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    # Extract and reshape features
    # CRITICAL FIX: Properly unpack nested arrays
    if "features" in df.columns:
        print("  Converting 'features' column to numpy array...")
        # Each row contains an array of 105 sub-arrays, each with 4 OHLC values
        # We need to stack them to get shape (N, 105, 4)
        X_unlabeled = np.stack([np.stack(x) for x in df["features"].values])
        print(f"  ✓ Converted to array shape: {X_unlabeled.shape}")
    else:
        print("ERROR: 'features' column not found in dataframe")
        sys.exit(1)

    # Validate shape
    if X_unlabeled.ndim != 3 or X_unlabeled.shape[1:] != (105, 4):
        print(f"ERROR: Invalid shape: expected [N, 105, 4], got {X_unlabeled.shape}")
        sys.exit(1)

    print(f"  ✓ Data validation passed")
    print(f"  Final shape: {X_unlabeled.shape}")
    print(f"  Memory size: {X_unlabeled.nbytes / 1024 / 1024:.2f} MB")
    print()

    # Initialize pre-trainer
    print("[MODEL INITIALIZATION]")
    pretrainer = MaskedLSTMPretrainer(
        input_dim=4,  # OHLC
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=0.2,
        mask_ratio=config["mask_ratio"],
        mask_strategy=config["mask_strategy"],
        patch_size=config["patch_size"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
        device=config["device"],
        seed=config["seed"],
    )

    # Calculate expected batches
    val_split = 0.1
    train_samples = int(len(X_unlabeled) * (1 - val_split))
    expected_batches = train_samples // config["batch_size"]
    print(f"  Model initialized")
    print(f"  Expected batches per epoch: ~{expected_batches}")
    print()

    # Create output directory
    config["output_path"].parent.mkdir(parents=True, exist_ok=True)

    # Pre-train encoder
    print("[PRE-TRAINING START]")
    print(f"  Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    start_time = time.time()

    history = pretrainer.pretrain(
        X_unlabeled=X_unlabeled,
        n_epochs=config["epochs"],
        val_split=0.1,
        patience=config["patience"],
        save_path=config["output_path"],
        verbose=True
    )

    end_time = time.time()
    duration_minutes = (end_time - start_time) / 60

    # Print final results
    print()
    print("=" * 80)
    print("PRE-TRAINING COMPLETE")
    print("=" * 80)
    print(f"  Duration: {duration_minutes:.1f} minutes")
    print(f"  Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"  Best val loss: {min(history['val_loss']):.6f}")
    print(f"  Total epochs: {len(history['train_loss'])}")
    print()
    print(f"  Encoder saved: {config['output_path']}")

    # Verify saved file
    if config["output_path"].exists():
        file_size_mb = config["output_path"].stat().st_size / 1024 / 1024
        print(f"  File size: {file_size_mb:.2f} MB")

        # Verify checkpoint structure
        checkpoint = torch.load(config["output_path"], map_location="cpu")
        print(f"  Checkpoint keys: {list(checkpoint.keys())}")
        print(f"  Encoder params: {len(checkpoint['encoder_state_dict'])} tensors")
    else:
        print("  ERROR: Encoder file not found after training!")
        sys.exit(1)

    print("=" * 80)
    print()
    print("SUCCESS! Pre-training completed successfully.")
    print()
    print("Next steps:")
    print("  1. Download encoder: scp -P 21856 -i ~/.ssh/id_ed25519 \\")
    print(f"       root@149.36.1.109:{config['output_path']} \\")
    print("       artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt")
    print("  2. Use in SimpleLSTM: model.load_pretrained_encoder(encoder_path)")
    print("  3. Fine-tune on labeled data for classification")
    print()


if __name__ == "__main__":
    main()
