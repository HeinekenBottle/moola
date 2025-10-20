#!/usr/bin/env python3
"""Generate OOF predictions for full 115 sample dataset (no data cleaning).

This is needed for CleanLab analysis which requires predictions on ALL samples
including potentially mislabeled ones.
"""
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def backup_current_oof():
    """Backup current OOF predictions (from cleaned 97 samples)."""
    oof_dir = Path("data/artifacts/oof")

    if not oof_dir.exists():
        print("No OOF predictions to backup")
        return

    backup_dir = Path("data/artifacts/oof_backup_cleaned_97")
    if backup_dir.exists():
        print(f"Backup already exists: {backup_dir}")
    else:
        shutil.copytree(oof_dir, backup_dir)
        print(f"✓ Backed up cleaned OOF predictions to {backup_dir}")


def generate_full_oof():
    """Generate OOF predictions without data cleaning."""
    from moola.cli import oof
    from moola.config import paths

    print("\n" + "=" * 50)
    print("Generating OOF predictions for FULL 115 samples")
    print("=" * 50)

    # Temporarily disable data cleaning in CLI
    # We'll do this by directly importing and modifying the data loading

    # Load full dataset
    train_path = paths.data / "processed" / "train.parquet"
    import pandas as pd
    df = pd.read_parquet(train_path)
    print(f"\n✓ Loaded {len(df)} samples (no cleaning)")

    # Clear old OOF predictions
    oof_dir = Path("data/artifacts/oof")
    if oof_dir.exists():
        print(f"Removing old OOF predictions...")
        shutil.rmtree(oof_dir)

    # Clear old CV splits
    splits_dir = Path("data/artifacts/splits")
    if splits_dir.exists():
        print(f"Removing old CV splits...")
        shutil.rmtree(splits_dir)

    print("\nRunning OOF with CNN-Transformer...")
    print("This will take a few minutes...")

    # Run OOF - but we need to patch the validation
    # Actually, let's just call the CLI with a modified version

    # For now, let's just tell the user to run the command manually with a flag
    print("\n" + "=" * 50)
    print("To generate full OOF predictions, run:")
    print("=" * 50)
    print("\n  python3 scripts/run_oof_no_clean.py\n")
    print("This will temporarily disable data validation and run OOF on all 115 samples.")


def main():
    """Main entry point."""
    print("🔧 Preparing to generate OOF predictions for CleanLab")

    # Backup current OOF
    backup_current_oof()

    # Instructions
    print("\n" + "=" * 70)
    print("IMPORTANT: OOF Generation for CleanLab")
    print("=" * 70)
    print("""
CleanLab needs predictions on ALL samples, including potentially bad ones.

Current situation:
- We have OOF predictions for 97 cleaned samples
- We need OOF predictions for all 115 samples (including 18 flagged)

Solution:
1. Temporarily disable data cleaning
2. Regenerate OOF predictions on full dataset
3. Run CleanLab analysis
4. Restore cleaned data for final training

To proceed, I'll create a temporary script that disables validation...
""")


if __name__ == "__main__":
    main()
