#!/usr/bin/env python3
"""Deployment verification script for 2-class stacking ensemble.

This script verifies that all model artifacts are present and functional
on RunPod network storage. Run this after deployment to ensure everything
is ready for production inference.

Usage:
    python scripts/verify_deployment.py

    Or on RunPod:
    cd /workspace/moola && source /tmp/moola-venv/bin/activate && python scripts/verify_deployment.py
"""

import sys
from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd


def verify_data():
    """Verify training data configuration."""
    print("=" * 80)
    print("DATA CONFIGURATION VERIFICATION")
    print("=" * 80)

    data_path = Path("data/processed/train.parquet")
    if not data_path.exists():
        print(f"❌ Training data not found: {data_path}")
        return False

    df = pd.read_parquet(data_path)
    print(f"✓ Training data loaded: {data_path}")
    print(f"  - Total samples: {len(df)}")
    print(f"  - Classes: {sorted(df['label'].unique())}")
    print(f"  - Distribution:")

    for label in sorted(df['label'].unique()):
        count = (df['label'] == label).sum()
        pct = count / len(df) * 100
        print(f"      {label}: {count} ({pct:.1f}%)")

    # Verify 2-class configuration
    if len(df) != 115:
        print(f"⚠️  Expected 115 samples, got {len(df)}")
        return False

    if set(df['label'].unique()) != {'consolidation', 'retracement'}:
        print(f"⚠️  Expected ['consolidation', 'retracement'], got {sorted(df['label'].unique())}")
        return False

    print("\n✓ Data configuration correct (2-class, 115 samples)")
    return True


def verify_artifacts():
    """Verify all model artifacts are present."""
    print("\n" + "=" * 80)
    print("MODEL ARTIFACTS VERIFICATION")
    print("=" * 80)

    artifacts_dir = Path("data/artifacts")
    if not artifacts_dir.exists():
        print(f"❌ Artifacts directory not found: {artifacts_dir}")
        return False

    # Check OOF predictions
    models = ['logreg', 'rf', 'xgb', 'rwkv_ts', 'cnn_transformer']
    print("\n1. OOF Predictions:")
    for model in models:
        oof_path = artifacts_dir / f"oof/{model}/v1/seed_1337.npy"
        if not oof_path.exists():
            print(f"   ❌ {model}: NOT FOUND")
            return False

        oof = np.load(oof_path)
        if oof.shape != (115, 2):
            print(f"   ❌ {model}: Wrong shape {oof.shape}, expected (115, 2)")
            return False

        print(f"   ✓ {model:15s}: shape {oof.shape}, range [{oof.min():.4f}, {oof.max():.4f}]")

    # Check stacking model
    print("\n2. Stacking Ensemble:")
    stack_path = artifacts_dir / "models/stack/stack.pkl"
    if not stack_path.exists():
        print(f"   ❌ Stack model not found: {stack_path}")
        return False

    with open(stack_path, 'rb') as f:
        stack_model = pickle.load(f)
    print(f"   ✓ Meta-learner loaded: {type(stack_model).__name__}")
    print(f"   ✓ Number of estimators: {stack_model.n_estimators}")

    # Check metrics
    metrics_path = artifacts_dir / "models/stack/metrics.json"
    if not metrics_path.exists():
        print(f"   ❌ Metrics not found: {metrics_path}")
        return False

    with open(metrics_path) as f:
        metrics = json.load(f)
    print(f"   ✓ Metrics loaded:")
    print(f"      - Accuracy: {metrics['accuracy']:.4f}")
    print(f"      - F1 Score: {metrics['f1']:.4f}")
    print(f"      - ECE: {metrics['ece']:.4f}")
    print(f"      - Log Loss: {metrics['logloss']:.4f}")

    # Check manifest
    manifest_path = artifacts_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"   ⚠️  Manifest not found: {manifest_path}")
    else:
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"\n3. Deployment Manifest:")
        print(f"   ✓ Created: {manifest['created_at']}")
        print(f"   ✓ Git SHA: {manifest['git_sha']}")
        print(f"   ✓ Seed: {manifest['seed']}")
        print(f"   ✓ CV Folds: {manifest['cv_folds']}")
        print(f"   ✓ Models: {', '.join(manifest['models'])}")

    print("\n✓ All artifacts present and valid")
    return True


def verify_inference():
    """Test inference pipeline."""
    print("\n" + "=" * 80)
    print("INFERENCE PIPELINE VERIFICATION")
    print("=" * 80)

    artifacts_dir = Path("data/artifacts")

    # Load all OOF predictions
    print("\n1. Loading OOF predictions...")
    models = ['logreg', 'rf', 'xgb', 'rwkv_ts', 'cnn_transformer']
    oof_list = []
    for model in models:
        oof_path = artifacts_dir / f"oof/{model}/v1/seed_1337.npy"
        oof = np.load(oof_path)
        oof_list.append(oof)

    X_meta = np.concatenate(oof_list, axis=1)
    print(f"   ✓ Concatenated OOF shape: {X_meta.shape} (115 samples × 10 features)")

    # Load stacking model
    print("\n2. Loading meta-learner...")
    stack_path = artifacts_dir / "models/stack/stack.pkl"
    with open(stack_path, 'rb') as f:
        stack_model = pickle.load(f)
    print(f"   ✓ Meta-learner loaded: {type(stack_model).__name__}")

    # Test prediction on first 5 samples
    print("\n3. Testing predictions...")
    y_pred = stack_model.predict(X_meta[:5])
    y_proba = stack_model.predict_proba(X_meta[:5])

    print(f"   ✓ Predictions (first 5): {y_pred}")
    print(f"   ✓ Probabilities shape: {y_proba.shape}")
    print(f"   ✓ Probability range: [{y_proba.min():.4f}, {y_proba.max():.4f}]")
    print(f"   ✓ Classes: {stack_model.classes_}")

    # Verify probability constraints
    assert np.allclose(y_proba.sum(axis=1), 1.0), "Probabilities don't sum to 1!"
    assert np.all((y_proba >= 0) & (y_proba <= 1)), "Invalid probabilities!"

    print("\n✓ Inference pipeline functional")
    return True


def main():
    """Run all verification checks."""
    print("\n" + "=" * 80)
    print("MOOLA DEPLOYMENT VERIFICATION")
    print("2-Class Stacking Ensemble (Consolidation vs Retracement)")
    print("=" * 80)

    checks = [
        ("Data Configuration", verify_data),
        ("Model Artifacts", verify_artifacts),
        ("Inference Pipeline", verify_inference),
    ]

    results = []
    for name, check_func in checks:
        try:
            success = check_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} FAILED with error: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    all_passed = True
    for name, success in results:
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"{name:30s} {status}")
        if not success:
            all_passed = False

    print("=" * 80)

    if all_passed:
        print("\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("Deployment is ready for production!")
        return 0
    else:
        print("\n❌❌❌ SOME CHECKS FAILED ❌❌❌")
        print("Please fix the issues before deploying.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
