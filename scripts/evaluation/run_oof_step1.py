#!/usr/bin/env python3
"""Simplified OOF generation for Step 1 - Clean Baseline"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.models import LogRegModel, RFModel, XGBModel, SimpleLSTMModel, CnnTransformerModel

def main():
    print("=" * 80)
    print("STEP 1: Generate Clean OOF Baseline (No Augmentation)")
    print("=" * 80)

    # Load data
    print("\n[1/6] Loading data...")
    df = pd.read_parquet('data/processed/train_clean.parquet')
    print(f"  ✓ Loaded {len(df)} samples")
    print(f"  ✓ Classes: {df['label'].value_counts().to_dict()}")

    # Extract features
    print("\n[2/6] Extracting features...")
    features_list = []
    for idx, row in df.iterrows():
        timesteps = row['features']
        timestep_matrix = np.stack(timesteps)
        features_list.append(timestep_matrix)

    X = np.stack(features_list)  # [N, T, F]
    y = df['label'].values
    print(f"  ✓ Feature matrix: {X.shape}")

    # Initialize OOF arrays
    print("\n[3/6] Preparing cross-validation...")
    splits_dir = Path('data/splits')
    k_folds = 5

    # Model configurations (clean mode - no augmentation)
    models_config = {
        "logreg": (LogRegModel, {}),
        "rf": (RFModel, {"n_estimators": 200, "max_depth": 10}),
        "xgb": (XGBModel, {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}),
        "simple_lstm": (SimpleLSTMModel, {
            "mixup_alpha": 0.0,
            "use_temporal_aug": False,
            "early_stopping_patience": 20,
        }),
        "cnn_transformer": (CnnTransformerModel, {
            "mixup_alpha": 0.0,
            "use_temporal_aug": False,
            "early_stopping_patience": 20,
        }),
    }

    # Train each model
    print(f"\n[4/6] Training {len(models_config)} models with {k_folds}-fold CV...")
    output_dir = Path('data/oof')
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for model_idx, (model_name, (model_class, model_kwargs)) in enumerate(models_config.items(), 1):
        print(f"\n  [{model_idx}/{len(models_config)}] {model_name.upper()}")
        print(f"    Config: {model_kwargs if model_kwargs else 'defaults'}")

        # Initialize OOF predictions
        oof_preds = np.zeros((len(X), 2))

        # K-fold CV
        for fold in range(k_folds):
            print(f"    Fold {fold+1}/{k_folds}...", end=" ", flush=True)

            # Load split
            train_idx = np.load(splits_dir / f"fold_{fold}_train.npy")
            val_idx = np.load(splits_dir / f"fold_{fold}_val.npy")

            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train
            device = "cpu" if model_name in ["logreg", "rf", "xgb"] else "cpu"  # Force CPU for now
            model = model_class(seed=1337, device=device, **model_kwargs)
            model.fit(X_train, y_train)

            # Predict
            val_preds = model.predict_proba(X_val)
            oof_preds[val_idx] = val_preds

            # Compute fold accuracy
            unique_labels = np.unique(y_val)
            label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
            y_val_numeric = np.array([label_to_idx[label] for label in y_val])
            y_val_pred = np.argmax(val_preds, axis=1)
            fold_acc = accuracy_score(y_val_numeric, y_val_pred)

            print(f"acc={fold_acc:.3f}", flush=True)

        # Save OOF predictions
        output_path = output_dir / f"{model_name}_phase2_clean.npy"
        np.save(output_path, oof_preds)

        # Compute overall accuracy
        unique_labels = np.unique(y)
        label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        y_numeric = np.array([label_to_idx[label] for label in y])
        y_pred = np.argmax(oof_preds, axis=1)
        overall_acc = accuracy_score(y_numeric, y_pred)

        results[model_name] = overall_acc
        print(f"    ✓ Overall accuracy: {overall_acc:.1%}")
        print(f"    ✓ Saved: {output_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("STEP 1 COMPLETE - Clean OOF Baseline Results")
    print("=" * 80)
    for model_name, acc in results.items():
        print(f"  {model_name:20s}: {acc:.1%}")

    avg_acc = np.mean(list(results.values()))
    print(f"\n  {'Average':20s}: {avg_acc:.1%}")
    print(f"\nExpected range: 58-62% (honest baseline, no leakage)")
    print("=" * 80)

    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        exit(1)
