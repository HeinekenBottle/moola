#!/usr/bin/env python3
"""Regenerate all OOF predictions with Phase 2 improvements.

This script regenerates OOF (out-of-fold) predictions for all models using:
- Phase 1 fixes: Per-fold SMOTE, fixed loss functions, SimpleLSTM (replaces RWKV-TS)
- Phase 2 improvements: Increased mixup (α=0.4), temporal augmentation, GPU optimizations

Expected improvements:
- Mixup (α=0.4): +2-4%
- Temporal augmentation: +2-3%
- CleanLab cleaning: +3-5%
- **Total expected**: 58-62% → 65-74% accuracy

Usage:
    # Generate clean baseline (no augmentation for fair comparison)
    python scripts/regenerate_oof_phase2.py --mode clean --experiment phase2-clean-baseline

    # Generate with full Phase 2 augmentation
    python scripts/regenerate_oof_phase2.py --mode augmented --experiment phase2-full-augmentation

    # Generate with CleanLab-cleaned data
    python scripts/regenerate_oof_phase2.py --mode augmented --use-cleaned-data --experiment phase2-cleaned
"""

import argparse
from pathlib import Path
import sys

import mlflow
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.models import (
    LogRegModel,
    RFModel,
    XGBModel,
    SimpleLSTMModel,
    CnnTransformerModel,
)


def generate_oof(
    X: np.ndarray,
    y: np.ndarray,
    model_class,
    model_name: str,
    model_kwargs: dict,
    seed: int,
    k: int,
    splits_dir: Path,
    output_path: Path,
    apply_smote: bool,
    mlflow_tracking: bool,
    mlflow_experiment: str,
    device: str = "cpu",
):
    """Generate OOF predictions for a single model.

    Args:
        X: Feature matrix [N, D]
        y: Target labels [N]
        model_class: Model class to instantiate
        model_name: Model name for logging
        model_kwargs: Keyword arguments for model initialization
        seed: Random seed
        k: Number of folds
        splits_dir: Directory containing split indices
        output_path: Where to save OOF predictions
        apply_smote: Whether to apply SMOTE (per-fold)
        mlflow_tracking: Enable MLflow tracking
        mlflow_experiment: MLflow experiment name
        device: Device for training ('cpu' or 'cuda')
    """
    from imblearn.over_sampling import SMOTE

    print(f"\n{'='*80}")
    print(f"Generating OOF: {model_name}")
    print(f"{'='*80}")
    print(f"Augmentation: {'SMOTE' if apply_smote else 'None'}")
    print(f"Device: {device}")
    print(f"Model kwargs: {model_kwargs}")

    # Initialize OOF predictions
    oof_preds = np.zeros((len(X), 2))  # Assuming 2 classes

    # Set MLflow experiment
    if mlflow_tracking:
        mlflow.set_experiment(mlflow_experiment)

    # K-fold cross-validation
    for fold in range(k):
        print(f"\n[FOLD {fold+1}/{k}]")

        # Load split indices
        train_idx = np.load(splits_dir / f"fold_{fold}_train.npy")
        val_idx = np.load(splits_dir / f"fold_{fold}_val.npy")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Apply SMOTE if requested (per-fold to avoid leakage)
        if apply_smote:
            # Reshape for SMOTE if needed
            if X_train.ndim == 3:
                N, T, F = X_train.shape
                X_train_flat = X_train.reshape(N, T * F)
            else:
                X_train_flat = X_train

            smote = SMOTE(random_state=seed, k_neighbors=5)
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)

            # Reshape back if needed
            if X_train.ndim == 3:
                X_train_resampled = X_train_resampled.reshape(-1, T, F)

            print(f"[SMOTE] Samples: {len(X_train)} → {len(X_train_resampled)}")
            X_train, y_train = X_train_resampled, y_train_resampled

        # Initialize model
        model = model_class(seed=seed, device=device, **model_kwargs)

        # MLflow tracking
        if mlflow_tracking:
            with mlflow.start_run(run_name=f"{model_name}_fold{fold}"):
                mlflow.log_params({
                    "model": model_name,
                    "fold": fold,
                    "train_size": len(X_train),
                    "val_size": len(X_val),
                    "smote": apply_smote,
                    **model_kwargs,
                })

                # Train model
                model.fit(X_train, y_train)

                # Predict on validation fold
                val_preds = model.predict_proba(X_val)
                oof_preds[val_idx] = val_preds

                # Log metrics
                from sklearn.metrics import accuracy_score, roc_auc_score
                y_val_pred = np.argmax(val_preds, axis=1)

                # Get label mapping
                unique_labels = np.unique(y_val)
                label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
                y_val_numeric = np.array([label_to_idx[label] for label in y_val])

                acc = accuracy_score(y_val_numeric, y_val_pred)
                auc = roc_auc_score(y_val_numeric, val_preds[:, 1])

                mlflow.log_metrics({
                    "val_accuracy": acc,
                    "val_auc": auc,
                })

                print(f"[METRICS] Fold {fold}: Acc={acc:.4f}, AUC={auc:.4f}")
        else:
            # Train without MLflow
            model.fit(X_train, y_train)
            val_preds = model.predict_proba(X_val)
            oof_preds[val_idx] = val_preds

    # Save OOF predictions
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, oof_preds)
    print(f"\n[SAVED] OOF predictions: {output_path}")

    # Overall metrics
    from sklearn.metrics import accuracy_score, roc_auc_score
    unique_labels = np.unique(y)
    label_to_idx = {label: idx for idx, label in enumerate(sorted(unique_labels))}
    y_numeric = np.array([label_to_idx[label] for label in y])

    y_pred = np.argmax(oof_preds, axis=1)
    overall_acc = accuracy_score(y_numeric, y_pred)
    overall_auc = roc_auc_score(y_numeric, oof_preds[:, 1])

    print(f"\n[OVERALL] {model_name}: Acc={overall_acc:.4f}, AUC={overall_auc:.4f}")

    if mlflow_tracking:
        with mlflow.start_run(run_name=f"{model_name}_overall"):
            mlflow.log_metrics({
                "overall_accuracy": overall_acc,
                "overall_auc": overall_auc,
            })

    return oof_preds


def main():
    parser = argparse.ArgumentParser(description="Regenerate OOF predictions with Phase 2 improvements")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["clean", "augmented"],
        default="augmented",
        help="Training mode: 'clean' (no augmentation) or 'augmented' (Phase 2 full)"
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/processed/train.parquet"),
        help="Training data file"
    )
    parser.add_argument(
        "--use-cleaned-data",
        action="store_true",
        help="Use CleanLab-cleaned dataset instead of original"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing fold split indices"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/oof/phase2"),
        help="Output directory for OOF predictions"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="phase2-oof-regeneration",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device for deep learning models"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed"
    )
    parser.add_argument(
        "--k-folds",
        type=int,
        default=5,
        help="Number of CV folds"
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Phase 2: OOF Regeneration")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_file}")
    print(f"Use cleaned data: {args.use_cleaned_data}")
    print(f"Device: {args.device}")
    print(f"MLflow: {'Disabled' if args.no_mlflow else 'Enabled'}")

    # Load data
    if args.use_cleaned_data:
        data_file = Path("data/processed/train_clean_phase2.parquet")
        if not data_file.exists():
            print(f"[ERROR] Cleaned dataset not found: {data_file}")
            print("[ERROR] Run scripts/run_cleanlab_phase2.py first")
            return 1
    else:
        data_file = args.data_file

    print(f"\n[LOAD] Loading data: {data_file}")
    df = pd.read_parquet(data_file)

    print(f"[LOAD] Samples: {len(df)}")
    print(f"[LOAD] Classes: {df['label'].value_counts().to_dict()}")

    # Extract features and labels
    if 'X' in df.columns:
        # X column already has proper feature arrays
        X = np.stack(df['X'].values)
    elif 'features' in df.columns:
        # Features column contains time series of OHLC arrays
        features_list = []
        for idx, row in df.iterrows():
            timesteps = row['features']  # Array of OHLC arrays
            timestep_matrix = np.stack(timesteps)  # Stack into [T, F] matrix
            features_list.append(timestep_matrix)
        X = np.stack(features_list)  # [N, T, F]
    else:
        # Fallback: all numeric columns except 'label' are features
        feature_cols = [col for col in df.columns if col != 'label']
        X = df[feature_cols].values

    y = df['label'].values

    print(f"[LOAD] Feature shape: {X.shape}")

    # Define models to run
    if args.mode == "clean":
        # Clean baseline: no augmentation
        models_config = {
            "logreg": (LogRegModel, {}),
            "rf": (RFModel, {"n_estimators": 200, "max_depth": 10}),
            "xgb": (XGBModel, {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}),
            "simple_lstm": (SimpleLSTMModel, {
                "mixup_alpha": 0.0,  # Disable mixup
                "use_temporal_aug": False,  # Disable temporal augmentation
                "early_stopping_patience": 20,
            }),
            "cnn_transformer": (CnnTransformerModel, {
                "mixup_alpha": 0.0,  # Disable mixup
                "use_temporal_aug": False,  # Disable temporal augmentation
                "early_stopping_patience": 20,
            }),
        }
    else:
        # Augmented: Phase 2 full improvements
        models_config = {
            "logreg": (LogRegModel, {}),
            "rf": (RFModel, {"n_estimators": 200, "max_depth": 10}),
            "xgb": (XGBModel, {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1}),
            "simple_lstm": (SimpleLSTMModel, {
                "mixup_alpha": 0.4,  # Phase 2: increased from 0.2
                "use_temporal_aug": True,  # Phase 2: enabled
                "early_stopping_patience": 20,  # Phase 2: reduced from 30
                "batch_size": 1024,  # Phase 2: increased from 512
            }),
            "cnn_transformer": (CnnTransformerModel, {
                "mixup_alpha": 0.4,  # Phase 2: increased from 0.2
                "use_temporal_aug": True,  # Phase 2: enabled
                "early_stopping_patience": 20,  # Phase 2: reduced from 30
                "batch_size": 1024,  # Phase 2: increased from 512
            }),
        }

    # Generate OOF for each model
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, (model_class, model_kwargs) in models_config.items():
        output_path = args.output_dir / f"{model_name}_{args.mode}.npy"

        try:
            generate_oof(
                X=X,
                y=y,
                model_class=model_class,
                model_name=model_name,
                model_kwargs=model_kwargs,
                seed=args.seed,
                k=args.k_folds,
                splits_dir=args.splits_dir,
                output_path=output_path,
                apply_smote=(args.mode == "augmented"),  # SMOTE only for augmented mode
                mlflow_tracking=(not args.no_mlflow),
                mlflow_experiment=args.experiment,
                device=args.device if model_name in ["simple_lstm", "cnn_transformer"] else "cpu",
            )
        except Exception as e:
            print(f"[ERROR] Failed to generate OOF for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 80)
    print("Phase 2 OOF Regeneration Complete!")
    print("=" * 80)
    print(f"Output directory: {args.output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Compare clean vs augmented performance")
    print(f"  2. Run CleanLab analysis: scripts/run_cleanlab_phase2.py")
    print(f"  3. Train on cleaned data and regenerate OOF")
    print(f"  4. Compare all three: original → augmented → cleaned")

    return 0


if __name__ == "__main__":
    exit(main())
