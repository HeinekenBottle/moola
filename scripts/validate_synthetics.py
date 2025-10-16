"""Validate synthetic samples quality using holdout real data.

Strategy:
1. Hold out 20% of real samples as validation set
2. Train XGBoost on synthetic samples only
3. Evaluate on real holdout
4. Threshold: Must beat 53% accuracy (50% + 3% margin)

This ensures synthetic samples are realistic enough to generalize to real data.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


def main():
    # Load datasets
    clean_path = Path('data/processed/train_clean.parquet')
    smote_path = Path('data/processed/train_smote_300.parquet')

    print("="*60)
    print("SMOTE Synthetic Validation")
    print("="*60)

    # Load real data
    print(f"\nLoading real data from {clean_path}")
    df_real = pd.read_parquet(clean_path)
    print(f"Real samples: {len(df_real)}")

    # Extract real features and labels
    X_real = np.stack([np.stack(f) for f in df_real['features']])
    y_real = df_real['label'].values

    # Encode labels to integers for XGBoost
    le = LabelEncoder()
    y_real_encoded = le.fit_transform(y_real)

    # Flatten for XGBoost: [N, T, F] -> [N, T*F]
    N, T, F = X_real.shape
    X_real_flat = X_real.reshape(N, T * F)
    print(f"Real feature shape: {X_real.shape} -> {X_real_flat.shape}")

    # Split real data: 80% train (for comparison), 20% holdout (for validation)
    X_real_train, X_real_holdout, y_real_train, y_real_holdout = train_test_split(
        X_real_flat, y_real_encoded, test_size=0.2, random_state=1337, stratify=y_real_encoded
    )
    print(f"\nReal data split:")
    print(f"  Train: {len(X_real_train)} samples")
    print(f"  Holdout: {len(X_real_holdout)} samples")

    # Load SMOTE augmented data
    print(f"\nLoading SMOTE data from {smote_path}")
    df_smote = pd.read_parquet(smote_path)
    print(f"SMOTE samples: {len(df_smote)}")
    print(f"Class distribution:\n{df_smote['label'].value_counts()}")

    # Extract only synthetic samples (exclude original 98 real samples)
    # Synthetic samples have window_id starting with "synthetic_"
    df_synthetic = df_smote[df_smote['window_id'].str.startswith('synthetic_')]
    print(f"\nFiltered to {len(df_synthetic)} pure synthetic samples")

    X_synthetic = np.stack([np.stack(f) for f in df_synthetic['features']])
    y_synthetic = df_synthetic['label'].values
    y_synthetic_encoded = le.transform(y_synthetic)
    X_synthetic_flat = X_synthetic.reshape(len(df_synthetic), T * F)

    # Train XGBoost on synthetic only
    print("\n" + "="*60)
    print("Training XGBoost on SYNTHETIC samples only")
    print("="*60)

    clf_synthetic = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=1337,
        eval_metric='logloss'
    )
    clf_synthetic.fit(X_synthetic_flat, y_synthetic_encoded)

    # Evaluate on real holdout
    y_pred_synthetic = clf_synthetic.predict(X_real_holdout)
    acc_synthetic = accuracy_score(y_real_holdout, y_pred_synthetic)

    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"\nModel trained on: {len(df_synthetic)} synthetic samples")
    print(f"Evaluated on: {len(X_real_holdout)} real holdout samples")
    print(f"\nAccuracy on real holdout: {acc_synthetic:.1%}")

    # Threshold check
    threshold = 0.53  # 50% baseline + 3% margin
    passed = acc_synthetic >= threshold

    print(f"\nThreshold: {threshold:.1%} (50% + 3% margin)")
    if passed:
        print(f"✅ VALIDATION PASSED ({acc_synthetic:.1%} >= {threshold:.1%})")
        print("\nSynthetic samples are realistic enough for training.")
        print("Proceed with SMOTE-augmented training.")
    else:
        print(f"❌ VALIDATION FAILED ({acc_synthetic:.1%} < {threshold:.1%})")
        print("\nSynthetic samples may not generalize well.")
        print("Consider:")
        print("  - Adjusting SMOTE k_neighbors parameter")
        print("  - Using different augmentation strategy")
        print("  - Collecting more real data instead")

    print(f"\n{'='*60}")
    print("Classification Report on Real Holdout:")
    print(f"{'='*60}")
    print(classification_report(y_real_holdout, y_pred_synthetic, target_names=le.classes_))

    # Baseline: Train on real train split for comparison
    print(f"\n{'='*60}")
    print("BASELINE: Training XGBoost on REAL training split")
    print(f"{'='*60}")

    clf_real = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=1337,
        eval_metric='logloss'
    )
    clf_real.fit(X_real_train, y_real_train)
    y_pred_real = clf_real.predict(X_real_holdout)
    acc_real = accuracy_score(y_real_holdout, y_pred_real)

    print(f"\nModel trained on: {len(X_real_train)} real samples")
    print(f"Evaluated on: {len(X_real_holdout)} real holdout samples")
    print(f"\nBaseline accuracy on real holdout: {acc_real:.1%}")

    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Synthetic-trained model: {acc_synthetic:.1%}")
    print(f"Real-trained baseline:   {acc_real:.1%}")
    diff = acc_synthetic - acc_real
    print(f"Difference:              {diff:+.1%}")

    if diff >= 0:
        print("\n✅ Synthetic training matches or exceeds real training!")
    else:
        print(f"\n⚠️  Synthetic training is {abs(diff):.1%} below real training")
        print("   (Still acceptable if above 53% threshold)")

    return passed


if __name__ == '__main__':
    passed = main()
    exit(0 if passed else 1)
