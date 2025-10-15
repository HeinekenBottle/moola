#!/usr/bin/env python3
"""Test Option 1: Simple 5-Feature Classical ML

Validates the hypothesis that 5 simple features outperform 37 complex features.
This is the HIGHEST ROI approach from the forensic audit.

Expected: 63-66% accuracy (+6.5-9.5% improvement from 56.5% baseline)

Features:
1. price_change: (close[-1] - close[0]) / close[0]
2. direction: sign(price_change)
3. range_ratio: (high.max() - low.min()) / close[0]
4. body_dominance: mean(|close - open| / (high - low))
5. wick_balance: mean((high - max(o,c)) / (max(o,c) - low))
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Add moola to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def extract_simple_features(X_raw: np.ndarray, expansion_start: np.ndarray, expansion_end: np.ndarray) -> np.ndarray:
    """Extract 5 simple features from expansion regions ONLY.

    Args:
        X_raw: Raw OHLC data [N, 105, 4]
        expansion_start: Start indices [N]
        expansion_end: End indices [N]

    Returns:
        Simple features [N, 5]
    """
    N = X_raw.shape[0]
    features = []

    for i in range(N):
        start = int(expansion_start[i])
        end = int(expansion_end[i])
        pattern = X_raw[i, start:end+1, :]

        if len(pattern) == 0:
            # Handle empty pattern
            features.append([0.0, 0.0, 0.0, 0.5, 0.5])
            continue

        o, h, l, c = pattern[:, 0], pattern[:, 1], pattern[:, 2], pattern[:, 3]

        # Feature 1: Price change
        if len(c) > 0:
            price_change = (c[-1] - c[0]) / (c[0] + 1e-10)
        else:
            price_change = 0.0

        # Feature 2: Direction
        direction = 1.0 if price_change > 0 else -1.0

        # Feature 3: Range ratio
        if len(h) > 0 and len(l) > 0 and len(c) > 0:
            range_ratio = (h.max() - l.min()) / (c[0] + 1e-10)
        else:
            range_ratio = 0.0

        # Feature 4: Body dominance
        if len(o) > 0:
            body = np.abs(c - o)
            total_range = h - l + 1e-10
            body_dominance = (body / total_range).mean()
        else:
            body_dominance = 0.5

        # Feature 5: Wick balance
        if len(o) > 0:
            upper_wick = h - np.maximum(o, c)
            lower_wick = np.minimum(o, c) - l
            total_wick = upper_wick + lower_wick + 1e-10
            wick_balance = (upper_wick / total_wick).mean()
        else:
            wick_balance = 0.5

        features.append([price_change, direction, range_ratio, body_dominance, wick_balance])

    feature_matrix = np.array(features, dtype=np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

    return feature_matrix


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def main():
    """Run Option 1 validation: 5 simple features."""
    print("=" * 80)
    print(f"{'OPTION 1 VALIDATION: 5 SIMPLE FEATURES':^80}")
    print("=" * 80)
    print("\nHypothesis: 5 simple features > 37 complex features")
    print("Expected: 63-66% accuracy (vs 56.5% baseline)")
    print("\nFeatures:")
    print("  1. price_change: (close[-1] - close[0]) / close[0]")
    print("  2. direction: sign(price_change)")
    print("  3. range_ratio: (high.max() - low.min()) / close[0]")
    print("  4. body_dominance: mean(|close - open| / (high - low))")
    print("  5. wick_balance: mean((high - max(o,c)) / (max(o,c) - low))")

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet('data/processed/train.parquet')
    print(f"Total samples: {len(df)}")

    # Convert features
    X_raw = np.array([np.array([np.array(bar) for bar in sample]) for sample in df['features']])
    y = df['label'].values
    expansion_start = df['expansion_start'].values
    expansion_end = df['expansion_end'].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    unique, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} ({100*count/len(y):.1f}%)")

    # Extract simple features
    print_section("EXTRACTING 5 SIMPLE FEATURES")
    print("\nExtracting from expansion regions only...")
    X_simple = extract_simple_features(X_raw, expansion_start, expansion_end)
    print(f"Feature matrix shape: {X_simple.shape}")

    # Show sample features
    print(f"\nSample features (first 5 rows):")
    feature_names = ['price_change', 'direction', 'range_ratio', 'body_dominance', 'wick_balance']
    print(f"{'Sample':<8} " + " ".join([f"{name:>14}" for name in feature_names]))
    print("─" * 80)
    for i in range(min(5, len(X_simple))):
        values_str = " ".join([f"{val:>14.6f}" for val in X_simple[i]])
        print(f"{i:<8} {values_str}")

    # Feature statistics
    print(f"\nFeature Statistics:")
    print(f"{'Feature':<20} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
    print("─" * 80)
    for i, name in enumerate(feature_names):
        feat_vals = X_simple[:, i]
        print(f"{name:<20} {feat_vals.mean():>12.6f} {feat_vals.std():>12.6f} "
              f"{feat_vals.min():>12.6f} {feat_vals.max():>12.6f}")

    # Train XGBoost with simple features
    print_section("TRAINING XGBOOST (5 SIMPLE FEATURES)")

    print("\nModel Configuration:")
    print("  n_estimators: 100")
    print("  max_depth: 3")
    print("  learning_rate: 0.1")
    print("  min_child_weight: 5")
    print("  subsample: 0.8")
    print("  colsample_bytree: 0.8")

    model_simple = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=1337,
        tree_method='hist'
    )

    # Cross-validation with stratified k-fold
    print(f"\nRunning 5-fold stratified cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)
    scores_simple = cross_val_score(model_simple, X_simple, y_encoded, cv=cv, scoring='accuracy')

    print(f"\nCross-Validation Results (5 simple features):")
    print(f"{'Fold':<8} {'Accuracy':>12}")
    print("─" * 25)
    for i, score in enumerate(scores_simple, 1):
        print(f"{i:<8} {score:>12.4f}")
    print("─" * 25)
    print(f"{'Mean':<8} {scores_simple.mean():>12.4f}")
    print(f"{'Std':<8} {scores_simple.std():>12.4f}")

    # Compare to baseline (complex features)
    print_section("COMPARISON TO BASELINE (37 COMPLEX FEATURES)")

    print("\nExtracting 37 complex features for comparison...")
    from moola.features.price_action_features import engineer_classical_features
    X_complex = engineer_classical_features(X_raw, expansion_start=expansion_start, expansion_end=expansion_end)
    print(f"Complex features shape: {X_complex.shape}")

    print(f"\nTraining XGBoost with 37 complex features...")
    model_complex = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=1337,
        tree_method='hist'
    )

    scores_complex = cross_val_score(model_complex, X_complex, y_encoded, cv=cv, scoring='accuracy')

    print(f"\nCross-Validation Results (37 complex features):")
    print(f"{'Fold':<8} {'Accuracy':>12}")
    print("─" * 25)
    for i, score in enumerate(scores_complex, 1):
        print(f"{i:<8} {score:>12.4f}")
    print("─" * 25)
    print(f"{'Mean':<8} {scores_complex.mean():>12.4f}")
    print(f"{'Std':<8} {scores_complex.std():>12.4f}")

    # Results comparison
    print_section("RESULTS COMPARISON")

    baseline = 0.565  # Current OOF accuracy
    simple_mean = scores_simple.mean()
    complex_mean = scores_complex.mean()

    improvement_simple = simple_mean - baseline
    improvement_complex = complex_mean - baseline
    simple_vs_complex = simple_mean - complex_mean

    print(f"\n{'Approach':<30} {'Accuracy':>12} {'vs Baseline':>15} {'Assessment':<30}")
    print("─" * 90)
    print(f"{'Baseline (Current OOF)':<30} {baseline:>12.4f} {'---':>15} {'---':<30}")
    print(f"{'5 Simple Features':<30} {simple_mean:>12.4f} {improvement_simple:>+14.4f} "
          f"{'(' + f'{improvement_simple/baseline*100:+.1f}%' + ')':<30}")
    print(f"{'37 Complex Features':<30} {complex_mean:>12.4f} {improvement_complex:>+14.4f} "
          f"{'(' + f'{improvement_complex/baseline*100:+.1f}%' + ')':<30}")
    print("─" * 90)
    print(f"{'Simple vs Complex':<30} {'---':>12} {simple_vs_complex:>+14.4f} "
          f"{'(' + f'{simple_vs_complex/complex_mean*100:+.1f}%' + ')':<30}")

    # Verdict
    print_section("VERDICT")

    target_min = 0.63
    target_max = 0.66

    if simple_mean >= target_min:
        print(f"\n🎉 SUCCESS! 5 simple features achieved {simple_mean:.4f} ({simple_mean*100:.1f}%)")
        print(f"✓ Target range: [{target_min:.4f}, {target_max:.4f}] ({target_min*100:.1f}%-{target_max*100:.1f}%)")
        print(f"✓ Improvement: {improvement_simple:+.4f} ({improvement_simple/baseline*100:+.1f}%)")

        if simple_mean > complex_mean + 0.01:
            print(f"✓ Outperforms complex features by {simple_vs_complex:+.4f}")
            print(f"\n🚀 RECOMMENDATION: Implement Option 1 (5 simple features)")
            print(f"   - Fastest implementation (~1-2 hours)")
            print(f"   - Highest ROI")
            print(f"   - Lowest complexity")
            print(f"   - Best for small dataset (115 samples)")
        else:
            print(f"\n→ RECOMMENDATION: Use 5 simple features (similar to complex, but simpler)")
            print(f"   - Easier to maintain")
            print(f"   - Faster feature extraction")
            print(f"   - Lower overfitting risk")

    elif simple_mean >= baseline + 0.02:
        print(f"\n✓ GOOD IMPROVEMENT: 5 simple features achieved {simple_mean:.4f} ({simple_mean*100:.1f}%)")
        print(f"✓ Improvement: {improvement_simple:+.4f} ({improvement_simple/baseline*100:+.1f}%)")
        print(f"⚠️  Below target range: [{target_min:.4f}, {target_max:.4f}]")
        print(f"\n🔧 RECOMMENDATION: Implement Option 1 with tweaks")
        print(f"   - Add 2-3 more simple features (e.g., volatility, momentum)")
        print(f"   - Or ensemble multiple simple models")

    elif simple_mean > complex_mean:
        print(f"\n→ MARGINAL WIN: 5 simple features slightly better")
        print(f"   Simple: {simple_mean:.4f} vs Complex: {complex_mean:.4f}")
        print(f"   Difference: {simple_vs_complex:+.4f}")
        print(f"\n🔧 RECOMMENDATION: Try Option 2 (Fix CNN-Transformer)")
        print(f"   - Simple features not enough")
        print(f"   - Deep learning may be needed")

    else:
        print(f"\n✗ HYPOTHESIS REJECTED: Complex features outperform simple")
        print(f"   Simple: {simple_mean:.4f} vs Complex: {complex_mean:.4f}")
        print(f"   Difference: {simple_vs_complex:+.4f}")
        print(f"\n🔧 RECOMMENDATION: Implement Option 2 (Fix CNN-Transformer)")
        print(f"   - Simple features insufficient")
        print(f"   - Need deeper feature engineering or architecture fix")

    # Feature importance
    print_section("FEATURE IMPORTANCE (5 SIMPLE FEATURES)")

    print("\nTraining full model to extract feature importance...")
    model_simple.fit(X_simple, y_encoded)
    importances = model_simple.feature_importances_

    print(f"\n{'Feature':<20} {'Importance':>15} {'Rank':>8}")
    print("─" * 50)
    importance_pairs = list(zip(feature_names, importances))
    importance_pairs.sort(key=lambda x: x[1], reverse=True)

    for rank, (name, imp) in enumerate(importance_pairs, 1):
        marker = " ⭐" if rank == 1 else ""
        print(f"{name:<20} {imp:>15.6f} {rank:>8}{marker}")

    print("\n" + "=" * 80)
    print(f"{'OPTION 1 VALIDATION COMPLETE':^80}")
    print("=" * 80)
    print(f"\nFinal Result: {simple_mean:.4f} ({simple_mean*100:.1f}% accuracy)")
    print(f"Baseline: {baseline:.4f} ({baseline*100:.1f}% accuracy)")
    print(f"Improvement: {improvement_simple:+.4f} ({improvement_simple/baseline*100:+.1f}%)")

    if simple_mean >= target_min:
        print(f"\n✅ VALIDATION SUCCESS - Proceed with Option 1 implementation")
    elif simple_mean >= baseline + 0.02:
        print(f"\n✓ VALIDATION PROMISING - Proceed with Option 1 (with tweaks)")
    else:
        print(f"\n⚠️  VALIDATION INCONCLUSIVE - Consider Option 2 (Fix CNN-Transformer)")


if __name__ == '__main__':
    main()
