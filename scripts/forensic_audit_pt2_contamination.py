#!/usr/bin/env python3
"""Phase 2: Feature Contamination Analysis

Identifies which features are helping vs hurting model performance by:
- Computing correlation of each feature with labels
- Ranking features by predictive power
- Testing ablation (removing weakest features)
- Identifying "poisoning" features (negative contributors)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

# Add moola to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from moola.features.price_action_features import engineer_classical_features


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def compute_feature_correlations(X_features: np.ndarray, y: np.ndarray) -> list:
    """Compute Pearson correlation of each feature with labels.

    Args:
        X_features: Feature matrix [N, F]
        y: Labels [N]

    Returns:
        List of (feature_idx, correlation) tuples, sorted by |correlation|
    """
    from scipy.stats import pearsonr

    # Encode labels to numeric if needed
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])

    correlations = []
    for i in range(X_features.shape[1]):
        feature_values = X_features[:, i]

        # Handle constant features
        if np.std(feature_values) < 1e-10:
            corr = 0.0
            p_value = 1.0
        else:
            corr, p_value = pearsonr(feature_values, y_numeric)

        correlations.append((i, corr, p_value))

    # Sort by absolute correlation (descending)
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    return correlations


def analyze_feature_correlations(correlations: list, feature_names: list = None):
    """Print feature correlation analysis."""
    print_section("FEATURE CORRELATION RANKING")

    print(f"\nTotal features: {len(correlations)}")
    print(f"\nTop 15 Most Correlated Features (Strongest Signals):")
    print(f"{'Rank':<6} {'Idx':<5} {'Correlation':>12} {'P-value':>10} {'Name':<30}")
    print("─" * 80)

    for rank, (idx, corr, p_val) in enumerate(correlations[:15], 1):
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        print(f"{rank:<6} {idx:<5} {corr:+12.6f} {p_val:10.6f} {name:<30} {sig}")

    print(f"\nBottom 15 Weakest Correlations (Potential Poisons):")
    print(f"{'Rank':<6} {'Idx':<5} {'Correlation':>12} {'P-value':>10} {'Name':<30}")
    print("─" * 80)

    weak_start = len(correlations) - 15
    for rank, (idx, corr, p_val) in enumerate(correlations[-15:], weak_start + 1):
        name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        print(f"{rank:<6} {idx:<5} {corr:+12.6f} {p_val:10.6f} {name:<30} {sig}")

    # Identify near-zero correlations
    near_zero = [(idx, corr, p_val) for idx, corr, p_val in correlations if abs(corr) < 0.05]

    print(f"\n⚠️  Near-Zero Correlations (|r| < 0.05): {len(near_zero)} features")
    print(f"These features contribute minimal signal and may add noise.")

    if near_zero:
        print(f"\nNear-zero features:")
        for idx, corr, p_val in near_zero[:20]:  # Show first 20
            name = feature_names[idx] if feature_names and idx < len(feature_names) else f"feature_{idx}"
            print(f"  Feature {idx:3d} ({name:<30}): r={corr:+.4f}, p={p_val:.4f}")


def test_feature_ablation(X_features: np.ndarray, y: np.ndarray, correlations: list, n_remove: int = 10):
    """Test impact of removing weakest features.

    Args:
        X_features: Feature matrix [N, F]
        y: Labels [N]
        correlations: List of (idx, corr, p_val) tuples
        n_remove: Number of weakest features to remove
    """
    print_section("FEATURE ABLATION STUDY")

    # Encode labels to numeric
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Baseline: Train with all features
    print(f"\nBaseline: Training XGBoost with ALL {X_features.shape[1]} features...")

    baseline_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=1337,
        tree_method='hist'
    )

    # Use cross-validation for robust estimate
    baseline_scores = cross_val_score(baseline_model, X_features, y_encoded, cv=5, scoring='accuracy')
    baseline_mean = baseline_scores.mean()
    baseline_std = baseline_scores.std()

    print(f"Baseline accuracy: {baseline_mean:.4f} ± {baseline_std:.4f}")
    print(f"CV scores: {[f'{s:.4f}' for s in baseline_scores]}")

    # Remove bottom N features (weakest correlations)
    weak_indices = [idx for idx, _, _ in correlations[-n_remove:]]
    mask = np.ones(X_features.shape[1], dtype=bool)
    mask[weak_indices] = False
    X_reduced = X_features[:, mask]

    print(f"\n{'─' * 80}")
    print(f"Ablation Test: Removing {n_remove} weakest features...")
    print(f"Removed feature indices: {sorted(weak_indices)}")
    print(f"Remaining features: {X_reduced.shape[1]}")

    reduced_scores = cross_val_score(baseline_model, X_reduced, y_encoded, cv=5, scoring='accuracy')
    reduced_mean = reduced_scores.mean()
    reduced_std = reduced_scores.std()

    print(f"Reduced model accuracy: {reduced_mean:.4f} ± {reduced_std:.4f}")
    print(f"CV scores: {[f'{s:.4f}' for s in reduced_scores]}")

    # Calculate improvement
    improvement = reduced_mean - baseline_mean
    improvement_pct = (improvement / baseline_mean) * 100

    print(f"\n{'─' * 80}")
    print(f"ABLATION RESULTS")
    print(f"{'─' * 80}")
    print(f"Accuracy change: {improvement:+.4f} ({improvement_pct:+.1f}%)")

    if improvement > 0.01:
        print(f"✓ IMPROVEMENT: Removing weak features HELPED")
        print(f"✓ Conclusion: {n_remove} features were POISONING the model")
        print(f"✓ Recommendation: Remove these features from pipeline")
    elif improvement < -0.01:
        print(f"✗ DEGRADATION: Removing weak features HURT")
        print(f"✗ Conclusion: Even weak features contribute some signal")
        print(f"✗ Recommendation: Keep all features")
    else:
        print(f"→ NEUTRAL: Minimal change")
        print(f"→ Conclusion: Weak features neither help nor hurt significantly")
        print(f"→ Recommendation: Can remove for simpler model without performance loss")

    # Test removing top N features (sanity check - should hurt)
    print(f"\n{'─' * 80}")
    print(f"Sanity Check: Removing {n_remove} STRONGEST features...")

    strong_indices = [idx for idx, _, _ in correlations[:n_remove]]
    mask_strong = np.ones(X_features.shape[1], dtype=bool)
    mask_strong[strong_indices] = False
    X_no_strong = X_features[:, mask_strong]

    no_strong_scores = cross_val_score(baseline_model, X_no_strong, y_encoded, cv=5, scoring='accuracy')
    no_strong_mean = no_strong_scores.mean()

    strong_impact = no_strong_mean - baseline_mean
    print(f"Accuracy without strong features: {no_strong_mean:.4f} ({strong_impact:+.4f})")

    if strong_impact < -0.05:
        print(f"✓ Sanity check PASSED: Removing strong features significantly hurt performance")
    else:
        print(f"⚠️  Sanity check WARNING: Removing strong features didn't hurt much")
        print(f"   This suggests feature redundancy or weak overall signal")


def test_simple_features(X_raw: np.ndarray, y: np.ndarray, expansion_start: np.ndarray, expansion_end: np.ndarray):
    """Test if simple features outperform complex engineered features.

    Args:
        X_raw: Raw OHLC data [N, 105, 4]
        y: Labels [N]
        expansion_start: Expansion start indices [N]
        expansion_end: Expansion end indices [N]
    """
    print_section("SIMPLE vs COMPLEX FEATURES")

    # Encode labels to numeric
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    print(f"\nHypothesis: Complex ICT features (FVG, order blocks, liquidity zones)")
    print(f"            may add more noise than signal for 6-bar patterns.")
    print(f"\nTest: Compare 5 simple features vs 40+ complex features")

    # Extract simple features manually
    N = X_raw.shape[0]
    simple_features = []

    for i in range(N):
        start = int(expansion_start[i])
        end = int(expansion_end[i])
        pattern = X_raw[i, start:end+1, :]

        o, h, l, c = pattern[:, 0], pattern[:, 1], pattern[:, 2], pattern[:, 3]

        features = []

        # 1. Price change
        price_change = (c[-1] - c[0]) / (c[0] + 1e-10) if len(c) > 0 else 0.0
        features.append(price_change)

        # 2. Direction
        direction = 1.0 if price_change > 0 else -1.0
        features.append(direction)

        # 3. Range ratio
        range_ratio = (h.max() - l.min()) / (c[0] + 1e-10) if len(h) > 0 else 0.0
        features.append(range_ratio)

        # 4. Body dominance
        body = np.abs(c - o)
        total_range = h - l + 1e-10
        body_dominance = (body / total_range).mean() if len(o) > 0 else 0.5
        features.append(body_dominance)

        # 5. Wick balance
        upper_wick = h - np.maximum(o, c)
        lower_wick = np.minimum(o, c) - l
        total_wick = upper_wick + lower_wick + 1e-10
        wick_balance = (upper_wick / total_wick).mean() if len(o) > 0 else 0.5
        features.append(wick_balance)

        simple_features.append(features)

    X_simple = np.array(simple_features, dtype=np.float32)
    X_simple = np.nan_to_num(X_simple, nan=0.0, posinf=1e6, neginf=-1e6)

    print(f"\nSimple features shape: {X_simple.shape}")
    print(f"Simple features: price_change, direction, range_ratio, body_dominance, wick_balance")

    # Train with simple features
    print(f"\nTraining XGBoost with 5 SIMPLE features...")
    simple_model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=1337,
        tree_method='hist'
    )

    simple_scores = cross_val_score(simple_model, X_simple, y_encoded, cv=5, scoring='accuracy')
    simple_mean = simple_scores.mean()
    simple_std = simple_scores.std()

    print(f"Simple features accuracy: {simple_mean:.4f} ± {simple_std:.4f}")
    print(f"CV scores: {[f'{s:.4f}' for s in simple_scores]}")

    # Train with complex features
    print(f"\nTraining XGBoost with 40+ COMPLEX features...")
    X_complex = engineer_classical_features(X_raw, expansion_start=expansion_start, expansion_end=expansion_end)

    complex_scores = cross_val_score(simple_model, X_complex, y_encoded, cv=5, scoring='accuracy')
    complex_mean = complex_scores.mean()
    complex_std = complex_scores.std()

    print(f"Complex features accuracy: {complex_mean:.4f} ± {complex_std:.4f}")
    print(f"CV scores: {[f'{s:.4f}' for s in complex_scores]}")

    # Compare
    difference = simple_mean - complex_mean
    difference_pct = (difference / complex_mean) * 100

    print(f"\n{'─' * 80}")
    print(f"SIMPLE vs COMPLEX RESULTS")
    print(f"{'─' * 80}")
    print(f"Accuracy difference: {difference:+.4f} ({difference_pct:+.1f}%)")

    if difference > 0.02:
        print(f"✓ SIMPLE WINS: 5 simple features outperform 40+ complex features!")
        print(f"✓ Conclusion: Complex features adding noise, not signal")
        print(f"✓ Recommendation: Use simpler feature set for 6-bar patterns")
    elif difference < -0.02:
        print(f"✓ COMPLEX WINS: Complex features add predictive power")
        print(f"✓ Conclusion: ICT features capturing useful signal")
        print(f"✓ Recommendation: Keep complex feature engineering")
    else:
        print(f"→ TIE: Similar performance")
        print(f"→ Conclusion: Complex features not significantly better")
        print(f"→ Recommendation: Use simple features for faster training")


def main():
    """Run Phase 2 forensic audit."""
    print("=" * 80)
    print(f"{'PHASE 2: FEATURE CONTAMINATION ANALYSIS':^80}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    df = pd.read_parquet('data/processed/train.parquet')
    print(f"Total samples: {len(df)}")

    # Convert features
    X_raw = np.array([np.array([np.array(bar) for bar in sample]) for sample in df['features']])
    y = df['label'].values
    expansion_start = df['expansion_start'].values
    expansion_end = df['expansion_end'].values

    print(f"X_raw shape: {X_raw.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")

    # Extract engineered features
    print(f"\nExtracting engineered features...")
    X_features = engineer_classical_features(X_raw, expansion_start=expansion_start, expansion_end=expansion_end)
    print(f"Engineered features shape: {X_features.shape}")

    # Phase 2.1: Compute correlations
    print(f"\nComputing feature correlations...")
    correlations = compute_feature_correlations(X_features, y)

    # Get feature names (if available)
    feature_names = [
        # Market structure (5)
        'num_peaks', 'num_troughs', 'higher_highs', 'lower_lows', 'trend_slope',
        # Liquidity zones (3)
        'equal_highs', 'equal_lows', 'pool_ratio',
        # Fair value gaps (3)
        'bullish_fvg', 'bearish_fvg', 'fvg_ratio',
        # Order blocks (3)
        'ob_count', 'ob_strength', 'dist_to_ob',
        # Imbalance ratios (5)
        'avg_body_ratio', 'avg_upper_shadow', 'avg_lower_shadow', 'avg_imbalance', 'avg_wick_dominance',
        # Geometry (4)
        'slope', 'r_squared', 'avg_curvature', 'price_angle',
        # Distance (3)
        'dist_to_support', 'dist_to_resistance', 'position_in_range',
        # Candles (5)
        'num_doji', 'bullish_engulf', 'bearish_engulf', 'num_hammer', 'num_shooting_star',
        # Williams R (1)
        'williams_r',
        # Buffer context (5)
        'left_return', 'left_vol', 'left_to_inner_gap', 'right_return', 'inner_to_right_gap',
    ]

    # Phase 2.2: Analyze correlations
    analyze_feature_correlations(correlations, feature_names)

    # Phase 2.3: Test ablation
    test_feature_ablation(X_features, y, correlations, n_remove=10)

    # Phase 2.4: Simple vs complex
    test_simple_features(X_raw, y, expansion_start, expansion_end)

    print("\n" + "=" * 80)
    print(f"{'PHASE 2 COMPLETE':^80}")
    print("=" * 80)
    print("\nKey Findings:")
    print("  - Identified strongest and weakest features by correlation")
    print("  - Tested if removing weak features improves performance")
    print("  - Compared simple vs complex feature sets")
    print("\nNext: Run Phase 3 (forensic_audit_pt3_smoothing.py) for averaging detection")


if __name__ == '__main__':
    main()
