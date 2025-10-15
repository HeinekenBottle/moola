#!/usr/bin/env python3
"""Phase 3: Averaging/Smoothing Detection

Identifies where signal is being destroyed through averaging and smoothing:
- Detects .mean(), .std(), rolling(), ewm() operations in feature code
- Compares raw vs smoothed feature values
- Tests if removing smoothing improves correlation with labels
- Quantifies signal loss from averaging
"""

import sys
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# Add moola to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from moola.features import price_action_features


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"{title:^80}")
    print("=" * 80)


def detect_smoothing_operations():
    """Detect smoothing operations in feature extraction code."""
    print_section("SMOOTHING OPERATION DETECTION")

    # Get all feature extraction functions
    feature_functions = [
        ('_extract_market_structure', price_action_features._extract_market_structure),
        ('_extract_liquidity_zones', price_action_features._extract_liquidity_zones),
        ('_extract_fair_value_gaps', price_action_features._extract_fair_value_gaps),
        ('_extract_order_blocks', price_action_features._extract_order_blocks),
        ('_extract_imbalance_ratios', price_action_features._extract_imbalance_ratios),
        ('_extract_geometry_features', price_action_features._extract_geometry_features),
        ('_extract_distance_measures', price_action_features._extract_distance_measures),
        ('_extract_candle_patterns', price_action_features._extract_candle_patterns),
        ('_extract_williams_r', price_action_features._extract_williams_r),
        ('_extract_buffer_context', price_action_features._extract_buffer_context),
    ]

    print(f"\nAnalyzing {len(feature_functions)} feature extraction functions...")
    print(f"\nDetecting patterns: .mean(), .std(), rolling(), ewm(), convolve(), linregress()")

    smoothing_found = {}
    total_smoothing_ops = 0

    for func_name, func in feature_functions:
        source = inspect.getsource(func)

        # Detect smoothing patterns
        has_mean = '.mean(' in source or 'mean(' in source
        has_std = '.std(' in source or 'std(' in source
        has_rolling = 'rolling(' in source
        has_ewm = 'ewm(' in source
        has_convolve = 'convolve' in source
        has_linregress = 'linregress' in source

        operations = []
        if has_mean:
            operations.append('mean')
            total_smoothing_ops += source.count('.mean(')
        if has_std:
            operations.append('std')
            total_smoothing_ops += source.count('.std(')
        if has_rolling:
            operations.append('rolling')
            total_smoothing_ops += 1
        if has_ewm:
            operations.append('ewm')
            total_smoothing_ops += 1
        if has_convolve:
            operations.append('convolve')
            total_smoothing_ops += 1
        if has_linregress:
            operations.append('linregress')
            total_smoothing_ops += source.count('linregress')

        if operations:
            smoothing_found[func_name] = operations

    print(f"\n{'Function':<35} {'Smoothing Operations':<40}")
    print("─" * 80)

    for func_name, operations in smoothing_found.items():
        ops_str = ', '.join(operations)
        marker = " ⚠️" if len(operations) >= 2 else ""
        print(f"{func_name:<35} {ops_str:<40}{marker}")

    print(f"\n{'─' * 80}")
    print(f"SMOOTHING DETECTION SUMMARY")
    print(f"{'─' * 80}")
    print(f"Functions with smoothing: {len(smoothing_found)}/{len(feature_functions)}")
    print(f"Total smoothing operations: {total_smoothing_ops}")

    if len(smoothing_found) > len(feature_functions) * 0.5:
        print(f"\n⚠️  WARNING: {len(smoothing_found)} out of {len(feature_functions)} functions use smoothing")
        print(f"⚠️  Averaging operations may be destroying 6-bar pattern signals")

    return smoothing_found


def extract_raw_features(X_raw: np.ndarray, expansion_start: np.ndarray, expansion_end: np.ndarray) -> np.ndarray:
    """Extract features WITHOUT any averaging operations.

    Uses only raw values from the expansion region:
    - Price change (no averaging)
    - Direction (no averaging)
    - Range (no averaging)
    - Individual bar properties (no aggregation)

    Args:
        X_raw: Raw OHLC [N, 105, 4]
        expansion_start: Start indices [N]
        expansion_end: End indices [N]

    Returns:
        Raw features [N, F_raw]
    """
    N = X_raw.shape[0]
    all_features = []

    for i in range(N):
        start = int(expansion_start[i])
        end = int(expansion_end[i])
        pattern = X_raw[i, start:end+1, :]

        if len(pattern) == 0:
            # Empty pattern - use defaults
            features = [0.0] * 10
        else:
            o, h, l, c = pattern[:, 0], pattern[:, 1], pattern[:, 2], pattern[:, 3]

            features = []

            # 1. Price change (raw, no averaging)
            price_change = (c[-1] - c[0]) / (c[0] + 1e-10)
            features.append(price_change)

            # 2. Direction (raw)
            direction = 1.0 if price_change > 0 else -1.0
            features.append(direction)

            # 3. High-Low range (raw)
            hl_range = (h.max() - l.min()) / (c[0] + 1e-10)
            features.append(hl_range)

            # 4. First bar body ratio (raw, no averaging)
            if len(o) > 0:
                body = abs(c[0] - o[0])
                total = h[0] - l[0] + 1e-10
                first_body_ratio = body / total
            else:
                first_body_ratio = 0.5
            features.append(first_body_ratio)

            # 5. Last bar body ratio (raw, no averaging)
            if len(o) > 0:
                body = abs(c[-1] - o[-1])
                total = h[-1] - l[-1] + 1e-10
                last_body_ratio = body / total
            else:
                last_body_ratio = 0.5
            features.append(last_body_ratio)

            # 6. Max candle range (raw)
            candle_ranges = (h - l) / (c + 1e-10)
            max_range = candle_ranges.max()
            features.append(max_range)

            # 7. Min candle range (raw)
            min_range = candle_ranges.min()
            features.append(min_range)

            # 8. Pattern length (no smoothing)
            pattern_length = len(pattern)
            features.append(pattern_length)

            # 9. Start price (raw)
            start_price = c[0] if len(c) > 0 else 0.0
            features.append(start_price)

            # 10. End price (raw)
            end_price = c[-1] if len(c) > 0 else 0.0
            features.append(end_price)

        all_features.append(features)

    feature_matrix = np.array(all_features, dtype=np.float32)
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

    return feature_matrix


def compare_raw_vs_smoothed(X_raw: np.ndarray, y: np.ndarray, expansion_start: np.ndarray, expansion_end: np.ndarray):
    """Compare raw features vs smoothed features."""
    print_section("RAW vs SMOOTHED FEATURE COMPARISON")

    # Extract raw features (no averaging)
    print(f"\nExtracting RAW features (no averaging)...")
    X_raw_features = extract_raw_features(X_raw, expansion_start, expansion_end)
    print(f"Raw features shape: {X_raw_features.shape}")

    # Extract smoothed features (with averaging)
    print(f"\nExtracting SMOOTHED features (with averaging)...")
    from moola.features.price_action_features import engineer_classical_features
    X_smoothed_features = engineer_classical_features(X_raw, expansion_start=expansion_start, expansion_end=expansion_end)
    print(f"Smoothed features shape: {X_smoothed_features.shape}")

    # Encode labels
    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])

    # Compute correlations for raw features
    print(f"\nComputing correlations for raw features...")
    raw_correlations = []
    for i in range(X_raw_features.shape[1]):
        if np.std(X_raw_features[:, i]) > 1e-10:
            corr, _ = pearsonr(X_raw_features[:, i], y_numeric)
        else:
            corr = 0.0
        raw_correlations.append(abs(corr))

    # Compute correlations for smoothed features
    print(f"Computing correlations for smoothed features...")
    smoothed_correlations = []
    for i in range(X_smoothed_features.shape[1]):
        if np.std(X_smoothed_features[:, i]) > 1e-10:
            corr, _ = pearsonr(X_smoothed_features[:, i], y_numeric)
        else:
            corr = 0.0
        smoothed_correlations.append(abs(corr))

    # Compare
    avg_raw_corr = np.mean(raw_correlations)
    avg_smoothed_corr = np.mean(smoothed_correlations)
    max_raw_corr = np.max(raw_correlations)
    max_smoothed_corr = np.max(smoothed_correlations)

    print(f"\n{'─' * 80}")
    print(f"CORRELATION COMPARISON")
    print(f"{'─' * 80}")
    print(f"{'Metric':<30} {'Raw Features':>20} {'Smoothed Features':>20}")
    print("─" * 80)
    print(f"{'Average |correlation|':<30} {avg_raw_corr:>20.6f} {avg_smoothed_corr:>20.6f}")
    print(f"{'Max |correlation|':<30} {max_raw_corr:>20.6f} {max_smoothed_corr:>20.6f}")
    print(f"{'Number of features':<30} {len(raw_correlations):>20} {len(smoothed_correlations):>20}")

    # Verdict
    difference = avg_raw_corr - avg_smoothed_corr
    difference_pct = (difference / avg_smoothed_corr) * 100 if avg_smoothed_corr > 0 else 0

    print(f"\n{'─' * 80}")
    print(f"VERDICT")
    print(f"{'─' * 80}")
    print(f"Correlation difference: {difference:+.6f} ({difference_pct:+.1f}%)")

    if difference > 0.02:
        print(f"\n✓ RAW FEATURES WIN")
        print(f"✓ Raw features have {difference_pct:.1f}% stronger correlation")
        print(f"✓ Smoothing/averaging is DESTROYING signal")
        print(f"✓ Recommendation: Remove averaging operations from feature extraction")
    elif difference < -0.02:
        print(f"\n✓ SMOOTHED FEATURES WIN")
        print(f"✓ Smoothed features have {-difference_pct:.1f}% stronger correlation")
        print(f"✓ Averaging is HELPING by reducing noise")
        print(f"✓ Recommendation: Keep smoothing operations")
    else:
        print(f"\n→ NEUTRAL")
        print(f"→ Smoothing has minimal impact on signal strength")
        print(f"→ Recommendation: Either approach acceptable")


def analyze_williams_r_period(X_raw: np.ndarray, y: np.ndarray, expansion_start: np.ndarray, expansion_end: np.ndarray):
    """Test different Williams %R periods to find optimal lookback."""
    print_section("WILLIAMS %R PERIOD ANALYSIS")

    print(f"\nHypothesis: Williams %R uses 10-14 bar lookback, but patterns are ~6 bars")
    print(f"            Longer lookback may smooth out the pattern signal")
    print(f"\nTest: Compare Williams %R with different periods")

    unique_labels = np.unique(y)
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y_numeric = np.array([label_map[label] for label in y])

    N = X_raw.shape[0]
    periods = [3, 5, 7, 10, 14, 20]
    results = []

    for period in periods:
        williams_r_values = []

        for i in range(N):
            start = int(expansion_start[i])
            end = int(expansion_end[i])
            pattern = X_raw[i, start:end+1, :]

            if len(pattern) == 0:
                williams_r_values.append(-50.0)
                continue

            h, l, c = pattern[:, 1], pattern[:, 2], pattern[:, 3]

            # Calculate Williams %R
            lookback = min(period, len(c))
            if lookback > 0:
                hh = h[-lookback:].max()
                ll = l[-lookback:].min()
                current_close = c[-1]

                if hh > ll:
                    wr = -100.0 * (hh - current_close) / (hh - ll)
                else:
                    wr = -50.0
            else:
                wr = -50.0

            williams_r_values.append(wr)

        # Compute correlation
        williams_r_array = np.array(williams_r_values)
        if np.std(williams_r_array) > 1e-10:
            corr, p_val = pearsonr(williams_r_array, y_numeric)
        else:
            corr, p_val = 0.0, 1.0

        results.append((period, abs(corr), p_val))

    # Display results
    print(f"\n{'Period':<10} {'|Correlation|':>15} {'P-value':>12} {'Assessment':<30}")
    print("─" * 80)

    best_period = max(results, key=lambda x: x[1])

    for period, corr, p_val in results:
        is_best = " ← BEST" if period == best_period[0] else ""
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        print(f"{period:<10} {corr:>15.6f} {p_val:>12.6f} {sig:<30}{is_best}")

    print(f"\n{'─' * 80}")
    print(f"OPTIMAL WILLIAMS %R PERIOD: {best_period[0]} bars")
    print(f"{'─' * 80}")

    # Check if shorter period is better
    short_periods = [p for p, c, pv in results if p <= 7]
    long_periods = [p for p, c, pv in results if p >= 10]

    avg_short_corr = np.mean([c for p, c, pv in results if p <= 7])
    avg_long_corr = np.mean([c for p, c, pv in results if p >= 10])

    print(f"\nShort periods (≤7): Avg |corr| = {avg_short_corr:.6f}")
    print(f"Long periods (≥10): Avg |corr| = {avg_long_corr:.6f}")

    if avg_short_corr > avg_long_corr * 1.1:
        print(f"\n✓ SHORT PERIODS WIN")
        print(f"✓ Shorter lookback preserves 6-bar pattern signal")
        print(f"✓ Recommendation: Use Williams %R period ≤ {best_period[0]} bars")
    elif avg_long_corr > avg_short_corr * 1.1:
        print(f"\n✓ LONG PERIODS WIN")
        print(f"✓ Longer lookback smooths noise effectively")
        print(f"✓ Recommendation: Keep Williams %R period ≥ {best_period[0]} bars")
    else:
        print(f"\n→ NEUTRAL")
        print(f"→ Period choice has minimal impact")


def main():
    """Run Phase 3 forensic audit."""
    print("=" * 80)
    print(f"{'PHASE 3: AVERAGING/SMOOTHING DETECTION':^80}")
    print("=" * 80)

    # Phase 3.1: Detect smoothing operations
    smoothing_found = detect_smoothing_operations()

    # Load data
    print(f"\nLoading data...")
    df = pd.read_parquet('data/processed/train.parquet')

    X_raw = np.array([np.array([np.array(bar) for bar in sample]) for sample in df['features']])
    y = df['label'].values
    expansion_start = df['expansion_start'].values
    expansion_end = df['expansion_end'].values

    # Phase 3.2: Compare raw vs smoothed
    compare_raw_vs_smoothed(X_raw, y, expansion_start, expansion_end)

    # Phase 3.3: Williams %R period analysis
    analyze_williams_r_period(X_raw, y, expansion_start, expansion_end)

    print("\n" + "=" * 80)
    print(f"{'PHASE 3 COMPLETE':^80}")
    print("=" * 80)
    print("\nKey Findings:")
    print("  - Identified all smoothing operations in feature extraction")
    print("  - Compared raw vs smoothed feature correlations")
    print("  - Tested optimal Williams %R period for 6-bar patterns")
    print("\nNext: Run Phase 4 (forensic_audit_pt4_regions.py) for window verification")


if __name__ == '__main__':
    main()
