#!/usr/bin/env python3
"""
Boundary Regression for Expansion Span Detection.

**Paradigm Shift**: Instead of classifying 22,050 timesteps (7.1% positive),
predict 420 boundary positions (210 starts + 210 ends) directly.

Objective: MAE < 5 bars for both start and end positions.
This is the correct formulation for reverse engineering expansion boundaries.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor


def extract_boundary_features(ohlc_array: np.ndarray) -> np.ndarray:
    """
    Extract features that describe temporal patterns within the window.

    Focus on patterns that might correlate with boundary positions:
    - Momentum indicators
    - Volatility profiles
    - Return distributions
    - Position-aware features (leverage the ICT structure)

    Args:
        ohlc_array: (105, 4) OHLC data

    Returns:
        Feature vector (fixed length)
    """
    opens = ohlc_array[:, 0]
    highs = ohlc_array[:, 1]
    lows = ohlc_array[:, 2]
    closes = ohlc_array[:, 3]

    features = []

    # 1. Momentum indicators (bar-to-bar changes)
    returns = np.diff(closes)  # (104,)
    features.extend(
        [
            np.percentile(returns, 10),
            np.percentile(returns, 50),
            np.percentile(returns, 90),
            returns.std(),
            returns.mean(),
        ]
    )

    # 2. Cumulative return profile (where does price move?)
    cum_returns = np.cumsum(returns)
    features.extend(
        [
            np.percentile(cum_returns, 10),
            np.percentile(cum_returns, 50),
            np.percentile(cum_returns, 90),
            cum_returns[-1],  # Total return
        ]
    )

    # 3. Volatility profile (ranges per 10-bar segments)
    ranges = highs - lows
    segment_size = 10
    n_segments = len(ranges) // segment_size
    segment_vols = []
    for i in range(n_segments):
        segment = ranges[i * segment_size : (i + 1) * segment_size]
        segment_vols.append(segment.mean())

    features.extend(segment_vols)  # 10 values

    # 4. Global statistics
    features.extend(
        [
            closes.std(),
            closes.ptp(),  # Peak-to-peak (range)
            (closes[-1] - closes[0]) / closes[0] if closes[0] > 0 else 0.0,  # Total return
        ]
    )

    # 5. Body/wick statistics (candle patterns)
    bodies = np.abs(closes - opens)
    upper_wicks = highs - np.maximum(opens, closes)
    lower_wicks = np.minimum(opens, closes) - lows

    features.extend(
        [
            bodies.mean(),
            bodies.std(),
            upper_wicks.mean(),
            lower_wicks.mean(),
        ]
    )

    # 6. Streak detection (consecutive up/down bars)
    streaks = []
    current_streak = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            current_streak = current_streak + 1 if current_streak > 0 else 1
        elif closes[i] < closes[i - 1]:
            current_streak = current_streak - 1 if current_streak < 0 else -1
        else:
            current_streak = 0
        streaks.append(abs(current_streak))

    features.extend(
        [
            max(streaks) if streaks else 0,
            np.mean(streaks) if streaks else 0,
        ]
    )

    # Total features: 5 + 4 + 10 + 3 + 4 + 2 = 28 features
    return np.array(features)


def create_boundary_dataset(df: pd.DataFrame):
    """
    Create dataset for boundary regression.

    Args:
        df: Parquet data with 'features' (105 x 4 OHLC) + expansion_start/end

    Returns:
        X: Feature matrix (N, n_features)
        y: Boundary targets (N, 2) - [start_position, end_position]
        window_ids: Window IDs
    """
    X_list = []
    y_start_list = []
    y_end_list = []
    window_ids = []

    for idx, row in df.iterrows():
        # Extract OHLC array (105 x 4)
        ohlc_array = np.array([np.array(x) for x in row["features"]])

        # Extract boundary features
        features = extract_boundary_features(ohlc_array)
        X_list.append(features)

        # Target: boundary positions
        y_start_list.append(int(row["expansion_start"]))
        y_end_list.append(int(row["expansion_end"]))

        window_ids.append(row.get("window_id", idx))

    X = np.array(X_list)
    y = np.column_stack([y_start_list, y_end_list])

    return X, y, window_ids


def train_boundary_regressor(X_train, y_train, X_val, y_val):
    """
    Train multi-output Random Forest regressor.

    Args:
        X_train: Training features (N x n_features)
        y_train: Training targets (N x 2) - [start, end]
        X_val: Validation features
        y_val: Validation targets

    Returns:
        Trained model
    """
    # Multi-output Random Forest
    base_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base_model)

    print("\n🌳 Training Multi-Output Random Forest...")
    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # Evaluate
    print("\n📊 Training Performance:")
    print(f"   Start MAE: {mean_absolute_error(y_train[:, 0], y_train_pred[:, 0]):.2f} bars")
    print(f"   End MAE: {mean_absolute_error(y_train[:, 1], y_train_pred[:, 1]):.2f} bars")
    print(f"   Start R²: {r2_score(y_train[:, 0], y_train_pred[:, 0]):.4f}")
    print(f"   End R²: {r2_score(y_train[:, 1], y_train_pred[:, 1]):.4f}")

    print("\n📊 Validation Performance:")
    mae_start = mean_absolute_error(y_val[:, 0], y_val_pred[:, 0])
    mae_end = mean_absolute_error(y_val[:, 1], y_val_pred[:, 1])
    r2_start = r2_score(y_val[:, 0], y_val_pred[:, 0])
    r2_end = r2_score(y_val[:, 1], y_val_pred[:, 1])

    print(f"   Start MAE: {mae_start:.2f} bars")
    print(f"   End MAE: {mae_end:.2f} bars")
    print(f"   Start R²: {r2_start:.4f}")
    print(f"   End R²: {r2_end:.4f}")

    # Success criteria
    print("\n✅ Success Criteria (MAE < 5 bars):")
    print(f"   Start: {'✅ PASS' if mae_start < 5 else '❌ FAIL'}")
    print(f"   End: {'✅ PASS' if mae_end < 5 else '❌ FAIL'}")

    return model, {
        "mae_start": mae_start,
        "mae_end": mae_end,
        "r2_start": r2_start,
        "r2_end": r2_end,
    }


def generate_pseudo_labels_from_boundaries(df: pd.DataFrame, model, X):
    """
    Generate pseudo-labels using predicted boundaries.

    For each window:
    1. Predict start and end positions
    2. Create soft mask with uncertainty buffer
    3. Save as pseudo-labels

    Args:
        df: Original dataframe
        model: Trained boundary regressor
        X: Feature matrix

    Returns:
        DataFrame with pseudo-labels
    """
    print("\n🔄 Generating pseudo-labels from predicted boundaries...")

    # Predict boundaries for all windows
    y_pred = model.predict(X)

    pseudo_labels = []

    for i in range(len(df)):
        start_pred = int(np.clip(y_pred[i, 0], 0, 104))
        end_pred = int(np.clip(y_pred[i, 1], 0, 104))

        # Create soft mask with uncertainty buffer (±2 bars)
        soft_mask = np.zeros(105)

        # Core region (high confidence)
        core_start = max(0, start_pred)
        core_end = min(104, end_pred)
        if core_start <= core_end:
            soft_mask[core_start : core_end + 1] = 1.0

        # Uncertainty buffer (gradient fade)
        buffer = 2
        for offset in range(1, buffer + 1):
            # Before start
            if start_pred - offset >= 0:
                soft_mask[start_pred - offset] = max(
                    soft_mask[start_pred - offset], 0.5 - offset * 0.25
                )
            # After end
            if end_pred + offset <= 104:
                soft_mask[end_pred + offset] = max(
                    soft_mask[end_pred + offset], 0.5 - offset * 0.25
                )

        pseudo_labels.append(soft_mask)

    # Create DataFrame
    pseudo_df = pd.DataFrame(
        {
            "window_id": df["window_id"],
            "predicted_start": y_pred[:, 0],
            "predicted_end": y_pred[:, 1],
            "true_start": df["expansion_start"],
            "true_end": df["expansion_end"],
            "pseudo_mask": pseudo_labels,
        }
    )

    return pseudo_df


def main():
    parser = argparse.ArgumentParser(description="Boundary Regression for Expansion Spans")
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/labeled/train_latest_overlaps_v2.parquet",
        help="Path to labeled parquet data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/boundary_regression",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Boundary Regression for Expansion Detection")
    print("=" * 60)

    # 1. Load data
    print(f"\n📥 Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"   Loaded {len(df)} windows")
    print(f"   Targets: {len(df) * 2} boundary positions (start + end)")

    # 2. Create boundary dataset
    print("\n🔧 Extracting boundary features (28 features per window)...")
    X, y, window_ids = create_boundary_dataset(df)
    print(f"   Features shape: {X.shape}")
    print(f"   Targets shape: {y.shape}")
    print(f"   Start range: [{y[:, 0].min():.0f}, {y[:, 0].max():.0f}]")
    print(f"   End range: [{y[:, 1].min():.0f}, {y[:, 1].max():.0f}]")

    # 3. Train/val split (by windows)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed)

    print("\n📊 Splitting (80/20)...")
    print(f"   Train: {len(X_train)} windows")
    print(f"   Val: {len(X_val)} windows")

    # 4. Train boundary regressor
    model, metrics = train_boundary_regressor(X_train, y_train, X_val, y_val)

    # 5. Save model
    import pickle

    model_file = output_dir / "boundary_regressor.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to {model_file}")

    # 6. Generate pseudo-labels
    pseudo_df = generate_pseudo_labels_from_boundaries(df, model, X)

    pseudo_file = output_dir / "pseudo_labels.parquet"
    pseudo_df.to_parquet(pseudo_file, index=False)
    print(f"💾 Pseudo-labels saved to {pseudo_file}")

    # 7. Analyze errors
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    errors_start = np.abs(pseudo_df["predicted_start"] - pseudo_df["true_start"])
    errors_end = np.abs(pseudo_df["predicted_end"] - pseudo_df["true_end"])

    print("Start position errors:")
    print(f"   Mean: {errors_start.mean():.2f} bars")
    print(f"   Median: {errors_start.median():.2f} bars")
    print(f"   90th percentile: {errors_start.quantile(0.9):.2f} bars")
    print(f"   Max: {errors_start.max():.2f} bars")

    print("\nEnd position errors:")
    print(f"   Mean: {errors_end.mean():.2f} bars")
    print(f"   Median: {errors_end.median():.2f} bars")
    print(f"   90th percentile: {errors_end.quantile(0.9):.2f} bars")
    print(f"   Max: {errors_end.max():.2f} bars")

    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Windows processed: {len(df)}")
    print(f"Validation MAE (start): {metrics['mae_start']:.2f} bars")
    print(f"Validation MAE (end): {metrics['mae_end']:.2f} bars")
    print(f"Validation R² (start): {metrics['r2_start']:.4f}")
    print(f"Validation R² (end): {metrics['r2_end']:.4f}")

    success = metrics["mae_start"] < 5 and metrics["mae_end"] < 5
    print(f"\n{'✅ SUCCESS' if success else '⚠️  NEEDS IMPROVEMENT'}: MAE < 5 bars threshold")

    if not success:
        print("\n💡 Recommendations:")
        print("   - Try Option 2 (changepoint detection) if MAE > 10")
        print("   - Add more features (zigzag swings, ATR-normalized)")
        print("   - Consider deep learning (BiLSTM encoder)")

    # Save summary
    summary = {
        "n_windows": len(df),
        "val_mae_start": float(metrics["mae_start"]),
        "val_mae_end": float(metrics["mae_end"]),
        "val_r2_start": float(metrics["r2_start"]),
        "val_r2_end": float(metrics["r2_end"]),
        "success": success,
        "error_stats": {
            "start_mean": float(errors_start.mean()),
            "start_median": float(errors_start.median()),
            "end_mean": float(errors_end.mean()),
            "end_median": float(errors_end.median()),
        },
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to {summary_file}")

    print("\n✅ Boundary regression complete!")


if __name__ == "__main__":
    main()
