#!/usr/bin/env python3
"""
Tree-based reverse engineering using PRODUCTION features from relativity.py.

Uses the same 12 features that the training pipeline uses:
- Candle shape (6): open_norm, close_norm, body_pct, upper_wick_pct, lower_wick_pct, range_z
- Swing-relative (4): dist_to_prev_SH, dist_to_prev_SL, bars_since_SH_norm, bars_since_SL_norm
- Proxies (2): expansion_proxy, consol_proxy

expansion_proxy and consol_proxy are specifically designed for this task!

Target: 0.20+ correlation (vs 0.1056 from simple features).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.features.relativity import RelativityConfig, build_features


def compute_production_features(ohlc_df: pd.DataFrame) -> np.ndarray:
    """
    Compute 12 production features from raw OHLC using relativity.py.

    Args:
        ohlc_df: DataFrame with columns [open, high, low, close]

    Returns:
        Features array (1, 105, 12) for this single window
    """
    cfg = RelativityConfig(window_length=105)
    X, mask, meta = build_features(ohlc_df, cfg)

    # X shape: (n_windows, 105, 12)
    # We only need the last window (most recent 105 bars)
    return X[-1]  # Shape: (105, 12)


def create_timestep_dataset_with_production_features(df: pd.DataFrame):
    """
    Create timestep-level dataset using production 12 features.

    Args:
        df: Parquet data with 'features' (105 x 4 OHLC) + expansion_start/end

    Returns:
        X: Features array (N_windows * 105, 12)
        y: Labels array (N_windows * 105,) - 1 if in-span, 0 if out-of-span
        window_ids: Window IDs for each timestep
        feature_names: List of 12 feature names
    """
    X_list = []
    y_list = []
    window_ids = []

    for idx, row in df.iterrows():
        # Extract OHLC array (105 x 4)
        ohlc_array = np.array([np.array(x) for x in row["features"]])

        # Convert to DataFrame for relativity.py
        ohlc_df = pd.DataFrame(ohlc_array, columns=["open", "high", "low", "close"])

        # Compute 12 production features
        try:
            features_12d = compute_production_features(ohlc_df)  # Shape: (105, 12)
        except Exception as e:
            print(f"Warning: Failed to compute features for window {idx}: {e}")
            continue

        # Get span boundaries
        start = int(row["expansion_start"])
        end = int(row["expansion_end"])

        # Add each timestep
        for t in range(105):
            X_list.append(features_12d[t])

            # Label: 1 if in-span, 0 if out-of-span
            label = 1 if start <= t <= end else 0
            y_list.append(label)

            window_ids.append(row.get("window_id", idx))

    feature_names = [
        "open_norm",
        "close_norm",
        "body_pct",
        "upper_wick_pct",
        "lower_wick_pct",
        "range_z",
        "dist_to_prev_SH",
        "dist_to_prev_SL",
        "bars_since_SH_norm",
        "bars_since_SL_norm",
        "expansion_proxy",
        "consol_proxy",
    ]

    return np.array(X_list), np.array(y_list), window_ids, feature_names


def fit_tree_with_grid_search(X_train, y_train, X_val, y_val):
    """
    Fit decision tree with grid-search over depth and min_samples_split.

    Args:
        X_train: Training features (N x 12)
        y_train: Training labels (N,) - binary in-span labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Best model, grid search results
    """
    param_grid = {
        "max_depth": [3, 4, 5, 6],  # Deeper trees since we have 12 features
        "min_samples_split": [10, 20, 30],
        "min_samples_leaf": [5, 10, 15],
    }

    clf = DecisionTreeClassifier(random_state=42, class_weight="balanced")

    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)

    # Evaluate best model on validation set
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)

    print(f"\n✅ Best params: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")
    print(f"Val F1: {f1_score(y_val, y_val_pred):.4f}")

    return best_model, grid_search


def extract_rules(model, feature_names):
    """
    Extract IF-THEN rules from fitted decision tree.

    Args:
        model: Fitted DecisionTreeClassifier
        feature_names: List of feature names

    Returns:
        String with human-readable rules
    """
    tree_rules = export_text(model, feature_names=feature_names)
    return tree_rules


def compute_correlation(y_true, y_pred_proba):
    """
    Compute Pearson correlation between true labels and predicted probabilities.

    Target: 0.20+ correlation
    """
    corr = np.corrcoef(y_true, y_pred_proba)[0, 1]
    return corr


def main():
    parser = argparse.ArgumentParser(
        description="Tree-based reverse engineering with production features"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/labeled/train_latest_overlaps_v2.parquet",
        help="Path to labeled parquet data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/reverse_engineering_v2",
        help="Output directory for rules and pseudo-labels",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tree-Based Reverse Engineering v2 (Production Features)")
    print("=" * 60)

    # 1. Load data
    print(f"\n📥 Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"   Loaded {len(df)} windows")

    # 2. Create timestep-level dataset with 12 production features
    print("\n🔧 Computing 12 production features per timestep (relativity.py)...")
    print("   Features: candle_shape(6) + swing_relative(4) + proxies(2)")
    X, y, window_ids, feature_names = create_timestep_dataset_with_production_features(df)
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   In-span ratio: {y.mean():.4f} ({y.sum()} / {len(y)})")

    # 3. Split by windows (not timesteps) to avoid leakage
    unique_windows = df["window_id"].unique()
    train_windows, val_windows = train_test_split(
        unique_windows, test_size=0.2, random_state=args.seed
    )

    # Create masks for train/val split
    train_mask = np.array([wid in train_windows for wid in window_ids])
    val_mask = ~train_mask

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    print("\n📊 Splitting by windows (80/20 - prevents leakage)...")
    print(f"   Train: {train_mask.sum()} timesteps from {len(train_windows)} windows")
    print(f"   Val: {val_mask.sum()} timesteps from {len(val_windows)} windows")

    # 4. Fit tree with grid-search
    print("\n🌳 Fitting decision tree with grid-search...")
    best_model, grid_search = fit_tree_with_grid_search(X_train, y_train, X_val, y_val)

    # 5. Extract rules
    print("\n📜 Extracting IF-THEN rules...")
    rules = extract_rules(best_model, feature_names)
    print("\n" + "=" * 60)
    print("DECISION TREE RULES")
    print("=" * 60)
    print(rules)

    # Save rules to file
    rules_file = output_dir / "tree_rules.txt"
    with open(rules_file, "w") as f:
        f.write(rules)
    print(f"\n💾 Rules saved to {rules_file}")

    # 6. Compute correlation on validation set
    y_val_proba = best_model.predict_proba(X_val)[:, 1]  # Prob of in-span
    corr = compute_correlation(y_val, y_val_proba)
    print(f"\n📈 Validation Correlation: {corr:.4f} (target: 0.20+)")

    if corr < 0.20:
        print(f"   ⚠️  Below target (improved from 0.1056 → {corr:.4f})")
    else:
        print("   ✅ Target achieved!")

    # 7. Save model and predictions
    import pickle

    model_file = output_dir / "tree_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    print(f"💾 Model saved to {model_file}")

    # Save predictions
    preds_df = pd.DataFrame(
        {
            "y_true": y_val,
            "y_pred_proba": y_val_proba,
            "y_pred": best_model.predict(X_val),
        }
    )
    preds_file = output_dir / "val_predictions.csv"
    preds_df.to_csv(preds_file, index=False)
    print(f"💾 Predictions saved to {preds_file}")

    # 8. Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Validation)")
    print("=" * 60)
    print(classification_report(y_val, best_model.predict(X_val)))

    # 9. Generate pseudo-labels for full dataset
    print("\n🔄 Generating pseudo-labels for full dataset...")
    y_full_proba = best_model.predict_proba(X)[:, 1]

    # Reshape to window format (N_windows, 105)
    pseudo_labels = y_full_proba.reshape(len(df), 105)

    # Save pseudo-labels
    pseudo_df = pd.DataFrame(
        {
            "window_id": df["window_id"],
            "pseudo_mask": list(pseudo_labels),  # List of 105 probs per window
        }
    )
    pseudo_file = output_dir / "pseudo_labels.parquet"
    pseudo_df.to_parquet(pseudo_file, index=False)
    print(f"💾 Pseudo-labels saved to {pseudo_file}")

    # 10. Feature importance
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i, idx in enumerate(indices[:5]):  # Top 5
        print(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # 11. Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Windows processed: {len(df)}")
    print(f"Timesteps total: {len(X)}")
    print(f"Validation correlation: {corr:.4f}")
    print(f"Improvement over v1: {corr / 0.1056:.2f}x")
    print(f"Best tree depth: {best_model.max_depth}")
    print(f"Best min_samples_split: {best_model.min_samples_split}")
    print(f"Pseudo-label std: {y_full_proba.std():.4f} (target: >0.05)")

    # Save summary
    summary = {
        "n_windows": len(df),
        "n_timesteps": len(X),
        "val_correlation": float(corr),
        "improvement_over_v1": float(corr / 0.1056),
        "best_params": grid_search.best_params_,
        "val_f1": float(f1_score(y_val, best_model.predict(X_val))),
        "pseudo_label_std": float(y_full_proba.std()),
        "feature_importances": {name: float(imp) for name, imp in zip(feature_names, importances)},
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to {summary_file}")

    print("\n✅ Tree-based reverse engineering v2 complete!")


if __name__ == "__main__":
    main()
