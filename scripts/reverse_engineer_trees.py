#!/usr/bin/env python3
"""
Tree-based reverse engineering of expansion spans.

Uses decision trees to distill non-linear rules from human marks.
Target: 0.20+ correlation (vs 0.017 from linear grid-search).

Approach:
1. Load 210 windows + marks
2. Engineer features per TIMESTEP (not per window)
3. Fit DecisionTree to predict in-span (1) vs out-of-span (0) per timestep
4. Extract IF-THEN rules via export_text
5. Score correlation/F1 on holdout (20%)
6. Augment with jitter σ=0.03 for pseudo-labels
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def engineer_timestep_features(ohlc_array: np.ndarray, timestep: int) -> dict:
    """
    Engineer features for a single timestep within a window.

    Features (9 total):
    - Local OHLC (4): open, high, low, close at this timestep
    - Recent momentum (1): (close[t] - close[t-3]) / close[t-3]
    - Local streak (1): consecutive up/down bars ending at t
    - Local volatility (1): (high[t] - low[t]) / close[t]
    - Position in window (1): t / 105 (normalized position)
    - Price vs window mean (1): close[t] / window_mean_close

    Args:
        ohlc_array: Shape (105, 4) OHLC data
        timestep: Current timestep index (0-104)

    Returns:
        Dict of features for this timestep
    """
    opens = ohlc_array[:, 0]
    highs = ohlc_array[:, 1]
    lows = ohlc_array[:, 2]
    closes = ohlc_array[:, 3]

    # 1. Local OHLC (normalized by close)
    c = closes[timestep]
    o_norm = opens[timestep] / c if c > 0 else 1.0
    h_norm = highs[timestep] / c if c > 0 else 1.0
    l_norm = lows[timestep] / c if c > 0 else 1.0
    c_norm = 1.0  # Always 1.0 since we normalize by close

    # 2. Recent momentum (3-bar)
    if timestep >= 3:
        momentum_3 = (
            (closes[timestep] - closes[timestep - 3]) / closes[timestep - 3]
            if closes[timestep - 3] > 0
            else 0.0
        )
    else:
        momentum_3 = 0.0

    # 3. Local streak (consecutive up/down bars ending at this timestep)
    streak = 0
    if timestep > 0:
        direction = 1 if closes[timestep] > closes[timestep - 1] else -1
        for i in range(timestep, 0, -1):
            if closes[i] > closes[i - 1]:
                if direction == 1:
                    streak += 1
                else:
                    break
            elif closes[i] < closes[i - 1]:
                if direction == -1:
                    streak -= 1
                else:
                    break
            else:
                break

    # 4. Local volatility
    vol = (highs[timestep] - lows[timestep]) / closes[timestep] if closes[timestep] > 0 else 0.0

    # 5. Position in window (normalized)
    position = timestep / 104.0  # 0 to 1

    # 6. Price vs window mean
    mean_close = closes.mean()
    price_vs_mean = closes[timestep] / mean_close if mean_close > 0 else 1.0

    return {
        "open_norm": o_norm,
        "high_norm": h_norm,
        "low_norm": l_norm,
        "close_norm": c_norm,
        "momentum_3": momentum_3,
        "streak": streak,
        "volatility": vol,
        "position": position,
        "price_vs_mean": price_vs_mean,
    }


def create_timestep_dataset(df: pd.DataFrame):
    """
    Create timestep-level dataset from window-level data.

    Args:
        df: Parquet data with 'features' (105 x 4 OHLC) + expansion_start/end

    Returns:
        X: Features array (N_windows * 105, 9)
        y: Labels array (N_windows * 105,) - 1 if in-span, 0 if out-of-span
        window_ids: Window IDs for each timestep
    """
    X_list = []
    y_list = []
    window_ids = []

    for idx, row in df.iterrows():
        # Extract OHLC array (105 x 4)
        ohlc_array = np.array([np.array(x) for x in row["features"]])

        # Get span boundaries
        start = int(row["expansion_start"])
        end = int(row["expansion_end"])

        # Engineer features for each timestep
        for t in range(105):
            feats = engineer_timestep_features(ohlc_array, t)
            X_list.append(list(feats.values()))

            # Label: 1 if in-span, 0 if out-of-span
            label = 1 if start <= t <= end else 0
            y_list.append(label)

            window_ids.append(row.get("window_id", idx))

    return np.array(X_list), np.array(y_list), window_ids


def fit_tree_with_grid_search(X_train, y_train, X_val, y_val):
    """
    Fit decision tree with grid-search over depth and min_samples_split.

    Args:
        X_train: Training features (N x 9)
        y_train: Training labels (N,) - binary in-span labels
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Best model, grid search results
    """
    param_grid = {
        "max_depth": [2, 3, 4],
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
        description="Tree-based reverse engineering of expansion spans"
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
        default="artifacts/reverse_engineering",
        help="Output directory for rules and pseudo-labels",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Tree-Based Reverse Engineering (Timestep-Level)")
    print("=" * 60)

    # 1. Load data
    print(f"\n📥 Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"   Loaded {len(df)} windows")

    # 2. Create timestep-level dataset
    print("\n🔧 Creating timestep-level features (9 per timestep)...")
    X, y, window_ids = create_timestep_dataset(df)
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   In-span ratio: {y.mean():.4f} ({y.sum()} / {len(y)})")

    # 3. Split by windows (not timesteps) to avoid leakage
    # Get unique window IDs and split
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
    feature_names = [
        "open_norm",
        "high_norm",
        "low_norm",
        "close_norm",
        "momentum_3",
        "streak",
        "volatility",
        "position",
        "price_vs_mean",
    ]
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
        print("   ⚠️  Below target - consider deeper trees or more features")
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

    # 10. Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Windows processed: {len(df)}")
    print(f"Timesteps total: {len(X)}")
    print(f"Validation correlation: {corr:.4f}")
    print(f"Best tree depth: {best_model.max_depth}")
    print(f"Best min_samples_split: {best_model.min_samples_split}")
    print(f"Pseudo-label std: {y_full_proba.std():.4f} (target: >0.05)")

    # Save summary
    summary = {
        "n_windows": len(df),
        "n_timesteps": len(X),
        "val_correlation": float(corr),
        "best_params": grid_search.best_params_,
        "val_f1": float(f1_score(y_val, best_model.predict(X_val))),
        "pseudo_label_std": float(y_full_proba.std()),
    }
    summary_file = output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Summary saved to {summary_file}")

    print("\n✅ Tree-based reverse engineering complete!")


if __name__ == "__main__":
    main()
