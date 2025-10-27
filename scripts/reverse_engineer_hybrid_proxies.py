#!/usr/bin/env python3
"""
Hybrid Proxy-Tree Distillation for Reverse Engineering Expansion Spans.

Strategy: Blend ChatGPT's 4 arithmetic proxies (soft probs from OHLC) with tree
fitting on 210 marks. Targets 0.20+ correlation by non-linearly adapting
forward-focused proxies to descriptive labels.

4 ChatGPT Arithmetic Proxies:
1. Multi-TF Blip: body/range spikes + SMA alignment
2. Heikin-Ashi Trend: HA streak + liquidity bonus
3. Shadow-Vol: wick/delta >2x ATR (early blip detector)
4. Consolidation Detector: Low volatility + small bodies

Random Forest distills these proxies + simple features (position, momentum)
to match your manual marks.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split


# Proxy 1: Multi-TF Blip (body/range spikes + SMA alignment)
def compute_multitf_blip(ohlc: np.ndarray, timestep: int) -> float:
    """
    Detect blips: sudden price expansion with SMA alignment.

    Args:
        ohlc: (105, 4) OHLC array
        timestep: Current timestep (0-104)

    Returns:
        Blip probability [0, 1]
    """
    if timestep < 20:  # Need SMA context
        return 0.0

    opens, highs, lows, closes = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]

    # Current bar metrics
    body = abs(closes[timestep] - opens[timestep])
    rng = highs[timestep] - lows[timestep]
    if rng == 0:
        return 0.0

    body_pct = body / rng

    # Average range over last 20 bars
    avg_range = np.mean(highs[timestep - 20 : timestep] - lows[timestep - 20 : timestep])
    if avg_range == 0:
        return 0.0

    range_spike = rng / avg_range

    # SMA alignment (20-period)
    sma_20 = np.mean(closes[timestep - 20 : timestep])
    above_sma = 1.0 if closes[timestep] > sma_20 else 0.5

    # Blip formula: range_spike × body_pct × SMA_alignment
    # Threshold: range_spike > 1.5, body_pct > 0.5
    blip_score = 0.0
    if range_spike > 1.5 and body_pct > 0.5:
        blip_score = min(1.0, (range_spike - 1.0) * body_pct * above_sma)

    return blip_score


# Proxy 2: Heikin-Ashi Trend (HA streak + liquidity bonus)
def compute_ha_trend(ohlc: np.ndarray, timestep: int) -> float:
    """
    Detect sustained trend using Heikin-Ashi candles.

    Args:
        ohlc: (105, 4) OHLC array
        timestep: Current timestep (0-104)

    Returns:
        Trend probability [0, 1]
    """
    if timestep < 5:
        return 0.0

    opens, highs, lows, closes = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]

    # Compute Heikin-Ashi for last 5 bars
    ha_close = (
        opens[timestep - 5 : timestep + 1]
        + highs[timestep - 5 : timestep + 1]
        + lows[timestep - 5 : timestep + 1]
        + closes[timestep - 5 : timestep + 1]
    ) / 4.0

    # Streak: consecutive HA bars in same direction
    streak = 0
    for i in range(1, len(ha_close)):
        if ha_close[i] > ha_close[i - 1]:
            streak += 1
        else:
            break

    # Liquidity bonus: high volume proxy (use range as proxy)
    avg_range = np.mean(highs[timestep - 10 : timestep] - lows[timestep - 10 : timestep])
    current_range = highs[timestep] - lows[timestep]
    liq_bonus = 1.0 if current_range > avg_range * 1.2 else 0.8

    # HA trend score: streak × liq_bonus, normalized
    ha_score = min(1.0, (streak / 5.0) * liq_bonus)

    return ha_score


# Proxy 3: Shadow-Vol (wick/delta >2x ATR) - Early blip detector
def compute_shadow_vol(ohlc: np.ndarray, timestep: int) -> float:
    """
    Detect early expansion blips via extreme wick/delta relative to ATR.

    Args:
        ohlc: (105, 4) OHLC array
        timestep: Current timestep (0-104)

    Returns:
        Shadow-vol probability [0, 1]
    """
    if timestep < 14:  # Need ATR context
        return 0.0

    opens, highs, lows, closes = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]

    # Compute ATR (14-period)
    tr_list = []
    for i in range(timestep - 14, timestep):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]) if i > 0 else 0,
            abs(lows[i] - closes[i - 1]) if i > 0 else 0,
        )
        tr_list.append(tr)

    atr = np.mean(tr_list)
    if atr == 0:
        return 0.0

    # Current bar metrics
    body = abs(closes[timestep] - opens[timestep])
    upper_wick = highs[timestep] - max(opens[timestep], closes[timestep])
    lower_wick = min(opens[timestep], closes[timestep]) - lows[timestep]

    # Shadow-vol: max(upper_wick, lower_wick, body) / ATR
    max_component = max(upper_wick, lower_wick, body)
    shadow_vol_ratio = max_component / atr

    # Threshold: >2x ATR
    shadow_score = 0.0
    if shadow_vol_ratio > 2.0:
        shadow_score = min(1.0, (shadow_vol_ratio - 2.0) / 2.0)  # Normalize to [0, 1]

    return shadow_score


# Proxy 4: Consolidation Detector (low volatility + small bodies)
def compute_consolidation(ohlc: np.ndarray, timestep: int) -> float:
    """
    Detect consolidation: low volatility, small bodies, time since swing.

    Returns HIGH values during consolidations (inverse of expansion).

    Args:
        ohlc: (105, 4) OHLC array
        timestep: Current timestep (0-104)

    Returns:
        Consolidation probability [0, 1]
    """
    if timestep < 10:
        return 0.0

    opens, highs, lows, closes = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]

    # Average range over last 10 bars
    avg_range = np.mean(highs[timestep - 10 : timestep] - lows[timestep - 10 : timestep])
    if avg_range == 0:
        return 0.0

    # Current range relative to average
    current_range = highs[timestep] - lows[timestep]
    range_ratio = current_range / avg_range

    # Body size
    body = abs(closes[timestep] - opens[timestep])
    body_pct = body / current_range if current_range > 0 else 0.0

    # Consolidation score: inverse of volatility and body
    # HIGH when range_ratio < 0.8 and body_pct < 0.3
    consol_score = 0.0
    if range_ratio < 0.8 and body_pct < 0.3:
        consol_score = (1.0 - range_ratio) * (1.0 - body_pct)

    return consol_score


def engineer_hybrid_features(ohlc: np.ndarray, timestep: int) -> dict:
    """
    Compute 4 proxies + 2 simple features per timestep.

    Args:
        ohlc: (105, 4) OHLC array
        timestep: Current timestep (0-104)

    Returns:
        Dict of 6 features: 4 proxies + position + momentum_3
    """
    # 4 ChatGPT proxies
    multitf_blip = compute_multitf_blip(ohlc, timestep)
    ha_trend = compute_ha_trend(ohlc, timestep)
    shadow_vol = compute_shadow_vol(ohlc, timestep)
    consolidation = compute_consolidation(ohlc, timestep)

    # Simple features (from v1)
    position = timestep / 104.0  # Normalized position in window

    closes = ohlc[:, 3]
    if timestep >= 3:
        momentum_3 = (
            (closes[timestep] - closes[timestep - 3]) / closes[timestep - 3]
            if closes[timestep - 3] > 0
            else 0.0
        )
    else:
        momentum_3 = 0.0

    return {
        "multitf_blip": multitf_blip,
        "ha_trend": ha_trend,
        "shadow_vol": shadow_vol,
        "consolidation": consolidation,
        "position": position,
        "momentum_3": momentum_3,
    }


def create_hybrid_dataset(df: pd.DataFrame):
    """
    Create timestep-level dataset with 6 hybrid features.

    Args:
        df: Parquet data with 'features' (105 x 4 OHLC) + expansion_start/end

    Returns:
        X: Features array (N_windows * 105, 6)
        y: Labels array (N_windows * 105,)
        window_ids: Window IDs for each timestep
        feature_names: List of 6 feature names
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
            feats = engineer_hybrid_features(ohlc_array, t)
            X_list.append(list(feats.values()))

            # Label: 1 if in-span, 0 if out-of-span
            label = 1 if start <= t <= end else 0
            y_list.append(label)

            window_ids.append(row.get("window_id", idx))

    feature_names = [
        "multitf_blip",
        "ha_trend",
        "shadow_vol",
        "consolidation",
        "position",
        "momentum_3",
    ]

    return np.array(X_list), np.array(y_list), window_ids, feature_names


def fit_random_forest_with_grid_search(X_train, y_train, X_val, y_val):
    """
    Fit Random Forest with grid-search.

    Args:
        X_train: Training features (N x 6)
        y_train: Training labels (N,)
        X_val: Validation features
        y_val: Validation labels

    Returns:
        Best model, grid search results
    """
    param_grid = {
        "n_estimators": [50],
        "max_depth": [4, 5, 6],
        "min_samples_split": [10, 20],
        "min_samples_leaf": [5, 10],
        "class_weight": ["balanced"],
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1)

    grid_search.fit(X_train, y_train)

    # Evaluate best model
    best_model = grid_search.best_estimator_
    y_val_pred = best_model.predict(X_val)

    print(f"\n✅ Best params: {grid_search.best_params_}")
    print(f"Best CV F1: {grid_search.best_score_:.4f}")
    print(f"Val F1: {f1_score(y_val, y_val_pred):.4f}")

    return best_model, grid_search


def compute_correlation(y_true, y_pred_proba):
    """Compute Pearson correlation. Target: 0.20+"""
    corr = np.corrcoef(y_true, y_pred_proba)[0, 1]
    return corr


def main():
    parser = argparse.ArgumentParser(
        description="Hybrid Proxy-Tree Distillation for Reverse Engineering"
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
        default="artifacts/reverse_engineering_hybrid",
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Hybrid Proxy-Tree Distillation")
    print("=" * 60)

    # 1. Load data
    print(f"\n📥 Loading data from {args.data}...")
    df = pd.read_parquet(args.data)
    print(f"   Loaded {len(df)} windows")

    # 2. Create hybrid dataset
    print("\n🔧 Computing 4 ChatGPT proxies + 2 simple features per timestep...")
    print("   Proxies: Multi-TF Blip, HA Trend, Shadow-Vol, Consolidation")
    print("   Simple: position, momentum_3")
    X, y, window_ids, feature_names = create_hybrid_dataset(df)
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    print(f"   In-span ratio: {y.mean():.4f} ({y.sum()} / {len(y)})")

    # 3. Split by windows
    unique_windows = df["window_id"].unique()
    train_windows, val_windows = train_test_split(
        unique_windows, test_size=0.2, random_state=args.seed
    )

    train_mask = np.array([wid in train_windows for wid in window_ids])
    val_mask = ~train_mask

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]

    print("\n📊 Splitting by windows (80/20)...")
    print(f"   Train: {train_mask.sum()} timesteps from {len(train_windows)} windows")
    print(f"   Val: {val_mask.sum()} timesteps from {len(val_windows)} windows")

    # 4. Fit Random Forest
    print("\n🌳 Fitting Random Forest with grid-search...")
    best_model, grid_search = fit_random_forest_with_grid_search(X_train, y_train, X_val, y_val)

    # 5. Feature importance
    print("\n📊 Feature Importance:")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    for i, idx in enumerate(indices):
        print(f"   {i + 1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # 6. Correlation
    y_val_proba = best_model.predict_proba(X_val)[:, 1]
    corr = compute_correlation(y_val, y_val_proba)
    print(f"\n📈 Validation Correlation: {corr:.4f} (target: 0.20+)")

    if corr >= 0.20:
        print("   ✅ Target achieved!")
    else:
        print(f"   ⚠️  Below target (v1: 0.1056, hybrid: {corr:.4f})")

    # 7. Save model
    import pickle

    model_file = output_dir / "rf_model.pkl"
    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    print(f"💾 Model saved to {model_file}")

    # 8. Classification report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Validation)")
    print("=" * 60)
    print(classification_report(y_val, best_model.predict(X_val)))

    # 9. Generate pseudo-labels
    print("\n🔄 Generating pseudo-labels for full dataset...")
    y_full_proba = best_model.predict_proba(X)[:, 1]

    pseudo_labels = y_full_proba.reshape(len(df), 105)

    pseudo_df = pd.DataFrame(
        {
            "window_id": df["window_id"],
            "pseudo_mask": list(pseudo_labels),
        }
    )
    pseudo_file = output_dir / "pseudo_labels.parquet"
    pseudo_df.to_parquet(pseudo_file, index=False)
    print(f"💾 Pseudo-labels saved to {pseudo_file}")

    # 10. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Windows processed: {len(df)}")
    print(f"Timesteps total: {len(X)}")
    print(f"Validation correlation: {corr:.4f}")
    print(f"Improvement over v1: {corr / 0.1056:.2f}x")
    print(f"Best tree depth: {best_model.max_depth}")
    print(f"Pseudo-label std: {y_full_proba.std():.4f} (target: >0.05)")

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

    print("\n✅ Hybrid proxy-tree distillation complete!")


if __name__ == "__main__":
    main()
