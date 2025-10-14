#!/usr/bin/env python3
"""Test multi-scale feature engineering and compare with classical features."""

import sys
sys.path.insert(0, '/Users/jack/projects/moola/src')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from moola.features.price_action_features import engineer_multiscale_features, engineer_classical_features

print("=" * 70)
print("MULTI-SCALE FEATURE ENGINEERING TEST")
print("=" * 70)

# Load data
print("\nLoading data...")
df = pd.read_parquet('data/processed/train.parquet')

X = np.array([np.array([np.array(bar) for bar in features]) for features in df['features']])
y = df['label'].values
expansion_start = df['expansion_start'].values
expansion_end = df['expansion_end'].values

# Encode labels for XGBoost
y_encoded = (y == 'consolidation').astype(int)

print(f"Dataset: {len(df)} samples")
print(f"X shape: {X.shape}")
print(f"Pattern lengths: mean={np.mean(expansion_end - expansion_start):.1f}, median={np.median(expansion_end - expansion_start):.0f}")

# Extract both feature sets
print("\n" + "=" * 70)
print("FEATURE EXTRACTION")
print("=" * 70)

print("\n1. Multi-scale features (NEW)...")
X_multiscale = engineer_multiscale_features(X, expansion_start, expansion_end)
print(f"   Shape: {X_multiscale.shape}")
print(f"   Features: {X_multiscale.shape[1]}")

print("\n2. Classical features (OLD - with expansion indices)...")
X_classical = engineer_classical_features(X, expansion_start, expansion_end)
print(f"   Shape: {X_classical.shape}")
print(f"   Features: {X_classical.shape[1]}")

print("\n3. Classical features (OLD - without expansion indices)...")
X_classical_no_exp = engineer_classical_features(X, expansion_start=None, expansion_end=None)
print(f"   Shape: {X_classical_no_exp.shape}")
print(f"   Features: {X_classical_no_exp.shape[1]}")

# Feature statistics
print("\n" + "=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

print("\nMulti-scale features:")
print(f"  Mean: {X_multiscale.mean(axis=0)[:10]}")
print(f"  Std:  {X_multiscale.std(axis=0)[:10]}")

# Feature correlation with labels
print("\n" + "=" * 70)
print("FEATURE-LABEL CORRELATIONS")
print("=" * 70)

y_numeric = y_encoded

print("\nMulti-scale features (top 10):")
correlations_multi = []
for i in range(X_multiscale.shape[1]):
    corr = np.corrcoef(X_multiscale[:, i], y_numeric)[0, 1]
    correlations_multi.append(abs(corr))
    if i < 10:
        print(f"  Feature {i+1:2d}: {abs(corr):.4f}")

print(f"\n  Max |correlation|: {max(correlations_multi):.4f}")
print(f"  Mean |correlation|: {np.mean(correlations_multi):.4f}")
print(f"  Features >0.15: {sum(c > 0.15 for c in correlations_multi)}/{len(correlations_multi)}")

print("\nClassical features (with expansion):")
correlations_classical = []
for i in range(X_classical.shape[1]):
    corr = np.corrcoef(X_classical[:, i], y_numeric)[0, 1]
    correlations_classical.append(abs(corr))
    if i < 10:
        print(f"  Feature {i+1:2d}: {abs(corr):.4f}")

print(f"\n  Max |correlation|: {max(correlations_classical):.4f}")
print(f"  Mean |correlation|: {np.mean(correlations_classical):.4f}")

# Quick model tests
print("\n" + "=" * 70)
print("MODEL PERFORMANCE COMPARISON (5-fold CV)")
print("=" * 70)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1337)

# LogReg with multi-scale (NORMALIZED)
print("\n1. LogisticRegression + Multi-scale features (normalized)")
scores_multi = []
for train_idx, val_idx in skf.split(X_multiscale, y):
    X_train, X_val = X_multiscale[train_idx], X_multiscale[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = LogisticRegression(random_state=1337, max_iter=2000)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_val_scaled, y_val)
    scores_multi.append(score)

print(f"   Accuracy: {np.mean(scores_multi):.4f} ± {np.std(scores_multi):.4f}")
print(f"   Folds: {[f'{s:.4f}' for s in scores_multi]}")

# LogReg with classical (for comparison)
print("\n2. LogisticRegression + Classical features (with expansion)")
scores_classical = []
for train_idx, val_idx in skf.split(X_classical, y):
    X_train, X_val = X_classical[train_idx], X_classical[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = LogisticRegression(random_state=1337, max_iter=2000)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    scores_classical.append(score)

print(f"   Accuracy: {np.mean(scores_classical):.4f} ± {np.std(scores_classical):.4f}")
print(f"   Folds: {[f'{s:.4f}' for s in scores_classical]}")

# XGBoost with multi-scale
print("\n3. XGBoost + Multi-scale features")
scores_xgb_multi = []
for train_idx, val_idx in skf.split(X_multiscale, y_encoded):
    X_train, X_val = X_multiscale[train_idx], X_multiscale[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    model = XGBClassifier(random_state=1337, n_estimators=100, max_depth=3, learning_rate=0.1, eval_metric='logloss')
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    scores_xgb_multi.append(score)

print(f"   Accuracy: {np.mean(scores_xgb_multi):.4f} ± {np.std(scores_xgb_multi):.4f}")
print(f"   Folds: {[f'{s:.4f}' for s in scores_xgb_multi]}")

# RandomForest with multi-scale (NORMALIZED)
print("\n4. RandomForest + Multi-scale features (normalized)")
scores_rf_multi = []
for train_idx, val_idx in skf.split(X_multiscale, y):
    X_train, X_val = X_multiscale[train_idx], X_multiscale[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(random_state=1337, n_estimators=100, max_depth=5)
    model.fit(X_train_scaled, y_train)
    score = model.score(X_val_scaled, y_val)
    scores_rf_multi.append(score)

print(f"   Accuracy: {np.mean(scores_rf_multi):.4f} ± {np.std(scores_rf_multi):.4f}")
print(f"   Folds: {[f'{s:.4f}' for s in scores_rf_multi]}")

# Summary
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)

improvements = {
    'LogReg': np.mean(scores_multi) - np.mean(scores_classical),
    'XGBoost': np.mean(scores_xgb_multi) - 0.4435,  # Previous best XGB with expansion
    'RandomForest': np.mean(scores_rf_multi) - 0.5391  # Baseline RF
}

print("\nMulti-scale vs Classical/Baseline:")
for model, improvement in improvements.items():
    print(f"  {model:15s}: {improvement:+.4f} ({improvement*100:+.1f}%)")

print(f"\nMax correlation improvement:")
print(f"  Classical: {max(correlations_classical):.4f}")
print(f"  Multi-scale: {max(correlations_multi):.4f}")
print(f"  Improvement: {max(correlations_multi) - max(correlations_classical):+.4f}")

print("\n" + "=" * 70)
print("SUCCESS CRITERIA CHECK")
print("=" * 70)

success_criteria = {
    'Max correlation >0.22': max(correlations_multi) > 0.22,
    'XGBoost >55%': np.mean(scores_xgb_multi) > 0.55,
    'Any model >60%': max(np.mean(scores_multi), np.mean(scores_xgb_multi), np.mean(scores_rf_multi)) > 0.60,
}

for criterion, passed in success_criteria.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {status} {criterion}")

if all(success_criteria.values()):
    print("\n🎯 ALL SUCCESS CRITERIA MET!")
else:
    print("\n⚠️  Some criteria not met yet - needs iteration")

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
