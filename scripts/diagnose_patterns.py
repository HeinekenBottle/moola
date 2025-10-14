#!/usr/bin/env python3
"""Diagnostic script to visualize patterns and feature distributions."""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
print("Loading data...")
df = pd.read_parquet('data/processed/train.parquet')

# Extract features
X = np.array([np.array([np.array(bar) for bar in features]) for features in df['features']])
y = df['label'].values

print(f"\n{'='*70}")
print("DATASET OVERVIEW")
print(f"{'='*70}")
print(f"Total samples: {len(df)}")
print(f"Feature shape: {X.shape}")
print(f"Consolidation: {(y == 'consolidation').sum()} ({(y == 'consolidation').mean()*100:.1f}%)")
print(f"Retracement: {(y == 'retracement').sum()} ({(y == 'retracement').mean()*100:.1f}%)")

# Create output directory
output_dir = Path('artifacts')
output_dir.mkdir(exist_ok=True)

# Split by class
consol_mask = y == 'consolidation'
retrace_mask = y == 'retracement'

X_consol = X[consol_mask]
X_retrace = X[retrace_mask]

# Sample for visualization
np.random.seed(42)
n_samples = min(10, min(len(X_consol), len(X_retrace)))
consol_indices = np.random.choice(len(X_consol), n_samples, replace=False)
retrace_indices = np.random.choice(len(X_retrace), n_samples, replace=False)

X_consol_sample = X_consol[consol_indices]
X_retrace_sample = X_retrace[retrace_indices]

# Plot 1: OHLC patterns (full sequence)
print("\nGenerating pattern comparison plot...")
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
fig.suptitle('Consolidation vs Retracement - Full 105-bar Sequence (OHLC)', fontsize=16)

for i in range(5):
    if i < len(X_consol_sample):
        ax = axes[0, i]
        ohlc = X_consol_sample[i]  # (105, 4)
        bars = np.arange(len(ohlc))

        # Plot close price
        ax.plot(bars, ohlc[:, 3], 'b-', label='Close', linewidth=1.5, alpha=0.8)

        # Fill between high and low
        ax.fill_between(bars, ohlc[:, 2], ohlc[:, 1], alpha=0.2, color='blue')

        # Add vertical line at bar 30 (prediction start)
        ax.axvline(x=30, color='green', linestyle='--', alpha=0.5, label='Prediction Start')

        ax.set_title(f'Consolidation {i+1}', fontsize=12)
        ax.set_xlabel('Bar Index', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

for i in range(5):
    if i < len(X_retrace_sample):
        ax = axes[1, i]
        ohlc = X_retrace_sample[i]  # (105, 4)
        bars = np.arange(len(ohlc))

        # Plot close price
        ax.plot(bars, ohlc[:, 3], 'r-', label='Close', linewidth=1.5, alpha=0.8)

        # Fill between high and low
        ax.fill_between(bars, ohlc[:, 2], ohlc[:, 1], alpha=0.2, color='red')

        # Add vertical line at bar 30
        ax.axvline(x=30, color='green', linestyle='--', alpha=0.5, label='Prediction Start')

        ax.set_title(f'Retracement {i+1}', fontsize=12)
        ax.set_xlabel('Bar Index', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('artifacts/pattern_comparison.png', dpi=150, bbox_inches='tight')
print("✅ Saved: artifacts/pattern_comparison.png")

# Plot 2: Feature statistics per OHLC component
print("\nGenerating feature distribution plots...")
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('OHLC Feature Distributions by Class', fontsize=16)

feature_names = ['Open', 'High', 'Low', 'Close']

for feat_idx, feat_name in enumerate(feature_names):
    ax = axes[feat_idx // 2, feat_idx % 2]

    # Extract feature across all timesteps
    consol_feat = X_consol[:, :, feat_idx].flatten()
    retrace_feat = X_retrace[:, :, feat_idx].flatten()

    # Plot distributions
    ax.hist(consol_feat, bins=50, alpha=0.5, label='Consolidation', color='blue', density=True)
    ax.hist(retrace_feat, bins=50, alpha=0.5, label='Retracement', color='red', density=True)

    ax.set_title(f'{feat_name} Distribution', fontsize=12)
    ax.set_xlabel('Price', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Print statistics
    print(f"\n{feat_name}:")
    print(f"  Consolidation: mean={consol_feat.mean():.2f}, std={consol_feat.std():.2f}")
    print(f"  Retracement:   mean={retrace_feat.mean():.2f}, std={retrace_feat.std():.2f}")

plt.tight_layout()
plt.savefig('artifacts/feature_distributions.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: artifacts/feature_distributions.png")

# Plot 3: Time-series feature engineering
print("\nAnalyzing engineered features...")

# Compute returns and volatility
def compute_features(X):
    """Compute statistical features from OHLC data."""
    features = []

    for sample in X:
        close = sample[:, 3]
        high = sample[:, 1]
        low = sample[:, 2]

        # Returns (log returns)
        returns = np.diff(np.log(close))

        # Volatility (std of returns in windows)
        vol_10 = np.std(returns[-10:]) if len(returns) >= 10 else 0
        vol_20 = np.std(returns[-20:]) if len(returns) >= 20 else 0

        # Range (high-low normalized by close)
        ranges = (high - low) / close
        avg_range = np.mean(ranges)

        # Trend (linear regression slope of close prices)
        x = np.arange(len(close))
        trend = np.polyfit(x, close, 1)[0]

        # Price change in prediction region (bars 30-74)
        if len(close) >= 75:
            pred_region_change = (close[74] - close[30]) / close[30]
        else:
            pred_region_change = 0

        features.append([
            np.mean(returns),      # avg return
            np.std(returns),       # return volatility
            vol_10,                # 10-bar volatility
            vol_20,                # 20-bar volatility
            avg_range,             # avg daily range
            trend,                 # price trend
            pred_region_change,    # change in prediction region
        ])

    return np.array(features)

feat_consol = compute_features(X_consol)
feat_retrace = compute_features(X_retrace)

feature_names_eng = [
    'Avg Return',
    'Return Volatility',
    '10-bar Vol',
    '20-bar Vol',
    'Avg Range',
    'Trend',
    'Pred Region Change'
]

# Compute correlations
print(f"\n{'='*70}")
print("ENGINEERED FEATURE ANALYSIS")
print(f"{'='*70}")

correlations = []
for i, feat_name in enumerate(feature_names_eng):
    # Combine features with labels
    combined = np.concatenate([feat_consol[:, i], feat_retrace[:, i]])
    labels = np.concatenate([np.zeros(len(feat_consol)), np.ones(len(feat_retrace))])

    # Calculate correlation
    corr = np.corrcoef(combined, labels)[0, 1]
    correlations.append(abs(corr))

    print(f"{feat_name:20s}: {corr:+.4f} (abs: {abs(corr):.4f})")

correlations = np.array(correlations)

# Plot feature importance
fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(feature_names_eng, correlations, color='steelblue')
ax.set_xlabel('Absolute Correlation with Label', fontsize=12)
ax.set_title('Engineered Feature Importance', fontsize=14)
ax.grid(True, alpha=0.3, axis='x')

# Add correlation values on bars
for i, (bar, val) in enumerate(zip(bars, correlations)):
    ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('artifacts/feature_importance.png', dpi=150, bbox_inches='tight')
print("\n✅ Saved: artifacts/feature_importance.png")

# Final diagnostic summary
print(f"\n{'='*70}")
print("DIAGNOSTIC SUMMARY")
print(f"{'='*70}")
print(f"Max correlation: {correlations.max():.4f}")
print(f"Features with corr >0.10: {(correlations > 0.10).sum()}")
print(f"Features with corr >0.20: {(correlations > 0.20).sum()}")
print(f"Features with corr >0.30: {(correlations > 0.30).sum()}")

print(f"\n{'='*70}")
print("DIAGNOSIS")
print(f"{'='*70}")

if correlations.max() < 0.10:
    print("❌ CRITICAL: No features correlate >0.10 with label")
    print("   → Task is fundamentally difficult with current features")
    print("   → Patterns may be indistinguishable or features inadequate")
    print("   → Expected accuracy ceiling: 52-57%")
elif correlations.max() < 0.20:
    print("⚠️  WARNING: Weak feature-label relationships (max <0.20)")
    print("   → Need better feature engineering")
    print("   → Current features have minimal predictive power")
    print("   → Expected accuracy ceiling: 55-62%")
elif correlations.max() < 0.30:
    print("⚠️  MODERATE: Some signal present (max <0.30)")
    print("   → Features show weak relationships")
    print("   → Model may struggle but has potential")
    print("   → Expected accuracy ceiling: 60-68%")
else:
    print("✅ GOOD: Strong feature-label relationships (max >=0.30)")
    print("   → Features capture meaningful differences")
    print("   → Model training issue, not feature issue")
    print("   → Expected accuracy ceiling: 65-75%+")

print(f"\n{'='*70}")
print("RECOMMENDATION")
print(f"{'='*70}")

if correlations.max() < 0.15:
    print("1. Accept 52-58% as realistic ceiling for this task")
    print("2. Focus on feature engineering (technical indicators, pattern matching)")
    print("3. Consider ensemble methods (already at 54.78%)")
    print("4. Semi-supervised learning (FixMatch) won't help with no signal")
elif correlations.max() < 0.25:
    print("1. Engineer better features (momentum, volume, pattern features)")
    print("2. Try simpler models (XGBoost with engineered features)")
    print("3. Semi-supervised learning has limited potential (<5% gain)")
else:
    print("1. Debug model training (current ~52% is far below potential)")
    print("2. Check for bugs in data loading or preprocessing")
    print("3. Try different model architectures")
    print("4. Semi-supervised learning could provide 5-10% boost")

print(f"\n{'='*70}")
