#!/usr/bin/env python3
"""Deep model diagnostic - examine what each model actually sees."""

import sys
import pandas as pd
import numpy as np
import torch

# Add moola to path
sys.path.insert(0, '/Users/jack/projects/moola/src')

from moola.models.cnn_transformer import CnnTransformerModel
from moola.features.price_action_features import engineer_classical_features

# Load one sample
print("Loading data...")
df = pd.read_parquet('data/processed/train.parquet')
sample = df.iloc[0]

print("="*60)
print("RAW DATA STRUCTURE")
print("="*60)
print(f"Total bars: {len(sample['features'])} timesteps × {len(sample['features'][0])} features")
print(f"Label: {sample['label']}")
if 'expansion_start' in df.columns:
    print(f"Expansion: [{sample['expansion_start']}:{sample['expansion_end']}] ({sample['expansion_end'] - sample['expansion_start']} bars)")

# Reconstruct OHLC from features
X_raw = np.array([np.array(bar) for bar in sample['features']])  # [105, 4]
print(f"OHLC shape: {X_raw.shape}")
print(f"Sample values (first 3 bars):")
for i in range(min(3, len(X_raw))):
    print(f"  Bar {i}: O={X_raw[i,0]:.2f} H={X_raw[i,1]:.2f} L={X_raw[i,2]:.2f} C={X_raw[i,3]:.2f}")

# Check each model
print("\n" + "="*60)
print("1. CNN-TRANSFORMER (Deep Learning)")
print("="*60)

print(f"Input shape: {X_raw.shape}")
print(f"Input region used: ALL 105 bars")
print(f"Prediction region: bars [30:75] (45 bars)")
print(f"Context buffers: [0:30] and [75:105]")

# Check if masking is applied
print("\nCreating model instance...")
try:
    model = CnnTransformerModel(seed=1337, device='cpu', n_epochs=1)
    x_tensor = torch.FloatTensor(X_raw).unsqueeze(0).transpose(1, 2)  # [1, 4, 105]
    print(f"Tensor shape fed to model: {x_tensor.shape}")
    print("  Format: [batch_size, features, timesteps] = [1, 4, 105]")
except Exception as e:
    print(f"Error creating model: {e}")

# Check model architecture
print("\nArchitecture flow:")
print("  1. Conv1d layers (kernels 3,5,7) → extract local patterns")
print("  2. Positional encoding added")
print("  3. Transformer (4 heads, 3 layers) → global attention")
print("  4. Classification head → [consol_prob, retrace_prob]")

print("\n❓ QUESTION: Does attention mask limit to [30:75]?")
# Check model code
import os
model_path = 'src/moola/models/cnn_transformer.py'
if os.path.exists(model_path):
    with open(model_path, 'r') as f:
        code = f.read()
        has_mask = 'attention_mask' in code or 'src_mask' in code
        has_region_extract = '[30:75]' in code or 'pred_region' in code
        print(f"   Attention masking found: {has_mask}")
        print(f"   Region extraction [30:75] found: {has_region_extract}")

        # Check for specific masking patterns
        if 'attention_mask' in code:
            print(f"   ✅ attention_mask parameter exists")
        if 'src_mask' in code:
            print(f"   ✅ src_mask parameter exists")
        if not has_mask:
            print(f"   ⚠️  NO MASKING - model sees all 105 bars equally")
else:
    print(f"   Model file not found at {model_path}")

print("\n" + "="*60)
print("2. XGBOOST (Classical ML)")
print("="*60)

# Check what XGBoost actually receives
if 'expansion_start' in df.columns:
    exp_start = int(sample['expansion_start'])
    exp_end = int(sample['expansion_end'])
    exp_region = X_raw[exp_start:exp_end+1, :]
    print(f"Uses expansion indices: YES")
    print(f"Expansion region: [{exp_start}:{exp_end}]")
    print(f"Region shape: {exp_region.shape} ({exp_region.shape[0]} bars)")
    print(f"\nExpansion region OHLC:")
    for i in range(min(3, len(exp_region))):
        idx = exp_start + i
        print(f"  Bar {idx}: O={exp_region[i,0]:.2f} H={exp_region[i,1]:.2f} L={exp_region[i,2]:.2f} C={exp_region[i,3]:.2f}")
else:
    print(f"Uses expansion indices: NO")
    print(f"Feature extraction region: FIXED [30:75] (45 bars)")

# Actually extract features to see them
X_batch = np.array([X_raw])
print("\n" + "="*60)
print("FEATURE EXTRACTION (WITH expansion indices)")
print("="*60)
if 'expansion_start' in df.columns:
    features_with = engineer_classical_features(
        X_batch,
        expansion_start=np.array([exp_start]),
        expansion_end=np.array([exp_end])
    )
    print(f"Engineered features shape: {features_with.shape}")
    print(f"Number of features: {features_with.shape[1]}")
    print(f"\nFeature values (first 10):")
    for i, val in enumerate(features_with[0][:10]):
        print(f"  Feature {i:2d}: {val:+.6f}")

    print("\n" + "="*60)
    print("FEATURE EXTRACTION (WITHOUT expansion indices)")
    print("="*60)
    features_without = engineer_classical_features(
        X_batch,
        expansion_start=None,
        expansion_end=None
    )
    print(f"Engineered features shape: {features_without.shape}")
    print(f"Number of features: {features_without.shape[1]}")
    print(f"\nFeature values (first 10):")
    for i, val in enumerate(features_without[0][:10]):
        print(f"  Feature {i:2d}: {val:+.6f}")

    print("\n" + "="*60)
    print("FEATURE COMPARISON")
    print("="*60)
    print(f"Feature count same: {features_with.shape[1] == features_without.shape[1]}")
    if features_with.shape[1] == features_without.shape[1]:
        diffs = np.abs(features_with[0] - features_without[0])
        print(f"Max difference: {diffs.max():.6f}")
        print(f"Mean difference: {diffs.mean():.6f}")
        print(f"Features with |diff| > 0.01: {(diffs > 0.01).sum()}/{len(diffs)}")

        # Show most different features
        top_diff_idx = np.argsort(diffs)[-5:][::-1]
        print(f"\nTop 5 most different features:")
        for idx in top_diff_idx:
            print(f"  Feature {idx:2d}: WITH={features_with[0,idx]:+.6f}, WITHOUT={features_without[0,idx]:+.6f}, diff={diffs[idx]:.6f}")
else:
    features = engineer_classical_features(X_batch, expansion_start=None, expansion_end=None)
    print(f"Engineered features shape: {features.shape}")
    print(f"Number of features: {features.shape[1]}")

print("\n❓ QUESTION: Are features computed on right region?")
if 'expansion_start' in df.columns:
    print(f"   Expected: Pattern region only ({exp_end - exp_start} bars)")
    print(f"   Actual: {'Pattern region' if 'expansion_start' in df.columns else 'Fixed [30:75] (45 bars)'}")
    print(f"   Feature differences detected: {(diffs > 0.01).sum()} / {len(diffs)}")
else:
    print(f"   Expected: Pattern region")
    print(f"   Actual: Fixed [30:75] (45 bars)")

print("\n" + "="*60)
print("3. OUTPUT PREDICTION CHECK")
print("="*60)

print("What each model would predict for this sample:")
print("  CNN-Transformer: [Need to load trained model weights]")
print("  XGBoost: [Need to load trained model weights]")
print("  Ensemble: [Need to load stacking weights]")
print(f"  Ground truth: {sample['label']}")

print("\n" + "="*60)
print("4. REGION VISUALIZATION")
print("="*60)

print(f"\nTimeline of 105 bars:")
print(f"  [0:30]   = Left buffer (30 bars)")
print(f"  [30:75]  = Default prediction region (45 bars)")
if 'expansion_start' in df.columns:
    print(f"  [{exp_start}:{exp_end}] = Actual pattern ({exp_end - exp_start} bars) ← GROUND TRUTH")
print(f"  [75:105] = Right buffer (30 bars)")

if 'expansion_start' in df.columns:
    # Show overlap
    overlap_start = max(30, exp_start)
    overlap_end = min(75, exp_end)
    overlap_bars = max(0, overlap_end - overlap_start)
    print(f"\nOverlap between [30:75] and [{exp_start}:{exp_end}]:")
    print(f"  Overlap region: [{overlap_start}:{overlap_end}]")
    print(f"  Overlap bars: {overlap_bars} / {exp_end - exp_start} pattern bars")
    print(f"  Coverage: {100 * overlap_bars / (exp_end - exp_start):.1f}%")

    if exp_end - exp_start > 0:
        signal_ratio = overlap_bars / (exp_end - exp_start)
        noise_ratio = (45 - overlap_bars) / 45
        print(f"\nSignal-to-noise in [30:75]:")
        print(f"  Signal: {overlap_bars} bars ({100*signal_ratio:.1f}%)")
        print(f"  Noise: {45 - overlap_bars} bars ({100*noise_ratio:.1f}%)")

print("\n" + "="*60)
print("5. DATA STATISTICS ACROSS DATASET")
print("="*60)

if 'expansion_start' in df.columns:
    exp_starts = df['expansion_start'].values
    exp_ends = df['expansion_end'].values
    pattern_lengths = exp_ends - exp_starts

    print(f"Dataset: {len(df)} samples")
    print(f"\nPattern length distribution:")
    print(f"  Min: {pattern_lengths.min()} bars")
    print(f"  Max: {pattern_lengths.max()} bars")
    print(f"  Mean: {pattern_lengths.mean():.1f} bars")
    print(f"  Median: {np.median(pattern_lengths):.0f} bars")
    print(f"  Std: {pattern_lengths.std():.1f} bars")

    print(f"\nPattern start position distribution:")
    print(f"  Min: {exp_starts.min()}")
    print(f"  Max: {exp_starts.max()}")
    print(f"  Mean: {exp_starts.mean():.1f}")
    print(f"  Median: {np.median(exp_starts):.0f}")

    print(f"\nPattern end position distribution:")
    print(f"  Min: {exp_ends.min()}")
    print(f"  Max: {exp_ends.max()}")
    print(f"  Mean: {exp_ends.mean():.1f}")
    print(f"  Median: {np.median(exp_ends):.0f}")

    # Check coverage
    overlaps = []
    for start, end in zip(exp_starts, exp_ends):
        overlap_start = max(30, start)
        overlap_end = min(75, end)
        overlap = max(0, overlap_end - overlap_start)
        pattern_len = end - start
        if pattern_len > 0:
            coverage = overlap / pattern_len
            overlaps.append(coverage)

    overlaps = np.array(overlaps)
    print(f"\nCoverage of patterns by [30:75] region:")
    print(f"  Fully contained (100%): {(overlaps >= 1.0).sum()} / {len(overlaps)}")
    print(f"  Mostly contained (>80%): {(overlaps >= 0.8).sum()} / {len(overlaps)}")
    print(f"  Partially contained (>50%): {(overlaps >= 0.5).sum()} / {len(overlaps)}")
    print(f"  Mean coverage: {overlaps.mean()*100:.1f}%")

print("\n" + "="*60)
print("DIAGNOSTIC SUMMARY")
print("="*60)
print("Issues to investigate:")
print("  [ ] Are buffer zones [0:30] and [75:105] being masked in attention?")
print("  [ ] Are expansion indices helping or hurting classical models?")
print("  [ ] Do deep models see too much context (105 bars vs 6-bar pattern)?")
print("  [ ] Should we use fixed [30:75] or variable expansion regions?")
print(f"  [ ] Are features too complex (37 engineered vs simple price action)?")

print("\n" + "="*60)
print("KEY FINDINGS")
print("="*60)
if 'expansion_start' in df.columns:
    print(f"✅ Expansion indices available in data")
    print(f"✅ Mean pattern length: {pattern_lengths.mean():.1f} bars (median {np.median(pattern_lengths):.0f})")
    print(f"✅ Mean coverage by [30:75]: {overlaps.mean()*100:.1f}%")
    if overlaps.mean() >= 0.8:
        print(f"✅ Most patterns well-covered by default [30:75] region")
    elif overlaps.mean() >= 0.5:
        print(f"⚠️  Patterns partially covered - expansion indices may help")
    else:
        print(f"❌ Poor coverage - expansion indices critical")
else:
    print(f"❌ Expansion indices NOT in data")
    print(f"⚠️  Using fixed [30:75] region (45 bars)")

print("\nRecommendations:")
if 'expansion_start' in df.columns and overlaps.mean() >= 0.8:
    print("  → Fixed [30:75] region captures most patterns")
    print("  → Expansion indices may add variance without benefit")
    print("  → Consider removing expansion index dependency")
elif 'expansion_start' in df.columns and overlaps.mean() < 0.8:
    print("  → Expansion indices provide meaningful signal")
    print("  → Keep using them for classical models")
    print("  → Deep models may need region-specific attention masks")
else:
    print("  → Need to create expansion indices if not present")
    print("  → Or verify that [30:75] contains actual patterns")

print("\n" + "="*60)
print("FORENSICS COMPLETE")
print("="*60)
