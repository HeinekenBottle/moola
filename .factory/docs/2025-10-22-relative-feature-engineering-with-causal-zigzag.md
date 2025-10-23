# Relative Feature Engineering with Causal Zigzag Implementation

## Overview
Replace existing feature system with price-relative, scale-invariant features that eliminate absolute price leakage while maintaining predictive power through pattern recognition.

## Core Implementation Components

### 1. CausalZigZag Class (`src/moola/features/zigzag.py`)
- **ATR Calculation**: Rolling True Range with configurable period (default 10)
- **Zigzag Detection**: K×ATR threshold for swing confirmation (K=1.2 default)
- **Hybrid Confirmation**: 5-bar lookback + 0.5×ATR retrace rule for early confirmation
- **Causal Design**: Only uses closed candles, no future information
- **Output**: (prev_SH_H, prev_SL_L, bars_since_SH, bars_since_SL)

### 2. Relativity Builder Update (`src/moola/features/relativity.py`)
- **Candle Shape Features (6 dims)**: open_norm, close_norm, body_pct, upper_wick_pct, lower_wick_pct, range_z
- **Swing-Relative Features (4 dims)**: dist_to_prev_SH, dist_to_prev_SL, bars_since_SH_norm, bars_since_SL_norm
- **Scale Invariance**: All features relative to ATR or EMA of range
- **Bounds**: [0,1] for shape features, [-3,3] for distance features
- **Output**: [N, 105, 10] feature tensor

### 3. Parquet Loader (`src/moola/data/parquet_loader.py`)
- **Simple Loading**: Concat parquet files, enforce dtypes, sort by timestamp
- **No Forward Fill**: Strict causality, no data leakage
- **Columns**: timestamp, open, high, low, close, volume
- **Typing**: Proper pandas dtypes for consistency

### 4. CLI Integration (`src/moola/cli.py`)
- **Replace dual_input**: Use parquet_loader.load() + relativity.build_features()
- **Feature Pipeline**: df → X, mask = build_features(df, cfg.relativity)
- **Backward Compatibility**: Maintain existing CLI interface

### 5. Configuration Updates
- **Jade Model**: input_size: 10 (down from 11)
- **Relativity Config**: ohlc, atr, zigzag, window sections
- **Feature Bounds**: Explicit range specifications for quality gates

### 6. Testing Suite
- **Invariance Tests**: Price scaling ×10 → features unchanged (1e-6 tolerance)
- **Bounds Tests**: Verify feature ranges and NaN-free output
- **Causality Tests**: Ensure no future leakage via sentinel checks
- **Hybrid Rule Tests**: Verify zigzag confirmation logic

## Acceptance Criteria
- Features invariant to linear price scaling within 1e-6 tolerance
- ≥99.9% finite values; warmup masked for first 15 bars
- Jade input_size matches produced feature dimension (10)
- No absolute price references in feature arrays or metadata
- All tests pass with synthetic and real data

## Files to Modify/Create
1. `src/moola/features/zigzag.py` (update existing)
2. `src/moola/features/relativity.py` (update existing)  
3. `src/moola/data/parquet_loader.py` (new)
4. `src/moola/cli.py` (update imports and feature pipeline)
5. `configs/model/jade.yaml` (input_size: 10)
6. `configs/features/relativity.yaml` (update schema)
7. `tests/test_zigzag.py` (new)
8. `tests/test_relativity.py` (update existing)

This implementation maintains AGENTS.md compliance while providing clean, causal, and scale-invariant features for the Jade model.