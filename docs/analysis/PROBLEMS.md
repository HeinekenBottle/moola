# Moola Persistent Problems & Solutions

## Overview
This document serves as the source of truth for ongoing issues in the Moola ML pipeline, focusing on training failures, model collapse, and optimization challenges. It incorporates insights from Grok's mathematical/programming/ML recommendations and ChatGPT research.

## Core Issues

### 1. Model Collapse on Retracement Classification
**Problem**: Fine-tuned models consistently predict "consolidation" for 100% of samples, achieving only ~23% accuracy despite 174 labeled samples with ~60% retracement types.

**Root Causes**:
- Class imbalance (57% consol in train vs 23% test)
- Extreme feature sparsity (99% zeros in early bars)
- Poor pre-training transfer
- Improper training configurations defaulting to easier predictions

**Solutions Implemented**:
- Stratified temporal splits (train 2020-09~2024-12, val 2025-01~03, test 04~06)
- Uncertainty weighting (Kendall method with learnable σ)
- WeightedRandomSampler for class balancing
- Center-length encoding for spans
- Logit bias initialization
- Batch sizes 32-64, unfreeze schedules

**Current Status**: Anti-collapse pipeline enforced, awaiting validation.

### 2. Feature Sparsity & Densification
**Problem**: 99% zeros in early bars due to ATR/zigzag lag, spotlighting only 5-6 bar expansions.

**Solutions**:
- 20-bar padding/warmup for context
- Expansion proxies (range_z * leg_dir * body_pct)
- Jitter σ=0.03, magnitude warping σ=0.2
- Non-zero density >50% target

### 3. Pre-training & Fine-tuning Mismatch
**Problem**: Pre-trained encoder not transferring well to supervised task.

**Solutions**:
- Relativity features: candle norms [0,1], swing ATR_10, causal zigzag k=1.2
- Batch=64, LR=1e-3 AdamW, dropout=0.6, 10-20 epochs
- Fine-tune with Huber loss (δ=0.08), stratified splits

### 4. Performance Bottlenecks
**Problem**: GPU utilization issues, memory constraints, slow data loading.

**Solutions**:
- Gradient checkpointing for memory efficiency
- Float32 enforcement
- Pre-computed features (1000x speedup)
- Optimized windowing (105-bar, 50% overlap)

### 5. Data Pipeline Instability
**Problem**: Broken zigzag CLI, missing TA-Lib, purged augmentations.

**Solutions**:
- TA-Lib integration
- Relativity pipeline implementation
- Dead code cleanup

## Multi-Phase Resolution Plan

### Phase 1: Data & Feature Engineering (Current)
- Pre-compute NQ features using relativity
- Verify 5-year data integrity (SHA256 checked)
- Implement densification techniques

### Phase 2: Training Optimization
- Enforce anti-collapse pipeline
- Validate parameter ratios (~95K params for 174 samples)
- Implement proper metrics (F1_macro, AUCPR)

### Phase 3: Performance Tuning
- GPU/memory optimizations
- Data loading efficiency
- Scale to larger datasets if needed

### Phase 4: Validation & Deployment
- Achieve 60-75% F1 target
- Bootstrap CIs for stability
- RunPod deployment with tightened sync

## Key Metrics & Targets
- **F1 Score**: 60-75% on test set
- **Parameter Ratio**: ~550 params per sample (95K/174)
- **Feature Density**: >50% non-zero
- **Training Stability**: No collapse, proper class distribution

## References
- Grok Chat Summary: Pre-train/fine-tune fixes, sparsity handling
- ChatGPT Research: Uncertainty weighting, stratified splits
- PDF Synthesis: BiLSTM multi-task, center-length encoding

## Next Steps
1. Complete feature pre-computation
2. Run full training cycle with anti-collapse measures
3. Validate on held-out test set
4. Iterate based on results</content>
</xai:function_call">Update sync scripts to exclude more unnecessary files, tighten dependencies in requirements.txt.