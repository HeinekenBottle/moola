# Moola Performance Optimization Plan

## Phase 1: Critical Architecture Fixes (Day 1-2)

### 1.1 Model Sizing for Small Dataset Regime

**Problem**: 565,636 parameters for 174 samples (3,250:1 ratio) is dangerously overparameterized

**Solution**: Implement Jade-Compact variant
- Hidden size: 128 → 64 (74.6% parameter reduction)
- Parameter count: 565K → 144K
- Parameter-to-sample ratio: 3,250:1 → 825:1 (acceptable)

**Implementation**:
```yaml
# configs/model/jade_compact.yaml
model:
  name: jade_compact
  architecture:
    hidden_size: 64  # Reduced from 128
    num_layers: 2
    input_dropout: 0.3  # Increased regularization
    recurrent_dropout: 0.7
    dense_dropout: 0.6
  # ... rest of optimized config
```

**Expected Impact**:
- Memory usage: 2.2MB → 0.5MB (1.7MB saved)
- Training speed: 8-12min/epoch → 2-3min/epoch
- Overfitting risk: High → Low

### 1.2 Data Pipeline Optimization

**Problem**: Object dtype arrays causing memory overhead and slow loading

**Solution**: Pre-compute float32 tensors
```python
# Convert object arrays to proper float32
features = np.zeros((n_samples, 105, 4), dtype=np.float32)
for i, feature_array in enumerate(df["features"]):
    features[i] = np.array([timestep.flatten() for timestep in feature_array], dtype=np.float32)
```

**Expected Impact**:
- Memory reduction: ~40%
- Loading speed: 2-3x faster
- Better GPU utilization

### 1.3 GPU Utilization Fix

**Problem**: Only 0.02GB GPU memory usage, suspiciously fast training

**Root Cause**: Model not actually using GPU or data not transferred properly

**Solution**:
```python
# Ensure proper GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)

# Verify GPU memory usage
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

## Phase 2: Performance Enhancements (Day 3-5)

### 2.1 Enhanced Data Augmentation

**Current**: Basic augmentation with 3x multiplier
**Optimized**: Strong augmentation for small dataset
- Jitter: σ=0.04, prob=0.9
- Magnitude warp: σ=0.25, prob=0.6
- Multiplier: 5x (more on-the-fly augmentation)

### 2.2 Learning Rate Optimization

**Current**: 3e-4 learning rate
**Optimized**: 1e-4 for small dataset stability
- Lower LR prevents overfitting
- Longer training with early stopping
- ReduceLROnPlateau with faster adaptation

### 2.3 Batch Size Optimization

**Current**: 29 samples per batch
**Optimized**: 16 samples per batch
- Better gradient estimates for small dataset
- More frequent weight updates
- Reduced memory usage

## Phase 3: Advanced Optimizations (Day 6-10)

### 3.1 Feature Engineering Upgrade

**Current**: 4D OHLC features
**Target**: 11D RelativeTransform features
- Price action features
- Technical indicators
- Statistical features

### 3.2 Ensemble Methods

**Small Dataset Strategy**: Use ensemble of smaller models
- 5-fold cross-validation
- Model diversity (different architectures)
- Stacking ensemble for final predictions

### 3.3 Pre-training Integration

**Optional**: BiLSTM masked autoencoder pre-training
- Use 2.2M unlabeled samples
- Transfer learning to small dataset
- Expected +3-5% accuracy boost

## Implementation Priority

### 🔴 IMMEDIATE (Day 1)
1. Deploy Jade-Compact configuration
2. Fix data pipeline object dtype issues
3. Verify GPU utilization

### 🟡 HIGH (Day 2-3)
1. Implement optimized data augmentation
2. Update learning rate and batch size
3. Add performance monitoring

### 🟢 MEDIUM (Day 4-5)
1. Upgrade to 11D RelativeTransform features
2. Implement ensemble methods
3. Add pre-training pipeline

## Expected Performance Gains

| Metric | Current | Optimized | Improvement |
|--------|---------|-----------|-------------|
| Parameters | 565K | 144K | 74.6% reduction |
| Memory Usage | 2.2MB | 0.5MB | 77.3% reduction |
| Training Speed | 8-12min/epoch | 2-3min/epoch | 3-4x faster |
| GPU Utilization | 0.02GB | 1-2GB | 50-100x improvement |
| Overfitting Risk | High | Low | Significant reduction |

## Monitoring & Validation

### Key Metrics to Track
1. **Parameter-to-sample ratio** (target: <1000:1)
2. **GPU memory utilization** (target: >1GB)
3. **Training time per epoch** (target: <3min)
4. **Validation accuracy stability** (target: <5% variance)

### Validation Steps
1. Baseline performance measurement
2. A/B testing of optimizations
3. Cross-validation for small dataset
4. Performance regression testing

## Risk Mitigation

### Potential Risks
1. **Underfitting** with smaller model
   - Mitigation: Strong augmentation, longer training
2. **Data pipeline breaking**
   - Mitigation: Incremental rollout, thorough testing
3. **GPU compatibility issues**
   - Mitigation: Fallback to CPU, device detection

### Rollback Plan
1. Git version control for all changes
2. Configuration-based model switching
3. Performance benchmarking before/after
4. Automated testing pipeline

## Success Criteria

### Performance Targets
- [ ] Parameter-to-sample ratio <1000:1
- [ ] GPU memory utilization >1GB
- [ ] Training time <3min/epoch
- [ ] No accuracy regression
- [ ] Stable validation performance

### Quality Gates
- [ ] All tests pass
- [ ] Performance benchmarks met
- [ ] Code review completed
- [ ] Documentation updated

## Next Steps

1. **Immediate**: Deploy Jade-Compact config
2. **Today**: Fix data pipeline issues
3. **Tomorrow**: Verify GPU utilization
4. **This week**: Implement full optimization pipeline
5. **Next week**: Performance validation and tuning

---

*This optimization plan prioritizes stability and performance for the small dataset regime while maintaining the Stones non-negotiables for production ML.*