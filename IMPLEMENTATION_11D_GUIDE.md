# 11-Dimensional Feature Integration Implementation Guide

## Overview

This guide provides a complete implementation strategy for incorporating 11-dimensional features into the Moola ML pipeline while maintaining backward compatibility with existing 4D OHLC data.

## Architecture Summary

### Current State
- **4D OHLC Data**: `[N, 105, 4]` - Open, High, Low, Close
- **SimpleLSTM Model**: Enhanced dual-input architecture supporting engineered features
- **RelativeFeatureTransform**: Converts 4D OHLC → 11D relative features
- **Storage**: Parquet files with schema validation

### Target State
- **11D Relative Features**: `[N, 105, 11]` - Log returns, candle ratios, rolling z-scores
- **Dual-Input Pipeline**: Both 4D OHLC and 11D features available
- **Backward Compatibility**: Existing 4D pipeline continues to work
- **Enhanced Storage**: New parquet schema with both feature sets

## Implementation Components

### 1. Data Source Integration

**Strategy**: Use existing `RelativeFeatureTransform` to generate 11D features from 4D OHLC data.

```python
from moola.features.relative_transform import RelativeFeatureTransform
from moola.data.feature_11d_integration import Feature11DIntegrator

# Load 4D OHLC and generate 11D features
integrator = Feature11DIntegrator()
ohlc_data, relative_11d_data = integrator.load_and_transform_data("data/raw/unlabeled_windows.parquet")
```

**11D Features Composition**:
- **4 Log Returns**: `log(price_t / price_t-1)` for O, H, L, C
- **3 Candle Ratios**: body/range, upper_wick/range, lower_wick/range  
- **4 Rolling Z-Scores**: 20-bar standardized values for O, H, L, C

### 2. Data Quality Framework

**Enhanced Schema Validation**:
```python
from moola.data_infra.schemas_11d import EnhancedTimeSeriesWindow

# Validate 11D data
window = EnhancedTimeSeriesWindow(
    window_id="sample_001",
    feature_dimension=FeatureDimension.DUAL_INPUT,
    ohlc_features=ohlc_data[0].tolist(),
    relative_features=relative_11d_data[0].tolist()
)
```

**Quality Checks**:
- Feature dimension validation (4 vs 11 vs 15)
- Missing value detection
- Numerical stability checks
- Cross-validation between OHLC and relative features

### 3. Feature Engineering Pipeline

**Automatic Feature Generation**:
```python
from moola.data.feature_11d_integration import create_enhanced_dataset

# Create enhanced dataset with both 4D and 11D features
df_enhanced = create_enhanced_dataset(
    input_path=Path("data/batches/batch_200.parquet"),
    output_path=Path("data/enhanced_11d/batch_200_enhanced.parquet")
)
```

**Feature Analysis**:
```python
# Analyze feature importance and quality
quality_report = integrator.validate_feature_quality(ohlc_data, relative_11d_data)
importance_analysis = integrator.get_feature_importance_analysis(relative_11d_data, labels)
```

### 4. Storage Architecture

**Directory Structure**:
```
data/
├── raw/                    # Original 4D OHLC (unchanged)
│   └── unlabeled_windows.parquet
├── enhanced_11d/           # New 11D enhanced datasets
│   ├── windows_enhanced.parquet
│   └── batch_*_enhanced.parquet
├── batches/               # Annotation batches (4D + 11D)
│   └── batch_*_11d.parquet
└── processed/             # Final training datasets
    └── train_enhanced.parquet
```

**Storage Schema**:
```python
from moola.data.storage_11d import get_storage_architecture

storage = get_storage_architecture()
schema = storage.get_storage_schema()
```

**Enhanced Parquet Schema**:
```json
{
  "window_id": "string",
  "features_ohlc": "array[105, 4]",
  "features_relative": "array[105, 11]", 
  "features_enhanced": "array[105, 15]",
  "feature_dimension": "string",
  "label": "string",
  "expansion_start": "int",
  "expansion_end": "int"
}
```

## Migration Strategy

### Phase 1: Backward Compatibility (Immediate)
1. **Deploy 11D Integration**: Install new modules without changing existing pipeline
2. **Generate Enhanced Datasets**: Create 11D versions alongside existing 4D data
3. **Validation**: Run quality checks on generated features

```bash
# Generate enhanced dataset
python3 -c "
from moola.data.storage_11d import migrate_to_11d
migrate_to_11d(
    Path('data/batches/batch_200.parquet'),
    Path('data/enhanced_11d/batch_200_enhanced.parquet')
)
"
```

### Phase 2: Dual-Input Training (Week 1-2)
1. **Model Integration**: Use existing Enhanced SimpleLSTM with dual-input
2. **Training Pipeline**: Update training to use both 4D + 11D features
3. **Performance Comparison**: A/B test 4D vs 11D performance

```python
# Train with 11D features
model = SimpleLSTMModel(use_engineered_features=True)
model.fit(
    X_ohlc, y,
    expansion_start=expansion_start,
    expansion_end=expansion_end
)
```

### Phase 3: Full Migration (Week 3-4)
1. **Pipeline Update**: Default to 11D features for new experiments
2. **Storage Migration**: Migrate all existing datasets to enhanced format
3. **Documentation Update**: Update all documentation and training guides

## CLI Integration

### New CLI Commands
```bash
# Migrate existing data to 11D
python3 -m moola.cli migrate-11d --input data/batches/batch_200.parquet

# Create 11D annotation batch  
python3 -m moola.cli create-batch-11d --window-ids window_001,window_002

# Validate 11D data quality
python3 -m moola.cli validate-11d --data-path data/enhanced_11d/

# Train with 11D features
python3 -m moola.cli train --model simple_lstm --use-11d-features --device cuda
```

### Enhanced Training Command
```bash
# Standard 4D training (unchanged)
python3 -m moola.cli train --model simple_lstm --device cuda

# Enhanced 11D training (new)
python3 -m moola.cli train --model simple_lstm --use-11d-features --device cuda

# Dual-input training with both feature sets
python3 -m moola.cli train --model simple_lstm --dual-input --device cuda
```

## Impact Analysis

### Data Pipeline Impact
- **Unlabeled Windows**: No impact, 4D data preserved
- **Annotation Batches**: Enhanced with 11D features, backward compatible
- **Training Datasets**: New enhanced format, 4D still supported
- **Candlesticks Integration**: No impact, continues with 4D OHLC

### Model Architecture Impact
- **SimpleLSTM**: Already supports dual-input, minimal changes needed
- **BiLSTM Pre-training**: No impact, continues with 4D OHLC
- **Ensemble Models**: Can leverage 11D features for improved performance

### Performance Impact
- **Memory**: ~3.75x increase (4→15 features per timestep)
- **Training Time**: ~2-3x increase due to additional features
- **Performance**: Expected 5-8% accuracy improvement from relative features
- **Storage**: ~3.75x increase in dataset size

## Quality Assurance

### Automated Validation
```python
from moola.data.feature_11d_integration import Feature11DIntegrator

integrator = Feature11DIntegrator()
quality_report = integrator.validate_feature_quality(ohlc_data, relative_11d_data)

# Check for issues
assert quality_report['ohlc_quality']['missing_values'] == 0
assert quality_report['relative_quality']['missing_values'] == 0
assert quality_report['cross_validation']['consistent_samples']
```

### Feature Importance Monitoring
```python
importance_analysis = integrator.get_feature_importance_analysis(
    relative_11d_data, labels
)

# Monitor top performing features
top_features = sorted(
    importance_analysis['supervised_importance'].items(),
    key=lambda x: abs(x[1]['correlation']),
    reverse=True
)[:5]
```

## Rollback Plan

### Immediate Rollback
1. **Switch to 4D**: Use `use_engineered_features=False` in SimpleLSTM
2. **Load Original Data**: Use existing 4D parquet files
3. **Disable 11D Pipeline**: Set environment variable `MOOLA_USE_11D=false`

### Complete Rollback
```bash
# Restore 4D-only training
python3 -m moola.cli train --model simple_lstm --use-engineered-features=false

# Verify 4D pipeline works
python3 -m moola.cli doctor --check-4d-only
```

## Success Metrics

### Technical Metrics
- [ ] 11D feature generation success rate > 99%
- [ ] Data validation pass rate > 95%
- [ ] Backward compatibility maintained (no breaking changes)
- [ ] Storage overhead < 4x (target: 3.75x)

### Performance Metrics  
- [ ] Model accuracy improvement > 3%
- [ ] Training time increase < 3x
- [ ] Memory usage increase < 4x
- [ ] No degradation in existing 4D model performance

### Operational Metrics
- [ ] Zero downtime during migration
- [ ] All existing experiments continue to work
- [ ] Documentation updated and validated
- [ ] Team training completed

## Next Steps

1. **Immediate (This Week)**:
   - Review and approve implementation plan
   - Set up enhanced storage architecture
   - Run initial 11D feature generation tests

2. **Short Term (Next 2 Weeks)**:
   - Deploy 11D integration to staging
   - Run performance benchmarks
   - Update training pipelines

3. **Medium Term (Next Month)**:
   - Full production deployment
   - Migrate existing datasets
   - Update all documentation

4. **Long Term (Ongoing)**:
   - Monitor 11D feature performance
   - Optimize feature engineering pipeline
   - Explore additional feature dimensions

## Support and Troubleshooting

### Common Issues
1. **Memory Errors**: Reduce batch size or use gradient accumulation
2. **Feature Validation Failures**: Check OHLC data quality first
3. **Performance Degradation**: Verify feature importance analysis
4. **Storage Issues**: Monitor disk space with 3.75x overhead

### Debug Commands
```bash
# Check 11D integration status
python3 -m moola.cli doctor --check-11d

# Validate specific dataset
python3 -m moola.cli validate-11d --data-path data/enhanced_11d/batch_200_enhanced.parquet

# Compare 4D vs 11D performance
python3 -m moola.cli benchmark --features-4d --features-11d
```

### Contact Support
- **Technical Issues**: Create GitHub issue with `11d-integration` label
- **Performance Issues**: Contact ML engineering team
- **Data Issues**: Contact data infrastructure team