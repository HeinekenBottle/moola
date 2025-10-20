# Dual-Input Data Pipeline Integration Guide

## Overview

This guide documents the enhanced data pipeline that integrates both raw OHLC data AND sophisticated engineered features for improved model performance in the Moola project.

## What's Been Implemented

### 🔄 Dual-Input Data Processing
- **Raw OHLC Data**: Always available as `[N, 105, 4]` temporal sequences
- **Engineered Features**: Optional sophisticated features extracted from OHLC data
- **Backward Compatibility**: Existing models continue to work without changes
- **Model-Specific Inputs**: Automatic selection of appropriate input format per model type

### 🎯 Enhanced CLI Commands
New CLI options for `train` and `evaluate` commands:
```bash
--use-engineered-features      # Enable engineered features (small dataset + price action)
--max-engineered-features 25   # Maximum number of engineered features to use
--use-hopsketch                # Include HopSketch features (1575 features, for XGBoost)
```

### 📊 Feature Types Available

#### 1. Small Dataset Optimized Features (up to 25 features)
- **Pattern Morphology**: Fractal dimension, symmetry, curvature, directionality
- **Relative Dynamics**: Volatility ratios, momentum continuity, positioning
- **Market Microstructure**: Body ratios, wick dominance, price efficiency
- **Geometric Invariants**: Path length, Hurst exponent, turning points
- **Temporal Signatures**: Autocorrelation, periodicity, partial autocorrelation

#### 2. Multi-Scale Price Action Features (21 features)
- **Pattern-level**: 5 features on expansion region (price change, direction, range)
- **Context-level**: 10 features on fixed [30:75] window (Williams %R, volatility, trend)
- **Relative Position**: 5 features (pattern vs context relationships)
- **Adaptive Williams %R**: 1 feature on pattern region

#### 3. HopSketch Features (1575 features)
- Per-bar geometric features: 15 features × 105 bars
- OHLC normalized, geometry, context features
- Designed for XGBoost and tree-based models

## Usage Examples

### Basic Training (Backward Compatible)
```bash
# Uses only raw OHLC data (existing behavior)
moola train --model simple_lstm --device cpu
```

### Training with Engineered Features
```bash
# Enable engineered features for XGBoost
moola train \
  --model xgboost \
  --use-engineered-features \
  --max-engineered-features 30

# Enable engineered features for LSTM (uses raw OHLC, features available for context)
moola train \
  --model simple_lstm \
  --device cuda \
  --use-engineered-features \
  --max-engineered-features 25
```

### Advanced Configuration
```bash
# Maximum engineered features for XGBoost with HopSketch
moola train \
  --model xgboost \
  --use-engineered-features \
  --max-engineered-features 100 \
  --use-hopsketch

# Conservative feature selection for small datasets
moola train \
  --model logreg \
  --use-engineered-features \
  --max-engineered-features 15
```

### Evaluation with Engineered Features
```bash
# Evaluation automatically uses same feature configuration as training
moola evaluate --model xgboost
moola evaluate --model logreg --use-engineered-features
```

## Model Input Formats

### Deep Learning Models (LSTM, CNN-Transformer)
- **Primary Input**: Raw OHLC sequences `[N, 105, 4]`
- **Engineered Features**: Available as additional context (model architecture update needed)
- **Expansion Indices**: `[N]` for pattern alignment

### Tree-Based Models (XGBoost, Random Forest, Logistic Regression)
- **With Engineered Features**: Engineered feature matrix `[N, F]`
- **Without Engineered Features**: Flattened OHLC `[N, 420]`

## Performance Optimization

### Feature Selection Strategy
1. **Mutual Information**: Automatic selection using labels when available
2. **Variance-Based**: Fallback selection for unsupervised scenarios
3. **Category Limits**: Maximum features per category to avoid overfitting
4. **Total Limit**: Configurable maximum total features (default: 50)

### Small Dataset Considerations
- **Conservative Defaults**: 25 features max for small datasets
- **Robust Scaling**: Outlier-resistant preprocessing
- **Cross-Validation Stability**: Features selected for consistent performance
- **Noise Reduction**: Built-in smoothing and outlier handling

## Technical Implementation

### Core Components

#### 1. DualInputDataProcessor (`src/moola/data/dual_input_pipeline.py`)
- Main data processing pipeline
- Handles feature extraction and integration
- Caches engineered features for efficiency
- Validates input data and expansion indices

#### 2. FeatureConfig (`src/moola/data/dual_input_pipeline.py`)
- Configuration class for feature processing
- Controls which feature types are enabled
- Sets limits and processing parameters

#### 3. Enhanced CLI (`src/moola/cli.py`)
- Updated `train` and `evaluate` commands
- Backward compatible with existing workflows
- Automatic feature metadata saving and loading

### Data Flow
```
Raw Data (parquet) → DualInputDataProcessor → Model Inputs
├── OHLC Features → Always preserved [N, 105, 4]
├── Small Dataset Features → Optional [N, F1]
├── Multi-Scale Features → Optional [N, F2]
└── HopSketch Features → Optional [N, F3]
```

### Error Handling
- **Invalid Data**: Graceful handling of malformed inputs
- **Missing Features**: Zero-filling or dropping problematic features
- **Expansion Indices**: Validation and clipping to valid ranges
- **Feature Selection**: Fallback strategies for edge cases

## Best Practices

### For Small Datasets (< 200 samples)
```bash
moola train \
  --model logreg \
  --use-engineered-features \
  --max-engineered-features 20
```

### For Deep Learning Models
```bash
moola train \
  --model simple_lstm \
  --device cuda \
  --use-engineered-features \
  --max-engineered-features 25
```

### For Tree-Based Models
```bash
moola train \
  --model xgboost \
  --use-engineered-features \
  --max-engineered-features 50 \
  --use-hopsketch
```

### For Production Models
```bash
# Train with engineered features
moola train --model logreg --use-engineered-features --max-engineered-features 30

# Evaluate automatically uses same configuration
moola evaluate --model logreg
```

## Troubleshooting

### Common Issues

#### 1. Feature Extraction Too Slow
- **Solution**: Reduce `--max-engineered-features` or disable HopSketch
- **Cause**: Too many features being computed for dataset size

#### 2. Overfitting with Engineered Features
- **Solution**: Reduce `--max-engineered-features` to 15-20
- **Cause**: Too many features for small dataset

#### 3. Memory Issues with HopSketch
- **Solution**: Disable `--use-hopsketch` for large datasets
- **Cause**: 1575 features per sample can be memory-intensive

#### 4. Model Performance Degradation
- **Solution**: Try different feature combinations or disable engineered features
- **Cause**: Feature noise or irrelevant features for specific model

### Debugging Commands

#### Check Feature Configuration
```bash
# Check saved feature metadata
cat data/artifacts/models/logreg/feature_metadata.json
```

#### Test Feature Extraction
```bash
# Run integration tests
python3 scripts/test_dual_input_integration.py
```

#### Compare Performance
```bash
# Train without engineered features
moola train --model logreg
moola evaluate --model logreg

# Train with engineered features
moola train --model logreg --use-engineered-features --max-engineered-features 25
moola evaluate --model logreg
```

## Integration with Existing Workflows

### SSH/SCP Workflow (RunPod)
The enhanced pipeline works seamlessly with existing SSH/SCP workflows:

1. **Code Development**: Edit on Mac with new CLI options
2. **Pre-commit Hooks**: Automatic formatting and validation
3. **SSH to RunPod**: `ssh -i ~/.ssh/runpod_key ubuntu@IP && cd /workspace/moola`
4. **Training**: `python3 -m moola.cli train --use-engineered-features`
5. **Results**: `scp ubuntu@IP:/workspace/moola/experiment_results.jsonl ./`

### Results Logging
- **Feature Metadata**: Saved with models for reproducibility
- **Configuration Tracking**: All feature settings logged
- **Performance Comparison**: Easy comparison between feature configurations

## Advanced Usage

### Custom Feature Configuration
```python
from moola.data.dual_input_pipeline import DualInputDataProcessor, FeatureConfig

# Create custom configuration
config = FeatureConfig(
    use_small_dataset_features=True,
    small_dataset_max_features=15,
    use_price_action_features=True,
    max_total_engineered_features=40,
    handle_missing_features="mean"
)

# Use custom processor
processor = DualInputDataProcessor(config)
```

### Programmatic Feature Extraction
```python
from moola.data.dual_input_pipeline import create_dual_input_processor

# Process data programmatically
processor = create_dual_input_processor(
    use_engineered_features=True,
    max_engineered_features=30
)

processed_data = processor.process_training_data(df)
feature_stats = processor.get_feature_statistics(processed_data['X_engineered'])
```

## Future Enhancements

### Planned Features
1. **Model Architecture Updates**: Enable direct use of engineered features in LSTM/Transformer models
2. **Feature Importance Tracking**: Automatic feature importance analysis
3. **Feature Versioning**: Track feature extraction versions for reproducibility
4. **Performance Optimization**: GPU acceleration for feature extraction
5. **Real-time Features**: Online feature extraction for live trading

### Model-Specific Optimizations
1. **LSTM Enhancement**: Concatenate engineered features to LSTM outputs
2. **Transformer Enhancement**: Use engineered features for attention mechanisms
3. **Ensemble Methods**: Combine raw OHLC and engineered feature models
4. **Feature Stacking**: Multi-level feature integration strategies

## Summary

The dual-input data pipeline successfully integrates:

✅ **Raw OHLC temporal data** (always available)
✅ **Small dataset optimized features** (up to 25 features)
✅ **Multi-scale price action features** (21 features)
✅ **HopSketch geometric features** (1575 features)
✅ **Backward compatibility** with existing models
✅ **Model-specific input formatting**
✅ **Performance optimization** for small datasets
✅ **Comprehensive error handling** and validation
✅ **Easy CLI integration** with minimal learning curve

The implementation provides a robust foundation for enhanced model performance while maintaining the simplicity and reliability of existing workflows.