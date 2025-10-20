# Integration Test Suite for Enhanced Moola Architecture

This comprehensive integration test suite validates the enhanced Moola architecture components, ensuring robustness, performance, and backward compatibility.

## Overview

The integration test suite covers:

1. **Data Pipeline Integration** - Raw OHLC → Feature Engineering → Model Training
2. **Enhanced SimpleLSTM Architecture** - Parameter optimization, gradient flow, memory efficiency
3. **Pre-training Integration** - Encoder transfer learning, two-phase training
4. **Backward Compatibility** - Existing models, CLI interfaces, data formats
5. **Performance Benchmarking** - Training speed, memory usage, prediction latency
6. **Small Dataset Validation** - Production constraints (98 samples)
7. **Regression Detection** - Performance regression prevention

## Test Structure

```
tests/integration/
├── __init__.py                     # Main integration test suite
├── test_data_pipeline.py          # Data flow validation
├── test_model_architecture.py      # SimpleLSTM architecture tests
├── test_pretraining_integration.py # Pre-training compatibility
├── test_backward_compatibility.py  # Legacy interface validation
├── validation_utils.py             # Performance benchmarks and validation
├── test_runner.py                  # Automated test execution
└── README.md                       # This documentation
```

## Running Tests

### Option 1: Run All Tests
```bash
python -m pytest tests/integration/ -v
```

### Option 2: Run Specific Test Suite
```bash
python -m pytest tests/integration/test_data_pipeline.py -v
python -m pytest tests/integration/test_model_architecture.py -v
python -m pytest tests/integration/test_pretraining_integration.py -v
python -m pytest tests/integration/test_backward_compatibility.py -v
```

### Option 3: Use Automated Test Runner
```bash
python tests/integration/test_runner.py
```

### Option 4: Run Individual Tests
```bash
python tests/integration/__init__.py
```

## Test Coverage

### Data Pipeline Integration
- OHLC → Feature Engineering transformation
- Data schema validation
- Model training with engineered features
- Data augmentation pipeline
- Memory efficiency with small datasets
- Error handling

### Enhanced SimpleLSTM Architecture
- Parameter count optimization (target: 40K-60K)
- Multi-head attention integration
- Gradient flow validation
- Forward pass compatibility
- Model save/load consistency
- Small dataset optimization
- Class imbalance handling

### Pre-training Integration
- Encoder loading capability
- Two-phase training workflow
- Architecture compatibility
- Layer count mismatch handling
- Freeze/unfreeze cycles
- Training consistency
- Error handling

### Backward Compatibility
- Existing model interfaces
- CLI interface maintenance
- Data format compatibility
- Parameter compatibility
- Seed reproducibility
- Model list enumeration
- Error handling patterns
- Cross-validation integration

### Performance Benchmarking
- Training time measurement
- Memory usage tracking
- Prediction latency
- Feature engineering speed
- Data augmentation performance
- Regression detection

### Small Dataset Validation
- 98-sample training compatibility
- Class balance robustness
- Overfitting detection
- Performance thresholds
- Production constraints

## Validation Utilities

### PerformanceBenchmark
```python
from tests.integration.validation_utils import PerformanceBenchmark

benchmark = PerformanceBenchmark()
results = benchmark.benchmark_model_training("simple_lstm", X, y)
```

### ArchitectureValidator
```python
from tests.integration.validation_utils import ArchitectureValidator

validator = ArchitectureValidator()
param_validation = validator.validate_parameter_count(model, (40000, 60000), "SimpleLSTM")
```

### RegressionDetector
```python
from tests.integration.validation_utils import PerformanceRegressionDetector

detector = PerformanceRegressionDetector()
regressions = detector.detect_regressions(current_results)
```

### SmallDatasetValidator
```python
from tests.integration.validation_utils import SmallDatasetValidator

validator = SmallDatasetValidator()
small_dataset_results = validator.validate_small_dataset_training("simple_lstm")
```

## Expected Results

### Success Criteria
- **Parameter Count**: 40K-60K parameters for SimpleLSTM
- **Training Time**: < 2 minutes for 98 samples
- **Memory Usage**: < 4GB on GPU
- **Accuracy**: > 80% on validation data
- **Success Rate**: ≥ 80% for all test suites
- **Backward Compatibility**: 100% maintenance

### Performance Targets
- Feature engineering: > 50 samples/second
- Prediction latency: < 100ms per sample
- Training throughput: > 10 samples/second
- Memory efficiency: < 100MB per model instance

## Test Configuration

### Test Data
- **Raw OHLC**: 100 samples × 105 time steps × 4 features
- **Engineered Features**: 20-40 features per time step
- **Labels**: Binary classification (class_A, class_B)
- **Imbalance**: Controlled class ratios for robustness testing

### Model Configurations
```python
# Enhanced SimpleLSTM configuration
{
    "hidden_size": 128,        # Optimized for pre-training transfer
    "num_layers": 1,           # Single LSTM layer
    "num_heads": 2,            # Multi-head attention
    "dropout": 0.1,            # Regularization
    "n_epochs": 60,            # Training epochs
    "batch_size": 512,         # Batch size
    "learning_rate": 5e-4,     # Optimized learning rate
    "early_stopping_patience": 20
}
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure src is in Python path
   export PYTHONPATH=/path/to/src:$PYTHONPATH
   ```

2. **CUDA Memory Issues**
   ```bash
   # Use CPU for testing
   python tests/integration/test_runner.py
   ```

3. **Test Timeouts**
   ```bash
   # Increase timeout
   python tests/integration/test_runner.py --timeout 600
   ```

4. **Missing Dependencies**
   ```bash
   # Install test dependencies
   pip install pytest numpy pandas torch scikit-learn loguru
   ```

### Debug Mode
```bash
# Run with verbose output
python tests/integration/test_runner.py --verbose --parallel-execution false
```

## Output Files

Test execution generates:

1. **integration_test_results.json** - Detailed test results
2. **integration_test_report.html** - HTML report with visualizations
3. **validation_report.md** - Comprehensive validation analysis
4. **validation_report.json** - Benchmarking metrics

## Integration with CI/CD

### GitHub Actions Example
```yaml
jobs:
  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run integration tests
        run: |
          python -m pytest tests/integration/ -v --tb=short
      - name: Run automated test runner
        run: |
          python tests/integration/test_runner.py
```

### Pre-commit Hook
```yaml
repos:
  - repo: local
    hooks:
      - id: integration-tests
        name: Run integration tests
        entry: python tests/integration/test_runner.py
        language: system
        pass_filenames: false
        always_run: true
```

## Best Practices

1. **Run tests before major changes**
2. **Monitor performance regressions**
3. **Maintain backward compatibility**
4. **Document test failures**
5. **Regularly update baselines**
6. **Test edge cases thoroughly**
7. **Monitor memory usage in production**

## Contributing

1. **Add new tests for new features**
2. **Maintain test documentation**
3. **Update baselines for performance improvements**
4. **Report test failures with detailed reproduction steps**
5. **Follow existing test patterns and naming conventions**

## Support

For questions or issues:
- Review test output in generated reports
- Check specific test suite documentation
- Run individual tests for debugging
- Monitor performance baselines for regressions