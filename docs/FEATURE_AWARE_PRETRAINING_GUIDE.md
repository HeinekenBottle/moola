# Feature-Aware Pre-training System Guide

## Overview

The Feature-Aware Pre-training System extends the original BiLSTM masked autoencoder to process both raw OHLC data and engineered features simultaneously. This enables richer representation learning and better transfer learning performance for financial time series classification.

## Architecture

### Original vs Feature-Aware

**Original BiLSTM Masked Autoencoder:**
```
OHLC [Batch, 105, 4] → BiLSTM Encoder → Decoder → OHLC Reconstruction
                     ↓
               Masked Reconstruction (15% masking)
```

**Feature-Aware BiLSTM Masked Autoencoder:**
```
OHLC [Batch, 105, 4] + Features [Batch, 105, 25-30] → Feature Fusion → BiLSTM Encoder → Dual Decoders → OHLC + Features Reconstruction
                                                                    ↓
                                                              Dual Masked Reconstruction
```

### Key Components

1. **Feature Fusion Strategies:**
   - **Concatenation (`concat`)**: Simple concatenation of OHLC and features
   - **Addition (`add`)**: Project to same dimension and element-wise addition
   - **Gated Fusion (`gate`)**: Learned gating mechanism for optimal combination

2. **Dual Masking:**
   - Separate masking for OHLC and engineered features
   - Supports all original masking strategies (random, block, patch)
   - Maintains temporal consistency where appropriate

3. **Dual Reconstruction Loss:**
   - OHLC reconstruction loss (masked positions only)
   - Feature reconstruction loss (masked positions only)
   - Latent regularization to prevent representation collapse

## Quick Start

### 1. Installation

```bash
# Ensure you're in the moola project directory
cd /path/to/moola

# Install dependencies (if not already installed)
pip3 install -r requirements.txt
```

### 2. Data Preparation

```python
import numpy as np
from moola.features.feature_engineering import AdvancedFeatureEngineer, FeatureConfig

# Load your OHLC data [N, 105, 4]
X_ohlc = np.load("data/your_ohlc_data.npy")

# Configure feature engineering
feature_config = FeatureConfig(
    use_returns=True,
    use_moving_averages=True,
    use_rsi=True,
    use_macd=True,
    use_volatility=True,
    use_bollinger=True,
    # ... more options
)

# Engineer features
engineer = AdvancedFeatureEngineer(feature_config)
X_features = engineer.transform(X_ohlc)  # [N, 105, ~30]

print(f"OHLC shape: {X_ohlc.shape}")
print(f"Features shape: {X_features.shape}")
print(f"Feature names: {engineer.feature_names}")
```

### 3. Feature-Aware Pre-training

```python
from moola.config.feature_aware_config import get_feature_aware_pretraining_config
from moola.utils.feature_aware_utils import run_feature_aware_pretraining

# Configure pre-training
config = get_feature_aware_pretraining_config(
    feature_fusion="concat",  # or "add", "gate"
    preset="default"           # or "fast", "high_quality"
)

# Run pre-training
encoder_path = run_feature_aware_pretraining(
    X_ohlc=X_ohlc_unlabeled,
    X_features=X_features_unlabeled,
    config=config,
    save_dir=Path("artifacts/pretrained/feature_aware")
)

print(f"Encoder saved: {encoder_path}")
```

### 4. Transfer Learning

```python
from moola.models.enhanced_simple_lstm import EnhancedSimpleLSTMModel
from sklearn.model_selection import train_test_split

# Prepare labeled data
X_train, X_val, y_train, y_val = train_test_split(
    X_ohlc_labeled, y_labeled, test_size=0.2, random_state=42
)

# Combine with features for feature-aware mode
X_train_combined = np.concatenate([X_train, X_features_train], axis=-1)
X_val_combined = np.concatenate([X_val, X_features_val], axis=-1)

# Create enhanced model
model = EnhancedSimpleLSTMModel(
    feature_fusion="concat",
    n_epochs=50,
    batch_size=256,
    device="cuda"
)

# Train with transfer learning
model.fit(
    X_train_combined, y_train,
    pretrained_encoder_path=encoder_path,
    freeze_encoder=True,
    unfreeze_encoder_after=10  # Two-phase training
)

# Evaluate
accuracy = model.score(X_val_combined, y_val)
print(f"Validation accuracy: {accuracy:.4f}")
```

## CLI Usage

### Feature-Aware Pre-training

```bash
# With pre-computed features
python -m moola.cli_feature_aware pretrain-features \
    --ohlc data/unlabeled_ohlc.npy \
    --features data/unlabeled_features.npy \
    --fusion concat \
    --preset high_quality \
    --epochs 100 \
    --device cuda

# Compute features automatically
python -m moola.cli_feature_aware pretrain-features \
    --ohlc data/unlabeled_ohlc.npy \
    --feature-config comprehensive \
    --fusion gate \
    --output artifacts/my_encoder.pt
```

### Transfer Learning Evaluation

```bash
python -m moola.cli_feature_aware evaluate-transfer \
    --train-ohlc data/train_ohlc.npy \
    --train-features data/train_features.npy \
    --train-labels data/train_labels.npy \
    --val-ohlc data/val_ohlc.npy \
    --val-features data/val_features.npy \
    --val-labels data/val_labels.npy \
    --encoder artifacts/pretrained/feature_aware_encoder.pt \
    --modes pretrained_features \
    --fusion concat
```

### Feature Importance Analysis

```bash
python -m moola.cli_feature_aware analyze-importance \
    --encoder artifacts/pretrained/feature_aware_encoder.pt \
    --ohlc data/sample_ohlc.npy \
    --features data/sample_features.npy \
    --samples 100 \
    --device cuda
```

## Configuration

### Pre-training Configurations

```python
from moola.config.feature_aware_config import FeatureAwarePretrainingConfig

config = FeatureAwarePretrainingConfig(
    # Architecture
    ohlc_dim=4,
    feature_dim=25,  # Adjust based on your feature engineering
    hidden_dim=128,
    num_layers=2,
    dropout=0.2,
    feature_fusion="concat",  # "concat", "add", "gate"

    # Masking
    mask_ratio=0.15,
    mask_strategy="patch",    # "random", "block", "patch"
    patch_size=7,

    # Loss weights
    loss_weights={
        'ohlc_weight': 0.4,
        'feature_weight': 0.4,
        'regularization_weight': 0.2
    },

    # Training
    learning_rate=1e-3,
    batch_size=256,
    n_epochs=50,
    early_stopping_patience=10,

    # Performance
    device="cuda",
    use_amp=True,
)
```

### Feature Engineering Configurations

```python
from moola.features.feature_engineering import FeatureConfig

feature_config = FeatureConfig(
    # Price transformations
    use_returns=True,
    use_zscore=True,

    # Technical indicators
    use_moving_averages=True,
    use_rsi=True,
    use_macd=True,
    use_volatility=True,
    use_bollinger=True,
    use_atr=True,

    # Pattern recognition
    use_candle_patterns=True,
    use_swing_points=True,
    use_gaps=True,

    # Volume proxies
    use_volume_proxy=True,

    # Hyperparameters
    ma_windows=[5, 10, 20],
    rsi_period=14,
    macd_fast=12,
    macd_slow=26,
    macd_signal=9,
    volatility_windows=[5, 10, 20],
    bollinger_window=20,
    bollinger_num_std=2.0,
    atr_period=14,
)
```

## Performance Expectations

### Pre-training Performance

| GPU | Batch Size | Training Time (11K samples, 50 epochs) |
|-----|------------|-----------------------------------------|
| RTX 4090 (24GB) | 512 | ~20 minutes |
| RTX 3080 (16GB) | 256 | ~35 minutes |
| RTX 3070 (8GB)  | 128 | ~60 minutes |

### Fine-tuning Improvements

| Method | Expected Accuracy Improvement | Training Time |
|--------|------------------------------|---------------|
| OHLC-only baseline | - | 5-10 minutes |
| Feature-aware from scratch | +5-8% | 8-15 minutes |
| Pre-trained OHLC only | +8-12% | 6-12 minutes |
| Pre-trained with features | +10-15% | 8-15 minutes |

## Best Practices

### 1. Feature Engineering

- **Start comprehensive**: Enable all feature categories initially
- **Monitor correlation**: Remove highly correlated features (>0.95)
- **Domain knowledge**: Include features specific to your market/domain
- **Scaling**: Use robust scaling for outlier resistance

### 2. Pre-training

- **Data quality**: Use clean, high-quality unlabeled data
- **Sufficient data**: Aim for 10K+ samples for effective pre-training
- **Masking strategy**: Use `patch` for temporal data, `block` for longer patterns
- **Fusion strategy**: Start with `concat`, experiment with `gate` for complex relationships

### 3. Transfer Learning

- **Two-phase training**: Freeze encoder initially, unfreeze after 10-15 epochs
- **Learning rate**: Use lower LR (3e-4 to 5e-4) for fine-tuning
- **Batch size**: Adjust based on your labeled dataset size
- **Regularization**: Use appropriate dropout for your dataset size

### 4. Evaluation

- **Multiple runs**: Use different random seeds for robust evaluation
- **Cross-validation**: For small datasets, use stratified K-fold
- **Baseline comparison**: Always compare against OHLC-only baseline
- **Ablation studies**: Test impact of different feature categories

## Troubleshooting

### Common Issues

1. **Memory Errors**:
   - Reduce batch size
   - Use gradient accumulation
   - Disable mixed precision (use_amp=False)

2. **Poor Pre-training Loss**:
   - Check data quality and normalization
   - Adjust mask ratio (try 0.1-0.2)
   - Verify feature engineering output

3. **No Transfer Learning Benefit**:
   - Check domain mismatch between pre-training and fine-tuning data
   - Try different fusion strategies
   - Adjust loss weights

4. **Slow Training**:
   - Enable mixed precision (use_amp=True)
   - Increase number of workers
   - Use appropriate batch size

### Debugging

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check data shapes and types
print(f"OHLC shape: {X_ohlc.shape}, dtype: {X_ohlc.dtype}")
print(f"Features shape: {X_features.shape}, dtype: {X_features.dtype}")

# Visualize reconstruction quality
from moola.pretraining.feature_aware_masked_lstm_pretrain import visualize_feature_aware_reconstruction

sample_ohlc = torch.FloatTensor(X_ohlc[:1])
sample_features = torch.FloatTensor(X_features[:1])

viz_results = visualize_feature_aware_reconstruction(
    model, sample_ohlc, sample_features,
    mask_strategy="patch", device="cuda"
)

# Plot reconstruction quality
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
# OHLC reconstruction
axes[0,0].plot(viz_results['ohlc_original'][0, :, 3], label='Original')
axes[0,0].plot(viz_results['ohlc_reconstructed'][0, :, 3], label='Reconstructed')
axes[0,0].set_title('OHLC Reconstruction (Close)')
axes[0,0].legend()

# Feature reconstruction
axes[0,1].plot(viz_results['features_original'][0, :, 0], label='Original')
axes[0,1].plot(viz_results['features_reconstructed'][0, :, 0], label='Reconstructed')
axes[0,1].set_title('Feature Reconstruction (First Feature)')
axes[0,1].legend()

plt.show()
```

## Advanced Usage

### Custom Fusion Strategies

```python
class CustomFeatureAwareBiLSTM(FeatureAwareBiLSTMMaskedAutoencoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Add custom fusion layers
        self.custom_fusion = nn.Sequential(
            nn.Linear(self.ohlc_dim + self.feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.ohlc_dim + self.feature_dim),
            nn.Sigmoid()
        )

    def _fuse_inputs(self, ohlc, features):
        concatenated = torch.cat([ohlc, features], dim=-1)
        gate = self.custom_fusion(concatenated)
        return concatenated * gate
```

### Custom Loss Functions

```python
def custom_loss(ohlc_recon, feature_recon, ohlc_orig, feature_orig,
                ohlc_mask, feature_mask, loss_weights):
    # Standard reconstruction losses
    ohlc_loss = F.mse_loss(ohlc_recon[ohlc_mask], ohlc_orig[ohlc_mask])
    feature_loss = F.mse_loss(feature_recon[feature_mask], feature_orig[feature_mask])

    # Add custom contrastive loss
    contrastive_loss = compute_contrastive_loss(ohlc_recon, feature_recon)

    # Combine losses
    total_loss = (
        loss_weights['ohlc_weight'] * ohlc_loss +
        loss_weights['feature_weight'] * feature_loss +
        loss_weights['contrastive_weight'] * contrastive_loss
    )

    return total_loss
```

## Integration with Existing Workflows

### 1. SSH/RunPod Workflow

```bash
# On your local machine
ssh -i ~/.ssh/runpod_key ubuntu@RUNPOD_IP

# On RunPod
cd /workspace/moola
python -m moola.cli_feature_aware pretrain-features \
    --ohlc data/unlabeled_ohlc.npy \
    --features data/unlabeled_features.npy \
    --fusion concat \
    --epochs 100 \
    --device cuda

# Transfer results back
scp -i ~/.ssh/runpod_key ubuntu@RUNPOD_IP:/workspace/moola/artifacts/pretrained/*.pt ./
```

### 2. Integration with Existing Models

```python
# The enhanced SimpleLSTM is backward compatible
from moola.models.simple_lstm import SimpleLSTMModel
from moola.models.enhanced_simple_lstm import EnhancedSimpleLSTMModel

# Original SimpleLSTM (OHLC-only)
original_model = SimpleLSTMModel()
original_model.fit(X_ohlc, y)

# Enhanced SimpleLSTM (automatic mode detection)
enhanced_model = EnhancedSimpleLSTMModel()
enhanced_model.fit(X_ohlc, y)  # OHLC-only mode
enhanced_model.fit(X_combined, y)  # Feature-aware mode
```

## References and Citations

If you use this feature-aware pre-training system in your research, please cite:

```bibtex
@software{feature_aware_bilstm,
  title={Feature-Aware Bidirectional LSTM for Financial Time Series Pre-training},
  author={Moola Team},
  year={2024},
  url={https://github.com/your-repo/moola}
}
```

## Support and Contributing

For issues, questions, or contributions:
1. Check the existing documentation and examples
2. Review the troubleshooting section
3. Create an issue with detailed information
4. Follow the contribution guidelines for pull requests

---

**Next Steps:**
1. Run the example script: `python examples/feature_aware_pretraining_example.py`
2. Experiment with your own data
3. Try different fusion strategies and configurations
4. Integrate with your existing workflows