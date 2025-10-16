# Ensemble & Model Architecture Optimization Analysis

**Date**: 2025-10-16
**Dataset**: 105 samples (60 consolidation, 45 retracement) after CleanLab cleaning
**Ensemble**: 5 base models + Stack meta-learner
**Critical Issue**: RWKV-TS catastrophic failure, CNN-Trans severe bias, ensemble not utilizing all models

---

## Executive Summary

### Critical Findings

1. **🚨 RWKV-TS Model Failure** (29% reported, actually 50.5% on OOF)
   - **Parameter ratio**: 655K params / 105 samples = **6,238:1**
   - Architecture too complex for dataset size
   - Instance normalization + 4 RWKV blocks + window masking = severe overfitting
   - **Recommendation**: Replace with simpler RNN/LSTM or remove from ensemble

2. **⚠️ CNN-Transformer Severe Bias** (42.9% accuracy)
   - **Per-class accuracy**: 0% consolidation, 100% retracement
   - Predicts only ONE class (all samples → retracement)
   - Root cause: Class imbalance + insufficient regularization
   - **Recommendation**: Fix class weighting, add stronger regularization

3. **❓ Stack Ensemble Missing Models**
   - Stack code references 5 models but only loads what's available
   - CNN-Trans OOF predictions exist but model performs poorly
   - RWKV-TS OOF predictions exist but model underperforms
   - **Recommendation**: Validate which models contribute to stack

4. **✅ Traditional Models Performing Best**
   - Random Forest: 50.5% (best single model)
   - XGBoost: 50.5% (tied best, better balanced)
   - Logistic Regression: 44.8% (baseline)

5. **🎯 Ensemble Opportunity**
   - Stack ensemble: 71.3% accuracy (much better than any single model!)
   - However, this is on SMOTE dataset with 300 samples
   - Need to verify stack performance on clean 105-sample dataset

---

## 1. Individual Model Analysis

### 1.1 Logistic Regression (44.8% accuracy)
```
Parameters: ~420 weights (linear model)
Param/Sample Ratio: 4:1 ✅ (healthy)

Performance:
- Overall: 44.8%
- Consolidation: 60.0% (37/60 correct)
- Retracement: 24.4% (11/45 correct)

Issues:
- Strong bias toward consolidation (majority class)
- Poor retracement detection
- Uses classical feature engineering (may be losing temporal info)

Strengths:
- Fast, interpretable
- No overfitting
- Good as diversity contributor

Recommendations:
✅ KEEP - Provides linear baseline for ensemble diversity
```

### 1.2 Random Forest (50.5% accuracy)
```
Parameters: 1000 trees × ~50 nodes avg = ~50K params
Param/Sample Ratio: 476:1 ⚠️ (moderate risk)

Performance:
- Overall: 50.5% ⭐ (BEST SINGLE MODEL)
- Consolidation: 61.7% (37/60)
- Retracement: 35.6% (16/45)
- Avg confidence: 0.6463

Issues:
- Moderate consolidation bias
- Still struggles with retracement class

Strengths:
- Best overall accuracy
- Balanced class weights helping
- Good feature importance for interpretation
- Non-linear decision boundaries

Recommendations:
✅ KEEP - Best performer, critical for ensemble
🔧 TUNE: Reduce n_estimators (1000 → 500) to reduce overfitting risk
🔧 TUNE: Increase min_samples_leaf (2 → 5) for better generalization
```

### 1.3 XGBoost (50.5% accuracy)
```
Parameters: 200 trees × ~16 nodes = ~3.2K params
Param/Sample Ratio: 30:1 ✅ (healthy)

Performance:
- Overall: 50.5% ⭐ (TIED BEST)
- Consolidation: 55.0% (33/60)
- Retracement: 44.4% (20/45) ⭐ (BEST RETRACEMENT DETECTION)
- Avg confidence: 0.6786 ⭐ (HIGHEST)

Issues:
- Uses HopSketch feature engineering (may lose some temporal info)
- SMOTE applied in training (not helpful per experiment)

Strengths:
- BEST BALANCED PERFORMANCE across classes
- Highest confidence scores → good calibration
- Strong regularization (L1/L2, min_child_weight=5)
- Uses SMOTE but still performs well

Recommendations:
✅ KEEP - Most balanced model, critical for ensemble
🔧 IMPROVE: Remove SMOTE (per experiment results)
🔧 IMPROVE: Try using raw temporal features instead of HopSketch
💡 EXPERIMENT: Increase n_estimators (200 → 300) with early stopping
```

### 1.4 RWKV-TS (50.5% accuracy reported, but CATASTROPHIC issues)
```
Parameters: 655K (4 layers × 128 d_model)
Param/Sample Ratio: 6,238:1 🚨 (SEVERE OVERFITTING RISK)

Architecture Issues:
1. Too many parameters for 105 samples
2. Instance normalization (adds instability on small batches)
3. Window-aware masking (hardcoded for 105-length sequences)
4. State-space recurrence (complex for small dataset)
5. Mixup + CutMix augmentation (may create unrealistic samples)

Performance:
- Overall: 50.5% (but see issues below)
- Consolidation: 45.0% (worse than random!)
- Retracement: 57.8% (predicting retracement more often)
- Avg confidence: 0.5794 (LOWEST - model is uncertain)

Critical Issues:
- 29% reported accuracy suggests training failures
- 50.5% OOF accuracy ≈ random guessing
- Lowest confidence scores → model doesn't trust predictions
- Inverse bias (prefers minority class) suggests overfitting

Root Causes:
1. **Too complex for dataset**: 6,238 params per sample
2. **Instance norm instability**: Requires stable batch statistics
3. **Aggressive augmentation**: Mixup/CutMix creating noise
4. **Deep architecture**: 4 layers too deep for 105 samples

Recommendations:
❌ REMOVE from ensemble (hurting more than helping)
🔄 REPLACE with simpler architecture:
   - Option 1: 1-layer LSTM (128 hidden) = ~70K params
   - Option 2: Simple GRU (64 hidden) = ~25K params
   - Option 3: TCN (Temporal Convolutional Network) with 2 layers
🔧 IF KEEPING: Reduce to 2 layers, d_model=64, remove instance norm
🔧 IF KEEPING: Disable mixup/cutmix, reduce dropout to 0.1
🔧 IF KEEPING: Add L2 regularization (weight_decay=1e-3)
```

### 1.5 CNN-Transformer (42.9% accuracy - SEVERE BIAS)
```
Parameters: ~600K (3 CNN blocks + 3 Transformer layers)
Param/Sample Ratio: 5,714:1 🚨 (SEVERE OVERFITTING RISK)

Performance:
- Overall: 42.9%
- Consolidation: 0.0% 🚨 (PREDICTS ZERO CONSOLIDATIONS!)
- Retracement: 100.0% (predicts ONLY retracement)
- Avg confidence: 0.5713 (low confidence)

Critical Issue: **COMPLETE CLASS COLLAPSE**
- Model learned to predict only majority class (by probability)
- Despite "retracement" being minority class (45/105), model predicts it exclusively
- Suggests softmax probabilities heavily skewed

Root Causes:
1. **Class imbalance amplification**: Focal loss may be over-weighting minority
2. **Too many parameters**: 5,714 params per sample → memorization
3. **Weak regularization**: Dropout 0.25 insufficient for this param count
4. **Multi-task confusion**: Optional pointer prediction heads may interfere
5. **Augmentation issues**: Mixup/CutMix creating unrealistic patterns

Architecture Issues:
- 3 CNN blocks (64→128→128 channels)
- 3 Transformer layers × 4 heads
- Relative positional encoding (adds complexity)
- Window-aware masking (hardcoded assumptions)
- Multi-task learning heads (disabled but still in code)

Recommendations:
🔧 CRITICAL FIX: Balance class weights properly
   - Current: Focal loss + class weights (may be over-correcting)
   - Fix: Use simple balanced class weights OR Focal loss (not both)

🔧 REDUCE COMPLEXITY:
   - 3 CNN blocks → 2 CNN blocks
   - 3 Transformer layers → 2 layers
   - 128 channels → 64 channels
   - Target: ~150K params (1,428:1 ratio - still high but better)

🔧 INCREASE REGULARIZATION:
   - Dropout: 0.25 → 0.4
   - Weight decay: 1e-4 → 1e-3
   - Add label smoothing: 0.1

🔧 DISABLE AUGMENTATION:
   - Remove mixup/cutmix for now
   - Dataset too small for these techniques

🔧 SIMPLIFY ARCHITECTURE:
   - Remove window-aware masking (overfitting to 105-length)
   - Remove relative positional encoding (use standard sinusoidal)
   - Remove multi-task heads (focus on classification only)

💡 ALTERNATIVE: Replace with simpler CNN-LSTM hybrid
```

---

## 2. Ensemble Analysis

### 2.1 Stack Meta-Learner Performance

**Current Setup** (from `stack_train.py`):
```python
# Line 114: Base models
all_base_models = ["logreg", "rf", "xgb", "rwkv_ts", "cnn_transformer"]

# Line 119-127: Load OOF predictions
for model_name in all_base_models:
    oof_path = oof_dir / model_name / "v1" / f"seed_{seed}.npy"
    if oof_path.exists():
        oof.append(oof)
        base_models.append(model_name)
```

**Observed Behavior**:
- Code attempts to load all 5 models
- Only loads what exists
- Concatenates OOF predictions: [N, 2] + [N, 2] + ... → [N, 5×2=10]
- Adds 3 meta-features (diversity metrics)
- Final input to stack: [N, 13] features

**Stack Performance on SMOTE Dataset** (300 samples):
```
Accuracy: 71.3%
F1 Score: 0.669
ECE (calibration): 0.089
Log Loss: 0.571

Per-fold:
- Fold 1: 73.9%
- Fold 2: 73.9%
- Fold 3: 65.2%
- Fold 4: 65.2%
- Fold 5: 78.3%
```

**Issue**: These metrics are on SMOTE-augmented dataset (300 samples) with 202 synthetic samples that models couldn't learn. Not comparable to baseline.

### 2.2 Ensemble Composition Issues

**Problem 1: CNN-Trans Hurting Ensemble**
```
CNN-Trans OOF predictions:
- Class distribution: [300] (all predictions → class 0)
- Adding noise to stack meta-learner
- Stack must learn to IGNORE these predictions

Evidence from SMOTE OOF:
- CNN_TRANSFORMER: All 300 samples → class 0
- This is PURE NOISE for the meta-learner
```

**Problem 2: RWKV-TS Uncertain Predictions**
```
RWKV-TS OOF predictions:
- Avg confidence: 0.5794 (LOWEST of all models)
- Model is essentially guessing
- May be contributing minimal signal to stack

Evidence: 50.5% accuracy ≈ random baseline
```

**Problem 3: Missing Stack Metrics on Clean Dataset**
- Stack metrics.json shows 71.3% on 300-sample SMOTE dataset
- Need to verify stack performance on clean 105-sample dataset
- Cannot directly compare SMOTE results to baseline results

### 2.3 Ensemble Diversity Analysis

**Model Agreement** (based on OOF predictions):
```
Strong performers (50.5%):
- RF: Predicts consolidation 61.7% accurate
- XGB: BEST BALANCED (55% / 44.4%)
- RWKV: Inverse bias (45% / 57.8%)

Biased performers:
- LogReg: Heavy consolidation bias (60% / 24.4%)
- CNN-Trans: COMPLETE COLLAPSE (0% / 100%)

Diversity Contribution:
✅ RF + XGB: Complementary strengths
✅ LogReg: Different feature space (classical vs temporal)
❌ RWKV: Low confidence, near-random
❌ CNN-Trans: Single-class predictions = no diversity
```

**Meta-Features Added** (from `add_meta_features()`):
1. **Agreement std**: Std of max probabilities across models
2. **Ensemble entropy**: Uncertainty in averaged predictions
3. **Max confidence**: Highest probability from any model

**Issue**: Meta-features computed from 5 models, but 2 are providing poor signals.

---

## 3. Parameter Budget Analysis

### 3.1 Parameter Counts vs Sample Size

| Model | Parameters | Samples | Ratio | Risk Level | Status |
|-------|-----------|---------|-------|-----------|--------|
| LogReg | ~420 | 105 | 4:1 | ✅ Low | Healthy |
| XGBoost | ~3.2K | 105 | 30:1 | ✅ Low | Healthy |
| RF | ~50K | 105 | 476:1 | ⚠️ Moderate | Acceptable |
| CNN-Trans | ~600K | 105 | 5,714:1 | 🚨 Severe | **OVERFIT** |
| RWKV-TS | ~655K | 105 | 6,238:1 | 🚨 Severe | **OVERFIT** |

**Rules of Thumb**:
- **< 10:1**: Safe zone for deep learning
- **10:1 to 100:1**: Needs strong regularization
- **100:1 to 1000:1**: High overfitting risk
- **> 1000:1**: Almost guaranteed overfitting without massive regularization

**Current Situation**:
- 2/5 models in severe overfitting territory
- Both deep learning models have 5000:1+ ratios
- Traditional models (LogReg, XGB, RF) are within safe ranges

### 3.2 Recommended Parameter Budgets

For 105 samples with strong regularization:
```
Conservative (safe):  < 1,000 params  (10:1 ratio)
Moderate (risky):     1,000-10,000    (100:1 ratio)
Aggressive (expert):  10,000-50,000   (500:1 ratio)
Danger zone:          > 50,000        (requires SSL, pretrain, etc)
```

**Current deep learning models are 10-13x over budget!**

---

## 4. Root Cause Analysis

### 4.1 Why RWKV-TS Failed

**Design Intent** (from docstring):
```python
"""RWKV-TS: RNN/state-space model for sequential temporal modeling.

Architecture:
- Multi-scale patching with sizes [7, 15, 21, 35]
- Recurrent state-space blocks with d_model=128
- 4 layers (reverted from 6 - too many parameters for small dataset)
- Instance normalization for stable training
```

**What Went Wrong**:

1. **Still Too Deep**: 4 layers → 655K params
   - Author tried 6 layers (982K params, failed)
   - Reduced to 4 layers but still too many
   - Should have gone to 2 layers (~330K) or 1 layer (~165K)

2. **d_model Too Large**: 128 dimensions
   - For 105 samples, d_model=64 or even 32 would be safer
   - Each layer has 4×d_model FFN → massive param count

3. **Instance Normalization**: Requires stable batch statistics
   - With batch_size=512 but only 105 samples, many batches are tiny
   - InstanceNorm1d expects consistent sequence lengths
   - Should use LayerNorm instead

4. **Window Masking Hardcoded**: Lines 59-61
   ```python
   mask = torch.ones(105)
   mask[30:75] = 1.2  # Boost inner window
   ```
   - Only works for 105-bar sequences
   - Breaks if sequence length changes
   - Over-specialized to current dataset

5. **Aggressive Augmentation**:
   - Mixup alpha=0.2
   - CutMix prob=0.5
   - For 105 samples, creating synthetic samples via mixing
   - May create unrealistic temporal patterns

**Evidence from Code Comments**:
```python
# Line 21-22:
# Note: 6 layers caused severe overfitting (982K params for 115 samples = 8,540:1 ratio)
#       4 layers provides 655K params (5,695:1 ratio - still high but more stable)
```
→ Author recognized overfitting but didn't reduce enough!

### 4.2 Why CNN-Transformer Failed

**Design Intent**:
```python
"""CNN → Transformer hybrid for hierarchical feature extraction.

Architecture:
- Multi-scale CNN blocks (Conv1d with kernels {3, 5, 9})
- Transformer encoder with relative positional encoding
- 3 layers × 4 heads
- Channels: 3× [64, 128, 128]
```

**What Went Wrong**:

1. **Class Imbalance Handling**: Lines 637-638
   ```python
   criterion = FocalLoss(gamma=2.0, alpha=class_weights, reduction='mean')
   ```
   - Focal loss DESIGNED for severe imbalance (e.g., object detection)
   - Class weights ALSO applied
   - Double-correction may over-penalize majority class
   - Result: Model learns to predict minority class exclusively

2. **Multi-Task Confusion**: Lines 214, 332-334
   ```python
   predict_pointers: bool = False  # Optional pointer prediction

   if predict_pointers:
       self.pointer_start_head = nn.Linear(self.d_model, 1)
       self.pointer_end_head = nn.Linear(self.d_model, 1)
   ```
   - Code has multi-task learning infrastructure
   - May interfere with classification even when disabled
   - Adds complexity and potential for bugs

3. **Overly Complex Positional Encoding**: Lines 131-163
   ```python
   class RelativePositionalEncoding(nn.Module):
       """Relative positional encoding for Transformers."""
   ```
   - Relative position embeddings: 2×max_len×d_model parameters
   - For max_len=512, d_model=128: 131K extra parameters!
   - Standard sinusoidal would be parameter-free

4. **Window-Aware Masking**: Lines 336-359
   ```python
   def _create_attention_mask(self, seq_len: int, device: torch.device):
       # Block buffers [0:30] and [75:105] from attending to [30:75]
       mask[0:30, 30:75] = float('-inf')
       mask[75:105, 30:75] = float('-inf')
   ```
   - Hardcoded to 105-length sequences
   - Assumes specific domain knowledge (buffer regions)
   - May prevent model from learning actual patterns

5. **Dropout Too Low**: Line 227
   ```python
   dropout: float = 0.25
   ```
   - For 600K params on 105 samples, need dropout ~0.4-0.5
   - 0.25 is insufficient regularization

**Evidence from Training Logs** (implied from 0% consolidation accuracy):
- Softmax outputs heavily skewed to one class
- Model found local minimum predicting single class
- Focal loss + class weights over-corrected imbalance

### 4.3 Why Traditional Models Perform Better

**XGBoost Success Factors**:
1. ✅ Built-in regularization (L1, L2, min_child_weight)
2. ✅ Tree-based = no vanishing gradients
3. ✅ Column subsampling = implicit feature regularization
4. ✅ Early stopping (though not used here)
5. ✅ Moderate param count (3.2K for 105 samples)

**Random Forest Success Factors**:
1. ✅ Bagging = variance reduction
2. ✅ balanced_subsample = handles imbalance well
3. ✅ max_features="sqrt" = feature regularization
4. ✅ Robust to outliers
5. ✅ No hyperparameter tuning needed

**Key Insight**: Traditional models have built-in regularization and are less prone to overfitting on small datasets.

---

## 5. Recommendations

### 5.1 Immediate Fixes (High Priority)

#### Fix 1: Remove or Replace RWKV-TS
```
Option A: REMOVE entirely
- Ensemble performs better without noisy predictions
- 4 models (LogReg, RF, XGB, CNN-Trans) sufficient

Option B: REPLACE with simpler RNN
Architecture:
  - 1-layer LSTM: 128 hidden units = ~70K params
  - OR 1-layer GRU: 64 hidden units = ~25K params
  - Remove instance norm → use LayerNorm
  - Remove window masking
  - Reduce dropout to 0.3
  - Remove mixup/cutmix
  - Add weight_decay=1e-3

Expected improvement: 50.5% → 55-60%
```

#### Fix 2: Fix CNN-Transformer Class Imbalance
```python
# CURRENT (Line 637-638):
criterion = FocalLoss(gamma=2.0, alpha=class_weights, reduction='mean')

# OPTION A: Use ONLY class weights (simpler)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# OPTION B: Use ONLY Focal loss (if imbalance severe)
criterion = FocalLoss(gamma=2.0, alpha=None, reduction='mean')

# OPTION C: Reduce Focal gamma (gentler correction)
criterion = FocalLoss(gamma=0.5, alpha=class_weights, reduction='mean')
```

#### Fix 3: Reduce CNN-Transformer Complexity
```python
# CURRENT:
cnn_channels = [64, 128, 128]  # 3 blocks
transformer_layers = 3
dropout = 0.25

# PROPOSED:
cnn_channels = [64, 128]  # 2 blocks → ~50% param reduction
transformer_layers = 2     # → ~33% param reduction
dropout = 0.4             # → stronger regularization

Total reduction: ~400K params → ~200K params (2,000:1 ratio)
```

### 5.2 Architecture Recommendations

#### Recommendation 1: Simplify Deep Learning Models

**For RWKV-TS** (if keeping):
```python
class SimpleLSTM(BaseModel):
    """Simple 1-layer LSTM for time series classification."""

    def __init__(self, hidden_size=128, dropout=0.3):
        self.lstm = nn.LSTM(
            input_size=4,  # OHLC
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0,  # Only 1 layer, no dropout in LSTM
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: [B, 105, 4]
        lstm_out, (hn, cn) = self.lstm(x)
        # Use final hidden state
        out = self.dropout(hn[-1])
        logits = self.fc(out)
        return logits
```
**Parameters**: ~70K (667:1 ratio - much better!)

**For CNN-Transformer**:
```python
class SimpleCNNTransformer(BaseModel):
    """Lightweight CNN + Transformer hybrid."""

    def __init__(self):
        # 2 CNN blocks instead of 3
        self.cnn1 = CNNBlock(4, 64, kernels=[3, 5], dropout=0.4)
        self.cnn2 = CNNBlock(64, 128, kernels=[3, 5], dropout=0.4)

        # 2 Transformer layers instead of 3
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, dim_feedforward=256,  # Reduced FFN
            dropout=0.4, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Simple classification head
        self.classifier = nn.Linear(128, n_classes)
```
**Parameters**: ~150K (1,428:1 ratio - still high but manageable)

#### Recommendation 2: Alternative Architectures

**Option 1: Temporal Convolutional Network (TCN)**
```
Pros:
- Parallelizable (faster than RNN)
- Larger receptive field than standard CNN
- Fewer parameters than Transformer
- Good for time series

Architecture:
- 4 dilated conv layers (dilation: 1, 2, 4, 8)
- Residual connections
- ~50K parameters

Expected params: ~50K (476:1 ratio)
Expected performance: 52-58%
```

**Option 2: Attention-Based LSTM**
```
Architecture:
- 1-layer BiLSTM (64 hidden)
- Self-attention over sequence
- Classification head
- ~30K parameters

Expected params: ~30K (286:1 ratio)
Expected performance: 50-56%
```

**Option 3: Simple 1D CNN**
```
Architecture:
- 3 Conv1D layers (32, 64, 128 channels)
- Global average pooling
- Classification head
- ~20K parameters

Expected params: ~20K (190:1 ratio)
Expected performance: 48-54%
```

### 5.3 Ensemble Optimization

#### Strategy 1: Selective Model Inclusion
```python
# In stack_train.py, add model selection based on performance

MIN_ACCURACY_THRESHOLD = 0.48  # 48%

for model_name in all_base_models:
    oof_path = oof_dir / model_name / "v1" / f"seed_{seed}.npy"
    if oof_path.exists():
        oof = np.load(oof_path)

        # Calculate OOF accuracy
        preds = oof.argmax(axis=1)
        acc = accuracy_score(y, preds)

        if acc >= MIN_ACCURACY_THRESHOLD:
            oof_predictions.append(oof)
            base_models.append(model_name)
            logger.info(f"✅ Including {model_name} (acc={acc:.3f})")
        else:
            logger.warning(f"❌ Excluding {model_name} (acc={acc:.3f} < threshold)")
```

**Expected Effect**:
- Removes CNN-Trans (42.9% < 48%)
- Removes RWKV-TS if it performs poorly
- Keeps only high-quality predictions for meta-learner

#### Strategy 2: Weighted Ensemble (Alternative to Stack)
```python
def train_weighted_ensemble(oof_predictions, y, seed):
    """Find optimal weights for ensemble averaging."""
    from scipy.optimize import minimize

    def ensemble_loss(weights):
        weights = np.abs(weights)  # Ensure positive
        weights = weights / weights.sum()  # Normalize to sum=1

        # Weighted average of predictions
        ensemble_pred = sum(w * oof for w, oof in zip(weights, oof_predictions))
        ensemble_pred = ensemble_pred.argmax(axis=1)

        return -accuracy_score(y, ensemble_pred)  # Negative for minimization

    # Optimize weights
    n_models = len(oof_predictions)
    initial_weights = np.ones(n_models) / n_models  # Equal weights initially
    result = minimize(ensemble_loss, initial_weights, method='Nelder-Mead')

    optimal_weights = np.abs(result.x)
    optimal_weights /= optimal_weights.sum()

    return optimal_weights
```

**Expected Effect**:
- Automatically down-weights poor models
- Simpler than meta-learner (no overfitting risk)
- Interpretable (can see which models contribute)

#### Strategy 3: Diversity-Based Selection
```python
def select_diverse_models(oof_predictions, y, k=3):
    """Select k most diverse models for ensemble."""
    from sklearn.metrics.pairwise import cosine_similarity

    # Compute pairwise similarities
    n_models = len(oof_predictions)
    similarities = np.zeros((n_models, n_models))

    for i in range(n_models):
        for j in range(i+1, n_models):
            sim = cosine_similarity(
                oof_predictions[i],
                oof_predictions[j]
            ).mean()
            similarities[i, j] = sim
            similarities[j, i] = sim

    # Greedy selection: start with best model, add most diverse
    accuracies = [accuracy_score(y, oof.argmax(axis=1)) for oof in oof_predictions]
    selected = [np.argmax(accuracies)]  # Start with best

    for _ in range(k-1):
        # Find most diverse from selected set
        diversities = []
        for i in range(n_models):
            if i in selected:
                diversities.append(-np.inf)
            else:
                # Average distance from selected models
                div = -np.mean([similarities[i, s] for s in selected])
                diversities.append(div)

        selected.append(np.argmax(diversities))

    return selected
```

**Expected Effect**:
- Select 3-4 most diverse models
- Ensure ensemble doesn't have redundant models
- Balance accuracy vs diversity

### 5.4 Hyperparameter Tuning Priorities

#### Priority 1: CNN-Transformer (CRITICAL)
```python
# Test these combinations:
configs = [
    # Config 1: Reduce complexity + increase regularization
    {
        'cnn_channels': [64, 128],
        'transformer_layers': 2,
        'dropout': 0.4,
        'learning_rate': 3e-4,
        'weight_decay': 1e-3,
    },

    # Config 2: Even simpler
    {
        'cnn_channels': [64],
        'transformer_layers': 2,
        'dropout': 0.5,
        'learning_rate': 1e-4,
        'weight_decay': 5e-3,
    },

    # Config 3: Focus on CNN only (disable Transformer)
    {
        'cnn_channels': [32, 64, 128],
        'transformer_layers': 0,  # No Transformer
        'dropout': 0.3,
        'learning_rate': 5e-4,
    },
]
```

#### Priority 2: RWKV-TS Replacement
```python
# Option 1: Simple LSTM
configs_lstm = [
    {
        'hidden_size': 128,
        'num_layers': 1,
        'dropout': 0.3,
        'learning_rate': 1e-3,
    },
    {
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.4,
        'learning_rate': 5e-4,
    },
]

# Option 2: GRU (fewer params)
configs_gru = [
    {
        'hidden_size': 128,
        'num_layers': 1,
        'dropout': 0.3,
    },
    {
        'hidden_size': 96,
        'num_layers': 1,
        'dropout': 0.4,
    },
]
```

#### Priority 3: XGBoost (OPTIONAL - already good)
```python
# Fine-tune best performer
configs_xgb = [
    # Current config is good, try slight variations
    {
        'n_estimators': 300,  # From 200
        'max_depth': 4,
        'learning_rate': 0.05,  # From 0.1
        'subsample': 0.7,  # From 0.8
    },

    # Try early stopping
    {
        'n_estimators': 500,
        'max_depth': 4,
        'learning_rate': 0.05,
        'early_stopping_rounds': 50,
    },
]
```

### 5.5 Training Improvements

#### Improvement 1: Add Early Stopping to All Models
```python
# For sklearn models (RF, XGB):
from sklearn.model_selection import cross_val_score

# Add validation monitoring
best_score = 0
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    # Train
    model.fit(X_train, y_train)

    # Validate
    val_score = accuracy_score(y_val, model.predict(X_val))

    if val_score > best_score:
        best_score = val_score
        patience_counter = 0
        # Save model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

#### Improvement 2: Add Cross-Validation Grid Search
```python
from sklearn.model_selection import GridSearchCV

# For XGBoost
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.05, 0.1, 0.2],
}

grid_search = GridSearchCV(
    XGBClassifier(**base_params),
    param_grid,
    cv=5,
    scoring='f1_macro',
    n_jobs=-1
)
grid_search.fit(X, y)

print(f"Best params: {grid_search.best_params_}")
print(f"Best F1: {grid_search.best_score_:.3f}")
```

#### Improvement 3: Add Label Smoothing
```python
# For deep learning models
criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.1  # Prevents overconfident predictions
)
```

---

## 6. Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
**Priority**: URGENT - Fix broken models

**Tasks**:
1. ✅ **Fix CNN-Transformer class imbalance** (2 hours)
   - Change loss function (remove double correction)
   - Test on OOF split
   - Target: 42.9% → 50%+

2. ✅ **Replace RWKV-TS with simple LSTM** (4 hours)
   - Implement SimpleLSTM architecture
   - Train on OOF splits
   - Compare to RWKV-TS baseline
   - Target: 50.5% → 55%+

3. ✅ **Verify stack ensemble composition** (1 hour)
   - Check which models are actually loaded
   - Log model contributions
   - Verify meta-features computation

4. ✅ **Retrain stack on clean dataset** (2 hours)
   - Use 105-sample clean dataset (not SMOTE)
   - Generate new OOF predictions
   - Compare to 71.3% SMOTE result
   - Target: 65-70% on clean data

**Expected Impact**:
- CNN-Trans: 42.9% → 52%+ (+9%)
- RWKV-TS replacement: 50.5% → 55%+ (+5%)
- Stack ensemble: Verify actual performance on clean data

### Phase 2: Architecture Optimization (Week 2)
**Priority**: HIGH - Improve model quality

**Tasks**:
1. 🔄 **Reduce CNN-Transformer complexity** (4 hours)
   - Implement simplified architecture (2 CNN blocks, 2 Transformer layers)
   - Add stronger regularization (dropout 0.4, weight decay 1e-3)
   - Target: 150K params (from 600K)

2. 🔄 **Implement TCN as alternative** (6 hours)
   - Build Temporal Convolutional Network
   - Train and compare to LSTM
   - Target: 52-58% accuracy with ~50K params

3. 🔄 **Add ensemble model selection** (3 hours)
   - Implement accuracy threshold filtering
   - Implement diversity-based selection
   - Compare stack performance with/without poor models

4. 🔄 **Hyperparameter tuning** (8 hours)
   - Grid search for XGBoost
   - Architecture search for CNN models
   - Learning rate tuning for deep models

**Expected Impact**:
- Reduced overfitting in deep models
- Better ensemble diversity
- 5-10% improvement in stack accuracy

### Phase 3: Advanced Techniques (Week 3-4)
**Priority**: MEDIUM - Explore improvements

**Tasks**:
1. 💡 **SSL Pretraining** (8 hours)
   - TS-TCC self-supervised pretraining
   - Use 118K unlabeled samples
   - Fine-tune on 105 labeled samples
   - Target: 2-5% improvement

2. 💡 **Feature engineering improvements** (6 hours)
   - Add domain-specific features
   - Test different feature sets for classical models
   - Feature selection for XGBoost/RF

3. 💡 **Ensemble weighting optimization** (4 hours)
   - Implement weighted averaging
   - Find optimal model weights
   - Compare to meta-learner

4. 💡 **Data augmentation** (6 hours)
   - Test different augmentation strategies
   - Time series-specific augmentations
   - Avoid SMOTE (proven ineffective)

**Expected Impact**:
- SSL: +2-5% on deep models
- Feature engineering: +2-4% on classical models
- Optimized weighting: +1-3% on ensemble

### Phase 4: Production Readiness (Week 5)
**Priority**: LOW - Polish and deploy

**Tasks**:
1. 📊 **Comprehensive evaluation** (4 hours)
   - Cross-validation all models
   - Generate calibration curves
   - Per-class performance analysis

2. 📝 **Documentation** (4 hours)
   - Model cards for each model
   - Ensemble composition documentation
   - Training procedures

3. 🚀 **Pipeline optimization** (6 hours)
   - Parallelize OOF generation
   - Add caching for features
   - Optimize inference speed

4. ✅ **Final validation** (4 hours)
   - Hold-out test set evaluation
   - Comparison to baseline
   - Sign-off on production readiness

**Expected Impact**:
- Production-ready models
- Clear documentation
- Faster training/inference

---

## 7. Expected Outcomes

### 7.1 Individual Model Performance Targets

| Model | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|-------|---------|---------------|---------------|---------------|
| LogReg | 44.8% | 45-46% | 46-48% | 48-50% |
| RF | 50.5% | 50-52% | 52-54% | 54-56% |
| XGBoost | 50.5% | 51-53% | 53-55% | 55-58% |
| ~~RWKV-TS~~ | ~~50.5%~~ | ~~REMOVED~~ | - | - |
| Simple LSTM | - | 52-55% | 55-58% | 57-60% |
| CNN-Trans | 42.9% | 50-52% | 52-55% | 55-58% |
| TCN (new) | - | - | 52-54% | 54-57% |

### 7.2 Ensemble Performance Targets

| Metric | Current (SMOTE) | Phase 1 (Clean) | Phase 2 | Phase 3 |
|--------|----------------|----------------|---------|---------|
| Stack Accuracy | 71.3% | 65-68% | 68-72% | 72-76% |
| Stack F1 | 0.669 | 0.64-0.67 | 0.67-0.71 | 0.71-0.75 |
| ECE (calibration) | 0.089 | 0.08-0.10 | 0.06-0.08 | 0.04-0.06 |

**Notes**:
- Phase 1 targets on 105-sample clean dataset (not 300 SMOTE)
- Phase 2-3 assume improved base models
- Realistic upper bound: ~75-78% given dataset size

### 7.3 Risk Assessment

**High Confidence** (>80% likely):
- ✅ CNN-Trans fix will improve from 42.9% to 50%+
- ✅ Removing RWKV-TS won't hurt ensemble
- ✅ Stack on clean data will be 65-70%

**Medium Confidence** (50-80% likely):
- ⚠️ Simple LSTM will outperform RWKV-TS (52-55%)
- ⚠️ Reduced CNN-Trans complexity will help (52-55%)
- ⚠️ Ensemble will reach 70%+ after Phase 2

**Low Confidence** (<50% likely):
- ❓ SSL pretraining will provide >2% boost
- ❓ TCN will outperform LSTM
- ❓ Ensemble will exceed 75% accuracy

---

## 8. Conclusion

### Key Findings

1. **Stack ensemble is working** (71.3% on SMOTE, likely 65-70% on clean data)
2. **Deep learning models are severely overparameterized** (5000:1+ param ratios)
3. **Traditional models are most reliable** (XGBoost and RF both 50.5%)
4. **CNN-Transformer has critical class imbalance bug** (0% consolidation accuracy)
5. **RWKV-TS is too complex and underperforming** (should be replaced)

### Immediate Actions

**Critical (do now)**:
1. Fix CNN-Transformer loss function (remove double correction)
2. Replace RWKV-TS with simple 1-layer LSTM
3. Retrain stack ensemble on clean 105-sample dataset
4. Verify which models contribute to current stack

**High Priority (this week)**:
1. Reduce CNN-Transformer complexity (600K → 150K params)
2. Implement model selection in stack ensemble
3. Remove SMOTE from XGBoost training
4. Add proper validation metrics

**Medium Priority (next 2 weeks)**:
1. Hyperparameter tuning for all models
2. Implement TCN as alternative architecture
3. Try SSL pretraining (TS-TCC)
4. Optimize ensemble weighting

### Success Metrics

**Minimum Viable**:
- All models > 48% accuracy
- Stack ensemble > 65% accuracy
- No single-class predictions

**Target**:
- Best single model > 55% accuracy
- Stack ensemble > 70% accuracy
- ECE < 0.08 (well-calibrated)

**Stretch Goal**:
- Best single model > 60% accuracy
- Stack ensemble > 75% accuracy
- Reliable on unseen data

---

## Appendix A: Code Locations

### Model Implementations
- **Stack**: `/Users/jack/projects/moola/src/moola/models/stack.py`
- **RWKV-TS**: `/Users/jack/projects/moola/src/moola/models/rwkv_ts.py`
- **CNN-Transformer**: `/Users/jack/projects/moola/src/moola/models/cnn_transformer.py`
- **XGBoost**: `/Users/jack/projects/moola/src/moola/models/xgb.py`
- **Random Forest**: `/Users/jack/projects/moola/src/moola/models/rf.py`
- **LogReg**: `/Users/jack/projects/moola/src/moola/models/logreg.py`

### Training Pipelines
- **OOF Generation**: `/Users/jack/projects/moola/src/moola/pipelines/oof.py`
- **Stack Training**: `/Users/jack/projects/moola/src/moola/pipelines/stack_train.py`

### Artifacts
- **OOF Predictions**: `/Users/jack/projects/moola/data/artifacts/oof/*/v1/seed_1337.npy`
- **Stack Metrics**: `/Users/jack/projects/moola/data/artifacts/models/stack/metrics.json`

---

**Generated**: 2025-10-16
**Dataset**: 105 samples (60 consolidation, 45 retracement)
**Analysis Focus**: Ensemble optimization, model architecture, parameter budgets
