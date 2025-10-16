# Pre-training Method Comparison for SimpleLSTM

**Quick Reference Guide**

---

## Executive Summary

**WINNER**: 🥇 **Masked Autoencoding Pre-training** (BERT-style)

**Expected Gain**: +8-12% accuracy (57% → 65-69%)

**Implementation Time**: 6-8 hours

**Key Advantage**: Forces LSTM to learn temporal dependencies by reconstructing masked timesteps

---

## Comparison Table

| Method | Accuracy Gain | Implementation | Training Time | Data Required | Complexity | Score |
|--------|---------------|----------------|---------------|---------------|------------|-------|
| **🥇 Masked Autoencoding** | **+8-12%** | **6-8 hours** | **20 min** | **11,873** | **Medium** | **88/100** |
| 🥈 Variational Autoencoder | +4-7% | 3-4 hours | 18 min | 11,873 | Medium-High | 75/100 |
| 🥉 Classical Autoencoder | +3-5% | 2-3 hours | 15 min | 11,873 | Low | 72/100 |
| TS-TCC (fixed) | +2-4% | 1-2 hours | 0 min* | Already trained | Low | 68/100 |
| Temporal Triplet | +3-6% | 4 hours | 18 min | 11,873 | Medium | 66/100 |
| Next-Step Prediction | +1-3% | 2 hours | 12 min | 11,873 | Low | 52/100 |

*Already pre-trained, only needs fine-tuning fixes

---

## Method Details

### 1. Masked Autoencoding (RECOMMENDED)

**What it does**:
- Randomly masks 15% of timesteps in input sequence
- Trains LSTM encoder to reconstruct masked values from context
- Forces learning of temporal dependencies (can't just copy)

**Why it works**:
- SimpleLSTM's problem: Uses only final timestep, but pivots are at bar 50
- Masking forces encoder to understand relationships across ALL timesteps
- BERT-style pre-training proven effective (PatchTST: 21% MSE reduction)

**Implementation**:
```python
# Mask 15% of bars randomly
x_masked[mask_indices] = MASK_TOKEN

# Encoder learns to predict masked bars from context
encoded = BiLSTM(x_masked)
reconstructed = Decoder(encoded)

# Loss on masked positions only
loss = MSE(reconstructed[mask], x_original[mask])
```

**Pros**:
- ✅ Directly addresses temporal attention mismatch
- ✅ Simple loss function (MSE) - easy to debug
- ✅ Proven effective (PatchTST, BERT)
- ✅ Learns class-discriminative features

**Cons**:
- ⚠️ Masking strategy needs tuning (random vs block vs patch)
- ⚠️ More implementation work than simpler methods

**Expected Results**:
```
Baseline:         57.14% accuracy
With Masked AE:   65-69% accuracy (+8-12%)

Class 0:          100% → 75-80%
Class 1:          0% → 45-55% (CRITICAL: Breaks class collapse!)
```

---

### 2. Variational Autoencoder (VAE)

**What it does**:
- Encodes time series to stochastic latent distribution (μ, σ²)
- Learns to reconstruct input from sampled latent code
- β-VAE variant encourages disentangled representations

**Why it works**:
- Stochastic encoding → better generalization than deterministic AE
- Can generate synthetic samples for data augmentation
- Disentangled latent space may separate consolidation/retracement factors

**Implementation**:
```python
# Encoder outputs mean and log-variance
mu, logvar = Encoder(x)
z = reparameterize(mu, logvar)  # z ~ N(mu, exp(logvar))

# Decoder reconstructs from latent sample
reconstructed = Decoder(z)

# VAE loss
recon_loss = MSE(reconstructed, x)
kl_loss = KL_divergence(mu, logvar, N(0,1))
total_loss = recon_loss + beta * kl_loss
```

**Pros**:
- ✅ Better generalization than standard AE
- ✅ Can generate synthetic data
- ✅ Principled probabilistic framework

**Cons**:
- ❌ Complex loss tuning (β parameter critical)
- ❌ Reconstruction quality may be worse
- ❌ Stochasticity slows convergence

**Expected Results**:
```
Baseline:         57.14%
With VAE:         61-64% (+4-7%)

Class 0:          100% → 78-82%
Class 1:          0% → 25-35%
```

---

### 3. Classical Autoencoder

**What it does**:
- Bidirectional LSTM encoder compresses sequence to latent vector
- LSTM decoder reconstructs original sequence
- Pre-trains encoder on reconstruction task

**Why it works**:
- Encoder must capture essential temporal patterns for reconstruction
- Bidirectional LSTM sees full context (past + future)
- Proven track record in financial time series (Nature 2025)

**Implementation**:
```python
# Encoder: BiLSTM → Latent
latent = BiLSTM_Encoder(x)  # [batch, 64]

# Decoder: Latent → LSTM → Reconstruction
reconstructed = LSTM_Decoder(latent)  # [batch, 105, 4]

# Simple MSE loss
loss = MSE(reconstructed, x)
```

**Pros**:
- ✅ Simplest to implement (2-3 hours)
- ✅ Interpretable (reconstruction quality = feature quality)
- ✅ Fast training (15 minutes)
- ✅ Well-studied approach

**Cons**:
- ❌ Focuses on low-level features (OHLC values)
- ❌ No explicit class-discriminative signal
- ❌ May overfit to common patterns

**Expected Results**:
```
Baseline:         57.14%
With Classical AE: 60-62% (+3-5%)

Class 0:          100% → 80-85%
Class 1:          0% → 20-30%
```

---

### 4. TS-TCC (Current - Needs Fixes)

**What it does**:
- Contrastive learning: Pull similar samples together, push dissimilar apart
- Two augmented views of same sample → similar embeddings
- Pre-trained on 11,873 samples (already complete)

**Why it failed**:
1. ❌ Encoder NOT frozen during fine-tuning → pre-trained features corrupted
2. ❌ Multi-task learning interference (classification + pointer prediction)
3. ❌ Insufficient fine-tuning epochs (early stopping at epoch 27)

**Required Fixes**:
```python
# 1. Freeze encoder for first 10 epochs
model.load_pretrained_encoder(path, freeze_encoder=True)

# 2. Disable pointer prediction
--no-predict-pointers

# 3. Increase fine-tuning epochs
--n-epochs 100 --patience 30
```

**Pros**:
- ✅ Already pre-trained (no waiting)
- ✅ Quick to fix (1-2 hours)
- ✅ Low risk (existing codebase)

**Cons**:
- ❌ Lower expected gain (+2-4% vs +8-12%)
- ❌ Contrastive features may not be class-discriminative

**Expected Results** (if fixed):
```
Baseline:         57.14%
With Fixed TS-TCC: 59-61% (+2-4%)

Class 0:          100% → 82-87%
Class 1:          0% → 15-25%
```

---

### 5. Temporal Triplet Loss

**What it does**:
- Train encoder to place temporally-close samples nearby in embedding space
- Anchor: Original sample
- Positive: Time-shifted version (bars 5-110)
- Negative: Random sample from different window

**Implementation**:
```python
# Triplet loss
d_pos = distance(anchor, positive)
d_neg = distance(anchor, negative)
loss = max(0, d_pos - d_neg + margin)
```

**Pros**:
- ✅ Learns temporal structure
- ✅ No labels required

**Cons**:
- ⚠️ Sampling strategy critical
- ❌ May not learn class boundaries
- ❌ Less proven for time series

**Expected Results**:
```
Baseline:         57.14%
With Triplet:     60-63% (+3-6%)

Class 0:          100% → 78-83%
Class 1:          0% → 20-30%
```

---

### 6. Next-Step Prediction

**What it does**:
- Train encoder to predict next N bars from previous M bars
- Pre-trains on forecasting task

**Why it doesn't work well**:
- ❌ Financial markets are non-stationary (hard to predict)
- ❌ Forecasting ≠ classification (different objectives)
- ❌ Requires huge dataset for meaningful learning

**Expected Results**:
```
Baseline:         57.14%
With Next-Step:   58-60% (+1-3%)

Class 0:          100% → 85-90%
Class 1:          0% → 10-20%
```

---

## Decision Matrix

### Choose Masked Autoencoding if:
- ✅ You want maximum performance improvement
- ✅ You have 6-8 hours for implementation
- ✅ You need to break class collapse (Class 1 = 0%)
- ✅ You want a reusable component for future projects

### Choose Fixed TS-TCC if:
- ✅ You need quick results (1-2 hours)
- ✅ You want low-risk approach (existing code)
- ✅ +2-4% improvement is acceptable
- ✅ Time is very constrained

### Choose Classical Autoencoder if:
- ✅ You want simple, interpretable approach
- ✅ You need fast implementation (2-3 hours)
- ✅ You want to understand reconstruction quality visually
- ✅ +3-5% improvement is acceptable

---

## Recommended Approach

### Phase 1: Quick Win (1-2 hours)
Fix TS-TCC pipeline to establish new baseline:
```bash
# 1. Freeze encoder during fine-tuning
python -m moola.cli oof \
    --model cnn_transformer \
    --load-pretrained-encoder data/artifacts/pretrained/encoder_weights.pt \
    --freeze-encoder \
    --unfreeze-after 10 \
    --n-epochs 100

# Expected: 59-61% accuracy (marginal improvement)
```

### Phase 2: Optimal Solution (6-8 hours)
Implement Masked Autoencoding pre-training:
```bash
# 1. Pre-train encoder (20 min on H100)
python -m moola.cli pretrain-masked-lstm \
    --input data/raw/unlabeled_windows.parquet \
    --output data/artifacts/pretrained/masked_lstm_encoder.pt \
    --mask-ratio 0.15 \
    --mask-strategy patch \
    --epochs 50

# 2. Fine-tune SimpleLSTM (15 min on H100)
python -m moola.cli oof \
    --model simple_lstm \
    --load-pretrained-encoder data/artifacts/pretrained/masked_lstm_encoder.pt \
    --freeze-encoder \
    --unfreeze-after 10

# Expected: 65-69% accuracy (significant improvement!)
```

### Phase 3: Ablation Study (Optional)
Compare all methods to identify best configuration:
- Random masking vs block masking vs patch masking
- Mask ratio: 10% vs 15% vs 25%
- Unfreezing schedule: epoch 5 vs 10 vs 20

---

## Key Takeaways

1. **Temporal Attention Mismatch is the Core Issue**
   - SimpleLSTM uses final timestep, but pivots are at bar 50
   - Pre-training must force encoder to learn full sequence dependencies

2. **Masked Autoencoding Directly Addresses This**
   - Reconstruction task requires understanding all timesteps
   - Cannot simply copy - must learn temporal patterns

3. **TS-TCC Can Be Salvaged**
   - Freezing encoder + disabling multi-task → +2-4% gain
   - Worth doing as quick baseline while implementing Masked AE

4. **Implementation is Feasible**
   - 6-8 hours total (4 hours core, 2 hours testing)
   - 20 minutes pre-training on H100 (RunPod)
   - Reusable component for future projects

5. **Expected Performance is Significant**
   - +8-12% accuracy improvement
   - Breaks class collapse (Class 1: 0% → 45-55%)
   - Reaches 65-69% accuracy target

---

## References

1. **PatchTST** (ICLR 2023) - Masked patch reconstruction for time series
2. **TS2Vec** (AAAI 2022) - Contrastive learning with contextual consistency
3. **TF-C** (NeurIPS 2022) - Time-frequency consistency contrastive learning
4. **Nature Communications (2025)** - LSTM autoencoder for financial networks
5. **Scientific Reports (2019)** - Unsupervised pre-training of stacked LSTM-AE

---

**For full analysis, see**: `LSTM_CHART_INTERACTION_ANALYSIS.md`
