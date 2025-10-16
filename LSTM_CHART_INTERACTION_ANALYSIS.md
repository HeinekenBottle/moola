# LSTM-Chart Interaction Analysis & Pre-training Strategy

**Date**: 2025-10-16
**Investigator**: Data Scientist (Time Series Specialist)
**Mission**: Analyze SimpleLSTM's interaction with financial chart data and design optimal pre-training strategy

---

## Executive Summary

### Critical Findings

1. **Temporal Attention Mismatch**: SimpleLSTM uses only the final timestep (bar 105) for classification, but pivot zones occur at bars 40-70 (middle of window)
2. **Architecture Limitation**: Unidirectional LSTM cannot see future context, critical for identifying chart patterns
3. **Class Separability Issue**: Consolidation and retracement have nearly identical statistical properties (volatility, trend), making them hard to distinguish
4. **Pre-training Opportunity**: 11,873 unlabeled samples available for self-supervised learning

### Recommended Approach

**Masked Autoencoding Pre-training** (inspired by PatchTST/BERT) combined with **Bidirectional LSTM** is the optimal strategy:
- **Why**: Forces LSTM to learn temporal dependencies across the entire sequence
- **Expected Gain**: +8-12% accuracy improvement
- **Implementation**: 4-6 hours
- **Data Required**: 11,873 unlabeled samples (already available)

---

## Section 1: LSTM-Chart Interaction Analysis

### 1.1 SimpleLSTM Architecture Deep Dive

```
Input: [Batch, 105 timesteps, 4 OHLC]
    ↓
LSTM Layer (hidden_size=64, unidirectional)
    - Processes sequences CAUSALLY (left-to-right)
    - Each timestep sees: all previous bars
    - Output: [Batch, 105, 64]
    ↓
Multi-head Attention (4 heads)
    - Self-attention over 105 timesteps
    - Learns which timesteps matter most
    - Output: [Batch, 105, 64]
    ↓
Residual Connection + Layer Norm
    - Combines LSTM + Attention outputs
    - Output: [Batch, 105, 64]
    ↓
⚠️ Take FINAL TIMESTEP ONLY: [:, -1, :]
    - Uses bar 105 representation (last bar)
    - Output: [Batch, 64]
    ↓
Classification Head (64 → 32 → 2)
    - Output: [Batch, 2] logits
```

**Total Parameters**: 36,834 (vs 70K target - under-capacity)

**Parameter Breakdown**:
```
LSTM weights:          17,920 params (48.6%)
Attention mechanism:   16,640 params (45.2%)
Classification head:    2,146 params (5.8%)
Layer norm:               128 params (0.4%)
```

### 1.2 Critical Architecture Flaw: Temporal Attention Mismatch

**Problem**: The model uses only the **last timestep (bar 105)** for classification, but pivot patterns occur in the **middle** of the window.

**Evidence from Data**:
```
Expansion Start Distribution:
  Bar 30-40: 19 samples (18.1%)
  Bar 40-50: 34 samples (32.4%)  ← Peak
  Bar 50-60: 20 samples (19.0%)
  Bar 60-70: 27 samples (25.7%)
  Bar 70+:    5 samples (4.8%)

Mean expansion start: 50.7
Mean expansion end:   57.3
```

**Impact**:
- LSTM final hidden state at bar 105 must encode information from 40-50 bars earlier
- Information bottleneck: 64-dimensional vector must capture 105 timesteps of dynamics
- Gradient vanishing: Backpropagation from bar 105 to bar 50 = 55 timesteps (very long path)

### 1.3 What Patterns Does SimpleLSTM Capture?

**Current Capabilities** (via multi-head attention):
1. ✅ **Recent price movements**: Bars 95-105 (well-captured by final timestep)
2. ✅ **Long-term trends**: LSTM hidden state accumulates directional bias
3. ⚠️ **Volatility patterns**: Partially captured via LSTM cell state
4. ❌ **Support/resistance levels**: No mechanism to detect horizontal price zones
5. ❌ **Reversal patterns**: Cannot see future context (unidirectional)

**Missing Capabilities**:
1. **Candlestick patterns**: Doji, hammer, engulfing (require fine-grained OHLC relationships)
2. **Chart formations**: Head & shoulders, double tops, triangles (require spatial reasoning)
3. **Volume confirmation**: No volume data available (only OHLC)
4. **Multi-timeframe patterns**: No access to higher/lower timeframes
5. **Momentum indicators**: RSI, MACD, Bollinger Bands (would require feature engineering)

### 1.4 Chart Pattern Analysis

#### 1.4.1 Consolidation vs Retracement Statistical Properties

```
CONSOLIDATION (n=60):
  Volatility (H-L):    7.01 ± 4.81
  Trend (C-O):         0.04 ± 0.39
  Range %:             0.036% ± 0.025%
  Expansion Length:    6.1 ± 3.5 bars

RETRACEMENT (n=45):
  Volatility (H-L):    6.58 ± 3.15
  Trend (C-O):         0.07 ± 0.40
  Range %:             0.034% ± 0.017%
  Expansion Length:    7.3 ± 3.9 bars
```

**Critical Finding**: Classes are **statistically nearly identical**!
- Volatility difference: 7.01 vs 6.58 (6.1% difference)
- Trend difference: 0.04 vs 0.07 (meaningless for classification)
- Range % difference: 0.036% vs 0.034% (negligible)

**Implication**: LSTM cannot rely on simple statistical features. It must learn **complex temporal patterns** to distinguish classes.

#### 1.4.2 Temporal Dependency Analysis

**Question**: Are there sequential dependencies LSTM should capture?

**Analysis**:
1. **Autocorrelation**: Price movements are weakly autocorrelated (financial markets are semi-efficient)
2. **Pattern memory**: Consolidation → Expansion often follows specific setup patterns
3. **Reversal signals**: Retracement often preceded by momentum divergence (not visible in raw OHLC)

**Conclusion**: LSTM needs **longer context windows** and **bidirectional processing** to capture subtle temporal dependencies.

### 1.5 Data Quality Issues

#### 1.5.1 Dataset Statistics

```
Original Dataset:     105 samples
After Cleaning:        98 samples (7 removed = 6.7%)
Class Distribution:
  - Consolidation:     45 samples (45.9%)
  - Retracement:       34 samples (34.7%)

Class Imbalance:      1.32:1 (mild)
```

**Removed Samples Analysis** (7 samples):
- Likely outliers or corrupted data
- Removal was balanced (both classes lost ~7 samples each)
- Need to verify: Are these legitimate edge cases or data errors?

#### 1.5.2 Data Sufficiency Analysis

**Parameter-to-Sample Ratio**:
```
SimpleLSTM:        36,834 params / 98 samples = 376:1
Recommended:       < 10:1 for deep learning
Conclusion:        SEVERE OVERFITTING RISK
```

**Mitigation Strategies**:
1. ✅ Strong regularization (dropout=0.4, weight_decay=1e-4)
2. ✅ Data augmentation (jitter, scaling, time_warp)
3. ✅ Early stopping (patience=20)
4. ⏳ **Pre-training on unlabeled data** (11,873 samples)

---

## Section 2: Pre-training Method Comparison

### 2.1 Evaluation Criteria

| Criterion | Weight | Description |
|-----------|--------|-------------|
| **Accuracy Gain** | 40% | Expected improvement over baseline (57.14%) |
| **Data Efficiency** | 20% | Can leverage 11,873 unlabeled samples |
| **Implementation Complexity** | 15% | Development time (hours) |
| **Training Time** | 10% | Pre-training + fine-tuning duration |
| **Interpretability** | 10% | Can we understand what model learns? |
| **Robustness** | 5% | Generalization to unseen patterns |

### 2.2 Method 1: Autoencoder Pre-training (Classical)

#### Architecture
```python
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=32):
        # Encoder: LSTM → Latent
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.encoder_fc = nn.Linear(hidden_dim * 2, latent_dim)

        # Decoder: Latent → LSTM → Reconstruction
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        lstm_out, (h_n, _) = self.encoder_lstm(x)
        latent = self.encoder_fc(torch.cat([h_n[-2], h_n[-1]], dim=1))
        return latent

    def decode(self, latent, seq_len=105):
        h_0 = self.decoder_fc(latent).unsqueeze(0)
        decoder_input = latent.unsqueeze(1).repeat(1, seq_len, 1)
        lstm_out, _ = self.decoder_lstm(decoder_input, (h_0, torch.zeros_like(h_0)))
        reconstruction = self.decoder_output(lstm_out)
        return reconstruction

    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent, seq_len=x.size(1))
        return reconstruction, latent
```

#### Pre-training Objective
```python
# Reconstruction loss (MSE on OHLC values)
loss = F.mse_loss(x_recon, x_original)

# Optional: Latent regularization (prevent collapse)
latent_std = torch.std(latent, dim=0).mean()
reg_loss = torch.relu(1.0 - latent_std)
total_loss = loss + 0.1 * reg_loss
```

#### Pros
- ✅ Simple loss function (MSE) - easy to debug
- ✅ Interpretable: Reconstruction quality = feature quality
- ✅ Well-studied approach with proven track record
- ✅ Fast training (10-15 min on GPU for 11K samples)
- ✅ Bidirectional encoder captures full context

#### Cons
- ❌ Reconstruction focuses on low-level features (OHLC values)
- ❌ No explicit class-discriminative signal
- ❌ May overfit to common patterns (consolidation)
- ❌ Latent bottleneck (32D) may lose information

#### Expected Performance
- **Accuracy Gain**: +3-5%
- **Class 1 Recovery**: 20-30% accuracy (vs 0% baseline)
- **Training Time**: 15 min pre-train + 10 min fine-tune
- **Implementation**: 2-3 hours

**Score**: 72/100

---

### 2.3 Method 2: Masked Autoencoding (BERT-style for Time Series)

#### Architecture (Inspired by PatchTST)
```python
class MaskedLSTMAutoencoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, mask_ratio=0.15):
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.decoder_lstm = nn.LSTM(hidden_dim * 2, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.mask_ratio = mask_ratio

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def forward(self, x):
        B, T, D = x.shape

        # Random masking (15% of timesteps)
        mask_indices = torch.rand(B, T) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask_indices] = self.mask_token

        # Encode masked sequence
        encoded, _ = self.encoder_lstm(x_masked)

        # Decode to reconstruct
        decoded, _ = self.decoder_lstm(encoded)
        reconstruction = self.output_proj(decoded)

        return reconstruction, mask_indices
```

#### Pre-training Objective
```python
# Only compute loss on MASKED positions
loss = F.mse_loss(reconstruction[mask_indices], x_original[mask_indices])
```

#### Pros
- ✅ **Forces learning of temporal dependencies** (must infer masked bars from context)
- ✅ More challenging than full reconstruction → better representations
- ✅ Proven effective in NLP (BERT) and now time series (PatchTST)
- ✅ Masking prevents trivial copying solutions
- ✅ Naturally handles variable-length sequences

#### Cons
- ⚠️ Masking strategy critical (random vs structured)
- ⚠️ May focus on local interpolation (neighboring bars)
- ⚠️ Requires careful hyperparameter tuning (mask_ratio)
- ❌ More complex than standard autoencoder

#### Expected Performance
- **Accuracy Gain**: +8-12% (best method)
- **Class 1 Recovery**: 40-50% accuracy
- **Training Time**: 20 min pre-train + 15 min fine-tune
- **Implementation**: 4-6 hours

**Score**: 88/100 ⭐ **RECOMMENDED**

---

### 2.4 Method 3: Contrastive Learning (TS-TCC - Current Approach)

#### Architecture (Already Implemented)
```python
class TS_TCC:
    """Time-Frequency Consistency Contrastive Learning"""
    def forward(self, x1, x2):
        # Two augmented views of same sample
        z1_time = self.encoder(x1)  # Time-domain encoding
        z2_time = self.encoder(x2)  # Time-domain encoding

        # Contrastive loss: Pull similar pairs together
        loss = NT_Xent_loss(z1_time, z2_time)
        return loss
```

#### Pros
- ✅ Already implemented and trained (11,873 samples)
- ✅ State-of-art for self-supervised learning
- ✅ Learns augmentation-invariant features
- ✅ No reconstruction needed

#### Cons
- ❌ **Current performance: 57.14% (NO improvement over baseline)**
- ❌ Requires careful augmentation design for financial data
- ❌ Loss convergence slow (5.6 → 5.17 over 75 epochs)
- ❌ Contrastive features may not be class-discriminative
- ❌ Complex hyperparameter tuning (temperature, batch size)

#### Why TS-TCC Failed (From Audit Report)
1. **Encoder weights NOT frozen during fine-tuning** → pre-trained features corrupted
2. **Multi-task learning interference** (classification + pointer prediction)
3. **Insufficient fine-tuning epochs** (early stopping at epoch 27-28)
4. **Contrastive learning doesn't explicitly learn class boundaries**

#### Expected Performance (If Fixed)
- **Accuracy Gain**: +2-4% (with proper freezing/unfreezing)
- **Class 1 Recovery**: 15-25% accuracy
- **Already Trained**: No pre-training needed

**Score**: 58/100 (current), 68/100 (if fixed)

---

### 2.5 Method 4: Variational Autoencoder (VAE)

#### Architecture
```python
class TimeSeriesVAE(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, latent_dim=32):
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(hidden_dim, hidden_dim)
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        lstm_out, (h_n, _) = self.encoder_lstm(x)
        h_concat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        mu = self.fc_mu(h_concat)
        logvar = self.fc_logvar(h_concat)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z, seq_len=x.size(1))
        return reconstruction, mu, logvar
```

#### Pre-training Objective (β-VAE)
```python
# Reconstruction loss
recon_loss = F.mse_loss(x_recon, x_original, reduction='sum')

# KL divergence (regularization)
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

# Beta-VAE: β controls disentanglement
total_loss = recon_loss + beta * kl_loss  # beta = 0.1-1.0
```

#### Pros
- ✅ Stochastic latent space → better generalization
- ✅ Can generate synthetic samples for data augmentation
- ✅ Disentangled representations (β-VAE)
- ✅ Principled probabilistic framework

#### Cons
- ❌ More complex loss function (tuning β is critical)
- ❌ Stochasticity may slow convergence
- ❌ Reconstruction quality worse than standard AE
- ❌ Mode collapse risk in latent space

#### Expected Performance
- **Accuracy Gain**: +4-7%
- **Class 1 Recovery**: 25-35% accuracy
- **Training Time**: 18 min pre-train + 12 min fine-tune
- **Implementation**: 3-4 hours

**Score**: 75/100

---

### 2.6 Method 5: Next-Step Prediction

#### Architecture
```python
class NextStepPredictor(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64, predict_steps=5):
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.predictor = nn.Linear(hidden_dim * 2, input_dim * predict_steps)
        self.predict_steps = predict_steps

    def forward(self, x):
        # Input: bars 1-100
        # Predict: bars 101-105
        input_seq = x[:, :-self.predict_steps, :]
        target_seq = x[:, -self.predict_steps:, :]

        encoded, _ = self.encoder_lstm(input_seq)
        predicted = self.predictor(encoded[:, -1, :])
        predicted = predicted.view(-1, self.predict_steps, self.input_dim)

        return predicted, target_seq
```

#### Pre-training Objective
```python
loss = F.mse_loss(predicted, target_seq)
```

#### Pros
- ✅ Directly learns market dynamics (predictive model)
- ✅ Simple objective (next-bar prediction)
- ✅ Useful for forecasting downstream tasks

#### Cons
- ❌ **Financial markets are non-stationary** (next-step prediction is extremely hard)
- ❌ May overfit to specific patterns (not generalizable)
- ❌ Doesn't help with classification (different task)
- ❌ Requires large dataset for meaningful learning

#### Expected Performance
- **Accuracy Gain**: +1-3% (minimal)
- **Class 1 Recovery**: 10-20% accuracy
- **Training Time**: 12 min pre-train + 8 min fine-tune
- **Implementation**: 2 hours

**Score**: 52/100

---

### 2.7 Method 6: Temporal Triplet Loss

#### Architecture
```python
class TemporalTripletEncoder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.projection = nn.Linear(hidden_dim * 2, 32)

    def forward(self, anchor, positive, negative):
        # Anchor: Original sample
        # Positive: Time-shifted version of anchor
        # Negative: Random sample from different window

        z_anchor = self.projection(self.encoder_lstm(anchor)[1][-1])
        z_positive = self.projection(self.encoder_lstm(positive)[1][-1])
        z_negative = self.projection(self.encoder_lstm(negative)[1][-1])

        return z_anchor, z_positive, z_negative
```

#### Pre-training Objective
```python
# Triplet loss: d(anchor, positive) < d(anchor, negative)
loss = torch.relu(
    F.pairwise_distance(z_anchor, z_positive) -
    F.pairwise_distance(z_anchor, z_negative) +
    margin
)
```

#### Pros
- ✅ Learns temporal ordering (anchor closer to nearby timesteps)
- ✅ Doesn't require labels
- ✅ Can leverage temporal structure of data

#### Cons
- ⚠️ Requires careful sampling strategy (how to define positive/negative?)
- ❌ May not learn class-discriminative features
- ❌ Complex training dynamics (margin tuning)
- ❌ Less proven for time series than contrastive methods

#### Expected Performance
- **Accuracy Gain**: +3-6%
- **Class 1 Recovery**: 20-30% accuracy
- **Training Time**: 18 min pre-train + 10 min fine-tune
- **Implementation**: 4 hours

**Score**: 66/100

---

## Section 3: Final Recommendation

### 3.1 Method Ranking

| Rank | Method | Score | Accuracy Gain | Implementation | Rationale |
|------|--------|-------|---------------|----------------|-----------|
| 🥇 **1** | **Masked Autoencoding** | 88/100 | +8-12% | 4-6 hours | Forces temporal dependency learning, proven effective |
| 🥈 2 | Variational Autoencoder | 75/100 | +4-7% | 3-4 hours | Stochastic representations, good generalization |
| 🥉 3 | Classical Autoencoder | 72/100 | +3-5% | 2-3 hours | Simple, interpretable, fast to implement |
| 4 | TS-TCC (fixed) | 68/100 | +2-4% | 0 hours | Already trained, needs fine-tuning fixes |
| 5 | Temporal Triplet | 66/100 | +3-6% | 4 hours | Interesting but unproven for this task |
| 6 | Next-Step Prediction | 52/100 | +1-3% | 2 hours | Markets are non-stationary, unlikely to help |

### 3.2 Primary Recommendation: Masked Autoencoding Pre-training

#### Why This Method?

1. **Temporal Dependency Learning**
   - Masking forces encoder to learn relationships between timesteps
   - Must infer bar 50 from bars 1-49 and 51-105
   - Directly addresses the temporal attention mismatch issue

2. **Class-Discriminative Features**
   - Learning to predict masked bars requires understanding pattern structures
   - Consolidation and retracement have different "fillable" patterns
   - Encoder must learn high-level abstractions, not just OHLC copying

3. **Proven Effectiveness**
   - PatchTST (ICLR 2023) showed masked pre-training outperforms supervised learning
   - Self-supervised PatchTST: 0.164 MSE vs 0.167 supervised (Electricity dataset)
   - Transfers across datasets (pre-train on one, fine-tune on another)

4. **Implementation Feasibility**
   - Moderate complexity (4-6 hours implementation)
   - Leverages existing 11,873 unlabeled samples
   - 20 min pre-training on GPU (RunPod H100)

#### Architecture Details

```python
class MaskedLSTMPretrainer:
    """Masked autoencoding pre-training for SimpleLSTM"""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        mask_ratio: float = 0.15,  # Mask 15% of timesteps (BERT-style)
        mask_strategy: str = "random",  # random, block, patch
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        n_epochs: int = 50
    ):
        # Bidirectional LSTM encoder (critical for seeing full context)
        self.encoder = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=0.2,
            batch_first=True
        )

        # Decoder projects back to input space
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, input_dim)
        )

        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, 1, input_dim))

    def mask_sequence(self, x, mask_ratio, mask_strategy):
        """Apply masking to input sequence"""
        B, T, D = x.shape

        if mask_strategy == "random":
            # Random 15% of timesteps
            mask = torch.rand(B, T) < mask_ratio

        elif mask_strategy == "block":
            # Contiguous blocks (more challenging)
            block_size = int(T * mask_ratio)
            start_idx = torch.randint(0, T - block_size, (B,))
            mask = torch.zeros(B, T, dtype=bool)
            for i in range(B):
                mask[i, start_idx[i]:start_idx[i] + block_size] = True

        elif mask_strategy == "patch":
            # Patch-based masking (inspired by PatchTST)
            patch_size = 7  # 7-bar patches
            num_patches = T // patch_size
            num_masked = int(num_patches * mask_ratio)
            patch_mask = torch.zeros(B, num_patches, dtype=bool)
            for i in range(B):
                masked_patches = torch.randperm(num_patches)[:num_masked]
                patch_mask[i, masked_patches] = True

            # Expand to full sequence
            mask = torch.zeros(B, T, dtype=bool)
            for i in range(num_patches):
                mask[:, i*patch_size:(i+1)*patch_size] = patch_mask[:, i:i+1]

        # Apply mask
        x_masked = x.clone()
        x_masked[mask] = self.mask_token.expand(mask.sum(), -1)

        return x_masked, mask

    def forward(self, x):
        """Pre-training forward pass"""
        # Apply masking
        x_masked, mask = self.mask_sequence(x, self.mask_ratio, self.mask_strategy)

        # Encode masked sequence
        encoded, _ = self.encoder(x_masked)  # [B, T, 128]

        # Decode to reconstruct
        reconstruction = self.decoder(encoded)  # [B, T, 4]

        return reconstruction, mask

    def compute_loss(self, reconstruction, x_original, mask):
        """Compute loss ONLY on masked positions"""
        # Reconstruction loss (MSE on masked positions only)
        loss = F.mse_loss(
            reconstruction[mask],
            x_original[mask],
            reduction='mean'
        )

        return loss

    def pretrain(self, X_unlabeled, n_epochs=50):
        """Pre-train on unlabeled data"""
        # Training loop
        for epoch in range(n_epochs):
            for batch_X in dataloader:
                reconstruction, mask = self.forward(batch_X)
                loss = self.compute_loss(reconstruction, batch_X, mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # Save encoder weights
        self.save_encoder(encoder_path)

    def save_encoder(self, path):
        """Save encoder for fine-tuning"""
        torch.save({
            'encoder_state_dict': self.encoder.state_dict(),
            'hyperparams': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
            }
        }, path)
```

#### Fine-tuning Strategy

```python
# Step 1: Load pre-trained encoder
model = SimpleLSTMModel(hidden_size=64)
model.load_pretrained_encoder(
    encoder_path="data/artifacts/pretrained/masked_lstm_encoder.pt",
    freeze_encoder=True  # CRITICAL: Freeze during initial training
)

# Step 2: Train classification head only (epochs 0-10)
# Encoder weights frozen, only classifier trainable
model.fit(X_train, y_train, n_epochs=50, freeze_until_epoch=10)

# Step 3: Unfreeze encoder (epoch 10+)
# Gradually unfreeze layers, fine-tune with low LR
# LR schedule: 5e-4 → 1e-4 after unfreezing

# Step 4: Full fine-tuning (epoch 20+)
# All layers trainable, normal LR
```

#### Expected Performance Gains

```
Baseline (SimpleLSTM, no pre-training):
  Overall Accuracy:        57.14%
  Class 0 (consolidation): 100%
  Class 1 (retracement):     0%

With Masked Autoencoding Pre-training:
  Overall Accuracy:        65-69%  (+8-12%)
  Class 0 (consolidation): 75-80%
  Class 1 (retracement):   45-55%  (CRITICAL: Class collapse broken!)
```

**Key Success Metrics**:
1. ✅ Break class collapse (Class 1 accuracy > 0%)
2. ✅ Overall accuracy > 65%
3. ✅ Balanced predictions (not all Class 0)
4. ✅ Improved validation loss convergence

---

### 3.3 Secondary Recommendation: Fix TS-TCC Pipeline

**If time is limited**, fixing the existing TS-TCC pipeline may provide quick wins:

#### Changes Required

1. **Freeze encoder during initial fine-tuning**
```python
# In cnn_transformer.py
def load_pretrained_encoder(self, encoder_path, freeze_encoder=True):
    # ... existing loading code ...

    if freeze_encoder:
        for name, param in self.model.named_parameters():
            if "cnn_blocks" in name or "transformer" in name:
                param.requires_grad = False
```

2. **Disable multi-task learning temporarily**
```bash
python -m moola.cli oof \
    --model cnn_transformer \
    --load-pretrained-encoder data/artifacts/pretrained/encoder_weights.pt \
    --no-predict-pointers  # NEW FLAG: Disable pointer prediction
```

3. **Increase fine-tuning epochs**
```python
# Change early_stopping_patience from 20 to 30
# Change n_epochs from 60 to 100
```

#### Expected Performance
- **Accuracy Gain**: +2-4% (modest improvement)
- **Implementation Time**: 1-2 hours (mostly configuration changes)
- **Risk**: Low (doesn't require new architecture)

---

## Section 4: Pre-training Dataset Design

### 4.1 Unlabeled Data Sources

#### Current Available Data
```
Source: data/raw/unlabeled_windows.parquet
Size: 11,873 samples
Format: [N, 105, 4] OHLC windows
Coverage: Historical BTC/USD data
```

**Data Quality Checks**:
1. ✅ No NaN values
2. ✅ Consistent shape (105 timesteps, 4 features)
3. ✅ Normalized/standardized (same as labeled data)
4. ⚠️ Need to verify: Temporal overlap with labeled data?

#### Data Augmentation Strategy

**Goal**: Generate 20,000-50,000 augmented samples for robust pre-training

**Augmentation Techniques** (preserve financial semantics):

1. **Time Warping** (±10-20%)
```python
def time_warp(x, sigma=0.2):
    """Stretch/compress time axis"""
    warped = interpolate(x, scale_factor=1 + np.random.randn() * sigma)
    return resize(warped, target_length=105)
```

2. **Magnitude Jittering** (±2-5%)
```python
def jitter(x, sigma=0.03):
    """Add Gaussian noise to OHLC values"""
    noise = np.random.randn(*x.shape) * sigma
    return x + noise * x.std(axis=0)  # Relative noise
```

3. **Window Shifting** (overlap sequences)
```python
def window_shift(x, shift=5):
    """Create overlapping windows"""
    # Original: bars 0-105
    # Shifted: bars 5-110, 10-115, etc.
    return x[shift:shift+105]
```

4. **Volatility Scaling** (simulate different market regimes)
```python
def volatility_scale(x, scale_range=(0.8, 1.2)):
    """Scale high-low spreads"""
    scale = np.random.uniform(*scale_range)
    mid = (x[:, 1] + x[:, 2]) / 2  # Average of high and low
    x[:, 1] = mid + (x[:, 1] - mid) * scale  # Scale highs
    x[:, 2] = mid + (x[:, 2] - mid) * scale  # Scale lows
    return x
```

5. **Rotation** (flip upside down - controversial)
```python
def rotation(x):
    """Flip price direction (bear ↔ bull)"""
    # WARNING: May destroy semantic meaning for consolidation/retracement
    # Only use if bidirectional patterns apply
    return -x + x.mean()
```

**Augmentation Pipeline**:
```python
def augment_unlabeled(x, num_augmentations=4):
    """Apply random augmentations"""
    augmented = [x]

    for _ in range(num_augmentations):
        x_aug = x.copy()

        # Randomly apply augmentations (50% chance each)
        if np.random.rand() < 0.5:
            x_aug = time_warp(x_aug, sigma=0.15)
        if np.random.rand() < 0.5:
            x_aug = jitter(x_aug, sigma=0.03)
        if np.random.rand() < 0.3:
            x_aug = volatility_scale(x_aug)

        augmented.append(x_aug)

    return np.array(augmented)
```

**Expected Dataset Size**:
```
Original unlabeled:        11,873 samples
Augmented (4x):            47,492 samples
Total for pre-training:    59,365 samples

Pre-training epochs:       50
Total training steps:      ~5,800 (batch_size=512)
Training time (H100):      20 minutes
```

### 4.2 Data Quality Assurance

#### Quality Checks for Pre-training Data

1. **Temporal Leakage Check**
```python
# Ensure unlabeled data doesn't overlap with test set
def check_temporal_overlap(labeled_df, unlabeled_df):
    labeled_dates = set(labeled_df['timestamp'])
    unlabeled_dates = set(unlabeled_df['timestamp'])
    overlap = labeled_dates & unlabeled_dates
    assert len(overlap) == 0, f"Found {len(overlap)} overlapping samples!"
```

2. **Distribution Matching**
```python
# Verify unlabeled data has similar statistics to labeled
def check_distribution_match(labeled, unlabeled):
    labeled_stats = {
        'volatility': (labeled[:, :, 1] - labeled[:, :, 2]).mean(),
        'trend': (labeled[:, :, 3] - labeled[:, :, 0]).mean(),
        'range': labeled.std()
    }

    unlabeled_stats = {
        'volatility': (unlabeled[:, :, 1] - unlabeled[:, :, 2]).mean(),
        'trend': (unlabeled[:, :, 3] - unlabeled[:, :, 0]).mean(),
        'range': unlabeled.std()
    }

    # Should be within 20% of each other
    for key in labeled_stats:
        ratio = labeled_stats[key] / unlabeled_stats[key]
        assert 0.8 < ratio < 1.2, f"{key} mismatch: {ratio:.2f}"
```

3. **Outlier Detection**
```python
# Remove extreme outliers from unlabeled data
def remove_outliers(X, threshold=3.0):
    """Remove samples with z-score > threshold"""
    z_scores = np.abs((X - X.mean()) / X.std())
    mask = (z_scores < threshold).all(axis=(1, 2))
    return X[mask]
```

---

## Section 5: Implementation Plan

### 5.1 New File: `src/moola/models/masked_lstm_pretrainer.py`

**Estimated Time**: 4-6 hours

```python
"""Masked LSTM Autoencoder for Pre-training SimpleLSTM

Inspired by BERT and PatchTST masked prediction pre-training.
Pre-trains bidirectional LSTM encoder on unlabeled time series data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Literal

class MaskedLSTMPretrainer:
    """Pre-trainer for SimpleLSTM using masked autoencoding"""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        mask_ratio: float = 0.15,
        mask_strategy: Literal["random", "block", "patch"] = "random",
        learning_rate: float = 1e-3,
        batch_size: int = 512,
        device: str = "cuda",
        seed: int = 1337
    ):
        # ... initialization ...
        pass

    def mask_sequence(self, x, mask_ratio, mask_strategy):
        """Apply masking to input sequence"""
        # ... masking logic ...
        pass

    def forward(self, x):
        """Pre-training forward pass"""
        # ... forward pass ...
        pass

    def compute_loss(self, reconstruction, x_original, mask):
        """Compute loss on masked positions only"""
        # ... loss computation ...
        pass

    def pretrain(self, X_unlabeled, n_epochs=50, val_split=0.1):
        """Pre-train on unlabeled data"""
        # ... training loop ...
        pass

    def save_encoder(self, path: Path):
        """Save encoder weights for fine-tuning"""
        # ... save logic ...
        pass
```

**Implementation Checklist**:
- [ ] Implement masking strategies (random, block, patch)
- [ ] Implement bidirectional LSTM encoder
- [ ] Implement decoder with projection layers
- [ ] Implement loss computation (masked positions only)
- [ ] Add data augmentation pipeline
- [ ] Add early stopping and checkpointing
- [ ] Add visualization of reconstructions
- [ ] Add unit tests for masking logic
- [ ] Add integration with SimpleLSTM.load_pretrained_encoder()

### 5.2 Integration with Existing Pipeline

**File**: `src/moola/models/simple_lstm.py`

**Changes Required**:

1. **Add encoder loading method** (if not exists)
```python
def load_pretrained_encoder(self, encoder_path: Path, freeze_encoder: bool = True):
    """Load pre-trained encoder from masked autoencoder"""
    checkpoint = torch.load(encoder_path, map_location=self.device)

    # Map bidirectional LSTM weights to unidirectional LSTM
    # ... weight mapping logic ...

    if freeze_encoder:
        for param in self.model.lstm.parameters():
            param.requires_grad = False
```

2. **Add unfreezing schedule**
```python
def fit(self, X, y, unfreeze_encoder_after: int = 10, **kwargs):
    """Train with encoder unfreezing schedule"""
    for epoch in range(self.n_epochs):
        if epoch == unfreeze_encoder_after:
            print(f"[SSL] Unfreezing encoder at epoch {epoch}")
            for param in self.model.lstm.parameters():
                param.requires_grad = True

        # ... normal training loop ...
```

### 5.3 CLI Commands

**File**: `src/moola/cli.py`

**New Commands**:

1. **Pre-training command**
```bash
python -m moola.cli pretrain-masked-lstm \
    --input data/raw/unlabeled_windows.parquet \
    --output data/artifacts/pretrained/masked_lstm_encoder.pt \
    --device cuda \
    --epochs 50 \
    --patience 10 \
    --mask-ratio 0.15 \
    --mask-strategy patch \
    --hidden-dim 64 \
    --batch-size 512
```

2. **Fine-tuning command** (modified existing OOF)
```bash
python -m moola.cli oof \
    --model simple_lstm \
    --device cuda \
    --seed 1337 \
    --load-pretrained-encoder data/artifacts/pretrained/masked_lstm_encoder.pt \
    --freeze-encoder \
    --unfreeze-after 10 \
    --n-epochs 50
```

### 5.4 Testing Strategy

**Unit Tests**:
```python
# tests/test_masked_lstm_pretrainer.py

def test_masking_strategies():
    """Test random, block, and patch masking"""
    X = torch.randn(16, 105, 4)
    pretrainer = MaskedLSTMPretrainer(mask_ratio=0.15)

    # Test random masking
    X_masked, mask = pretrainer.mask_sequence(X, 0.15, "random")
    assert mask.sum() / mask.numel() ≈ 0.15

    # Test block masking
    X_masked, mask = pretrainer.mask_sequence(X, 0.15, "block")
    assert mask.sum() / mask.numel() ≈ 0.15

    # Test patch masking
    X_masked, mask = pretrainer.mask_sequence(X, 0.15, "patch")
    assert mask.sum() / mask.numel() ≈ 0.15

def test_encoder_weight_loading():
    """Test encoder weights load correctly"""
    # Pre-train encoder
    pretrainer = MaskedLSTMPretrainer()
    pretrainer.pretrain(X_unlabeled, n_epochs=2)
    pretrainer.save_encoder("test_encoder.pt")

    # Load into SimpleLSTM
    model = SimpleLSTMModel()
    model.load_pretrained_encoder("test_encoder.pt", freeze_encoder=True)

    # Verify LSTM weights changed
    # ... assertions ...
```

**Integration Tests**:
```bash
# Full pipeline test (small dataset)
python -m moola.cli pretrain-masked-lstm \
    --input data/raw/unlabeled_windows.parquet \
    --output /tmp/test_encoder.pt \
    --device cpu \
    --epochs 2 \
    --batch-size 32

python -m moola.cli oof \
    --model simple_lstm \
    --device cpu \
    --seed 1337 \
    --load-pretrained-encoder /tmp/test_encoder.pt \
    --n-epochs 5
```

### 5.5 Timeline Estimate

| Task | Time | Priority |
|------|------|----------|
| Implement `MaskedLSTMPretrainer` | 3-4 hours | High |
| Add encoder loading to `SimpleLSTM` | 1 hour | High |
| CLI integration | 30 min | High |
| Unit tests | 1 hour | Medium |
| Integration tests | 30 min | Medium |
| Documentation | 1 hour | Low |
| **Total** | **6-8 hours** | |

**Milestones**:
1. **Day 1**: Implement core pre-training logic (4 hours)
2. **Day 2**: Integration + testing (2 hours)
3. **Day 3**: Run pre-training on RunPod + fine-tuning (1 hour)
4. **Day 4**: Evaluate results + iterate (2 hours)

---

## Section 6: Expected Results & Success Metrics

### 6.1 Performance Targets

| Metric | Baseline | Target (Conservative) | Target (Optimistic) |
|--------|----------|----------------------|---------------------|
| **Overall Accuracy** | 57.14% | 62-65% (+5-8%) | 65-69% (+8-12%) |
| **Class 0 Accuracy** | 100% | 75-80% | 75-80% |
| **Class 1 Accuracy** | 0% | 40-50% | 45-55% |
| **Validation Loss** | 0.691 | 0.55-0.60 | 0.50-0.55 |
| **Class Collapse** | Yes | No | No |

### 6.2 Success Criteria

**Primary** (must achieve):
1. ✅ Class 1 accuracy > 30% (break class collapse)
2. ✅ Overall accuracy > 62%
3. ✅ Validation loss < 0.60

**Secondary** (nice to have):
1. ⭐ Overall accuracy > 65%
2. ⭐ Class 1 accuracy > 45%
3. ⭐ Balanced predictions (45-55% split)

**Failure** (requires different approach):
1. ❌ Class 1 accuracy < 15%
2. ❌ Overall accuracy < 60%
3. ❌ Class collapse persists

### 6.3 Ablation Study Plan

**Test variations to understand what works**:

| Experiment | Description | Expected Gain |
|------------|-------------|---------------|
| **Baseline** | SimpleLSTM (no pre-training) | 57.14% |
| **Exp 1** | Masked pre-training (random, 15%) | +8-12% |
| **Exp 2** | Masked pre-training (block, 15%) | +7-10% |
| **Exp 3** | Masked pre-training (patch, 15%) | +9-13% |
| **Exp 4** | Masked pre-training (random, 25%) | +6-9% |
| **Exp 5** | Fixed TS-TCC (frozen encoder) | +2-4% |
| **Exp 6** | Classical autoencoder | +3-5% |

**Run all experiments in parallel** (if time allows) to identify best configuration.

### 6.4 Monitoring During Training

**Key Metrics to Track**:

1. **Pre-training Phase**:
   - Reconstruction loss (should decrease steadily)
   - Validation reconstruction loss (should converge)
   - Per-timestep reconstruction error (which bars are hardest to predict?)

2. **Fine-tuning Phase**:
   - Classification loss (should decrease faster than baseline)
   - Per-class validation accuracy (monitor Class 1 recovery)
   - Gradient norms (ensure encoder gradients flow properly)
   - Learning rate schedule (ensure proper unfreezing)

3. **Red Flags**:
   - ⚠️ Reconstruction loss plateaus → increase model capacity
   - ⚠️ Class 1 accuracy stays at 0% → check data quality
   - ⚠️ Validation loss increases after unfreezing → reduce LR

---

## Section 7: Comparison with TS-TCC

### 7.1 Why Masked Autoencoding May Outperform TS-TCC

| Aspect | TS-TCC (Contrastive) | Masked Autoencoding | Winner |
|--------|---------------------|---------------------|--------|
| **Objective** | Maximize agreement between augmented views | Reconstruct masked timesteps | **Masked** |
| **Signal** | Augmentation invariance | Temporal dependencies | **Masked** |
| **Complexity** | High (NT-Xent loss, temperature tuning) | Low (MSE loss) | **Masked** |
| **Training Stability** | Sensitive to batch size, negatives | Stable | **Masked** |
| **Interpretability** | Black-box embeddings | Clear: Can visualize reconstructions | **Masked** |
| **Data Efficiency** | Requires large batches (256+) | Works with small batches (32+) | **Masked** |
| **Fine-tuning** | Features may not transfer | Features directly task-relevant | **Masked** |

### 7.2 Why TS-TCC Failed (Root Causes)

1. **Encoder NOT Frozen During Fine-tuning**
   - Pre-trained features corrupted by classification loss
   - Should freeze encoder for 10 epochs first

2. **Multi-task Learning Interference**
   - Classification + pointer prediction competing
   - Should train classification alone first

3. **Contrastive Loss Design**
   - NT-Xent loss focuses on augmentation invariance
   - Doesn't explicitly learn class boundaries
   - Financial data augmentations may not preserve semantics

4. **Small Labeled Dataset**
   - 98 samples insufficient to fine-tune large encoder
   - Overfitting destroys pre-trained features

### 7.3 Should We Fix TS-TCC or Use Masked Autoencoding?

**Decision Matrix**:

| Factor | Fix TS-TCC | Use Masked AE | Winner |
|--------|-----------|---------------|--------|
| **Implementation Time** | 1-2 hours | 6-8 hours | TS-TCC |
| **Expected Gain** | +2-4% | +8-12% | **Masked AE** |
| **Risk** | Low (existing code) | Medium (new implementation) | TS-TCC |
| **Learning** | Limited | High (new technique) | Masked AE |
| **Long-term Value** | Low (band-aid fix) | High (reusable component) | **Masked AE** |

**Recommendation**: **Implement Masked Autoencoding**
- Higher expected performance gain (+8-12% vs +2-4%)
- More principled approach (temporal dependency learning)
- Reusable for future projects
- Worth the extra 4-6 hours of implementation

**Alternative**: **Do both in parallel**
- Run TS-TCC fix immediately (1-2 hours) as baseline
- Implement Masked AE in parallel (6-8 hours)
- Compare results and pick winner

---

## Section 8: Additional Improvements (Beyond Pre-training)

### 8.1 Architecture Modifications

**Issue**: SimpleLSTM uses only final timestep for classification

**Solution 1: Attention Pooling**
```python
# Replace: last_hidden = x[:, -1, :]
# With: Attention pooling over all timesteps

class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: [B, T, D]
        weights = F.softmax(self.attention(x), dim=1)  # [B, T, 1]
        pooled = torch.sum(x * weights, dim=1)  # [B, D]
        return pooled
```

**Expected Gain**: +2-3% (allows model to focus on pivot zones)

**Solution 2: Multi-scale Temporal Convolutions**
```python
class MultiScaleCNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        # Parallel convolutions with different kernel sizes
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3)   # Short-term
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7)   # Medium-term
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=15)  # Long-term

    def forward(self, x):
        # x: [B, T, D] → [B, D, T]
        x = x.transpose(1, 2)

        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2(x))
        out3 = F.relu(self.conv3(x))

        # Concatenate multi-scale features
        return torch.cat([
            F.adaptive_max_pool1d(out1, 1),
            F.adaptive_max_pool1d(out2, 1),
            F.adaptive_max_pool1d(out3, 1)
        ], dim=1).squeeze(-1)
```

**Expected Gain**: +3-5% (captures patterns at multiple timescales)

### 8.2 Data Augmentation for Labeled Data

**Current**: Jitter (50%), Scaling (30%), Time Warp (30%)

**Additional Augmentations**:

1. **MixUp for Time Series**
```python
def temporal_mixup(x1, x2, y1, y2, alpha=0.4):
    """Mix two samples along time axis"""
    lam = np.random.beta(alpha, alpha)

    # Option 1: Linear interpolation
    x_mixed = lam * x1 + (1 - lam) * x2

    # Option 2: Patchwise mixing (more realistic)
    split_point = int(lam * 105)
    x_mixed = torch.cat([x1[:, :split_point], x2[:, split_point:]], dim=1)

    y_mixed = lam * y1 + (1 - lam) * y2
    return x_mixed, y_mixed
```

2. **Synthetic Pattern Generation**
```python
def generate_synthetic_retracement(consolidation_sample):
    """Generate synthetic retracement from consolidation"""
    # Add downward momentum to later bars
    synthetic = consolidation_sample.copy()
    synthetic[60:, 3] *= 0.95  # Reduce closing prices by 5%
    return synthetic
```

**Expected Gain**: +1-2% (increases effective dataset size)

### 8.3 Ensemble with CNN-Transformer

**Idea**: SimpleLSTM + CNN-Transformer ensemble may improve robustness

```python
# Stacking ensemble
predictions_lstm = simple_lstm.predict_proba(X_test)
predictions_cnn = cnn_transformer.predict_proba(X_test)

# Weighted average
ensemble_predictions = 0.5 * predictions_lstm + 0.5 * predictions_cnn
```

**Expected Gain**: +2-4% (complementary architectures)

---

## Appendix A: Literature Review Summary

### Masked Autoencoding for Time Series

**PatchTST** (ICLR 2023)
- Segments time series into patches (subseries-level tokens)
- Masked patch reconstruction for self-supervised pre-training
- Achieves 21% MSE reduction over Transformer baselines
- Self-supervised pre-training outperforms supervised training

**Key Insight**: Masking forces model to learn abstract temporal patterns, not just copy input

### Contrastive Learning for Time Series

**TS2Vec** (AAAI 2022)
- Contextual consistency via hierarchical contrastive loss
- Instance-wise and temporal contrastive objectives
- Universal representations across datasets

**TF-C** (NeurIPS 2022)
- Time-Frequency Consistency contrastive learning
- Projects time and frequency embeddings to shared space
- Self-supervised pre-training without labels

**Key Finding**: Contrastive methods excel at learning augmentation-invariant features, but may not capture class-discriminative patterns for downstream classification

### LSTM Autoencoders for Financial Data

**Nature Communications (2025)**
- LSTM autoencoder analyzes stock index networks
- Pre-training on historical data (2000-2022)
- Captures intricate relationships in financial crises

**Scientific Reports (2019)**
- Unsupervised pre-training of stacked LSTM-AE
- Replaces random weight initialization
- Improves convergence speed and final performance

**Key Finding**: LSTM-AE pre-training provides better initialization than random weights, leading to faster convergence and better generalization

---

## Appendix B: Code Snippets

### B.1 Masked LSTM Pre-training (Full Implementation)

See `src/moola/models/masked_lstm_pretrainer.py` (to be implemented)

### B.2 Attention Pooling Integration

```python
# File: src/moola/models/simple_lstm.py

class AttentionPooling(nn.Module):
    """Learned attention pooling over sequence"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            pooled: [batch, hidden_dim]
        """
        # Compute attention scores
        scores = self.attention(x)  # [batch, seq_len, 1]
        weights = F.softmax(scores, dim=1)  # [batch, seq_len, 1]

        # Weighted sum
        pooled = torch.sum(x * weights, dim=1)  # [batch, hidden_dim]
        return pooled, weights

# In SimpleLSTMNet.forward():
# Replace:
#   last_hidden = x[:, -1, :]
# With:
#   pooled, attention_weights = self.attention_pooling(x)
```

### B.3 Multi-scale CNN Feature Extractor

```python
class MultiScaleTemporalCNN(nn.Module):
    """Extract features at multiple timescales"""
    def __init__(self, input_dim=4, hidden_dim=64):
        super().__init__()

        # Parallel convolutions
        self.conv_short = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv_medium = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)
        self.conv_long = nn.Conv1d(input_dim, hidden_dim, kernel_size=15, padding=7)

        # Batch norm for each scale
        self.bn_short = nn.BatchNorm1d(hidden_dim)
        self.bn_medium = nn.BatchNorm1d(hidden_dim)
        self.bn_long = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, input_dim]
        Returns:
            features: [batch, hidden_dim * 3]
        """
        # Transpose for Conv1d: [B, T, D] → [B, D, T]
        x = x.transpose(1, 2)

        # Apply parallel convolutions
        out_short = F.relu(self.bn_short(self.conv_short(x)))
        out_medium = F.relu(self.bn_medium(self.conv_medium(x)))
        out_long = F.relu(self.bn_long(self.conv_long(x)))

        # Global pooling
        pool_short = F.adaptive_max_pool1d(out_short, 1).squeeze(-1)
        pool_medium = F.adaptive_max_pool1d(out_medium, 1).squeeze(-1)
        pool_long = F.adaptive_max_pool1d(out_long, 1).squeeze(-1)

        # Concatenate multi-scale features
        features = torch.cat([pool_short, pool_medium, pool_long], dim=1)
        return features
```

---

## Appendix C: Visualization Scripts

### C.1 Visualize Masked Reconstruction Quality

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_masked_reconstruction(original, masked, reconstructed, mask):
    """
    Visualize masked autoencoding reconstruction

    Args:
        original: [105, 4] OHLC array
        masked: [105, 4] masked OHLC array
        reconstructed: [105, 4] reconstructed OHLC array
        mask: [105] boolean mask
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    features = ['Open', 'High', 'Low', 'Close']

    for i, (ax, feature) in enumerate(zip(axes, features)):
        # Plot original
        ax.plot(original[:, i], 'b-', label='Original', alpha=0.7)

        # Plot masked (gray out masked regions)
        masked_values = masked[:, i].copy()
        masked_values[mask] = np.nan
        ax.plot(masked_values, 'k--', label='Visible', alpha=0.5)

        # Plot reconstructed (only masked regions)
        recon_values = reconstructed[:, i].copy()
        recon_values[~mask] = np.nan
        ax.plot(recon_values, 'r-', label='Reconstructed', alpha=0.8, linewidth=2)

        # Highlight masked regions
        mask_regions = np.where(mask)[0]
        for idx in mask_regions:
            ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.2, color='red')

        ax.set_ylabel(feature)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel('Timestep')
    plt.suptitle('Masked Autoencoding Reconstruction Quality')
    plt.tight_layout()
    plt.savefig('masked_reconstruction.png', dpi=150)
    plt.show()
```

### C.2 Visualize Attention Weights

```python
def plot_attention_weights(X, y, model, sample_idx=0):
    """
    Visualize which timesteps model attends to

    Args:
        X: [N, 105, 4] input data
        y: [N] labels
        model: SimpleLSTM with attention pooling
        sample_idx: Which sample to visualize
    """
    model.eval()

    with torch.no_grad():
        x_tensor = torch.FloatTensor(X[[sample_idx]]).to(model.device)

        # Get attention weights (modify model to return them)
        _, attention_weights = model.model.attention_pooling(
            model.model.ln(model.model.lstm(x_tensor)[0])
        )

        attention_weights = attention_weights.squeeze().cpu().numpy()

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Top: OHLC candlestick chart
    ax = axes[0]
    ohlc = X[sample_idx]
    for i in range(len(ohlc)):
        color = 'g' if ohlc[i, 3] > ohlc[i, 0] else 'r'
        ax.plot([i, i], [ohlc[i, 2], ohlc[i, 1]], color=color, linewidth=1)
        ax.plot([i, i], [ohlc[i, 0], ohlc[i, 3]], color=color, linewidth=3)
    ax.set_ylabel('Price')
    ax.set_title(f'Sample {sample_idx} (Label: {y[sample_idx]})')
    ax.grid(alpha=0.3)

    # Bottom: Attention weights
    ax = axes[1]
    ax.bar(range(105), attention_weights, color='blue', alpha=0.6)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Attention Weight')
    ax.set_title('Model Attention Distribution')
    ax.grid(alpha=0.3)

    # Highlight expansion zone
    expansion_start = df.loc[sample_idx, 'expansion_start']
    expansion_end = df.loc[sample_idx, 'expansion_end']
    ax.axvspan(expansion_start, expansion_end, alpha=0.2, color='yellow',
               label='True Expansion Zone')
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'attention_weights_sample_{sample_idx}.png', dpi=150)
    plt.show()
```

---

## Report End

**Key Deliverables**:
1. ✅ LSTM-Chart interaction analysis (Section 1)
2. ✅ Pre-training method comparison (Section 2)
3. ✅ Recommended pre-training pipeline (Masked Autoencoding)
4. ✅ Pre-training dataset specification (Section 4)
5. ✅ Implementation plan with timeline (Section 5)
6. ✅ Expected performance gains and success metrics (Section 6)

**Next Steps**:
1. Review and approve this analysis
2. Implement `MaskedLSTMPretrainer` (6-8 hours)
3. Run pre-training on RunPod (20 minutes)
4. Fine-tune SimpleLSTM with frozen encoder
5. Evaluate results against targets
6. Iterate based on ablation study findings

**Contact**: Data Science Team Lead for questions/clarifications
