# Masked LSTM Pre-training Quickstart

**TL;DR**: Implement masked autoencoding pre-training to achieve +8-12% accuracy gain (57% → 65-69%)

---

## Why Masked Autoencoding?

**Problem**: SimpleLSTM uses only the **final timestep** for classification, but pivots occur at **bars 40-70** (middle of window)

**Solution**: Masked autoencoding forces LSTM to learn temporal dependencies across the **entire sequence**

**Evidence**: PatchTST (ICLR 2023) achieved 21% MSE reduction using masked pre-training

---

## Quick Commands

### 1. Pre-train Encoder (20 min on H100)

```bash
python -m moola.cli pretrain-masked-lstm \
    --input data/raw/unlabeled_windows.parquet \
    --output data/artifacts/pretrained/masked_lstm_encoder.pt \
    --device cuda \
    --epochs 50 \
    --mask-strategy patch \
    --batch-size 512
```

### 2. Fine-tune SimpleLSTM (15 min on H100)

```bash
python -m moola.cli oof \
    --model simple_lstm \
    --device cuda \
    --load-pretrained-encoder data/artifacts/pretrained/masked_lstm_encoder.pt \
    --freeze-encoder \
    --unfreeze-after 10 \
    --n-epochs 50
```

### 3. Evaluate Results

```bash
python -m moola.cli ensemble --device cuda
```

**Expected**:
- Overall accuracy: 65-69% (vs 57.14% baseline)
- Class 0 accuracy: 75-80% (vs 100%)
- Class 1 accuracy: 45-55% (vs 0% - **class collapse broken!**)

---

## How It Works

### Masking Strategy

```python
# Original sequence: bars 1-105
original = [O, H, L, C] × 105

# Random masking (15% of bars)
masked = [O, H, L, C, O, H, L, C, [MASK], [MASK], O, H, L, C, ...]

# Encoder learns to predict masked bars from context
encoded = BiLSTM(masked)
reconstructed = Decoder(encoded)

# Loss on masked positions only
loss = MSE(reconstructed[mask], original[mask])
```

### Pre-training Flow

```
Unlabeled Data (11,873 samples)
    ↓
Random Masking (15% of timesteps)
    ↓
Bidirectional LSTM Encoder
    ↓
Decoder (reconstruction)
    ↓
MSE Loss on masked positions
    ↓
Pre-trained Encoder Weights
```

### Fine-tuning Flow

```
Pre-trained Encoder
    ↓
Load into SimpleLSTM
    ↓
Freeze encoder (epochs 0-10)
    ↓
Train classification head only
    ↓
Unfreeze encoder (epoch 10+)
    ↓
Full fine-tuning with low LR
    ↓
Final Classifier
```

---

## Implementation Checklist

### Phase 1: Core Pre-training (3-4 hours)
- [ ] Create `src/moola/models/masked_lstm_pretrainer.py`
- [ ] Implement masking strategies (random, block, patch)
- [ ] Implement bidirectional LSTM encoder
- [ ] Implement decoder
- [ ] Implement loss computation (masked positions only)

### Phase 2: Training Infrastructure (1.5 hours)
- [ ] Implement pre-training loop
- [ ] Add validation loop
- [ ] Add early stopping
- [ ] Add data augmentation

### Phase 3: Integration (1 hour)
- [ ] Add `load_pretrained_encoder()` to SimpleLSTM
- [ ] Add unfreezing schedule to `fit()`
- [ ] Map bidirectional → unidirectional weights

### Phase 4: CLI & Testing (1.5 hours)
- [ ] Add `pretrain-masked-lstm` CLI command
- [ ] Add unit tests
- [ ] Add integration tests

### Phase 5: Training & Evaluation (1 hour)
- [ ] Run pre-training on RunPod
- [ ] Fine-tune SimpleLSTM
- [ ] Evaluate results

**Total Time**: 8-9 hours

---

## Key Design Decisions

### 1. Masking Strategy: **Patch** (Recommended)

**Why**: Patches force long-range dependency learning

**Alternatives**:
- Random (15% of bars) - simpler, may focus on local patterns
- Block (contiguous segment) - more challenging, but may be too hard

**Recommendation**: Start with patch, ablate later

### 2. Mask Ratio: **15%** (BERT-style)

**Why**: Proven effective in NLP (BERT) and time series (PatchTST)

**Alternatives**:
- 10% - easier task, may not learn enough
- 25% - harder task, may be too difficult

**Recommendation**: Start with 15%, ablate with 10% and 25%

### 3. Unfreezing Schedule: **Epoch 10**

**Why**: Allows classification head to align with pre-trained features

**Alternatives**:
- Epoch 5 - may not be enough warm-up
- Epoch 20 - may waste training time

**Recommendation**: Start with epoch 10, monitor validation loss

### 4. Bidirectional Encoder: **Yes**

**Why**: Sees full context (past + future), critical for reconstruction

**Note**: Must map bidirectional → unidirectional during fine-tuning

---

## Troubleshooting

### Issue: Reconstruction loss not decreasing

**Possible causes**:
- Learning rate too high/low
- Batch size too small
- Mask ratio too high

**Solutions**:
- Adjust LR (try 5e-4 or 2e-3)
- Increase batch size (512 → 1024)
- Reduce mask ratio (15% → 10%)

### Issue: Class collapse persists after fine-tuning

**Possible causes**:
- Encoder not frozen initially
- Unfreezing too early
- Multi-task learning interference

**Solutions**:
- Verify encoder is frozen (check `requires_grad=False`)
- Unfreeze later (epoch 10 → epoch 20)
- Disable pointer prediction (`--no-predict-pointers`)

### Issue: Validation loss increases after unfreezing

**Possible causes**:
- Learning rate too high after unfreezing
- Encoder features corrupted

**Solutions**:
- Reduce LR by 50% after unfreezing
- Try gradual unfreezing (layer by layer)
- Monitor gradient norms

---

## Comparison with Alternatives

| Method | Accuracy Gain | Implementation | Training Time |
|--------|---------------|----------------|---------------|
| **Masked AE** | **+8-12%** | **6-8 hours** | **20 min** |
| TS-TCC (fixed) | +2-4% | 1-2 hours | 0 min* |
| Classical AE | +3-5% | 2-3 hours | 15 min |
| VAE | +4-7% | 3-4 hours | 18 min |

*Already pre-trained, only needs fine-tuning fixes

**Winner**: Masked Autoencoding (best accuracy gain, reasonable implementation time)

---

## References

1. **PatchTST** (ICLR 2023) - A Time Series is Worth 64 Words
   - Masked patch reconstruction: 21% MSE reduction
   - Self-supervised pre-training outperforms supervised

2. **BERT** (NAACL 2019) - Pre-training of Deep Bidirectional Transformers
   - Masked language modeling: State-of-art NLP
   - 15% masking ratio proven effective

3. **TS-TCC** (NeurIPS 2022) - Time-Frequency Consistency
   - Contrastive learning for time series
   - Current approach (needs fixes)

4. **Scientific Reports (2019)** - LSTM-based Stacked Autoencoder
   - Unsupervised pre-training improves convergence
   - Better than random initialization

---

## Next Steps

1. **Review Implementation Roadmap**: `MASKED_LSTM_IMPLEMENTATION_ROADMAP.md`
2. **Read Full Analysis**: `LSTM_CHART_INTERACTION_ANALYSIS.md`
3. **Compare Methods**: `PRETRAINING_METHOD_COMPARISON.md`
4. **Implement Core Architecture**: Start with Phase 1 (3-4 hours)
5. **Run Pre-training on RunPod**: H100 instance (20 min)
6. **Evaluate Results**: Compare with baseline

---

## Expected Timeline

| Day | Task | Duration |
|-----|------|----------|
| **Day 1** | Implement core architecture (Phase 1-2) | 5 hours |
| **Day 2** | Integration + testing (Phase 3-4) | 3 hours |
| **Day 3** | Pre-training on RunPod | 20 min |
| **Day 3** | Fine-tuning + evaluation | 1 hour |
| **Day 4** | Ablation study (optional) | 2 hours |

**Total**: 2-3 days (8-11 hours of work)

---

## Success Metrics

### Primary (Must Achieve)
- [  ] Class 1 accuracy > 30%
- [  ] Overall accuracy > 62%
- [  ] Class collapse broken

### Secondary (Nice to Have)
- [  ] Overall accuracy > 65%
- [  ] Class 1 accuracy > 45%
- [  ] Balanced predictions

### Failure (Requires Different Approach)
- [  ] Class 1 accuracy < 15%
- [  ] Overall accuracy < 60%
- [  ] Class collapse persists

---

**Questions?** Contact Data Science Team Lead

**Ready to Start?** See `MASKED_LSTM_IMPLEMENTATION_ROADMAP.md` for detailed implementation steps
