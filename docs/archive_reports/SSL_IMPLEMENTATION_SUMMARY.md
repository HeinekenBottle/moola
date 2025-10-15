# TS-TCC Semi-Supervised Learning Implementation Summary

**Date**: October 14, 2025
**Task**: Task 17 - Implement SSL with TS-TCC contrastive pre-training
**Status**: Implementation in progress (Phase 1 & 2 core modules complete)

---

## Implementation Progress

### ✅ Completed Components

#### 1. Temporal Augmentation Module (`src/moola/utils/temporal_augmentation.py`)
- **Jitter**: Add Gaussian noise (σ=0.03)
- **Scaling**: Random magnitude scaling (σ=0.1)
- **Time Warping**: Smooth temporal distortion (σ=0.2)
- **Permutation**: Segment shuffling (disabled by default - breaks temporal structure)
- **Rotation**: Feature space rotation (disabled for OHLC)
- **TemporalAugmentation class**: Generates two augmented views for contrastive learning

**Key Features**:
- Configurable augmentation probabilities
- Preserves OHLC data characteristics
- Designed for financial time series

#### 2. TS-TCC Contrastive Learning Module (`src/moola/models/ts_tcc.py`)

**Components**:
- `TSTCCEncoder`: Shared CNN-Transformer backbone (reuses architecture from `cnn_transformer.py`)
- `ProjectionHead`: Maps encoder output to embedding space for contrastive loss
- `TSTCC`: Complete model combining encoder + projection head
- `info_nce_loss()`: InfoNCE contrastive loss implementation
- `TSTCCPretrainer`: Pre-training manager with training loop, early stopping, mixed precision

**Architecture**:
```
Input [batch, 105, 4]
  ↓
CNN Blocks [64, 128, 128] with multi-scale kernels [3, 5, 9]
  ↓
Transformer (3 layers, 4 heads)
  ↓
Global Average Pooling
  ↓
Projection Head (128 → 64)
  ↓
L2 Normalized Embeddings [batch, 64]
  ↓
InfoNCE Loss (temperature=0.07)
```

**Training Features**:
- Automatic mixed precision (FP16) support
- Early stopping with patience=15
- Validation split=10%
- Batch size=128 for pre-training
- Learning rate=1e-3 with AdamW optimizer

---

## Remaining Implementation (Pipelines)

### Phase 1: SSL Pre-training Pipeline (`src/moola/pipelines/ssl_pretrain.py`)
**Status**: Core logic in `TSTCCPretrainer`, need wrapper script

**Functionality**:
- Load 118k unlabeled windows from parquet
- Initialize TSTCCPretrainer
- Run contrastive pre-training (~100 epochs, 2-3 hours on GPU)
- Save pre-trained encoder weights
- Visualize embeddings with t-SNE

**Expected Output**:
- `data/artifacts/pretrained/encoder_weights.pt`
- `data/artifacts/pretrained/training_history.json`
- `data/artifacts/pretrained/embeddings_tsne.png`

### Phase 2: Fine-tuning Pipeline (`src/moola/pipelines/ssl_finetune.py`)
**Status**: Need to create

**Functionality**:
- Load pre-trained encoder weights
- Modify CNN-Transformer model to load pre-trained weights
- Fine-tune on 115 labeled samples with 5-fold CV
- Compare performance vs random initialization
- Generate OOF predictions for stacking

**Expected Output**:
- 5-fold CV accuracy (target: 64-68%)
- OOF predictions: `data/artifacts/oof/cnn_transformer_ssl/v1/seed_1337.npy`
- Performance comparison report

### Phase 3: Adaptive Pseudo-Labeling (`src/moola/pipelines/ssl_pseudo_label.py`)
**Status**: Need to create

**Functionality**:
- Load fine-tuned model from Phase 2
- Generate predictions on 118k unlabeled windows
- Apply class-aware confidence thresholds:
  - Consolidation: τ=0.92 (majority class, higher threshold)
  - Retracement: τ=0.85 (minority class, lower threshold)
- Select ~185 high-confidence pseudo-labeled samples
- Balance class distribution in pseudo-labeled set
- Save pseudo-labeled dataset

**Expected Output**:
- `data/processed/pseudo_labeled.parquet` (~185 samples)
- Confidence distribution analysis report
- Manual validation sample (20 random samples)

### Phase 4: Combined Dataset Retraining (`src/moola/pipelines/ssl_retrain.py`)
**Status**: Need to create

**Functionality**:
- Load 115 real labeled samples
- Load ~185 pseudo-labeled samples
- Combine into 300-sample dataset
- Retrain CNN-Transformer with 5-fold CV
- Generate OOF predictions for stacking ensemble
- Evaluate performance improvements

**Expected Output**:
- 5-fold CV accuracy (target: 65-70%)
- OOF predictions: `data/artifacts/oof/cnn_transformer_ssl_full/v1/seed_1337.npy`
- Final performance report vs baseline

---

## CLI Integration

### Proposed Commands

```bash
# Phase 1: Pre-train encoder on unlabeled data
moola ssl-pretrain \
  --unlabeled data/processed/unlabeled_118k.parquet \
  --output data/artifacts/pretrained/encoder_weights.pt \
  --epochs 100 \
  --batch-size 128 \
  --device cuda \
  --seed 1337

# Phase 2: Fine-tune on labeled data
moola ssl-finetune \
  --encoder data/artifacts/pretrained/encoder_weights.pt \
  --labeled data/processed/train.parquet \
  --output data/artifacts/oof/cnn_transformer_ssl \
  --folds 5 \
  --device cuda \
  --seed 1337

# Phase 3: Generate pseudo-labels
moola ssl-pseudo-label \
  --model data/artifacts/models/cnn_transformer_ssl/model.pt \
  --unlabeled data/processed/unlabeled_118k.parquet \
  --output data/processed/pseudo_labeled.parquet \
  --threshold-cons 0.92 \
  --threshold-retr 0.85 \
  --seed 1337

# Phase 4: Retrain on combined dataset
moola ssl-retrain \
  --labeled data/processed/train.parquet \
  --pseudo data/processed/pseudo_labeled.parquet \
  --output data/artifacts/oof/cnn_transformer_ssl_full \
  --folds 5 \
  --device cuda \
  --seed 1337
```

---

## File Structure

```
src/moola/
├── models/
│   ├── ts_tcc.py                    ✅ COMPLETE
│   └── cnn_transformer.py           (existing, needs load_pretrained_encoder() method)
├── pipelines/
│   ├── ssl_pretrain.py              ⏳ TO DO
│   ├── ssl_finetune.py              ⏳ TO DO
│   ├── ssl_pseudo_label.py          ⏳ TO DO
│   └── ssl_retrain.py               ⏳ TO DO
└── utils/
    └── temporal_augmentation.py     ✅ COMPLETE

data/
├── processed/
│   ├── train.parquet                (existing, 115 samples)
│   ├── unlabeled_118k.parquet       (need to create/locate on RunPod)
│   └── pseudo_labeled.parquet       (will be generated in Phase 3)
└── artifacts/
    ├── pretrained/
    │   ├── encoder_weights.pt       (Phase 1 output)
    │   └── training_history.json    (Phase 1 output)
    ├── oof/
    │   ├── cnn_transformer_ssl/     (Phase 2 output)
    │   └── cnn_transformer_ssl_full/ (Phase 4 output)
    └── models/
        └── cnn_transformer_ssl/     (Phase 2 model checkpoint)
```

---

## Next Steps

### Immediate (This Session)
1. ✅ Create temporal augmentation utilities
2. ✅ Create TS-TCC contrastive learning module
3. ⏳ Create Phase 1 pipeline script (ssl_pretrain.py)
4. ⏳ Create Phase 2 pipeline script (ssl_finetune.py)
5. ⏳ Create Phase 3 pipeline script (ssl_pseudo_label.py)
6. ⏳ Create Phase 4 pipeline script (ssl_retrain.py)
7. ⏳ Add CLI commands to `src/moola/cli.py`
8. ⏳ Update `cnn_transformer.py` with pre-trained weight loading

### Testing on RunPod
1. Locate/create 118k unlabeled windows dataset
2. Run Phase 1 pre-training (~2-3 hours)
3. Run Phase 2 fine-tuning (~30 min)
4. Run Phase 3 pseudo-labeling (~10 min)
5. Run Phase 4 combined retraining (~30 min)
6. Evaluate final performance vs 60.9% baseline

### Success Criteria
- **Phase 1-2 Target**: 64-68% accuracy (+3-8% vs random init baseline)
- **Phase 3-4 Target**: 65-70% accuracy (+4-9% vs baseline)
- **Calibration**: ECE < 0.1 (maintained or improved)
- **Fold Variance**: Std dev < 5% (improved stability)

---

## Implementation Notes

### Unlabeled Data Acquisition
The 118k unlabeled windows are likely located in:
- `/workspace/data/processed/unlabeled_windows.parquet` (RunPod)
- Or need to be extracted from full market history in pivot-experiments repo

If not available, can generate from:
1. Load full OHLC market history
2. Apply sliding window (window_size=105, stride=1)
3. Exclude 134 labeled samples
4. Save to parquet

### Pre-trained Weight Loading
Need to add method to `CnnTransformerModel`:
```python
def load_pretrained_encoder(self, encoder_path: Path) -> None:
    """Load pre-trained encoder weights from TS-TCC.

    Args:
        encoder_path: Path to encoder_weights.pt from Phase 1
    """
    checkpoint = torch.load(encoder_path, map_location=self.device)
    encoder_state_dict = checkpoint['encoder_state_dict']

    # Load weights into CNN blocks and Transformer
    # (careful mapping required due to class structure differences)
    ...
```

### Performance Tracking
Create comprehensive tracking:
- Training curves (loss, accuracy per epoch)
- t-SNE visualizations of learned embeddings
- Confusion matrices at each phase
- ECE calibration curves
- Per-class F1 scores

---

## Estimated Timeline

- **Core Implementation** (Phases 1-4 pipelines): 2-3 hours
- **CLI Integration**: 30 minutes
- **Testing on RunPod**: 4-5 hours (mostly GPU time)
- **Total**: ~1 day implementation + testing

---

## References

- TS-TCC Paper: https://arxiv.org/abs/2106.14112
- SimCLR (InfoNCE Loss): https://arxiv.org/abs/2002.05709
- FixMatch (Pseudo-labeling): https://arxiv.org/abs/2001.07685

---

**Status**: Ready to proceed with pipeline implementation
**Next**: Create `ssl_pretrain.py`, `ssl_finetune.py`, `ssl_pseudo_label.py`, `ssl_retrain.py`
