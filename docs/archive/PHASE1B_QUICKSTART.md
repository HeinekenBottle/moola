# Phase 1b Quickstart: EnhancedSimpleLSTM with Strict Pretrained Loading

## Summary
Phase 1b registered EnhancedSimpleLSTM as the PRIMARY model and implemented strict pretrained weight validation.

## Quick Commands

### 1. Basic Training (No Pretrained)
```bash
moola train --model enhanced_simple_lstm \
  --split data/splits/fwd_chain_v3.json \
  --device cuda
```

### 2. Training with Pretrained Encoder
```bash
moola train --model enhanced_simple_lstm \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/pretrained/bilstm_encoder.pt \
  --freeze-encoder \
  --device cuda
```

### 3. Smoke Test (Baseline)
```bash
moola train --model simple_lstm \
  --split data/splits/fwd_chain_v3.json \
  --device cuda
```

## Key Files

| File | Purpose |
|------|---------|
| `src/moola/models/__init__.py` | Model registry (PRIMARY: enhanced_simple_lstm) |
| `src/moola/models/pretrained_utils.py` | Strict pretrained loader (≥80% match, 0 shape mismatches) |
| `src/moola/models/enhanced_simple_lstm.py` | Uses strict loader |
| `tests/models/test_pretrained_loading.py` | 9 unit tests (all passing) |
| `PHASE1B_COMPLETE.md` | Full documentation |

## Model Hierarchy

1. **enhanced_simple_lstm** - PRIMARY (BiLSTM + attention, pretrained support)
2. **simple_lstm** - BASELINE (lightweight, smoke tests)
3. **cnn_transformer** - EXPERIMENTAL (multi-task)
4. **rwkv_ts** - EXPERIMENTAL (RWKV)
5. **logreg, rf, xgb** - CLASSICAL ML (for stacking)

## Strict Validation Rules

- ✅ Match ratio ≥ 80% (configurable)
- ✅ Shape mismatches = 0 (configurable)
- ✅ ABORT on validation failure
- ✅ Comprehensive reporting

## Python API

```python
from moola.models import get_model

# Get model
model = get_model("enhanced_simple_lstm", seed=1337, device="cuda")

# Train with pretrained encoder
model.fit(
    X_train, 
    y_train,
    pretrained_encoder_path="artifacts/pretrained/bilstm_encoder.pt",
    freeze_encoder=True
)

# Check pretrained stats
print(model.pretrained_stats)
```

## Testing

```bash
# Run pretrained loading tests
pytest tests/models/test_pretrained_loading.py -v

# Expected: 9 passed
```

## Next Steps

1. Run pretraining on RunPod GPU
2. Generate pretrained encoder checkpoint
3. Train EnhancedSimpleLSTM with pretrained encoder
4. Compare performance vs. from-scratch baseline
5. Implement two-phase training (frozen → unfrozen)

## Troubleshooting

### Error: "match ratio 45.0% < 80.0%"
**Cause:** Incompatible encoder architecture
**Fix:** Verify encoder hidden_dim matches model hidden_size

### Error: "shape mismatch"
**Cause:** Tensor dimension mismatch
**Fix:** Check encoder and model architectures are compatible

### Error: "Pretrained checkpoint not found"
**Cause:** Invalid file path
**Fix:** Verify checkpoint path exists

## Documentation

- Full report: `PHASE1B_COMPLETE.md`
- Unit tests: `tests/models/test_pretrained_loading.py`
- Implementation: `src/moola/models/pretrained_utils.py`

---

**Status:** ✅ COMPLETE
**Tests:** ✅ 9/9 passing
**Date:** 2025-10-18
