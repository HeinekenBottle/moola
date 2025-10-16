# RunPod Training Guide v2: Full Pipeline with TS-TCC Pretraining

## Goal
Retrain all models on cleaned 105-sample dataset with optional TS-TCC pretraining for deep learning models.

---

## Setup Instructions for Claude Agent

**CRITICAL: Give this to the Claude agent on RunPod to avoid setup issues:**

```
Environment Setup (DO NOT SKIP):
1. PyTorch 2.1 is already installed (do not reinstall)
2. Install dependencies in this order:
   pip install "numpy<2.0" pandas scikit-learn xgboost loguru hydra-core pydantic
3. Verify numpy version: python -c "import numpy; print(numpy.__version__)"
   - Must be 1.x.x (NOT 2.x.x)
4. Navigate to: cd /workspace
5. Clone project: git clone <repo_url> OR sync from S3

Expected setup time: 2-3 minutes (NOT 30+ minutes)
```

---

## Training Pipeline

### Option A: Without TS-TCC (Faster, 20-30 minutes total)

```bash
cd /workspace/moola

# 1. Verify data (should be 105 samples)
python -c "import pandas as pd; df = pd.read_parquet('data/processed/train.parquet'); print(f'{len(df)} samples')"

# 2. Generate OOF predictions for base models
python -m moola.cli oof --model logreg --seed 1337 --device cpu
python -m moola.cli oof --model rf --seed 1337 --device cpu
python -m moola.cli oof --model xgb --seed 1337 --device cpu
python -m moola.cli oof --model rwkv_ts --seed 1337 --device cuda
python -m moola.cli oof --model cnn_transformer --seed 1337 --device cuda

# 3. Train stack ensemble
python -m moola.cli stack-train --seed 1337

# 4. Upload artifacts to S3
runpod-cli sync upload data/artifacts/ s3://bucket/data/artifacts/
```

### Option B: With TS-TCC Pretraining (Recommended, 30-40 minutes total)

```bash
cd /workspace/moola

# 1. Verify data
python -c "import pandas as pd; df = pd.read_parquet('data/processed/train.parquet'); print(f'{len(df)} samples')"

# 2. Pretrain TS-TCC encoder (10-15 minutes)
python -m moola.cli pretrain-tcc --device cuda --epochs 100 --patience 15
# This saves: data/artifacts/models/ts_tcc/pretrained_encoder.pt

# 3. Generate OOF predictions (traditional models: 5 min)
python -m moola.cli oof --model logreg --seed 1337 --device cpu
python -m moola.cli oof --model rf --seed 1337 --device cpu
python -m moola.cli oof --model xgb --seed 1337 --device cpu

# 4. Generate OOF predictions (deep learning with TS-TCC: 15-20 min)
# CRITICAL: Tell models to load pretrained encoder
python -m moola.cli oof --model rwkv_ts --seed 1337 --device cuda --load-pretrained
python -m moola.cli oof --model cnn_transformer --seed 1337 --device cuda --load-pretrained

# 5. Train stack ensemble (1 min)
python -m moola.cli stack-train --seed 1337

# 6. Upload artifacts to S3
runpod-cli sync upload data/artifacts/ s3://bucket/data/artifacts/
```

---

## Expected Training Times (RTX 4090)

| Step | Without TS-TCC | With TS-TCC | Notes |
|------|---------------|-------------|-------|
| Setup | 2-3 min | 2-3 min | pip install only |
| TS-TCC Pretrain | - | 10-15 min | 100 epochs, patience=15 |
| LogReg OOF | 10 sec | 10 sec | CPU, simple model |
| RF OOF | 30 sec | 30 sec | CPU, ensemble |
| XGB OOF | 1 min | 1 min | CPU, boosting |
| RWKV-TS OOF | 8-10 min | 8-10 min | GPU, 4 layers |
| CNN-Trans OOF | 8-10 min | 8-10 min | GPU, 3 layers |
| Stack Train | 1 min | 1 min | CPU, meta-learner |
| **Total** | **20-25 min** | **35-40 min** | Excludes setup |

---

## TS-TCC Pretraining Details

**What it does:**
- Learns temporal representations using contrastive learning
- Creates two augmented views of each time series
- Encoder learns to identify same sample in different views
- Captures temporal patterns without labels

**Augmentations used:**
- Jitter: Add Gaussian noise (sigma=0.03)
- Scaling: Random amplitude scaling
- Time masking: Random temporal dropout
- Creates 2 views per sample → 210 augmented samples from 105

**Architecture:**
- Same CNN backbone as CNN-Transformer
- Projection head: 128 → 64 dimensions
- InfoNCE loss with temperature=0.5
- 100 epochs, early stopping patience=15

**Output:**
- Pretrained encoder weights: `data/artifacts/models/ts_tcc/pretrained_encoder.pt`
- Training curves: `data/artifacts/models/ts_tcc/pretraining_loss.png`
- These weights get loaded into CNN-Transformer and RWKV-TS

---

## Verification Checklist

After training completes, verify:

```bash
# 1. Check OOF predictions exist
ls -lh data/artifacts/oof/*/v1/seed_1337.npy
# Expected: 5 files (logreg, rf, xgb, rwkv_ts, cnn_transformer)

# 2. Check stack model exists
ls -lh data/artifacts/models/stack/stack.pkl
# Expected: 4.1 MB file

# 3. Check TS-TCC encoder (if used)
ls -lh data/artifacts/models/ts_tcc/pretrained_encoder.pt
# Expected: 2-3 MB file

# 4. View stack metrics
cat data/artifacts/models/stack/metrics.json | jq '.accuracy, .f1, .ece'
# Expected: accuracy ~0.55-0.65, f1 ~0.50-0.60, ece ~0.10-0.20
```

---

## Troubleshooting

**Issue: numpy version error**
```
ImportError: numpy.core.multiarray failed to import
```
**Fix:**
```bash
pip uninstall numpy -y
pip install "numpy<2.0"
```

**Issue: CUDA out of memory**
```
RuntimeError: CUDA out of memory
```
**Fix:** Reduce batch size in model files:
- CNN-Transformer: `batch_size=256` (default 512)
- RWKV-TS: `batch_size=256` (default 512)

**Issue: TS-TCC pretraining fails**
```
No module named 'moola.cli.pretrain_tcc'
```
**Fix:** TS-TCC might not be implemented yet. Skip to Option A (without TS-TCC).

---

## Success Criteria

**Minimum acceptance:**
- ✓ All 5 base models OOF predictions generated
- ✓ Stack ensemble trained
- ✓ Stack accuracy ≥ 55% (current: 59%)
- ✓ All artifacts uploaded to S3

**Stretch goals (with TS-TCC):**
- ✓ CNN-Transformer accuracy improves by 3-5%
- ✓ Stack ensemble accuracy ≥ 60%
- ✓ Pretraining loss converges smoothly

---

## After Training: Generate CleanLab Predictions

Once models are trained and downloaded:

```bash
# 1. Generate predictions
python -m moola.cli predict \
  --model stack \
  --input data/processed/train.parquet \
  --output data/artifacts/predictions_v2_raw.csv

# 2. Convert to CleanLab format
python scripts/convert_to_cleanlab_format.py \
  data/artifacts/predictions_v2_raw.csv \
  data/artifacts/predictions_v2_cleanlab.csv

# 3. Upload to CleanLab Studio
# https://app.cleanlab.ai/
```

---

## Questions to Ask Claude Agent

**Before training:**
1. "Can you verify numpy version is <2.0?"
2. "Can you check if TS-TCC pretraining is implemented in the CLI?"
3. "How long do you estimate training will take?"

**During training:**
1. "What's the current OOF accuracy for each model?"
2. "Is TS-TCC pretraining loss decreasing?"

**After training:**
1. "What was the final stack ensemble accuracy?"
2. "Did TS-TCC pretraining improve CNN-Transformer accuracy?"
3. "Can you compare these results to the previous run (59% stack accuracy)?"
