# RunPod Retraining: Final Summary & Decision

## Critical Issue: Model Mismatch

**YOU OVERWROTE THE RUNPOD MODELS! ⚠️**

The locally-trained models you just created are INCOMPATIBLE with the RunPod-trained stack ensemble. Here's why:

| Model | RunPod (CV) | Local (Single Split) | Issue |
|-------|------------|---------------------|-------|
| LogReg | 42.8% | 38.1% | Different training |
| RF | 50.5% | **33.3%** | Worse than random! |
| XGB | 50.5% | 61.9% | Different split |
| RWKV | 50.5% | 38.1% | Different training |
| CNN | 42.9% | 57.1% | Different split |
| **Stack** | **59.0%** | - | **Trained on RunPod models!** |

**The predictions you generated are GARBAGE** because the stack model was trained on RunPod base models, not your local ones.

---

## Your Two Options

### Option A: Retrain Everything on RunPod (RECOMMENDED)

**Pros:**
- ✅ Consistent training environment
- ✅ Full GPU acceleration
- ✅ Cross-validation (more reliable)
- ✅ Can try TS-TCC pretraining
- ✅ Get reliable CleanLab predictions

**Cons:**
- Takes 35-40 minutes with TS-TCC
- Or 20-25 minutes without

**Commands:**
```bash
# Setup (2-3 min)
pip install "numpy<2.0" pandas scikit-learn xgboost loguru hydra-core pydantic

# Option 1: With TS-TCC (RECOMMENDED)
python -m moola.cli pretrain-tcc --device cuda --epochs 100 --patience 15
python -m moola.cli oof --model logreg --seed 1337 --device cpu
python -m moola.cli oof --model rf --seed 1337 --device cpu
python -m moola.cli oof --model xgb --seed 1337 --device cpu
python -m moola.cli oof --model rwkv_ts --seed 1337 --device cuda
python -m moola.cli oof --model cnn_transformer --seed 1337 --device cuda
python -m moola.cli stack-train --seed 1337

# Option 2: Without TS-TCC (Faster)
python -m moola.cli oof --model logreg --seed 1337 --device cpu
python -m moola.cli oof --model rf --seed 1337 --device cpu
python -m moola.cli oof --model xgb --seed 1337 --device cpu
python -m moola.cli oof --model rwkv_ts --seed 1337 --device cuda
python -m moola.cli oof --model cnn_transformer --seed 1337 --device cuda
python -m moola.cli stack-train --seed 1337
```

### Option B: Use Local Models (NOT RECOMMENDED)

**Pros:**
- Already done

**Cons:**
- ❌ Results unreliable (single split)
- ❌ RF scored 33.3% (worse than random!)
- ❌ No cross-validation
- ❌ Can't use RunPod stack model
- ❌ Need to retrain stack locally
- ❌ CleanLab predictions will be unreliable

---

## TS-TCC Decision

### Should You Use TS-TCC?

**YES - Try it on RunPod** (My recommendation)

**Expected outcomes:**
- **Realistic**: +2-5% improvement (42.9% → 45-48%)
- **Optimistic**: +5-10% improvement (42.9% → 48-53%)
- **Pessimistic**: No improvement

**Why try it:**
1. You're retraining anyway (sunk cost)
2. Only adds 10-15 minutes
3. Low risk, potential reward
4. Proper experiment setup
5. Learn if SSL helps with small datasets

**Why it might not help much:**
- 105 samples is borderline too small for SSL
- Typical SSL needs 500-1000+ samples
- Current bottleneck is data quality/quantity

---

## Setup Instructions for RunPod Claude Agent

**Give this to avoid 30-minute setup delays:**

```
CRITICAL SETUP (do this first, takes 2-3 minutes):

1. PyTorch 2.1 is already installed - DO NOT reinstall
2. Install dependencies in ONE command:
   pip install "numpy<2.0" pandas scikit-learn xgboost loguru hydra-core pydantic

3. Verify numpy < 2.0:
   python -c "import numpy; print(numpy.__version__)"
   Must be 1.x.x (NOT 2.x.x)

4. Navigate to project:
   cd /workspace/moola

Expected setup time: 2-3 minutes MAX (not 30+ minutes!)
```

---

## Training Workflow Comparison

### With TS-TCC (35-40 minutes total)

```
1. Setup                     2-3 min   ←─ Critical: numpy<2.0
2. TS-TCC Pretrain          10-15 min  ←─ NEW: Self-supervised learning
3. LogReg OOF               10 sec
4. RF OOF                   30 sec
5. XGB OOF                  1 min
6. RWKV-TS OOF (w/ pretrain) 8-10 min  ←─ Uses TS-TCC encoder
7. CNN-Trans OOF (w/ pretrain) 8-10 min  ←─ Uses TS-TCC encoder
8. Stack Train              1 min
────────────────────────────────────────
Total: 35-40 minutes
```

### Without TS-TCC (20-25 minutes total)

```
1. Setup                     2-3 min
2. LogReg OOF               10 sec
3. RF OOF                   30 sec
4. XGB OOF                  1 min
5. RWKV-TS OOF              8-10 min
6. CNN-Trans OOF            8-10 min
7. Stack Train              1 min
────────────────────────────────────────
Total: 20-25 minutes
```

---

## After Training: Generate CleanLab Predictions

Once RunPod training completes:

```bash
# 1. Download ALL models from RunPod (not just stack!)
runpod-cli sync download \
  s3://bucket/data/artifacts/models/ \
  data/artifacts/models/ \
  --overwrite

# 2. Generate predictions with CONSISTENT models
python -m moola.cli predict \
  --model stack \
  --input data/processed/train.parquet \
  --output data/artifacts/predictions_v2_raw.csv

# 3. Convert to CleanLab format
python scripts/convert_to_cleanlab_format.py \
  data/artifacts/predictions_v2_raw.csv \
  data/artifacts/predictions_v2_cleanlab.csv

# 4. Upload to CleanLab Studio
# Go to https://app.cleanlab.ai/
# Upload predictions_v2_cleanlab.csv
```

---

## Expected Results

### Iteration 1 (Before Cleaning)
- Dataset: 115 samples
- Flagged: 28 expansions
- Model: ??? accuracy

### Iteration 2 (After Cleaning)
- Dataset: 105 samples
- Flagged: ??? (target < 15)
- Model: 59% accuracy (without TS-TCC)
- Model: 60-62% accuracy? (with TS-TCC, optimistic)

---

## My Recommendation

**Train on RunPod with TS-TCC pretraining:**

1. **Low risk**: Only 10-15 extra minutes
2. **Proper setup**: SSL → Supervised (the right way)
3. **Learning opportunity**: See if SSL helps
4. **Best case**: +5-10% accuracy boost
5. **Worst case**: No improvement, but you tried

**Even if TS-TCC doesn't help, you'll have:**
- Clean, consistent models
- Reliable cross-validation results
- Proper CleanLab predictions for Iteration 2
- Knowledge that SSL doesn't help with 105 samples

---

## Files Created

1. `/Users/jack/projects/moola/scripts/convert_to_cleanlab_format.py` - Prediction converter
2. `/Users/jack/projects/moola/CLEANLAB_ITERATION_2_GUIDE.md` - CleanLab workflow
3. `/Users/jack/projects/moola/RUNPOD_TRAINING_GUIDE_V2.md` - Full training guide
4. `/Users/jack/projects/moola/src/moola/cli.py` - Added `pretrain-tcc` command

---

## Next Steps

1. **Decide**: TS-TCC or not?
2. **Train on RunPod**: Use the setup instructions above
3. **Download models**: Get ALL models (not just stack)
4. **Generate predictions**: Use consistent models
5. **Upload to CleanLab**: Compare Iteration 1 vs 2

**Don't use the predictions you just generated locally** - they're from mismatched models!
