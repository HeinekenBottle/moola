# Phase 0: Executive Summary - Baseline Discovery Complete

**Date:** 2025-10-18
**Status:** ✅ COMPLETE - All critical assets discovered and verified
**Decision:** 🟢 **PROCEED TO PHASE 1** with modifications

---

## Critical Questions: ANSWERED

### 1. Where is the 105-sample labeled set?
**Answer:** Actually **98 samples** (not 105)

**Canonical Dataset:**
```
Path: /Users/jack/projects/moola/data/processed/train_clean.parquet
Samples: 98 (binary: consolidation=56, retracement=42)
Features: OHLC windows (105 bars × 4 features per sample)
Quality: Cleanlab-reviewed, thoroughly filtered
Last modified: 2025-10-16
```

**Augmented Version (SMOTE):**
```
Path: /Users/jack/projects/moola/data/processed/train_smote_300.parquet
Samples: 300 (perfectly balanced 150/150)
Method: SMOTE augmentation
```

**Verdict:** ✅ Found and verified. Use `train_clean.parquet` as canonical v1.

---

### 2. Where is unlabeled OHLC for TS2Vec?
**Answer:** Found - 11,873 samples ready for pre-training

**Primary Corpus:**
```
Path: /Users/jack/projects/moola/data/raw/unlabeled_windows.parquet
Samples: 11,873 unlabeled OHLC windows (105 bars each)
Size: 2.20 MB
Shape per window: (105, 4) [open, high, low, close]
```

**Cached Version (Fast Loading):**
```
Path: /Users/jack/projects/moola/data/pretraining/unlabeled_ohlc.npy
Shape: (11873, 105, 4)
Size: 38.05 MB
Format: NumPy array (memory-mapped)
```

**Verdict:** ✅ Found and verified. Ready for TS2Vec pretraining.

---

### 3. How are splits currently chosen?
**Answer:** 🚨 **CRITICAL DISCREPANCY FOUND** 🚨

**Data files say:** "5-fold forward-chaining with purge"
```
Location: /Users/jack/projects/moola/data/artifacts/splits/v1/
Strategy: Forward-chaining (temporal)
Folds: 5 folds with stratification
Seed: 1337
```

**Code actually uses:** "Stratified random K-fold" (LEAKAGE RISK!)
```
Evidence:
- splits.py:68-94 uses StratifiedKFold (shuffles data!)
- cli.py:126 uses train_test_split(shuffle=True, random_state=42)
- simple_lstm.py:534 uses StratifiedKFold
```

**Verdict:** ⚠️ **CRITICAL ISSUE** - Code is NOT using the forward-chaining splits that exist in data/artifacts/splits/v1/. This creates look-ahead bias in financial time series!

**Action Required:** Phase 1 MUST fix this immediately.

---

### 4. Does artifacts/ts2vec/encoder_v1.pt exist?
**Answer:** Not exactly, but pretrained BiLSTM encoder exists

**Found:**
```
Path: /Users/jack/projects/moola/data/artifacts/pretrained/bilstm_encoder_correct.pt
Size: 2.03 MB (135K params)
Format: PyTorch state_dict
Purpose: BiLSTM masked autoencoder weights
Last modified: 2025-10-16
```

**Pretrained Loading Logic:**
```
Location: src/moola/models/simple_lstm.py:197-274
Features:
  ✅ Tensor name matching
  ✅ Shape validation
  ✅ Mismatch detection
  ✅ Encoder freezing support
  ⚠️ NO abort on <80% match (just warns)
```

**Verdict:** ✅ Pretrained encoder exists. Need to create `artifacts/ts2vec/` directory structure and move/alias this file.

---

### 5. Where are synthetic augmentations cached?
**Answer:** SMOTE augmentation deployed, pseudo-generation experimental

**Production Augmentation:**
```
Method: SMOTE (Synthetic Minority Oversampling)
Output: /Users/jack/projects/moola/data/processed/train_smote_300.parquet
Samples: 300 (from 98 originals)
Balance: Perfect 150/150
Quality: No KS p-value tracked (SMOTE default)
```

**Experimental Augmentation:**
```
Location: src/moola/utils/pseudo_sample_generation.py (1867 LOC)
Methods: Temporal, pattern-based, statistical, hybrid
Status: Code exists but unclear if deployed
Cache: Not found in data/
```

**Out-of-Fold Model Predictions (pseudo-labels):**
```
Path: /Users/jack/projects/moola/data/processed/oof/
Models: CNN, LSTM, XGB, RF, LogReg (10 files total)
Purpose: Stacking ensemble features
```

**Verdict:** ✅ SMOTE deployed. ⚠️ Need to create `data/synthetic_cache/` with versioned augmentation + quality metrics.

---

### 6. Which models are experimental?
**Answer:** Clear division between active and experimental

**ACTIVE (Production - Keep in models/):**
1. **SimpleLSTM** (`simple_lstm.py`, 921 LOC)
   - Status: ✅ Main production model
   - Registry: ✅ YES (models/__init__.py)
   - CLI: ✅ YES (train, evaluate, oof commands)
   - Params: 921 params (lightweight for 98 samples)
   - Features: BiLSTM, dual-input (OHLC + features), mixup/cutmix, pretrained encoder support

2. **EnhancedSimpleLSTM** (`enhanced_simple_lstm.py`, 778 LOC)
   - Status: ⚠️ Experimental variant
   - Registry: ❌ NO (not in __init__.py!)
   - CLI: ❌ NO (not wired up)
   - Note: Feature-aware transfer learning variant
   - **Action:** Clarify if this should replace SimpleLSTM or stay experimental

3. **LogReg, RF, XGBoost** (logreg.py, rf.py, xgb.py)
   - Status: ✅ Active baseline models
   - Purpose: Stacking ensemble base learners
   - Registry: ✅ YES

4. **Stack** (stack.py, 87 LOC)
   - Status: ✅ Active meta-learner
   - Purpose: Stacking ensemble

**EXPERIMENTAL (Move to models_extras/):**
1. **BiLSTM Masked Autoencoder** (`bilstm_masked_autoencoder.py`, 359 LOC)
   - Purpose: Pre-training only (not inference)
   - Used by: CLI pretrain-bilstm command
   - **Move to:** models_extras/pretraining/

2. **Feature-Aware BiLSTM Autoencoder** (`feature_aware_bilstm_masked_autoencoder.py`, 471 LOC)
   - Purpose: Pre-training variant
   - Recent work: Oct 18 (active development?)
   - **Move to:** models_extras/pretraining/

3. **BiLSTM Autoencoder (non-masked)** (`bilstm_autoencoder.py`, 403 LOC)
   - Purpose: Deprecated (replaced by masked version)
   - **Move to:** models_extras/deprecated/

4. **CNN-Transformer** (`cnn_transformer.py`, 2277 LOC)
   - Params: 56,384 (over-parameterized for 98 samples!)
   - Status: Experimental
   - **Move to:** models_extras/experimental/

5. **RWKV-TS** (`rwkv_ts.py`, 1041 LOC)
   - Params: 409,088 (SEVERELY over-parameterized!)
   - Status: Prototype
   - **Move to:** models_extras/experimental/

6. **TS-TCC** (`ts_tcc.py`, 408 LOC)
   - Purpose: Time-series contrastive learning
   - **Move to:** models_extras/pretraining/

**Verdict:** ✅ Clear division identified. Move 6 experimental models to `models_extras/`.

---

### 7. Ready to start Phase 1?
**Answer:** YES, with critical modification

**Critical Issue Found:** Split strategy mismatch
- Data files have forward-chaining splits
- Code uses random stratified splits (LEAKAGE!)
- **Must fix immediately in Phase 1**

**Proceed with modified priority:**
1. ~~Phase 1: Data Registry~~ → **Phase 1a: FIX SPLIT STRATEGY FIRST**
2. Phase 1b: Data Registry structure
3. Phase 2: Model clarity
4. Phase 3: Validation & logging

---

## Summary: What We Found

### ✅ Data Assets (All Verified)

| Asset | Status | Location | Count/Size |
|-------|--------|----------|------------|
| Labeled dataset | ✅ FOUND | `data/processed/train_clean.parquet` | 98 samples |
| Unlabeled corpus | ✅ FOUND | `data/raw/unlabeled_windows.parquet` | 11,873 samples |
| SMOTE augmentation | ✅ FOUND | `data/processed/train_smote_300.parquet` | 300 samples |
| Forward-chaining splits | ✅ FOUND | `data/artifacts/splits/v1/` | 5 folds |
| Pretrained encoder | ✅ FOUND | `data/artifacts/pretrained/bilstm_encoder_correct.pt` | 2.03 MB |
| OOF predictions | ✅ FOUND | `data/processed/oof/` | 10 models |

### ⚠️ Critical Issues

1. **Split Strategy Discrepancy** (CRITICAL)
   - Files exist for forward-chaining
   - Code uses random stratified splits
   - Creates look-ahead bias in time series
   - **Impact:** Results are unreliable
   - **Fix:** Enforce forward-chaining splits in code

2. **EnhancedSimpleLSTM Not Registered** (HIGH)
   - Exists but not wired into CLI
   - Unclear if production or experimental
   - **Action:** Clarify status with user

3. **No Validation Guards** (HIGH)
   - No enforcement of forward-chaining
   - No synthetic contamination prevention
   - No pretrained match threshold (<80% abort)
   - **Fix:** Implement in Phase 3

### ✅ What's Working

1. **Data Quality:** Clean, documented, versioned
2. **Pretrained Loading:** Robust implementation exists
3. **Metrics:** Basic accuracy, F1, precision, recall, ECE
4. **Logging:** Manifest with git SHA, timestamps
5. **Model Pipeline:** SimpleLSTM production-ready

### ❌ What's Missing

1. **Reliability Diagrams:** ECE computed but not visualized
2. **Advanced Metrics:** No PR-AUC, Brier score, class-wise F1
3. **Validation Guards:** No enforcement of data integrity
4. **Synthetic Quality:** No KS p-value tracking for SMOTE
5. **Data Registry:** No versioned, immutable data structure

---

## Modified Refactor Plan

### Phase 1a: FIX SPLIT STRATEGY (URGENT - 2-3 hours)

**Problem:** Code uses random splits despite forward-chaining splits existing.

**Solution:**
1. Modify `splits.py` to use forward-chaining from `data/artifacts/splits/v1/`
2. Update CLI to load split indices from JSON (not generate randomly)
3. Add guard: raise error if random split detected
4. Verify: all training uses temporal splits

**Files to modify:**
- `src/moola/data/splits.py` (remove StratifiedKFold)
- `src/moola/cli.py` (load splits from JSON)
- `src/moola/models/simple_lstm.py` (use provided splits)

**Acceptance:**
```bash
moola train --model simple_lstm --split fwd_chain_v3
# Should load splits/v1/fold_0.json and use those indices
# Should raise error if split=random
```

---

### Phase 1b: Data Registry Structure (4-5 hours)

**Create versioned, immutable data structure:**

```
data/
├── labeled_windows/
│   └── v1/
│       ├── X_ohlc.npy          # From train_clean.parquet (98, 105, 4)
│       ├── y.npy               # Labels (98,)
│       ├── meta.json           # T=105, D_ohlc=4, classes, distribution
│       └── manifest.json       # Audit trail
│
├── unlabeled_windows/
│   └── v1/
│       ├── X_ohlc.npy          # From unlabeled_windows.parquet (11873, 105, 4)
│       ├── meta.json
│       └── manifest.json
│
├── synthetic_cache/
│   └── v1_smote_300/
│       ├── X_ohlc.npy          # From train_smote_300.parquet (300, 105, 4)
│       ├── y.npy               # Labels (300,)
│       ├── augmentation_config.json  # SMOTE params
│       ├── quality_metrics.json      # Add KS p-value
│       └── manifest.json
│
└── splits/
    └── fwd_chain_v3.json       # Alias to data/artifacts/splits/v1/fold_0.json
```

**Implementation:**
1. Create DataRegistry class (load_labeled, load_unlabeled)
2. Convert parquet files to .npy arrays
3. Generate meta.json and manifest.json
4. Implement split purity validation (val/test = 0 synthetic)

---

### Phase 2: Model Clarity (3-4 hours)

**Actions:**
1. Clarify SimpleLSTM vs EnhancedSimpleLSTM (ask user)
2. Move experimental models to `models_extras/`
3. Create ModelRegistry with ACTIVE/LEGACY lists
4. Update documentation

---

### Phase 3: Validation & Logging (4-5 hours)

**Implement guards:**
1. Forbid random splits (raise error)
2. Abort if pretrained match <80%
3. Abort if synthetic KS p < 0.1
4. Validate split purity (val/test no synthetic)

**Enhance logging:**
1. Add PR-AUC, Brier score, class-wise F1
2. Generate reliability diagrams (calibration plots)
3. Create run manifest with full lineage

---

## Baseline Survey Report

**Generated Documents (8 files, ~97 KB):**

| Document | Purpose | Size |
|----------|---------|------|
| PHASE0_README.md | Overview and guide | - |
| PHASE0_DATA_SURVEY.md | Main reference (all data paths) | 580 lines |
| PHASE0_CODE_SURVEY.md | Models, splits, loading | 661 lines |
| PHASE0_METRICS_SURVEY.md | Validation and logging | - |
| PHASE0_INDEX.md | Navigation by use case | - |
| PHASE0_COMPLETE.txt | Executive findings | - |
| PHASE0_DATA_SUMMARY.txt | One-pager quick ref | - |
| PHASE0_QUICK_REFERENCE.md | Python loading snippets | - |

**All reports located in:** `/Users/jack/projects/moola/PHASE0*.md`

---

## Decision: Proceed to Phase 1a

**Status:** 🟢 **GO**

**Immediate Action:**
1. ✅ Phase 0 complete - all critical assets discovered
2. 🚨 **START PHASE 1a** - Fix split strategy (URGENT)
3. Then proceed with Phase 1b (Data Registry)

**Critical Path:**
```
Phase 1a: Fix splits (2-3h) → URGENT
  ↓
Phase 1b: Data registry (4-5h)
  ↓
Phase 2: Model clarity (3-4h)
  ↓
Phase 3: Validation (4-5h)
```

**Total Estimated Time:** 13-17 hours (reduced from original 15-20h due to found assets)

---

## Questions for User (Before Proceeding)

1. **SimpleLSTM vs EnhancedSimpleLSTM:**
   - SimpleLSTM is registered and used in CLI
   - EnhancedSimpleLSTM exists but not registered
   - Which should be the main production model?
   - Should EnhancedSimpleLSTM replace SimpleLSTM or remain experimental?

2. **Split Strategy Fix:**
   - Confirm: Use forward-chaining splits from `data/artifacts/splits/v1/`?
   - Confirm: Forbid random splits entirely (raise error)?
   - OK to break backward compatibility with old random split code?

3. **Dataset Count:**
   - You mentioned "105 samples" but data shows 98
   - Is this expected? (7 samples removed in quality filtering?)
   - Should we document this discrepancy?

4. **TS2Vec vs BiLSTM:**
   - Current pretrained encoder is BiLSTM masked autoencoder
   - Do you also want TS2Vec pretraining implemented?
   - Or is BiLSTM sufficient for transfer learning?

5. **Ready to start Phase 1a?**
   - Should I fix the split strategy immediately?
   - Or do you want to review Phase 0 findings first?

---

## Next Steps

**Awaiting User Confirmation:**
- Answer questions above
- Approve Phase 1a start (split strategy fix)
- Clarify SimpleLSTM vs EnhancedSimpleLSTM

**Once approved, I will:**
1. Start Phase 1a: Fix split strategy (enforce forward-chaining)
2. Add guard: forbid random splits
3. Update CLI to load split indices from JSON files
4. Verify all training uses temporal splits
5. Generate Phase 1a completion report

**Phase 0 Status:** ✅ COMPLETE
**Phase 1a Status:** ⏸️ AWAITING USER APPROVAL
**All Reports:** `/Users/jack/projects/moola/PHASE0*.md`
