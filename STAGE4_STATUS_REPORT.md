# Stage 4: Verify Integrity - Status Report

**Date:** 2025-10-22  
**Branch:** `reset/stones-only`  
**Commits:** `236ac4d`, `87f3ef4`

## Summary

Stage 4 is **PARTIALLY COMPLETE** with critical blockers identified. The codebase has been significantly cleaned up (32 files deleted, 11,500+ lines removed), but fundamental architectural issues prevent full functionality.

## ✅ Completed Tasks

### 1. Clean up CLI references to deleted models

**Status:** MOSTLY COMPLETE

**Changes Made:**
- ✅ Updated model selection to Stones-only with `type=click.Choice(["jade", "sapphire", "opal"])`
- ✅ Removed `simple_lstm`, `logreg`, `rf`, `xgb`, `stack` from help text
- ✅ Updated `--pretrained-encoder` help: "for opal/sapphire transfer learning"
- ✅ Updated `--predict-pointers` help: removed `enhanced_simple_lstm` reference
- ✅ Cleaned up model-specific logic blocks:
  - Removed `enhanced_simple_lstm` conditionals
  - Removed `simple_lstm` conditionals
  - Updated to use `model in ["jade", "opal", "sapphire"]` pattern
- ✅ Updated MC dropout condition: `model in ["jade", "opal", "sapphire"]`
- ✅ Updated temperature scaling condition: `model in ["jade", "opal", "sapphire"]`
- ✅ Updated pretrained encoder loading: `model in ["opal", "sapphire"]`

**Remaining Issues:**
- ❌ CLI still imports `create_dual_input_processor` and `prepare_model_inputs` from deleted `dual_input_pipeline.py`
- ❌ These functions are used at lines 625, 695, 1444, 1445, 1495, 1512
- ❌ Need to replace with simpler data loading approach or recreate minimal versions

### 2. Fix broken imports in remaining files

**Status:** COMPLETE

**Files Fixed:**
- ✅ `src/moola/models/__init__.py` - Updated to import from `jade.py` instead of `jade_compact.py`
- ✅ `src/moola/models/registry.py` - Updated to import from `jade.py` instead of `jade_compact.py`
- ✅ Deleted `src/moola/utils/feature_aware_utils.py` (628 lines, heavily dependent on deleted models)
- ✅ Deleted `src/moola/config/feature_aware_config.py` (dependent on deleted models)

**Verification:**
```bash
grep -r "enhanced_simple_lstm\|simple_lstm\|logreg\|xgb\|stack" src/ --include="*.py" | wc -l
# Result: 0 (no more imports of deleted models)
```

### 3. Verify basic functionality

**Status:** PARTIAL - Critical blocker found

**Test Results:**

✅ **PASS:** Import JadeModel
```bash
python3 -c "from moola.models import JadeModel; print('✅ JadeModel import successful')"
# Output: ✅ JadeModel import successful
```

✅ **PASS:** Import get_model
```bash
python3 -c "from moola.models import get_model; print('✅ get_model import successful')"
# Output: ✅ get_model import successful
```

❌ **FAIL:** Instantiate Jade model
```bash
python3 -c "from moola.models import get_model; m = get_model('jade', input_size=11); print('✅ Jade model instantiation successful')"
# Error: NameError: name 'seed' is not defined
```

**Root Cause:** `jade.py` has a **malformed structure** with duplicate `__init__` methods:
- First `__init__` at line 168-192: Architecture definition (input_size, hidden_size, etc.)
- Second `__init__` at line 231+: Training wrapper (seed, n_epochs, device, etc.)
- The second `__init__` references `seed` parameter that doesn't exist in first signature
- This is **invalid Python** - the second `__init__` would override the first

### 4. Run remaining tests

**Status:** NOT ATTEMPTED (blocked by jade.py malformation)

**Reason:** Cannot run tests until `jade.py` is fixed. The malformed `__init__` prevents model instantiation.

**Expected Test Status:**
- `tests/test_jade_model.py` - Will fail due to:
  - JadeModel is `nn.Module`, not `BaseModel` (no `seed` parameter in first `__init__`)
  - Tests expect `BaseModel` interface with `fit()`, `predict()`, `predict_proba()`
  - Malformed `__init__` prevents instantiation
- `tests/test_stones_augmentation.py` - Unknown (not checked)
- `tests/test_import.py` - Likely to pass (basic import tests)

### 5. Check for remaining broken references

**Status:** COMPLETE

**Search Results:**
```bash
# Check for references to deleted files
grep -r "dual_input_pipeline\|enhanced_pipeline\|latent_mixup\|pretrain_pipeline" src/ --include="*.py"
# Found: src/moola/cli.py imports dual_input_pipeline (lines 440-442, deleted in commit)
```

**Broken References:**
1. ❌ `src/moola/cli.py` - Imports `create_dual_input_processor`, `prepare_model_inputs` (deleted)
2. ❌ `src/moola/cli.py` - Uses these functions at 6 locations
3. ❌ `src/moola/models/jade.py` - Malformed structure prevents usage

## 🚨 Critical Blockers

### Blocker 1: jade.py Malformed Structure

**Issue:** `jade.py` has duplicate `__init__` methods in the same class, causing `NameError: name 'seed' is not defined`

**Location:** `src/moola/models/jade.py` lines 168-192 and 231+

**Impact:** 
- Cannot instantiate JadeModel
- Cannot run any training
- Cannot run tests
- Blocks all Stage 4 verification

**Options to Fix:**
1. **Option A (Recommended):** Create minimal `JadeModel` wrapper that extends `BaseModel`
   - Keep first `__init__` (architecture) as `JadeNet` inner class
   - Create `JadeModel(BaseModel)` wrapper with second `__init__` (training)
   - Implement `fit()`, `predict()`, `predict_proba()` methods
   - ~200 lines of code

2. **Option B:** Fix jade.py structure
   - Separate architecture (`JadeNet`) from training wrapper (`JadeModel`)
   - Requires understanding full 1,254-line file
   - Risk of breaking existing functionality

3. **Option C:** Use registry.py approach only
   - Keep `JadeModel` as pure `nn.Module` (first `__init__` only)
   - Remove second `__init__` entirely
   - Create separate training script/function
   - Tests would need complete rewrite

### Blocker 2: CLI Data Pipeline Missing

**Issue:** CLI imports `create_dual_input_processor` and `prepare_model_inputs` from deleted `dual_input_pipeline.py`

**Location:** `src/moola/cli.py` lines 440-442 (import), 625, 695, 1444, 1445, 1495, 1512 (usage)

**Impact:**
- Cannot run `moola train` command
- Cannot load training data
- Blocks CLI functionality

**Options to Fix:**
1. **Option A (Recommended):** Create minimal data loading functions
   - Simple parquet loader
   - Basic train/test split
   - ~50 lines of code

2. **Option B:** Restore `dual_input_pipeline.py`
   - Contradicts Stones-only objective
   - Adds complexity

## 📊 Files Changed Summary

**Total Changes:**
- 34 files deleted (11,500+ lines removed)
- 3 files modified
- 2 commits

**Deleted Files (34):**
- 6 root-level training scripts
- 9 model files (including jade_compact.py)
- 5 data pipeline files
- 2 pretraining files
- 1 CLI patch file
- 8 test files
- 2 config/utils files (feature_aware)

**Modified Files (3):**
- `src/moola/models/__init__.py` - Updated import
- `src/moola/models/registry.py` - Updated import
- `src/moola/cli.py` - Cleaned up model references (partial)

## 🎯 Recommendations

### Immediate Actions (to unblock Stage 4)

1. **Fix jade.py malformation** (Option A recommended):
   ```python
   # Create minimal wrapper in jade.py
   class JadeModel(BaseModel):
       def __init__(self, seed=1337, hidden_size=96, ...):
           super().__init__(seed=seed)
           self.model = JadeNet(input_size=11, hidden_size=hidden_size, ...)
       
       def fit(self, X, y, ...):
           # Training logic
           pass
       
       def predict(self, X):
           # Prediction logic
           pass
   ```

2. **Create minimal data loading** in CLI:
   ```python
   # Replace dual_input_pipeline imports with:
   def load_training_data(path):
       df = pd.read_parquet(path)
       X = np.stack(df['features'].values)
       y = df['label'].values
       return X, y
   ```

3. **Run basic sanity tests**:
   ```bash
   python3 -c "from moola.models import get_model; m = get_model('jade', input_size=11); print('✅ Works')"
   pytest tests/test_import.py -v
   ```

### Alternative: Minimal Viable Product (MVP) Approach

If fixing jade.py is too complex, consider creating a **minimal Stones-only training script** that bypasses the broken CLI:

```python
# scripts/train_stones_minimal.py
import torch
from moola.models.registry import build
from moola.data.load import load_parquet

# Load data
X, y = load_parquet("data/processed/train_latest.parquet")

# Build model via registry
cfg = load_config("configs/model/jade.yaml")
model = build(cfg)

# Train (manual loop)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
# ... training loop ...
```

This would:
- ✅ Bypass broken CLI
- ✅ Use working registry.py
- ✅ Demonstrate Stones models work
- ✅ Provide baseline for comparison
- ❌ Lose CLI convenience
- ❌ Require manual training scripts

## 📝 Next Steps

**If continuing Stage 4:**
1. Fix jade.py malformation (Option A: minimal wrapper)
2. Create minimal data loading functions
3. Run sanity tests
4. Run test_jade_model.py (expect failures, document)
5. Complete Stage 4 verification

**If skipping to Stage 5:**
1. Document current state as "Stones-only codebase (CLI broken, needs fixing)"
2. Create DATA_NORMALIZATION.md
3. Update README.md with known issues
4. Update .gitignore
5. Mark Stage 4 as "PARTIAL" and Stage 5 as "COMPLETE"

**Recommended:** Fix blockers first, then proceed to Stage 5 with working codebase.

## 🔄 Recovery

If needed, rollback to pre-clean state:
```bash
git reset --hard pre-clean-legacy
# or
git bundle unbundle archive/pre-clean-legacy.bundle
```

## 📈 Progress Metrics

- **Lines Removed:** 11,500+
- **Files Deleted:** 34
- **Commits:** 2
- **Stage 1:** ✅ COMPLETE
- **Stage 2:** ✅ COMPLETE
- **Stage 3:** ✅ COMPLETE
- **Stage 4:** ⚠️ PARTIAL (60% complete, 2 critical blockers)
- **Stage 5:** ⏳ NOT STARTED

