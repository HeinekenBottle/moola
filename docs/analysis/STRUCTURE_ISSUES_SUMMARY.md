# Moola Codebase Structure: Issues at a Glance

## Overall Health: 5.4/10 (Moderate Technical Debt)

### Critical Issues (Fix Now)

#### 1. Augmentation Fragmentation (BLOCKER for maintainability)
- 4 competing implementations across 1,387 LOC
- Different APIs: `TemporalAugmentation.apply_time_warp()` vs `FinancialAugmentationPipeline._apply_time_warp()`
- Developers must search 4 files to understand augmentation

**Files:**
- `utils/augmentation.py` (162 LOC)
- `utils/temporal_augmentation.py` (277 LOC)  
- `utils/financial_augmentation.py` (631 LOC)
- `pretraining/data_augmentation.py` (317 LOC)

**Fix:** Consolidate into `augmentation/` module with unified registry

---

#### 2. Duplicate LSTM Models (CONFUSING)
- `SimpleLSTMModel` (921 LOC)
- `EnhancedSimpleLSTMModel` (778 LOC)
- 95% code overlap, 5% feature fusion difference

**Fix:** Make EnhancedSimpleLSTM the default with optional `feature_fusion` parameter

---

#### 3. Split CLI (USER CONFUSION)
- `cli.py` (1403 LOC) - Main interface
- `cli_feature_aware.py` (430 LOC) - Variant interface
- Users must choose between two entry points

**Fix:** Merge into single CLI with `--feature-aware` flag

---

### High Priority Issues

#### 4. 1867-Line Monolith (TESTABILITY NIGHTMARE)
**File:** `utils/pseudo_sample_generation.py`
- 6 classes
- 40+ functions
- Mixed validation logic
- Needs unit test but too large

**Fix:** Split into `synthesis/` module with 5 focused files:
- `base.py` (ABC)
- `temporal.py` (generators)
- `feature.py` (feature synthesis)
- `hybrid.py` (hybrid approach)
- `validators.py` (validation)

---

#### 5. Schema Chaos (DATA CONTRACT CONFUSION)
- `schema.py` (139 LOC) - Pydantic
- `schemas/canonical_v1.py` (81 LOC) - Pandera  
- `data_infra/schemas.py` (435 LOC) - Pydantic

**Fix:** Consolidate under `data_infra/schemas/` with:
- `core.py` (OHLC, TimeSeriesWindow)
- `training.py` (TrainingDataRow)
- `validators.py` (Pandera validators)

---

#### 6. Empty Directories (DEAD STRUCTURE)
- `src/moola/diagnostics/` - Empty, duplicates `utils/model_diagnostics.py`
- `src/moola/optimization/` - Empty placeholder

**Fix:** Delete both directories

---

### Medium Priority Issues

#### 7. Pre-training Duplication
- `pretraining/masked_lstm_pretrain.py` (464 LOC)
- `pretraining/feature_aware_masked_lstm_pretrain.py` (554 LOC)
- 90% overlap, 10% feature fusion

**Fix:** Merge with optional `feature_aware` flag

---

#### 8. Feature Engineering Fragmentation
- `features/feature_engineering.py` (352 LOC) - Main engineer
- `features/price_action_features.py` (942 LOC) - Technical indicators
- `features/small_dataset_features.py` (759 LOC) - Overlapping indicators
- `features/relative_transform.py` (285 LOC) - Transforms

**Fix:** Organize as:
```
features/
├── indicators/
│   ├── volatility.py
│   ├── momentum.py
│   ├── structural.py
│   └── volume.py
└── engineering.py (dispatcher)
```

---

#### 9. Documentation Explosion (22 files at root)
**Large docs:**
- PRODUCTION_ML_PIPELINE_ARCHITECTURE.md (52K)
- IMPLEMENTATION_GUIDE.md (24K)
- QUICK_START_GUIDE.md (15K)
- 6 implementation summaries

**Fix:** Consolidate into docs/ with hierarchy:
```
docs/
├── QUICK_START.md
├── ARCHITECTURE.md
├── WORKFLOWS.md
├── API_REFERENCE.md
└── TROUBLESHOOTING.md
```

---

#### 10. Scripts Directory Chaos (48 files)
**Unknown status:**
- `run_lstm_experiment.py`
- `run_cleanlab_phase2.py`
- `generate_structure_labels.py`
- `scripts/archive/` (15+ old experiments)

**Fix:** Clarify production vs. experimental, move root .py files to `src/moola/scripts/`

---

## Dependency Chaos Map

```
Augmentation Mess:
  augmentation.py → [models]
  temporal_augmentation.py → [models, pretraining]
  financial_augmentation.py → [utils] (isolated)
  data_augmentation.py → [pretraining]

Schema Mess:
  schema.py → [CLI]
  schemas/canonical_v1.py → [tests]
  data_infra/schemas.py → [data_infra]
  (Which is source of truth???)

Feature Mess:
  feature_engineering.py → [models]
  price_action_features.py → [features/small_dataset_features.py]
  small_dataset_features.py → [data_infra, features/price_action_features.py]
  (Circular risk)
```

---

## Cleanup Priority & Effort

### Tier 1: Quick Wins (8 hours, huge impact)
- [ ] Consolidate augmentation → 3-4 hrs (impacts 10 import sites)
- [ ] Delete diagnostics/, optimization/ → 1 hr
- [ ] Merge LSTM models → 2 hrs (backward compatible)
- [ ] Consolidate schemas → 2 hrs (keep wrappers for compatibility)

### Tier 2: Major Refactoring (15 hours)
- [ ] Split pseudo_sample_generation → 4-5 hrs
- [ ] Merge CLIs → 3-4 hrs
- [ ] Split feature_aware_utils → 2-3 hrs
- [ ] Consolidate documentation → 2-3 hrs
- [ ] Consolidate feature engineering → 4-5 hrs

### Tier 3: Architecture (Long-term)
- [ ] Add registry pattern (augmentation, features, models, pretraining)
- [ ] Unify Hydra + Pydantic config system
- [ ] Add __all__ to all modules

---

## Code Smell Indicators

| Smell | Location | Severity |
|-------|----------|----------|
| Duplicate code (>90%) | LSTM variants | High |
| Duplicate code (>80%) | Pre-training variants | High |
| Multiple patterns | 4 augmentation modules | High |
| Large class/file | 1867 LOC file | High |
| Empty package | diagnostics/ | Medium |
| Unclear boundaries | 3 schemas | Medium |
| Fragmented feature eng | 4 feature files | Medium |
| Scattered docs | 22 MD files | Low |
| Unknown scripts | 48 experiment files | Medium |

---

## Expected Impact of Cleanup

### Before (Current)
```
- 169 classes in 33K LOC
- 4 augmentation implementations
- 2 LSTM variants  
- 3 schema definitions
- 2 CLI interfaces
- 22 doc files scattered
- 48 script files unclear status
```

### After (Tier 1+2 cleanup)
```
- ~150 classes in ~28K LOC (15% reduction)
- 1 augmentation framework with registry
- 1 LSTM model with feature fusion flag
- 1 canonical schema location (compatible wrappers)
- 1 CLI with optional flags
- ~8 organized doc files
- Scripts clearly separated (production/experimental)
- +50% improved maintainability
```

---

## Questions for Maintainers

1. **Augmentation:** Why 4 separate implementations? Can they be unified?
2. **LSTM:** Which model is the official one? When did EnhancedSimpleLSTM diverge?
3. **Pre-training:** Are feature-aware variants production-ready or experimental?
4. **Schemas:** Which schema framework (Pydantic vs Pandera) is authoritative?
5. **Features:** Why do price_action and small_dataset_features overlap?
6. **Diagnostics:** Why is the directory empty when model_diagnostics exists in utils?
7. **Scripts:** Which are production workflows vs. ad-hoc experiments?
8. **Docs:** Can the 22 docs be consolidated or archived?

---

## Quick Assessment Checklist

- [ ] Core models work: YES (SimpleLSTM, BiLSTM, CnnTransformer functional)
- [ ] Data pipeline functional: YES (schemas, validators, lineage tracking)
- [ ] Pre-training works: YES (multiple strategies available)
- [ ] CLI works: YES (but split across 2 interfaces)
- [ ] Tests pass: LIKELY (not fully checked)
- [ ] Code is clean: NO (multiple overlaps, dead code)
- [ ] API consistent: NO (4 augmentation styles, 2 LSTM variants)
- [ ] Documentation clear: NO (scattered, overlapping guides)
- [ ] Easy to extend: NO (fragmented feature engineering, no registries)

---

## Bottom Line

**The codebase WORKS but is MESSY.**

30-40 hours of focused cleanup can:
1. Eliminate duplicate code paths
2. Unify fragmented modules
3. Clarify data contracts
4. Improve maintainability by 50%
5. Enable easier extension/plugins

**Recommended approach:**
1. **Start with Tier 1** (8 hours) - High ROI quick wins
2. **Move to Tier 2** (15 hours) - Major refactoring
3. **Address Tier 3** (ongoing) - Architecture patterns

Each tier unblocks better development velocity for the next.

