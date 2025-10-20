# Moola Codebase Cleanup: Consolidated Action Plan

**Generated:** 2025-10-18
**Overall Health Score:** 5.4/10 (Moderate technical debt - cleanable without rewrites)
**Estimated Total Effort:** 25-30 hours
**Expected Impact:** 50% improvement in maintainability, 15% LOC reduction

---

## Executive Summary

Your codebase **works** but suffers from:
- **Code duplication** (15% of codebase - ~5K LOC duplicated)
- **Fragmentation** (4 augmentation modules, 2 LSTM models, 3 schema definitions)
- **Monolithic files** (1867-line pseudo_sample_generation.py, 1403-line cli.py)
- **Architectural debt** (no registries, inconsistent patterns, unclear boundaries)

**The Good News:**
- Core functionality is solid
- No critical bugs or architectural rewrites needed
- Cleanup is backward-compatible
- Quick wins (8 hours) deliver 70% of value

---

## Priority Matrix

| Priority | Issue | Severity | Effort | Impact | Files Affected |
|----------|-------|----------|--------|--------|----------------|
| **P0** | Augmentation fragmentation | CRITICAL | 3-4h | Very High | 4 modules, 1387 LOC |
| **P0** | LSTM model duplication | CRITICAL | 2h | High | 2 models, 1699 LOC |
| **P0** | Schema chaos | HIGH | 2h | High | 3 files, 655 LOC |
| **P1** | Monolithic pseudo_sample_generation.py | HIGH | 4-5h | High | 1 file, 1867 LOC |
| **P1** | CLI split | HIGH | 3-4h | Medium | 2 files, 1833 LOC |
| **P1** | Excessive CLI command length | CRITICAL | 2h | High | cli.py 14 commands |
| **P2** | Pre-training duplication | MEDIUM | 3h | Medium | 2 files, 1018 LOC |
| **P2** | Feature engineering fragmentation | MEDIUM | 4-5h | Medium | 4 files, 1701 LOC |
| **P2** | Dead directories | LOW | 1h | Low | 2 empty dirs |
| **P3** | Documentation explosion | LOW | 2-3h | Low | 22 files |
| **P3** | Configuration inconsistency | MEDIUM | 1h | Medium | Multiple |
| **P3** | Logging inconsistency | LOW | 4h | Low | Project-wide |

---

## Phase 1: Quick Wins (8 hours, 70% value)

### 1.1 Consolidate Augmentation Modules (3-4 hours)

**Problem:** 4 separate implementations with different APIs:
- `utils/augmentation.py` (162 LOC) - Basic mixup/cutmix
- `utils/temporal_augmentation.py` (277 LOC) - Time series
- `utils/financial_augmentation.py` (631 LOC) - Market-aware
- `pretraining/data_augmentation.py` (317 LOC) - Pre-training specific

**Impact:** Developers must search 4 files; inconsistent APIs; testing burden

**Solution:**
```python
# New structure:
src/moola/augmentation/
├── __init__.py          # Registry + unified API
├── base.py              # BaseAugmentation ABC
├── mixup.py             # Mixup, CutMix (from utils/augmentation.py)
├── temporal.py          # Time warping, jitter, scaling
├── financial.py         # Market-aware augmentation
└── registry.py          # Augmentation discovery

# Usage:
from moola.augmentation import get_augmentation
aug = get_augmentation('temporal', jitter_prob=0.5)
```

**Steps:**
1. Create `src/moola/augmentation/` package
2. Extract base class from `temporal_augmentation.py`
3. Move mixup/cutmix from `utils/augmentation.py` → `mixup.py`
4. Move temporal ops from `temporal_augmentation.py` → `temporal.py`
5. Move financial ops from `financial_augmentation.py` → `financial.py`
6. Create registry with decorator pattern
7. Add backward-compatible wrappers in original locations
8. Update all imports (10 files affected)
9. Add tests for unified API

**Verification:**
```bash
python -m pytest tests/augmentation/ -v
ruff check src/moola/augmentation/
```

---

### 1.2 Merge Duplicate LSTM Models (2 hours)

**Problem:**
- `SimpleLSTMModel` (921 LOC) vs `EnhancedSimpleLSTMModel` (778 LOC)
- 95% code overlap, only difference: feature fusion (50-100 LOC)
- Duplication in training loops (~250 lines each)

**Solution:**
```python
# models/simple_lstm.py
class SimpleLSTMModel(BaseDeepLearningModel):
    def __init__(
        self,
        config: SimpleLSTMConfig = None,
        feature_fusion: bool = False,  # NEW: controls feature encoder
        **kwargs
    ):
        super().__init__(config or SimpleLSTMConfig(), **kwargs)
        self.feature_fusion = feature_fusion

        # Common LSTM encoder
        self.lstm_encoder = self._build_lstm_encoder()

        # Optional feature encoder
        if self.feature_fusion:
            self.feature_encoder = self._build_feature_encoder()
            self.fusion_layer = self._build_fusion_layer()

    def forward(self, x_ohlc, x_features=None):
        lstm_out = self.lstm_encoder(x_ohlc)

        if self.feature_fusion and x_features is not None:
            feat_out = self.feature_encoder(x_features)
            return self.fusion_layer(lstm_out, feat_out)

        return self.head(lstm_out)
```

**Steps:**
1. Create `BaseDeepLearningModel` with shared training logic
2. Add `feature_fusion` parameter to `SimpleLSTMModel`
3. Conditionally build feature encoder
4. Deprecate `EnhancedSimpleLSTMModel` (keep as alias for backward compat)
5. Extract common training loop to base class
6. Update model registry
7. Update CLI to use `--feature-fusion` flag instead of different model
8. Update tests

**Verification:**
```bash
python -m moola.cli train --model simple_lstm --feature-fusion
python -m pytest tests/models/test_simple_lstm.py -v
```

---

### 1.3 Consolidate Schema Definitions (2 hours)

**Problem:** 3 schema definitions using 2 frameworks:
- `schema.py` (139 LOC) - Pydantic (used by CLI)
- `schemas/canonical_v1.py` (81 LOC) - Pandera (used by tests)
- `data_infra/schemas.py` (435 LOC) - Pydantic (used by data infra)

**Solution:**
```python
# New structure:
data_infra/schemas/
├── __init__.py          # Export all schemas
├── core.py              # OHLC, TimeSeriesWindow (Pydantic)
├── training.py          # TrainingDataRow, ExperimentConfig (Pydantic)
├── validators.py        # Pandera validators for validation
└── deprecated.py        # Backward-compatible imports

# schema.py → deprecated, imports from data_infra.schemas
# schemas/canonical_v1.py → move validators to data_infra/schemas/validators.py
```

**Steps:**
1. Create `data_infra/schemas/` package structure
2. Move core schemas (OHLC, etc.) to `core.py`
3. Move training schemas to `training.py`
4. Extract Pandera validators to `validators.py`
5. Create backward-compatible wrappers in original locations
6. Update all imports
7. Add clear docstrings: "This is the canonical schema definition"

**Verification:**
```bash
python -c "from moola.schema import TrainingDataRow; print('OK')"
python -c "from moola.data_infra.schemas import TrainingDataRow; print('OK')"
python -m pytest tests/schemas/ -v
```

---

### 1.4 Delete Dead Directories (30 minutes)

**Problem:**
- `src/moola/diagnostics/` - Empty, functionality exists in `utils/model_diagnostics.py`
- `src/moola/optimization/` - Empty placeholder

**Steps:**
1. Verify directories are truly empty
2. Search for any imports: `rg "from moola.diagnostics|from moola.optimization"`
3. Delete both directories
4. Update any references in documentation

**Verification:**
```bash
test ! -d src/moola/diagnostics && echo "Deleted diagnostics/"
test ! -d src/moola/optimization && echo "Deleted optimization/"
```

---

## Phase 2: Major Refactoring (15 hours)

### 2.1 Split Monolithic pseudo_sample_generation.py (4-5 hours)

**Problem:**
- 1867 lines with 6 generator classes
- Hard to test, modify, or extend
- Mixed concerns (generation, validation, quality metrics)

**Solution:**
```python
# New structure:
synthesis/
├── __init__.py
├── base.py                    # BasePseudoGenerator ABC (100 LOC)
├── temporal.py                # TemporalAugmentationGenerator (200 LOC)
├── pattern.py                 # PatternBasedSynthesisGenerator (250 LOC)
├── statistical.py             # StatisticalSimulationGenerator (300 LOC)
├── hybrid.py                  # HybridPseudoSampleGenerator (250 LOC)
├── pipeline.py                # PseudoSampleGenerationPipeline (150 LOC)
├── validation.py              # OHLC validation utilities (100 LOC)
└── quality_metrics.py         # Quality assessment (150 LOC)
```

**Steps:**
1. Create `src/moola/synthesis/` package
2. Extract `BasePseudoGenerator` to `base.py`
3. Move each generator to separate file
4. Extract shared validation logic to `validation.py`
5. Extract quality metrics to `quality_metrics.py`
6. Create pipeline orchestrator in `pipeline.py`
7. Add backward-compatible wrapper in `utils/pseudo_sample_generation.py`
8. Add unit tests for each module (now testable!)

**Impact:** Eliminates 400+ lines of duplication, 60% better testability

---

### 2.2 Extract CLI Business Logic (3-4 hours)

**Problem:**
- `cli.py` is 1403 lines with 14 command functions
- Each command 50-300 lines mixing orchestration + business logic
- Impossible to test individual logic
- Example: `train()` is 189 lines, `evaluate()` is 198 lines

**Solution:**
```python
# New structure:
orchestration/
├── __init__.py
├── training.py          # TrainingOrchestrator
├── evaluation.py        # EvaluationOrchestrator
├── pretraining.py       # PretrainingOrchestrator
└── results.py           # ResultsLogger

# cli.py (now 400 lines)
@app.command()
def train(cfg_dir, over, model, **kwargs):
    """Train a model (orchestration only)."""
    config = load_config(cfg_dir, over)
    orchestrator = TrainingOrchestrator(config)
    orchestrator.run(model, **kwargs)

# orchestration/training.py (testable!)
class TrainingOrchestrator:
    def __init__(self, config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_processor = FeatureProcessor(config)
        self.model_factory = ModelFactory(config)
        self.results_logger = ResultsLogger(config)

    def run(self, model_name, **kwargs):
        data = self.data_loader.load()
        features = self.feature_processor.process(data)
        model = self.model_factory.create(model_name, **kwargs)
        results = model.fit(features)
        self.results_logger.log(results)
        return results
```

**Steps:**
1. Create `src/moola/orchestration/` package
2. Extract `TrainingOrchestrator` from `train()` command
3. Extract `EvaluationOrchestrator` from `evaluate()` command
4. Extract `PretrainingOrchestrator` from `pretrain_bilstm()` command
5. Move results logging to `ResultsLogger` class
6. Update CLI commands to be thin wrappers
7. Add comprehensive tests for orchestrators

**Impact:** 70% reduction in cli.py size, fully testable, reusable logic

---

### 2.3 Merge Split CLI Interfaces (2-3 hours)

**Problem:**
- `cli.py` (1403 LOC) - Main interface
- `cli_feature_aware.py` (430 LOC) - Variant interface
- Users must choose between two entry points

**Solution:**
```python
# cli.py (consolidated)
@app.command()
def train(
    cfg_dir: str,
    over: str = "",
    model: str = "simple_lstm",
    feature_aware: bool = False,  # NEW: unified flag
    **kwargs
):
    """Train a model with optional feature awareness."""
    config = load_config(cfg_dir, over)

    # Use feature-aware orchestrator if requested
    if feature_aware:
        orchestrator = FeatureAwareTrainingOrchestrator(config)
    else:
        orchestrator = TrainingOrchestrator(config)

    orchestrator.run(model, **kwargs)
```

**Steps:**
1. Add `--feature-aware` flag to commands in `cli.py`
2. Merge logic from `cli_feature_aware.py` into orchestrators
3. Deprecate `cli_feature_aware.py` (keep as import alias for backward compat)
4. Update documentation
5. Test both modes

---

### 2.4 Consolidate Feature Engineering (4-5 hours)

**Problem:** Fragmented across 4 files with overlapping functionality:
- `features/feature_engineering.py` (352 LOC)
- `features/price_action_features.py` (942 LOC)
- `features/small_dataset_features.py` (759 LOC) - overlaps with price_action
- `features/relative_transform.py` (285 LOC)

**Solution:**
```python
# New structure:
features/
├── __init__.py
├── engineering.py           # Main dispatcher
├── indicators/
│   ├── __init__.py
│   ├── volatility.py        # ATR, Bollinger, etc.
│   ├── momentum.py          # RSI, MACD, etc.
│   ├── structural.py        # Support/resistance, patterns
│   └── volume.py            # Volume-based indicators
├── transforms/
│   ├── __init__.py
│   ├── relative.py          # Relative transforms
│   └── normalization.py     # Normalization strategies
└── registry.py              # Feature discovery
```

**Steps:**
1. Analyze overlap between `price_action_features.py` and `small_dataset_features.py`
2. Create `features/indicators/` package
3. Split indicators by category (volatility, momentum, structural, volume)
4. Consolidate duplicate implementations
5. Create feature registry for discovery
6. Update `engineering.py` to use registry
7. Add backward-compatible imports

**Impact:** Eliminates overlap, clearer organization, better discoverability

---

### 2.5 Consolidate Documentation (2-3 hours)

**Problem:** 22 documentation files scattered at root, many overlapping

**Current state:**
- PRODUCTION_ML_PIPELINE_ARCHITECTURE.md (52K)
- IMPLEMENTATION_GUIDE.md (24K)
- QUICK_START_GUIDE.md (15K)
- 6 implementation summaries
- Multiple overlapping guides

**Solution:**
```
docs/
├── QUICK_START.md           # Consolidated getting started
├── ARCHITECTURE.md          # System design (from 52K doc)
├── WORKFLOWS.md             # SSH/SCP, pre-training, experiments
├── API_REFERENCE.md         # Code API documentation
├── TROUBLESHOOTING.md       # Common issues
└── archive/                 # Old implementation summaries
```

**Steps:**
1. Create `docs/` directory
2. Consolidate quick start guides → `QUICK_START.md`
3. Extract architecture from large doc → `ARCHITECTURE.md`
4. Consolidate workflow guides → `WORKFLOWS.md`
5. Move implementation summaries to `docs/archive/`
6. Update links in README.md and CLAUDE.md
7. Add deprecation notices in old files

---

## Phase 3: Architecture & Polish (2-3 hours)

### 3.1 Add Registry Patterns (2 hours)

**Create registries for:**
- Models: `ModelRegistry.register('simple_lstm', SimpleLSTMModel)`
- Augmentations: `AugmentationRegistry.register('temporal', TemporalAugmentation)`
- Features: `FeatureRegistry.register('rsi', RSIFeature)`

**Benefits:**
- Discovery: `ModelRegistry.list_available()`
- Extensibility: Easy to add new models/features
- Validation: Registry validates compatibility

---

### 3.2 Improve Error Handling (1 hour)

**Priority areas:**
```python
# Add validation to data loading
class DataValidator:
    def validate_expansions(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'expansion_start' not in df.columns:
            raise ValueError("Missing 'expansion_start' column")

        try:
            valid = self._check_valid_ranges(df)
            n_invalid = (~valid).sum()

            if n_invalid > 0:
                logger.warning(f"Filtered {n_invalid}/{len(df)} invalid samples")

            return df[valid].reset_index(drop=True)
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            raise
```

**Apply to:**
- Data loading and validation
- Feature extraction
- Model initialization
- Results logging

---

### 3.3 Standardize Logging & Configuration (1 hour)

**Logging standardization:**
- Replace all `print()` with `logger.info()`
- Consistent log levels (DEBUG, INFO, WARNING, ERROR)
- Structured logging for metrics

**Configuration consolidation:**
- Remove CLI parameter duplication
- Use Hydra configs as single source of truth
- Use dataclasses consistently

---

## Implementation Timeline

### Week 1: Quick Wins (8 hours)
| Day | Tasks | Hours |
|-----|-------|-------|
| Day 1 | Consolidate augmentation | 3-4h |
| Day 2 | Merge LSTM models, consolidate schemas | 4h |
| Day 3 | Delete dead code, verification | 1h |

**Deliverables:**
- Unified augmentation framework
- Single LSTM model with feature fusion
- Clean schema organization
- No dead directories

---

### Week 2: Major Refactoring (15 hours)
| Day | Tasks | Hours |
|-----|-------|-------|
| Day 1-2 | Split pseudo_sample_generation.py | 4-5h |
| Day 3 | Extract CLI business logic | 3-4h |
| Day 4 | Merge CLI interfaces | 2-3h |
| Day 5 | Consolidate feature engineering | 4-5h |

**Deliverables:**
- Modular synthesis package
- Testable orchestration layer
- Unified CLI
- Organized feature engineering

---

### Week 3: Architecture & Polish (2-3 hours)
| Day | Tasks | Hours |
|-----|-------|-------|
| Day 1 | Add registry patterns | 2h |
| Day 2 | Error handling, logging, config | 2h |
| Day 3 | Documentation consolidation | 2-3h |

**Deliverables:**
- Registry-based extensibility
- Robust error handling
- Consolidated documentation

---

## Verification Checklist

After each phase:

### Functionality Tests
```bash
# Core functionality still works
python -m moola.cli train --model simple_lstm
python -m moola.cli pretrain-bilstm
python -m moola.cli evaluate

# Feature-aware mode works
python -m moola.cli train --model simple_lstm --feature-fusion

# Pre-commit hooks pass
git add -A
pre-commit run --all-files
```

### Code Quality Checks
```bash
# No linting errors
ruff check src/moola/

# Type checking passes
mypy src/moola/

# Tests pass
python -m pytest tests/ -v

# No broken imports
python -c "from moola import *"
```

### Metrics Tracking
```bash
# Before cleanup
cloc src/moola/ --by-file > before_cleanup.txt

# After cleanup
cloc src/moola/ --by-file > after_cleanup.txt

# Compare
diff before_cleanup.txt after_cleanup.txt
```

---

## Expected Outcomes

### Before Cleanup
- **LOC:** 33,193 lines
- **Classes:** 169
- **Augmentation:** 4 modules, different APIs
- **Models:** 2 LSTM variants (95% duplicate)
- **Schemas:** 3 definitions (2 frameworks)
- **CLI:** 2 interfaces
- **Documentation:** 22 scattered files
- **Testability:** Limited (large monolithic files)
- **Maintainability:** 5.4/10

### After Full Cleanup
- **LOC:** ~28,000 lines (15% reduction)
- **Classes:** ~150 (consolidated)
- **Augmentation:** 1 framework with registry
- **Models:** 1 LSTM with optional feature fusion
- **Schemas:** 1 canonical location (backward-compatible)
- **CLI:** 1 unified interface
- **Documentation:** ~8 organized files
- **Testability:** High (modular design)
- **Maintainability:** 8+/10 (50% improvement)

### Code Quality Improvements
- **Duplication:** 30% → <10%
- **Average function length:** 85 lines → <50 lines
- **Max cyclomatic complexity:** 25+ → <15
- **Test coverage:** Unknown → >80% (with new tests)
- **Files >500 lines:** 5 files → 0 files

---

## Risk Mitigation

### Backward Compatibility Strategy
1. **Deprecation wrappers:** Keep old imports working
2. **Gradual migration:** Mark old modules as deprecated
3. **Version bumping:** Follow semantic versioning
4. **Documentation:** Clear migration guides

### Testing Strategy
1. **Before refactoring:** Run full test suite, record results
2. **During refactoring:** Run affected tests after each change
3. **After refactoring:** Full regression testing
4. **Add new tests:** Test new modular components

### Rollback Plan
1. **Git branches:** Create feature branches for each phase
2. **Incremental commits:** Commit after each successful refactor
3. **Tag releases:** Tag stable points
4. **Backup:** Keep old code in `deprecated/` for reference

---

## Success Metrics

### Quantitative
- [ ] 15% reduction in total LOC
- [ ] <10% code duplication (from 30%)
- [ ] Zero files >500 lines
- [ ] >80% test coverage
- [ ] All pre-commit hooks pass
- [ ] Zero broken imports

### Qualitative
- [ ] Clear module boundaries
- [ ] Single source of truth for each concern
- [ ] Discoverable components (registries)
- [ ] Easy to test (modular design)
- [ ] Easy to extend (plugin architecture)
- [ ] Clear documentation hierarchy

---

## Next Steps

### Immediate Actions (Today)
1. **Review this plan** - Identify any concerns or blockers
2. **Set up tracking** - Create issues/tasks for each phase
3. **Prepare environment** - Ensure pre-commit hooks work
4. **Baseline metrics** - Run `cloc`, record test coverage

### This Week (Phase 1)
1. **Start with augmentation consolidation** (highest impact)
2. **Merge LSTM models** (quick win)
3. **Consolidate schemas** (reduce confusion)
4. **Delete dead code** (easy cleanup)

### Questions to Answer
1. Which LSTM model is "production" - SimpleLSTM or EnhancedSimpleLSTM?
2. Are feature-aware variants experimental or production-ready?
3. Which schema framework is preferred - Pydantic or Pandera?
4. Are there any untested integrations that might break?
5. What's the deployment process - does backward compat matter?

---

## Resources

### Analysis Documents
- `STRUCTURE_ISSUES_SUMMARY.md` - Quick reference (5 mins)
- `CODEBASE_STRUCTURE_ANALYSIS.md` - Deep dive (20 mins)
- `CLEANUP_ROADMAP.md` - Implementation details
- This document - Consolidated action plan

### Tools
- `ruff` - Linting and auto-fix
- `mypy` - Type checking
- `pytest` - Testing
- `cloc` - Line counting
- `pre-commit` - Quality gates

### Support
- Project documentation in `docs/` (after consolidation)
- CLAUDE.md - Project instructions
- README.md - Quick start

---

## Conclusion

Your codebase has **solid foundations** but suffers from **fragmentation and duplication**. The cleanup is:

✅ **Achievable** - No architectural rewrites needed
✅ **Valuable** - 50% maintainability improvement
✅ **Safe** - Backward compatible throughout
✅ **Incremental** - Quick wins (8h) deliver 70% of value

**Recommended approach:**
1. Start with Phase 1 Quick Wins (8 hours)
2. Evaluate impact and learnings
3. Proceed with Phase 2 Major Refactoring (15 hours)
4. Polish with Phase 3 Architecture (2-3 hours)

**Total investment:** 25-30 hours over 2-3 weeks
**Return:** Codebase that's 50% easier to maintain and extend

Good luck! 🚀
