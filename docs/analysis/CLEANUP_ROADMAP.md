# Moola Codebase Cleanup Roadmap

## Phase 1: Quick Wins (8 hours)

### 1.1 Delete Dead Directories (1 hour)

**Action:** Remove empty module placeholders
```bash
rm -rf src/moola/diagnostics/
rm -rf src/moola/optimization/
```

**Verify no imports:**
```bash
grep -r "from moola.diagnostics\|from moola.optimization" src/
# Should return: 0 results
```

**Note:** `model_diagnostics.py` stays in utils/

---

### 1.2 Consolidate Schemas (2 hours)

**Current State:**
- `src/moola/schema.py` (139 LOC) - Pydantic
- `src/moola/schemas/canonical_v1.py` (81 LOC) - Pandera
- `src/moola/data_infra/schemas.py` (435 LOC) - Pydantic

**Action Plan:**

1. **Keep canonical source**: `src/moola/data_infra/schemas.py` (authoritative)

2. **Create backward compatibility wrappers**:

```python
# src/moola/schema.py (new - thin wrapper)
"""Backward compatibility wrapper. Use data_infra.schemas instead."""

import warnings
from .data_infra.schemas import (
    TrainingDataRow,
    OHLCBar,
    TimeSeriesWindow,
)

warnings.warn(
    "Importing from moola.schema is deprecated. "
    "Use moola.data_infra.schemas instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["TrainingDataRow"]
```

3. **Update imports** (5 import sites in CLI):
```bash
grep -r "from.*schema import\|from moola.schema" src/moola/cli.py
# Update to: from moola.data_infra.schemas import ...
```

4. **Keep schemas/canonical_v1.py** for now (pandera validator):
```python
# Add deprecation note at top
"""Deprecated. Use data_infra.schemas.validators instead."""
```

**Files to Update:**
- `src/moola/cli.py` (line ~38: update TrainingDataRow import)
- Keep old files with deprecation warnings (backward compatible)

---

### 1.3 Merge LSTM Models (2 hours)

**Current State:**
- `SimpleLSTMModel` (921 LOC) - OHLC-only
- `EnhancedSimpleLSTMModel` (778 LOC) - OHLC + features

**Action Plan:**

1. **Expand SimpleLSTMModel with feature fusion flag**:
```python
# src/moola/models/simple_lstm.py
class SimpleLSTMModel(BaseModel):
    def __init__(
        self,
        ...,
        use_feature_fusion: bool = False,  # NEW
        feature_input_size: int = 0,        # NEW
        ...
    ):
        self.use_feature_fusion = use_feature_fusion
        self.feature_input_size = feature_input_size
        
        # Forward pass logic already handles both modes
        # or add: if use_feature_fusion: [feature fusion code]
```

2. **Delete EnhancedSimpleLSTM** (move unique code to SimpleLSTM if any):
```bash
rm src/moola/models/enhanced_simple_lstm.py
```

3. **Update model registry** (`src/moola/models/__init__.py`):
```python
# Remove from registry:
# - "enhanced_simple_lstm": EnhancedSimpleLSTMModel

# Keep:
# - "simple_lstm": SimpleLSTMModel
```

4. **Update CLI** (`src/moola/cli.py`):
```bash
grep -n "enhanced_simple_lstm\|EnhancedSimpleLSTM" src/moola/cli.py
# Remove references, use simple_lstm with --feature-fusion flag (Phase 2)
```

5. **Check imports**:
```bash
grep -r "EnhancedSimpleLSTM\|enhanced_simple_lstm" src/ tests/ examples/
# Update all to SimpleLSTMModel
```

**Backward Compatibility:**
```python
# src/moola/models/enhanced_simple_lstm.py (new - thin wrapper)
"""Backward compatibility alias. Use SimpleLSTMModel with use_feature_fusion=True."""

import warnings
from .simple_lstm import SimpleLSTMModel as EnhancedSimpleLSTMModel

warnings.warn(
    "EnhancedSimpleLSTMModel is deprecated. "
    "Use SimpleLSTMModel(use_feature_fusion=True) instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["EnhancedSimpleLSTMModel"]
```

---

### 1.4 Consolidate Augmentation Modules (3-4 hours)

**Current State:** 4 competing modules, 1,387 LOC
- `utils/augmentation.py` (162 LOC) - Mixup/CutMix
- `utils/temporal_augmentation.py` (277 LOC) - Time warp, jitter, scaling
- `utils/financial_augmentation.py` (631 LOC) - Market-aware
- `pretraining/data_augmentation.py` (317 LOC) - TimeSeriesAugmenter

**Action Plan:**

1. **Create new unified module**:
```
src/moola/augmentation/
├── __init__.py (exports + registry)
├── base.py (AugmentationStrategy ABC)
├── temporal.py (TimeWarp, Jitter, Scaling)
├── financial.py (Market-aware strategies)
├── mixup.py (Mixup, CutMix)
└── registry.py (get_augmentation function)
```

2. **Create base class** (`augmentation/base.py`):
```python
from abc import ABC, abstractmethod
import torch

class AugmentationStrategy(ABC):
    """Base class for all augmentation strategies."""
    
    @abstractmethod
    def apply(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Apply augmentation to input tensor."""
        pass
```

3. **Move + adapt each module**:
   - `temporal_augmentation.py` → `augmentation/temporal.py` (convert to Strategy classes)
   - `financial_augmentation.py` → `augmentation/financial.py` (keep existing, inherit from base)
   - `augmentation.py` → `augmentation/mixup.py` (adapt to strategy pattern)
   - `data_augmentation.py` → `augmentation/TimeSeriesAugmenter` in `temporal.py`

4. **Create registry** (`augmentation/registry.py`):
```python
from typing import Dict, Type
from .base import AugmentationStrategy

_REGISTRY: Dict[str, Type[AugmentationStrategy]] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_augmentation(name: str, **kwargs) -> AugmentationStrategy:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown augmentation: {name}")
    return _REGISTRY[name](**kwargs)

def list_augmentations() -> List[str]:
    return list(_REGISTRY.keys())
```

5. **Export from __init__.py**:
```python
# src/moola/augmentation/__init__.py
from .registry import get_augmentation, list_augmentations
from .base import AugmentationStrategy
from .temporal import (
    TemporalAugmentation,
    TimeWarpAugmentation,
    JitterAugmentation,
    ScalingAugmentation,
)
from .financial import FinancialAugmentationPipeline
from .mixup import MixupAugmentation, CutMixAugmentation

__all__ = [
    "AugmentationStrategy",
    "get_augmentation",
    "list_augmentations",
    # ... rest of exports
]
```

6. **Update imports** (10+ import sites):
```bash
# Before:
from moola.utils.temporal_augmentation import TemporalAugmentation
from moola.utils.augmentation import mixup_cutmix
from moola.pretraining.data_augmentation import TimeSeriesAugmenter

# After:
from moola.augmentation import (
    get_augmentation,
    TemporalAugmentation,
    TimeSeriesAugmenter,
    MixupAugmentation,
)

# Update these files:
grep -l "from.*augmentation import" src/moola/models/*.py  # 5+ files
grep -l "from.*augmentation import" src/moola/pretraining/*.py  # 2+ files
grep -l "from.*temporal_augmentation import" src/moola/models/*.py  # 3+ files
```

7. **Deprecate old modules**:
```python
# utils/augmentation.py (new)
"""Backward compatibility wrapper. Use moola.augmentation instead."""

import warnings
from moola.augmentation import (
    MixupAugmentation,
    CutMixAugmentation,
)

warnings.warn(
    "utils.augmentation is deprecated. Use moola.augmentation instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for compatibility
def mixup_criterion(*args, **kwargs):
    return MixupAugmentation()(*args, **kwargs)

def mixup_cutmix(*args, **kwargs):
    return CutMixAugmentation()(*args, **kwargs)
```

---

## Phase 2: Major Refactoring (15 hours)

### 2.1 Split Pseudo-Sample Generation (4-5 hours)

**File:** `utils/pseudo_sample_generation.py` (1867 LOC)

**New Structure:**
```
src/moola/synthesis/
├── __init__.py
├── base.py (BasePseudoGenerator)
├── temporal.py (TemporalAugmentationGenerator, TemporalAugmentationAdvanced)
├── feature.py (FeatureAugmentationGenerator)
├── hybrid.py (HybridPseudoGenerator)
└── validators.py (PseudoSampleValidator)
```

**Action:**
1. Extract each class to its own file
2. Create shared utilities module if needed
3. Update imports (check: `grep -r "pseudo_sample_generation" src/`)
4. Move validators inline or to validators.py

---

### 2.2 Split Feature-Aware Utils (2-3 hours)

**File:** `utils/feature_aware_utils.py` (624 LOC)

**New Structure:**
```
src/moola/transfer_learning/
├── __init__.py
├── data.py (prepare_feature_aware_data)
├── pretraining.py (run_feature_aware_pretraining)
├── evaluation.py (evaluate_transfer_learning)
└── analysis.py (analyze_encoder_importance, create_experiment_report)
```

**Action:**
1. Move functions to appropriate modules
2. Update imports (check: `grep -r "feature_aware_utils" src/`)
3. Create __init__.py with unified exports

---

### 2.3 Merge CLI Interfaces (3-4 hours)

**Files:** `cli.py` (1403 LOC) + `cli_feature_aware.py` (430 LOC)

**Action Plan:**

1. **Keep cli.py as primary**

2. **Migrate feature-aware commands** to cli.py with `--feature-aware` flag:
```python
@app.command()
@click.option("--feature-aware", is_flag=True, help="Use feature-aware pre-training")
def pretrain_bilstm(..., feature_aware):
    if feature_aware:
        from .pretraining.feature_aware_masked_lstm_pretrain import ...
    else:
        from .pretraining.masked_lstm_pretrain import ...
```

3. **Keep cli_feature_aware.py** as thin wrapper:
```python
"""Backward compatibility wrapper. Use cli.py with --feature-aware flag."""

import warnings
from .cli import app

warnings.warn(
    "cli_feature_aware is deprecated. Use main CLI with --feature-aware flag.",
    DeprecationWarning,
    stacklevel=2
)

if __name__ == "__main__":
    app()
```

4. **Update pyproject.toml** (keep both entry points for now):
```toml
[project.scripts]
moola = "moola.cli:app"
moola-feature-aware = "moola.cli_feature_aware:app"  # deprecated
```

---

### 2.4 Consolidate Documentation (2-3 hours)

**Current:** 22 files at root

**New Structure:**
```
docs/
├── README.md (project overview)
├── QUICK_START.md (consolidate 3 guides)
├── ARCHITECTURE.md 
├── WORKFLOWS.md (SSH/SCP + pre-training + monitoring)
├── API_REFERENCE.md (models, CLI)
├── TROUBLESHOOTING.md
└── archive/ (old implementation summaries)
```

**Action:**
1. Create docs/ hierarchy
2. Move PRODUCTION_ML_PIPELINE_ARCHITECTURE.md to docs/ARCHITECTURE.md
3. Consolidate QUICK_START guides
4. Archive old implementation summaries
5. Update root README to point to docs/

---

### 2.5 Consolidate Feature Engineering (4-5 hours)

**Files:** 4 fragmented modules (1,701 LOC)

**New Structure:**
```
src/moola/features/
├── __init__.py
├── indicators/
│   ├── __init__.py
│   ├── volatility.py (rolling volatility)
│   ├── momentum.py (returns, momentum indicators)
│   ├── structural.py (gaps, swing points)
│   └── volume.py (tick volume proxy)
├── engineering.py (AdvancedFeatureEngineer - dispatcher)
├── relative_transform.py (keep as-is)
└── small_dataset_features.py (merge into engineering or keep specialized)
```

**Action:**
1. Refactor price_action_features.py into indicators/ submodule
2. Keep relative_transform.py as-is
3. Consolidate small_dataset_features overlaps with indicators/
4. Create dispatcher in engineering.py
5. Update imports across codebase

---

## Phase 3: Cleanup Tasks (1-2 hours)

### 3.1 Organize Scripts (1 hour)

**Action:**
1. Create `src/moola/scripts/` with clear subdirs:
   ```
   src/moola/scripts/
   ├── experiments/ (runpod experiments)
   ├── analysis/ (data analysis)
   ├── deployment/ (deploy_to_fresh_pod.py, etc.)
   └── README.md (which script does what)
   ```

2. Move:
   - Root `.py` files to appropriate subdirs
   - scripts/archive/* to scripts/archive/ or delete

3. Document purpose of each script

### 3.2 Add __all__ Exports (30 mins)

**Action:** Add to all modules:
```python
__all__ = [
    "PublicClass",
    "public_function",
    # ...
]
```

Priority modules:
- models/__init__.py
- augmentation/__init__.py (after creation)
- features/__init__.py
- utils/__init__.py

---

## Checklist for Completion

### Phase 1: Quick Wins
- [ ] Delete diagnostics/ and optimization/
- [ ] Consolidate schemas (keep wrappers)
- [ ] Merge LSTM models (keep wrapper)
- [ ] Create augmentation/ module with registry
- [ ] Update all imports (10+ sites)
- [ ] Run tests: `pytest tests/`
- [ ] Verify backward compatibility

### Phase 2: Refactoring
- [ ] Split pseudo_sample_generation.py
- [ ] Split feature_aware_utils.py
- [ ] Merge CLI interfaces
- [ ] Consolidate documentation
- [ ] Consolidate feature engineering
- [ ] Run full test suite
- [ ] Update README

### Phase 3: Cleanup
- [ ] Organize scripts/
- [ ] Add __all__ exports
- [ ] Final test run
- [ ] Document changes in CHANGELOG.md

---

## Rollout Strategy

### Branch: `refactor/structure-cleanup`

```bash
git checkout -b refactor/structure-cleanup

# Phase 1: Quick Wins (commit after each logical change)
git add -A && git commit -m "cleanup: delete empty modules"
git add -A && git commit -m "refactor: consolidate schemas with backward compat"
git add -A && git commit -m "refactor: merge LSTM models with feature_fusion flag"
git add -A && git commit -m "refactor: create unified augmentation module"

# Phase 2: Major Refactoring (iterate)
git add -A && git commit -m "refactor: split pseudo_sample_generation"
git add -A && git commit -m "refactor: split feature_aware_utils"
git add -A && git commit -m "refactor: merge CLI interfaces"
git add -A && git commit -m "docs: consolidate documentation"
git add -A && git commit -m "refactor: consolidate feature engineering"

# Phase 3: Cleanup
git add -A && git commit -m "docs: organize scripts and add __all__ exports"

# PR with comprehensive description
```

---

## Verification Steps

After each phase:

```bash
# Type checking
python -m mypy src/moola/ --ignore-missing-imports

# Linting
python -m ruff check src/moola/

# Tests
pytest tests/ -v

# Import check
python -c "from moola import cli; from moola.models import get_model; print('OK')"

# Backward compat
python -c "from moola.schema import TrainingDataRow; print('Deprecated wrapper works')"
python -c "from moola.models import EnhancedSimpleLSTMModel; print('LSTM alias works')"
```

---

## Estimated Timeline

| Phase | Duration | Impact |
|-------|----------|--------|
| Phase 1 (Quick Wins) | 8 hours | High clarity, 70% of value |
| Phase 2 (Major) | 15 hours | Architecture improvement, API consistency |
| Phase 3 (Cleanup) | 2 hours | Polish, discovery improvement |
| **Total** | **25 hours** | **50% maintainability improvement** |

Can be done incrementally without blocking development.

