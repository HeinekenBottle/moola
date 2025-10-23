# AGENTS GUIDE

## Model Inventory (Section 7)

**Current model families discovered from repo:**

### Jade – BiLSTM Encoder with Multi-task Learning
**Implementation:** `src/moola/models/jade_core.py`  
**Registry:** `src/moola/models/registry.py` with `build(cfg)` function  
**Configs:** `configs/model/jade.yaml`, `configs/model/jade-optimized.yaml`

#### JadeCore (Full Variant)
- **Architecture:** 2-layer BiLSTM (128 hidden units × 2 directions)
- **Parameters:** ~85K total (~70K trainable)
- **Features:** Multi-head attention, global average pooling, dual task heads
- **Use case:** Large datasets, best performance
- **Config:** Full Jade settings with dropout ranges per AGENTS.md

#### JadeCompact (Small Dataset Variant)  
- **Architecture:** 1-layer BiLSTM (96 hidden units × 2 directions)
- **Parameters:** ~52K total (~45K trainable)
- **Features:** Projection head (64-dim), dual task heads
- **Use case:** Small datasets (174-sample regime), reduced overfitting
- **Config:** Compact settings with stronger regularization

#### Stones Compliance
✅ **Recurrent dropout:** 0.6-0.7 (configurable)  
✅ **Dense dropout:** 0.4-0.5 (configurable)  
✅ **Input dropout:** 0.2-0.3 (configurable)  
✅ **Uncertainty weighting:** Kendall et al. CVPR 2018 support  
✅ **Multi-task learning:** Type + pointer prediction heads  
✅ **Float32 precision:** Enforced throughout pipeline  

**Note:** Only Jade family is currently implemented. Opal and Sapphire referenced in legacy configs are not available in codebase.

## Feature Engineering (Section 6)

### Relativity Features
**Implementation:** `src/moola/features/relativity.py`  
**Config:** `configs/features/relativity.yaml`  
**CLI:** `python3 -m moola.features.relativity --config configs/features/relativity.yaml`

**Features:**
- **Price-relative:** `open_rel`, `high_rel`, `low_rel`, `close_rel` (range: [-1, 1])
- **Volume-normalized:** `volume_rel` (range: [0, 1])
- **Properties:** Scale-invariant, volatility-adjusted, no absolute price leakage

**AGENTS.md Compliance:**
✅ **Invariance:** Price scaling ×10 → features unchanged within 1e-6  
✅ **Bounds:** Relative features in [-1, 1], volume in [0, 1]  
✅ **Causality:** No future information used  
✅ **Float32:** All features cast to float32  

### Zigzag Features  
**Implementation:** `src/moola/features/zigzag.py`  
**Config:** `configs/features/zigzag.yaml`  
**CLI:** `python3 -m moola.features.zigzag --config configs/features/zigzag.yaml`

**Features:**
- **Pivot positions:** `pivot_1_pos`, `pivot_2_pos`, `pivot_3_pos`, `pivot_4_pos` (range: [0, 1])
- **Amplitude ratios:** `amplitude_1_ratio`, `amplitude_2_ratio` (range: [0, 1])
- **Pattern metrics:** `n_pivots_norm`, `pattern_symmetry` (range: [-1, 1])
- **Properties:** Pattern-based, distance features in [-3, 3], scale-invariant

**AGENTS.md Compliance:**
✅ **Invariance:** Price scaling ×10 → features unchanged within 1e-6  
✅ **Bounds:** Distance features in [-3, 3] (AGENTS.md requirement)  
✅ **Causality:** No future information used  
✅ **Pattern validation:** 3-20 pivots min/max limits  

## Quick Start Commands (Updated)

### Build Features
```bash
# Build relativity features (200-bar sample)
python3 -m moola.features.relativity \
  --config configs/features/relativity.yaml \
  --in data/sample_ohlcv.parquet \
  --out artifacts/features/relativity_sample.parquet

# Build zigzag features  
python3 -m moola.features.zigzag \
  --config configs/features/zigzag.yaml \
  --in data/sample_ohlcv.parquet \
  --out artifacts/features/zigzag_sample.parquet
```

### Train Jade Models
```bash
# Train Jade model with current config
python3 -m moola.cli train --model jade --device cuda

# Train Jade-Compact for small datasets
python3 -m moola.cli train --model jade --use-compact true --device cuda
```

### Test Quality Gates
```bash
# Run feature invariance tests
python3 -m pytest tests/test_feature_invariance.py -v

# Run CLI integration tests (200-bar sample requirement)
python3 -m pytest tests/test_cli_integration.py -v

# Check model parameter counts and overfit capability
python3 -m pytest tests/test_feature_invariance.py::TestQualityGates -v
```

- Python 3.10+ only; run `pip3 install -r requirements.txt` then `pre-commit install`.
- Dev install: `pip3 install -e .` keeps CLI entrypoints packaged for notebooks/tests.
- Build/format via `make format`; `make lint` runs black --check, isort --check-only, ruff.
- Test all with `make test` or `python3 -m pytest tests/ -v --tb=short`.
- Single test example: `python3 -m pytest tests/test_pipeline.py::test_simple_lstm_training -v`.
- Favor Make/CLI commands; manual runs should go through `python3 -m moola.cli`.
- RunPod sync lives in `scripts/sync_to_runpod.sh`; uses rsync + `~/.ssh/id_ed25519_runpod`.
- After syncing, `pip install --no-cache-dir -r requirements-runpod.txt` installs GPU deps remotely.
- Keep artifacts/data out of git; rsync script already excludes heavy folders.
- Imports/formatting: isort black profile (stdlib→third-party→local), Black 100 cols, trailing commas.
- Types: annotate new functions; prefer TypedDict/Protocol/dataclasses for structured payloads.
- Naming: snake_case funcs/vars, PascalCase classes, UPPER_SNAKE constants/config keys.
- Error handling: raise specific exceptions, never bare except, chain with `raise ... from e`.
- Logging: route through `moola.logging_setup.configure_logging()`; prints only in CLI entrypoints.
- Config: Hydra/YAML driven (`src/moola/config`, `defaults.yaml`); override configs instead of hardcoding.
- Stones doctrine is mandatory—enhanced models must keep uncertainty-weighted loss toggles enabled.
- Seed randomness via `moola.utils.seeds`; mirror src layout when adding tests.
- Pre-commit runs GitOps/data/security guards—fix hook failures; no Cursor/Copilot rules, follow `CLAUDE.md`.
- Watch for numpy<2/pandas/pyarrow wheel mismatches on RunPod; reinstall with `pip --no-cache-dir` if needed.
