# Moola Agents.md Alignment Plan

## Current State Analysis

### ✅ **What's Compliant:**
- **Jade model family**: Found in `src/moola/models/jade_core.py` with proper architecture
- **Model registry**: Implemented in `src/moola/models/registry.py` with `build()` function
- **Config structure**: YAML configs exist in `configs/model/` (jade.yaml, jade_optimized.yaml)
- **Stones dropout compliance**: Jade models implement required dropout ranges (0.6-0.7 recurrent, 0.4-0.5 dense, 0.2-0.3 input)
- **Float32 enforcement**: Registry includes precision utilities
- **Test coverage**: Found `test_jade_model.py` and `test_stones_augmentation.py`

### ❌ **Critical Violations Found:**

1. **Missing Model Families (Section 7):**
   - Registry claims `{"jade", "sapphire", "opal"}` but only **Jade** exists
   - Opal and Sapphire configs exist but no corresponding model classes
   - AGENTS.md rule violated: "Do not invent families not present in code"

2. **Missing Feature Engineering (Section 6):**
   - No `src/moola/features/relativity.py` (referenced in AGENTS.md)
   - No `src/moola/features/zigzag.py` (referenced in AGENTS.md)
   - No feature configs in `configs/features/`
   - Missing `feat_*` prefixed files entirely

3. **Missing Scratch Directory (Section 4):**
   - No `scratch/` directory exists
   - Missing `scratch/README.md` with ownership/expiry tracking

4. **Config Gaps (Section 9):**
   - No `configs/_base/` directory for defaults
   - Missing feature configs entirely
   - Configs don't follow naming conventions (should be kebab-case)

5. **CLI Integration Missing (Section 15):**
   - No CLI commands for feature building
   - Missing `moola features` subcommand
   - No output contracts for scripts

## Implementation Plan

### Phase 1: Fix Model Taxonomy (High Priority)
1. **Remove phantom model families** from registry
   - Update `ALLOWED = {"jade"}` in `registry.py`
   - Delete `configs/model/opal.yaml` and `sapphire.yaml`
   - Update AGENTS.md to reflect actual model inventory

2. **Add missing Jade variants** if needed
   - Implement Jade-Compact if not present
   - Add proper model IDs following naming convention

### Phase 2: Implement Feature Engineering (High Priority)
1. **Create feature modules:**
   - `src/moola/features/relativity.py` - Relative price transformation
   - `src/moola/features/zigzag.py` - Zigzag pattern detection
   - `src/moola/features/__init__.py`

2. **Create feature configs:**
   - `configs/features/relativity.yaml` 
   - `configs/features/zigzag.yaml`
   - `configs/_base/features.yaml` for defaults

3. **Add CLI integration:**
   - `moola features --config configs/features/relativity.yaml`
   - Implement output contracts (path, rows, time)

### Phase 3: Create Scratch Infrastructure (Medium Priority)
1. **Create scratch directory structure:**
   - `scratch/README.md` with ownership/expiry tracking
   - `scratch/jade/`, `scratch/features/` subdirectories

2. **Add scratch policy enforcement:**
   - Document 7-day expiry rule
   - Add to gitignore
   - Create cleanup script

### Phase 4: Config Compliance (Medium Priority)
1. **Restructure configs:**
   - Create `configs/_base/` with defaults
   - Convert to kebab-case naming
   - Add config composition examples

2. **Add effective config persistence:**
   - Save `config.effective.yaml` alongside outputs
   - Implement in model training pipeline

### Phase 5: Testing & Validation (Medium Priority)
1. **Add feature tests:**
   - Invariance tests (price scaling)
   - Bounds tests ([0,1] features)
   - Causality tests (no future leakage)

2. **Add CLI tests:**
   - Integration tests for each subcommand
   - Sample file tests (200 bars)

3. **Gate compliance:**
   - Overfit test: ≥99% train accuracy in 20 epochs
   - Parameter count validation
   - Uncertainty quantification tests

### Phase 6: Documentation Updates (Low Priority)
1. **Update AGENTS.md:**
   - Fix model inventory section
   - Update quick start commands
   - Add actual discovered families

2. **Add model cards:**
   - Jade architecture documentation
   - Parameter count ranges
   - Training procedures

## Success Criteria

### Functional Requirements:
- ✅ Model inventory matches code reality
- ✅ Feature engineering pipeline with causality guarantees
- ✅ CLI commands for all major operations
- ✅ Scratch directory with proper lifecycle management

### Quality Gates:
- ✅ Feature invariance: price scaling → unchanged features
- ✅ Feature bounds: relative features in [0,1], distances in [-3,3]
- ✅ Model overfit: ≥99% train accuracy on 200 bars
- ✅ Test coverage: ≥80% for touched lines

### Architectural Requirements:
- ✅ No absolute price leakage
- ✅ Float32 precision throughout
- ✅ Reproducible seeding
- ✅ Config-driven behavior

## Estimated Timeline
- **Phase 1-2 (Critical)**: 2-3 days
- **Phase 3-4 (Important)**: 1-2 days  
- **Phase 5-6 (Polish)**: 1-2 days

Total: **4-7 days** for full AGENTS.md compliance

This plan ensures Moola aligns with its governance standards while maintaining the existing Jade model functionality and adding the missing feature engineering capabilities essential for the financial ML pipeline.