# Archived Experimental Scripts

This directory contains experimental and one-off training scripts that are no longer part of the core training pipeline. These scripts are preserved for reference and historical analysis but should not be used for current training.

## Scripts

### Training Experiments

#### `train_augmented_20ep.py`
**Purpose:** Data augmentation via Gaussian jitter
**Status:** Experimental (low performer)
**Details:** Expanded 210 samples to 630 via σ=0.03 jitter, trained 20 epochs
**Result:** Proof-of-concept augmentation; data augmentation did not improve performance as expected
**When to use:** Reference only; prefer baseline models

#### `train_baseline_fast.py`
**Purpose:** Fast baseline training on subset
**Status:** Testing/Debug variant
**Details:** Quick version of baseline for testing purposes (not intended for production)
**When to use:** Never; use train_baseline_100ep.py instead

#### `train_boundary_regression.py`
**Purpose:** Boundary regression for expansion span detection
**Status:** Experimental (alternative paradigm)
**Details:** Predicts expansion start/end positions directly (420 regression targets) instead of classification
**Result:** Different approach; kept for reference but classification approach preferred
**When to use:** Reference if reconsidering span detection formulation

#### `train_expansion_local.py`
**Purpose:** Local expansion detection training
**Status:** Local development variant
**Details:** Version used for local CPU testing
**When to use:** Never; use RunPod + train_baseline_100ep.py instead

#### `train_stones_only_v2.py`
**Purpose:** Stones-only variant (v2)
**Status:** Deprecated variant
**Details:** Alternative implementation of stones-only approach
**When to use:** Never; use train_stones_only.py instead

### Fine-tuning Experiments

#### `finetune_jade.py`
**Purpose:** Generic Jade model fine-tuning
**Status:** Experimental
**Details:** General-purpose fine-tuning script
**When to use:** Reference; prefer position-encoding approach

#### `finetune_position_20ep.py`
**Purpose:** Position encoding fine-tuning (20 epochs)
**Status:** Experimental
**Details:** Quick variant of position encoding fine-tuning
**When to use:** Reference only

#### `finetune_position_crf.py`
**Purpose:** Position encoding with CRF
**Status:** Experimental (advanced technique)
**Details:** Combines position encoding with Conditional Random Fields for sequence modeling
**Result:** Interesting approach; not adopted due to complexity vs. benefit
**When to use:** Reference if adding structured prediction layer

#### `finetune_position_crf_20ep.py`
**Purpose:** CRF variant (20 epochs)
**Status:** Experimental
**Details:** Fast version of CRF approach
**When to use:** Reference only

### Experiment/Threshold Search

#### `experiment_a_threshold_grid.py`
**Purpose:** Threshold grid search (Experiment A)
**Status:** Analysis/Debug
**Details:** Systematic search for optimal classification threshold
**When to use:** Reference; use threshold_optimization.py in scripts/ instead

#### `experiment_b_augmentation.py`
**Purpose:** Augmentation experiments (Experiment B)
**Status:** Analysis/Comparison
**Details:** Comprehensive augmentation approach comparison
**When to use:** Reference for augmentation effectiveness

#### `quick_threshold_test.py`
**Purpose:** Quick threshold testing
**Status:** Quick-and-dirty variant
**Details:** Fast threshold evaluation for debugging
**When to use:** Never; use proper tools in scripts/

## Why These Are Archived

1. **Low/No Performance Gain:** Some experiments (augmentation, regression) didn't improve over baseline
2. **Superseded by Better Approaches:** Position encoding + CRF superseded by simpler methods
3. **Development Artifacts:** Local testing variants not needed once SSH/SCP established
4. **Deprecated Variants:** v2 variants when v1 proved sufficient

## Resurrection Process

If you need to revive any of these scripts:

1. **Copy to scripts/:** `cp scripts/archive/experiments/SCRIPT.py scripts/`
2. **Update imports/paths:** Paths may have changed since archival
3. **Test locally:** Verify functionality before running on RunPod
4. **Check dependencies:** Ensure required packages still available
5. **Document restoration:** Update CHANGELOG with reason and date

## Related Files

- `scripts/` - Core training scripts (keep these updated)
- `artifacts/README.md` - Artifact organization
- `CLAUDE.md` - Model and training documentation
- `CLEANUP_SUMMARY_2025-10-27.md` - Details on why each was archived

## Notes

- All scripts use Python 3.10+
- Require moola package installed: `pip3 install -e .`
- Use RunPod for GPU training (see WORKFLOW_SSH_SCP_GUIDE.md)
- Results saved to `artifacts/` (see artifacts/README.md)
