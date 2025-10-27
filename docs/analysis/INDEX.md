# Analysis & Experiment Documentation Index

**Last Updated:** 2025-10-27
**Total Documents:** 32
**Organization:** Consolidated from 106 scattered files across project root

---

## Quick Navigation

- [Deployment Guides](#deployment-guides) - Production deployment instructions
- [Experimental Results](#experimental-results) - Analysis of all experiments
- [Technical Analysis](#technical-analysis) - Deep dives and investigations
- [Planning & Checklists](#planning--checklists) - Future improvements
- [Archived Documents](#archived-documents) - Historical/legacy files

---

## Deployment Guides

**Consolidated guide:** `docs/DEPLOYMENT_GUIDE.md` (PRIMARY - use this!)

### Individual Deployment Variants

1. **Position Encoding Deployment** (RECOMMENDED)
   - File: `POSITION_ENCODING_DEPLOYMENT.md`
   - Status: ✅ PRODUCTION READY
   - F1 Score: 0.220 (exceeds 0.20 target)
   - Features: 13 (12 base + position_encoding)
   - Key fix: Class weighting (pos_weight=13.1)
   - Duration: 15-20 min on RTX 4090

2. **Stones-Only Deployment** (FAST BASELINE)
   - File: `STONES_ONLY_DEPLOYMENT.md`
   - Status: ✅ READY
   - F1 Score: >0.10 (expected)
   - Key change: Removed countdown task (was 91% of loss)
   - GPU utilization: 10-15% (5x better than baseline)
   - Duration: 15 min on RTX 4090

3. **Baseline 100-Epoch Deployment** (REFERENCE)
   - File: `BASELINE_100EP_DEPLOYMENT.md`
   - Status: ✅ READY
   - Purpose: Comprehensive baseline with extensive logging
   - Tasks: 4 (type, ptr, span, countdown)
   - Metrics collected: 6 CSV files (loss, uncertainty, gradients, features)
   - Duration: 20 min on RTX 4090

4. **RunPod Deployment Guide** (CPU PRE-COMPUTATION)
   - File: `RUNPOD_DEPLOYMENT.md`
   - Purpose: 32-core CPU pod for feature pre-computation
   - Target time: 5-10 minutes for 1.8M bars
   - Cost: ~$0.15 per run
   - Output: 8GB pre-computed features

5. **RunPod Deployment Plan** (EXPANSION-FOCUSED)
   - File: `RUNPOD_DEPLOYMENT_PLAN.md`
   - Status: ✅ LOCAL TEST PASSED
   - Architecture: 97K params with expansion heads
   - Key issue: Countdown loss scale (26.9)
   - Recommendation: Clip countdown to ±20 bars

---

## Experimental Results

### Summary Documents

1. **Final Results & Recommendations**
   - File: `FINAL_RESULTS_AND_RECOMMENDATIONS.md`
   - Content: Comprehensive conclusions and next steps
   - Date: 2025-10-27

2. **Executive Summary - Experiment Analysis**
   - File: `EXECUTIVE_SUMMARY_EXPERIMENT_ANALYSIS.md`
   - Content: High-level overview of all experiments
   - Date: 2025-10-26

3. **Baseline 100-Epoch Results**
   - File: `BASELINE_100EP_RESULTS.md`
   - Content: Detailed analysis of baseline training
   - Includes: Loss trajectory, probability calibration, gradient analysis
   - Location: `docs/analysis/`

### Experiment-Specific Analysis

1. **Augmentation Failure Analysis**
   - File: `AUGMENTATION_FAILURE_ANALYSIS.md`
   - Finding: Data augmentation (jitter) did NOT improve F1
   - Data: 174 samples, batch_size=32

2. **Expansion Heads Summary**
   - File: `EXPANSION_HEADS_SUMMARY.md`
   - Content: Multi-task learning with expansion-focused heads
   - Architecture: 97K params with pointer + span predictions

3. **Experiments Index**
   - File: `EXPERIMENTS_INDEX.md`
   - Content: Catalog of all experiments run
   - Format: Phase, date, model, F1, notes

4. **Experiment Results Summary**
   - File: `EXPERIMENT_RESULTS_SUMMARY_v1.md`
   - Content: Quick reference of best results per model

5. **Experiments Quickstart**
   - File: `EXPERIMENTS_QUICKSTART.md`
   - Content: How to run experiments locally and on RunPod

6. **Parallel Experiments README**
   - File: `PARALLEL_EXPERIMENTS_README.md`
   - Content: Running 3+ experiments simultaneously on RunPod
   - Recommendation: Use for hyperparameter sweeps

---

## Technical Analysis

### Loss Function & Training Improvements

1. **Loss Normalization Root Cause Analysis** (PRIMARY)
   - File: `LOSS_NORMALIZATION_ROOT_CAUSE_ANALYSIS.md`
   - Finding: Countdown loss = 10.08 (91% of total)
   - Solution: Normalization or task removal
   - Impact: Enables other tasks to learn

2. **Loss Normalization Evidence**
   - File: `LOSS_NORMALIZATION_EVIDENCE.md`
   - Content: Data supporting countdown loss dominance
   - Includes: Loss breakdown by component

3. **Loss Normalization Fixes**
   - File: `LOSS_NORMALIZATION_FIXES.md`
   - Content: Practical implementations (removed countdown)
   - Status: ✅ Working (stones_only model)

4. **Loss Normalization Technical Summary**
   - File: `LOSS_NORMALIZATION_TECHNICAL_SUMMARY.md`
   - Content: Concise explanation with code
   - Target audience: Developers

5. **Loss Normalization Analysis Index**
   - File: `LOSS_NORMALIZATION_ANALYSIS_INDEX.md`
   - Purpose: Navigate all loss-related documents

### Model Comparison & Validation

1. **Soft Span vs CRF Comparison** (KEY DECISION POINT)
   - File: `SOFT_SPAN_VS_CRF_COMPARISON.md`
   - Comparison: Soft span loss vs CRF head
   - Winner: Soft span loss (simpler, better F1)
   - Details: Implementation, metrics, trade-offs

2. **Final Comparison - Soft Span vs CRF**
   - File: `FINAL_COMPARISON_SOFT_SPAN_VS_CRF.md`
   - Content: Complete evaluation summary
   - Includes: Code, results, deployment recommendation

3. **Debugging Summary**
   - File: `DEBUGGING_SUMMARY.md`
   - Content: Issues found and resolved
   - Topics: Class imbalance, gradient flow, feature scaling

### Feature Engineering & Pipeline

1. **Feature Pipeline Validation**
   - File: `FEATURE_PIPELINE_VALIDATION.md`
   - Content: Validation of 12-feature relativity pipeline
   - Status: ✅ Working correctly
   - Features: 6 candle + 4 swing + 1 expansion + 1 pattern

2. **Pretraining Feature Validation**
   - File: `PRETRAINING_FEATURE_VALIDATION.md`
   - Content: Validation of features for MAE pre-training
   - Finding: 11-feature pipeline sufficient (position optional)

### Experimental Components

1. **Metrics Enhancement Summary**
   - File: `METRICS_ENHANCEMENT_SUMMARY.md`
   - Content: Improvements to Hit@K and span F1 metrics
   - Impact: Better evaluation

2. **Normalized Countdown Results**
   - File: `NORMALIZED_COUNTDOWN_RESULTS.md`
   - Content: Results after countdown normalization
   - F1 improvement: +25% vs baseline

3. **Overlapping Windows Summary**
   - File: `OVERLAPPING_WINDOWS_SUMMARY.md`
   - Content: Analysis of overlapping window generation
   - Impact: Better coverage of labeled data

4. **Threshold Optimization Analysis**
   - File: `THRESHOLD_OPTIMIZATION_ANALYSIS.md`
   - Content: Finding optimal prediction threshold
   - Result: 0.50 is optimal (validates default)

5. **Stones Reset Analysis**
   - File: `STONES_RESET_ANALYSIS.md`
   - Content: Impact of removing countdown task
   - Result: Enables other tasks to learn

---

## Planning & Checklists

1. **Experiments Checklist**
   - File: `EXPERIMENTS_CHECKLIST.md`
   - Content: Step-by-step for running experiments
   - Checklist format: Pre-deployment, deployment, monitoring

2. **Validation Plan**
   - File: `VALIDATION_PLAN.md`
   - Content: How to validate new models
   - Includes: Metrics, test procedures

3. **Training Analysis Summary**
   - File: `TRAINING_ANALYSIS_SUMMARY.txt`
   - Content: Summary of training runs and findings
   - Format: Structured text

---

## Archived Documents

1. **Agents.md** - Agent setup instructions (legacy)
2. **Problems.md** - Known issues list (mostly resolved)
3. **Quick Diagnostic.md** - Diagnostic commands (reference)
4. **Ready to Deploy.md** - Deployment readiness checklist

---

## Document Statistics

| Category | Count | Status |
|----------|-------|--------|
| Deployment Guides | 5 | ✅ Consolidated |
| Result Analysis | 8 | ✅ Organized |
| Technical Deep Dives | 10 | ✅ Organized |
| Planning Documents | 3 | ✅ Organized |
| Archived | 4 | ✅ Archived |
| **Total** | **32** | ✅ Organized |

---

## How to Use This Index

### For Deployment
→ Start with `docs/DEPLOYMENT_GUIDE.md` (consolidated)

### For Understanding Experiments
1. Read `EXECUTIVE_SUMMARY_EXPERIMENT_ANALYSIS.md` (overview)
2. Check `FINAL_RESULTS_AND_RECOMMENDATIONS.md` (conclusions)
3. Deep dive: `EXPERIMENTS_INDEX.md` (all experiments)

### For Technical Details
1. **Loss issues** → `LOSS_NORMALIZATION_ROOT_CAUSE_ANALYSIS.md`
2. **Model selection** → `FINAL_COMPARISON_SOFT_SPAN_VS_CRF.md`
3. **Features** → `FEATURE_PIPELINE_VALIDATION.md`

### For Quick Reference
- Best F1: `BASELINE_100EP_RESULTS.md` (0.220)
- Fastest training: `STONES_ONLY_DEPLOYMENT.md` (15 min)
- Most detailed logging: `BASELINE_100EP_DEPLOYMENT.md`

---

## Key Findings Summary

### ✅ What Worked
- Position encoding feature (+17.8% F1)
- Class weighting (pos_weight=13.1)
- Soft span loss > CRF head
- Stones-only (3 tasks) > baseline (4 tasks)
- BiLSTM architecture (simple, effective)

### ❌ What Didn't Work
- Data augmentation (jitter)
- Countdown task (91% of loss, no benefit)
- CRF head (complex, no improvement over soft span)

### 🎯 Current Best Model
- **Architecture:** Jade-Compact (96 hidden, 1 layer)
- **Features:** 13 (position encoding enabled)
- **Loss:** Uncertainty-weighted soft span
- **F1 Score:** 0.220 (production-ready)
- **Status:** ✅ DEPLOYED

---

## Related Top-Level Documentation

- `README.md` - Project overview
- `CLAUDE.md` - Project context for Claude Code
- `docs/DEPLOYMENT_GUIDE.md` - Consolidated deployment guide
- `docs/ARCHITECTURE.md` - Technical architecture
- `docs/GETTING_STARTED.md` - Quick start guide

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-10-27 | Initial consolidated index (32 docs organized) |

