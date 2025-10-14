# Data Collection Strategy for 200+ Samples Target

**Date**: October 14, 2025
**Task**: 16.3 - Plan data collection strategy
**Current State**: 115 labeled samples, 60.9% accuracy baseline
**Target**: 200+ samples minimum for breaking through 61% performance ceiling

---

## Executive Summary

Four-phase approach to scale from 115 to 500+ samples over 12 months:

| Phase | Timeline | Method | Expected Samples | Target Accuracy | Effort |
|-------|----------|--------|------------------|-----------------|--------|
| **Phase 1** | 0-2 weeks | Semi-Supervised Learning | +185 pseudo | 65-68% | Low |
| **Phase 2** | 1-3 months | Manual Labeling | +100-150 real | 64-66% | Medium |
| **Phase 3** | 3-6 months | Active Learning | +100 targeted | 68-70% | Low-Medium |
| **Phase 4** | 6-12 months | Automated Pipeline | +240/year | 70-72% | Low |

**Key Insight**: Leverage existing 118k unlabeled windows (Phase 1) for immediate gains while building sustainable data collection infrastructure (Phases 2-4).

---

## Current State Assessment

### Dataset Composition
- **Labeled Training**: 115 samples (65 consolidation, 50 retracement)
- **Reversal Holdout**: 19 samples (insufficient for 3-class problem)
- **Unlabeled Pool**: 118,000 windows from market data
- **Class Balance**: 1.3:1 ratio (moderately imbalanced)

### Performance Analysis
- **Baseline Accuracy**: 60.9% (Tasks 11-13 configuration)
- **Individual Models**: 43-53% range (high variance)
- **Ensemble Gain**: +7.9% over best individual
- **Fundamental Limitation**: Small dataset size creates hard ceiling at ~61%

### Root Cause Findings (from ML_TRAINING_ANOMALIES_REPORT.md)
- High fold variance (30-70% range) due to small validation sets (23 samples/fold)
- Individual models limited to ~50% average performance
- Stacking ensemble limited by poor base model performance
- **Conclusion**: 200+ samples required for individual models to exceed 50%, enabling better ensemble

---

## Phase 1: Semi-Supervised Learning (Immediate - 0-2 weeks)

### Objective
Leverage 118k unlabeled windows to expand effective dataset size without manual labeling

### Implementation (Task 17)

#### 1.1 TS-TCC Contrastive Pre-training
```
Approach: Pre-train CNN-Transformer encoder on unlabeled data
Method: InfoNCE loss with temporal augmentations
- Jitter: shift OHLC by ±2 timesteps
- Scale: multiply by 0.95-1.05x
- Time-warp: elastic deformation of time axis

Training: 118k windows, ~2-3 hours on RunPod RTX 4090
Output: Pre-trained encoder with learned representations
```

#### 1.2 Fine-tuning on Labeled Data
```
Approach: Replace contrastive head with 2-class classifier
Method: Fine-tune entire network on 115 labeled samples
Expected: +3-5% vs random initialization (learned features help)
```

#### 1.3 Adaptive Pseudo-Labeling
```
Approach: Generate predictions on 118k unlabeled pool
Method: Class-aware confidence thresholds
- Consolidation: τ = 0.92 (majority class, higher bar)
- Retracement: τ = 0.85 (minority class, lower bar)

Selection: ~185 high-confidence pseudo-labeled samples
Balance: Maintain ~1:1 class distribution in pseudo-labels
Combined Dataset: 115 real + 185 pseudo = 300 effective (2.6x expansion)
```

### Expected Outcomes
- **Accuracy Target**: 65-68% (+5-8% over 60.9% baseline)
- **Calibration**: Maintain ECE < 0.1
- **Timeline**: 2-3 days implementation + 1 day validation
- **Cost**: ~$10-20 RunPod GPU usage
- **Risk**: Low (no manual work, leverages existing data)

### Success Criteria
- Phase 1-2 minimum: 64% accuracy (+3.1% vs baseline)
- Pseudo-label precision: >80% when manually validated
- Model doesn't degrade on original 115 labeled samples

---

## Phase 2: Manual Labeling Campaign (Short-term - 1-3 months)

### Objective
Collect 100-150 high-quality labeled samples through systematic historical data review

### Data Sources

#### Priority 1: Historical Market Data
```
Coverage: Past 2-5 years of market history
Focus: Clear, unambiguous pattern instances
Quality Bar: Only include patterns with ≥4/5 clarity score
Target: 100-150 samples
```

#### Priority 2: Multiple Timeframes
```
Current: Single timeframe patterns
Expansion: 1H, 4H, 1D charts
Benefit: Diverse pattern scales and characteristics
Target: 50-75 additional samples
```

#### Priority 3: Cross-Asset Patterns (if applicable)
```
Current: Single asset focus
Expansion: Correlated assets with transferable patterns
Validation: Ensure feature distributions match
Target: 25-50 samples
```

### Labeling Protocol

#### Step 1: Pattern Identification
- Visual chart inspection for clear pattern structure
- Confirm OHLC data matches consolidation/retracement/reversal definition
- Document expansion_start and expansion_end indices
- Record pattern characteristics (volatility, duration, strength)

#### Step 2: Quality Control
- Two-person verification for ambiguous patterns
- Exclude borderline cases (maintain high precision over recall)
- Document uncertainty score (1-5 scale, only accept 4-5)
- Review conflicts through consensus discussion

#### Step 3: Metadata Tracking
```
Required Fields:
- Timestamp, asset, timeframe
- Pattern type (consolidation/retracement/reversal)
- Confidence score (1-5)
- Labeler ID and review status
- Feature statistics snapshot (for distribution monitoring)
```

#### Step 4: Balance Maintenance
- Track class distribution continuously
- Prioritize underrepresented classes each week
- Target: ~1:1:1 ratio for eventual 3-class restoration (150:150:150)

### Milestones
- **Month 1**: +50 samples (165 total) → Expected 62-63% accuracy
- **Month 2**: +50 samples (215 total) → Expected 64-65% accuracy
- **Month 3**: +50 samples (265 total) → Expected 66-68% accuracy

### Resource Requirements
- **Time**: 2-4 hours/week labeling + QC
- **Expertise**: Pattern recognition domain knowledge
- **Infrastructure**: Historical data access, labeling tool/spreadsheet
- **Cost**: $0 (manual effort only)

### Success Criteria
- Reach 200+ samples by end of Month 2
- Maintain class balance within ±15% of target ratio
- Inter-annotator agreement >85% on dual-labeled samples
- Feature distributions within historical ±15% variance

---

## Phase 3: Active Learning (Medium-term - 3-6 months)

### Objective
Maximize labeling efficiency by targeting most informative samples (2-3x better than random)

### Methodology

#### 3.1 Uncertainty Sampling
```python
def select_uncertain_samples(model, unlabeled_pool, n=25):
    """Select samples where model is most uncertain."""
    predictions = model.predict_proba(unlabeled_pool)

    # Calculate prediction entropy
    entropy = -np.sum(predictions * np.log(predictions + 1e-10), axis=1)

    # Select top-N most uncertain
    uncertain_idx = np.argsort(entropy)[-n:]
    return unlabeled_pool[uncertain_idx]
```

Benefits:
- Targets decision boundary samples
- Avoids redundant patterns already learned
- Maximum information gain per label

#### 3.2 Disagreement Sampling
```python
def select_disagreement_samples(ensemble_models, unlabeled_pool, n=25):
    """Select samples where ensemble models disagree most."""
    predictions = [m.predict(unlabeled_pool) for m in ensemble_models]

    # Calculate prediction variance across models
    disagreement = np.std(predictions, axis=0)

    # Select top-N highest disagreement
    disagreement_idx = np.argsort(disagreement)[-n:]
    return unlabeled_pool[disagreement_idx]
```

Benefits:
- Finds edge cases for ensemble
- Improves model diversity
- Reduces ensemble variance

#### 3.3 Iterative Retraining Loop
```
Cycle (every 2 weeks):
1. Run uncertainty + disagreement sampling → 25 candidate samples
2. Manual label 25 samples (1-2 hours)
3. Add to training set → Retrain ensemble
4. Evaluate performance improvement
5. Track ROI (% gain per labeling batch)
6. Continue if ROI > 0.5% per batch

Stop Condition: ROI diminishes below threshold OR reach 300-400 samples
```

### Expected Outcomes
- **Efficiency**: 2-3x better than random sampling
- **Target**: +100 high-value samples over 3 months
- **Accuracy**: 68-70% at 300 total samples
- **Timeline**: 1 week setup + 25 samples every 2 weeks

### Success Criteria
- Demonstrate >2x efficiency vs random baseline
- Fold variance reduced to <5% standard deviation
- Accuracy improves >0.5% per 25-sample batch

---

## Phase 4: Automated Collection Pipeline (Long-term - 6-12+ months)

### Objective
Build sustainable data collection infrastructure for continuous learning

### Infrastructure

#### 4.1 Live Pattern Detection
```python
class PatternMonitor:
    def __init__(self, confidence_threshold=0.75):
        self.threshold = confidence_threshold
        self.pending_labels = []
        self.model = load_production_model()

    def monitor_live_market(self, ohlc_stream):
        """Real-time pattern detection on live market data."""
        # Sliding window feature extraction
        windows = extract_windows(ohlc_stream, window_size=105)

        # Model predictions
        predictions = self.model.predict_proba(windows)
        confidences = np.max(predictions, axis=1)

        # Flag high-confidence predictions for review
        high_conf_idx = confidences > self.threshold

        for idx in np.where(high_conf_idx)[0]:
            self.pending_labels.append({
                'timestamp': windows[idx].timestamp,
                'prediction': predictions[idx],
                'confidence': confidences[idx],
                'verified': False,
                'outcome': None
            })

    def weekly_review(self, n=50):
        """Human reviews top-N predictions."""
        # Sort by confidence, return top-N
        sorted_pending = sorted(
            self.pending_labels,
            key=lambda x: x['confidence'],
            reverse=True
        )
        return sorted_pending[:n]
```

#### 4.2 Outcome Verification
```python
def verify_pattern_outcome(prediction_record, days_elapsed=7):
    """Verify if predicted pattern actually occurred."""
    # Wait 1-7 days for pattern to resolve
    if days_elapsed < 1:
        return None

    # Check actual market outcome
    actual_pattern = analyze_pattern_outcome(
        prediction_record['timestamp'],
        lookback=105,
        lookforward=days_elapsed
    )

    predicted_class = np.argmax(prediction_record['prediction'])

    # Auto-label if prediction matches reality
    if actual_pattern == predicted_class:
        prediction_record['verified'] = True
        prediction_record['outcome'] = actual_pattern
        add_to_training_set(prediction_record)
        return True

    return False
```

#### 4.3 Data Quality Monitoring
```python
def assess_label_quality(newly_labeled_samples):
    """Monitor distribution drift and feature statistics."""
    # Check class distribution
    new_dist = get_class_distribution(newly_labeled_samples)
    historical_dist = get_class_distribution(training_set)

    kl_div = kl_divergence(new_dist, historical_dist)
    if kl_div > 0.3:
        print(f"⚠️  Distribution shift detected (KL={kl_div:.3f})")

    # Check feature statistics
    new_features = extract_features(newly_labeled_samples)
    historical_features = extract_features(training_set)

    mean_shift = np.abs(new_features.mean() - historical_features.mean())
    std_shift = np.abs(new_features.std() - historical_features.std())

    if mean_shift > 0.2 * historical_features.mean():
        print(f"⚠️  Feature mean shift: {mean_shift:.3f}")

    if std_shift > 0.2 * historical_features.std():
        print(f"⚠️  Feature variance shift: {std_shift:.3f}")
```

### Workflow
```
Daily: Monitor live market → Flag high-confidence predictions
Weekly: Review top 50 flagged patterns (30 min manual effort)
Weekly: Verify outcomes for predictions from 7 days ago → Auto-add verified samples
Monthly: Retrain ensemble on accumulated samples → Deploy updated model
```

### Expected Outcomes
- **Target**: 20-30 verified samples/month (240-360/year)
- **Effort**: <1 hour/week manual review
- **Accuracy**: 70-72% at 500+ total samples
- **Timeline**: Month 6 deploy, Month 12 reach 500 samples

### Success Criteria
- Sustain ≥20 samples/month collection rate
- Verification precision >85% (auto-labels correct)
- Weekly review time <1 hour
- Distribution drift KL divergence <0.3

---

## Resource Allocation Summary

| Phase | Timeline | Manual Effort | Compute Cost | Expected Samples | Expected Accuracy |
|-------|----------|---------------|--------------|------------------|-------------------|
| **Phase 1: SSL** | 1-2 weeks | 2-3 days dev | $20 GPU | +185 pseudo | 65-68% |
| **Phase 2: Manual** | 1-3 months | 2-4 hrs/week | $0 | +100-150 real | 64-66% |
| **Phase 3: Active** | 3-6 months | 1-2 hrs/week | $50 compute | +100 targeted | 68-70% |
| **Phase 4: Auto** | 6-12 months | <1 hr/week | $100/month infra | +240/year | 70-72% |

**Total Investment (Year 1)**:
- Manual Effort: ~150-200 hours
- Compute Cost: ~$1,500
- Expected Samples: 500-700 total
- Expected Accuracy: 70-73%

---

## Success Metrics

### Data Quality
- **Class Balance**: ±10% of target 1:1 ratio (2-class) or 1:1:1 (3-class)
- **Pattern Clarity**: Minimum 4/5 confidence score for inclusion
- **Inter-annotator Agreement**: >85% on dual-labeled samples
- **Feature Variance**: Within historical ±15% range

### Model Performance
- **200 samples**: 64-66% accuracy (meaningful data threshold reached)
- **300 samples**: 67-69% accuracy (active learning benefits visible)
- **500 samples**: 70-73% accuracy (ensemble stability plateau)

### Process Efficiency
- **Phase 2 Manual**: 0.3-0.5% accuracy gain per 10 samples
- **Phase 3 Active**: 0.5-0.8% accuracy gain per 10 samples (2x better)
- **Phase 4 Auto**: Sustain 20+ samples/month with <1hr/week effort

---

## Risk Mitigation

### Risk 1: SSL Doesn't Improve Performance
- **Likelihood**: Medium (pseudo-labels may have errors)
- **Impact**: Low (Phase 2 continues regardless, just without +5% boost)
- **Mitigation**: Validate pseudo-label quality on small sample before full deployment, have direct Phase 2 fallback

### Risk 2: Manual Labeling Bottleneck
- **Likelihood**: High (time-intensive manual process)
- **Impact**: Medium (delays reaching 200+ target by months)
- **Mitigation**:
  - Build simple labeling UI for efficiency gains
  - Consider limited outsourcing with strict QC
  - Reduce minimum target to 150 samples if needed

### Risk 3: Pattern Distribution Shift
- **Likelihood**: Medium (market regimes change over time)
- **Impact**: High (new samples don't generalize to historical patterns)
- **Mitigation**:
  - Monitor feature statistics continuously with KL divergence
  - Implement domain adaptation if shift detected
  - Version datasets with timestamps for regime analysis

### Risk 4: Diminishing Returns Plateau
- **Likelihood**: High (performance plateaus exist for all models)
- **Impact**: Medium (may not reach 70%+ even with 500 samples)
- **Mitigation**:
  - Track ROI per labeling batch, stop if <0.5% gain
  - Pivot to architecture improvements if data plateau confirmed
  - Set realistic expectations: 65-68% may be achievable ceiling

---

## Decision Points

### After Phase 1 (Week 2)
**Continue to Phase 2 if:**
- ✅ SSL achieves 64-66% accuracy (+3-5% over baseline)
- ✅ Pseudo-labels show >80% precision when manually validated
- ✅ Model calibration maintained (ECE < 0.1)

**Alternative**: If SSL fails (<63%), skip directly to Phase 2 manual labeling

### After Phase 2 (Month 3)
**Invest in Phase 3 if:**
- ✅ Reached 200+ samples
- ✅ Accuracy improved to 64-66% range
- ✅ Fold variance reduced to <5% standard deviation

**Alternative**: If manual labeling too slow, reduce target to 150 and accelerate to Phase 4 with lower quality threshold

### After Phase 3 (Month 6)
**Deploy Phase 4 pipeline if:**
- ✅ Active learning demonstrated 2x efficiency vs random
- ✅ 300+ samples achieved
- ✅ Production infrastructure ready for monitoring

**Alternative**: Continue manual/active labeling if automated pipeline not ready

---

## Immediate Next Steps (This Week)

### Week 1 Actions
1. ✅ **Complete Task 16.3 Planning** (this document)
2. 🔄 **Begin Task 17 Implementation**
   - Set up TS-TCC contrastive learning framework
   - Pre-train encoder on 118k unlabeled windows
   - Implement adaptive pseudo-labeling
   - Target: Phase 1 complete within 2 weeks

3. 📋 **Prepare Phase 2 Infrastructure**
   - Document detailed pattern labeling guidelines
   - Create tracking spreadsheet (class balance, metadata, QC status)
   - Identify historical data sources and access methods
   - Design simple labeling UI (optional but helpful)

4. 🎯 **Set Success Criteria**
   - Baseline: 60.9% (current, Tasks 11-13)
   - Phase 1 target: 65-68% (SSL pseudo-labeling)
   - Phase 2 target: 64-66% at 200 real samples
   - Phase 3 target: 68-70% at 300 samples
   - Ultimate goal: 70-73% at 500+ samples

---

## Conclusion

This multi-phase strategy balances:

**Short-term gains**: SSL leverages existing 118k unlabeled data for immediate +5-8% boost without manual labeling effort

**Medium-term foundation**: Manual + active learning build to 200-300 samples for sustainable improvements and reduced model variance

**Long-term scalability**: Automated pipeline enables continuous data collection and model refinement with minimal ongoing effort

**Key Insight**: Phase 1 (SSL) provides critical bridge to Phase 2 (manual labeling), delivering immediate performance gains while buying time to execute slower data collection processes.

**Recommendation**: Execute Task 17 immediately while planning Phase 2 labeling campaign in parallel. This dual-track approach maximizes short-term results while building long-term data infrastructure.

---

**Document Status**: Complete
**Next Task**: Task 17 - Implement SSL with TS-TCC contrastive pre-training
**Dependencies**: None (Task 16 complete, ready to proceed)
