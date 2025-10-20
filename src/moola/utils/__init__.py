"""Utility functions for Moola ML pipeline."""

from .splits import load_splits, make_splits
from .augmentation.mixup import mixup_data, augment_dataset, mixup_criterion_sklearn
from .validation.pseudo_sample_generation import (
    PseudoSampleGenerationPipeline,
    TemporalAugmentationGenerator,
    PatternBasedSynthesisGenerator,
    StatisticalSimulationGenerator,
    MarketConditionSimulationGenerator,
    SelfSupervisedPseudoLabelingGenerator
)
# Re-export training utilities for backward compatibility
from .training.early_stopping import EarlyStopping

__all__ = [
    "make_splits",
    "load_splits",
    "mixup_data",
    "augment_dataset",
    "mixup_criterion_sklearn",
    "PseudoSampleGenerationPipeline",
    "TemporalAugmentationGenerator",
    "PatternBasedSynthesisGenerator",
    "StatisticalSimulationGenerator",
    "MarketConditionSimulationGenerator",
    "SelfSupervisedPseudoLabelingGenerator",
    "EarlyStopping",
]
