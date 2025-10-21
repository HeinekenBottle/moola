"""Utility functions for Moola ML pipeline."""

from .augmentation.mixup import augment_dataset, mixup_criterion_sklearn, mixup_data
from .splits import load_splits, make_splits

# Re-export training utilities for backward compatibility
from .training.early_stopping import EarlyStopping
from .validation.pseudo_sample_generation import (
    MarketConditionSimulationGenerator,
    PatternBasedSynthesisGenerator,
    PseudoSampleGenerationPipeline,
    SelfSupervisedPseudoLabelingGenerator,
    StatisticalSimulationGenerator,
    TemporalAugmentationGenerator,
)

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
