"""Experiment management utilities for LSTM pre-training optimization.

This package provides infrastructure for running parallel MLOps experiments:
- Data versioning and caching
- Augmentation with configurable parameters
- Quality validation gates
- Performance benchmarking
"""

from moola.experiments.data_manager import (
    AugmentationConfig,
    ExperimentDataManager,
    ExperimentMetadata,
    OHLCValidator,
    TemporalAugmentor,
)

__all__ = [
    "ExperimentDataManager",
    "AugmentationConfig",
    "ExperimentMetadata",
    "OHLCValidator",
    "TemporalAugmentor",
]
