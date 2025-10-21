"""Data loading and validation specifications.

Defines expected data formats, validation ranges, and integrity checks
to prevent silent data corruption and model-data mismatches.

Philosophy:
- Centralize all data format expectations
- Enable early validation before expensive training
- Prevent data leakage between train/val/test splits
- Catch malformed data immediately with clear error messages
"""

import hashlib
from pathlib import Path

# ============================================================================
# EXPECTED DATA FORMAT
# ============================================================================

EXPECTED_FEATURES_PER_WINDOW = 4  # OHLC: Open, High, Low, Close
EXPECTED_WINDOW_LENGTH = 105  # 30 past bars + 45 prediction + 30 future bars

# Window breakdown
PAST_WINDOW_START = 0
PAST_WINDOW_END = 30
PREDICTION_WINDOW_START = 30
PREDICTION_WINDOW_END = 75
FUTURE_WINDOW_START = 75
FUTURE_WINDOW_END = 105


# ============================================================================
# DATA VALIDATION RANGES
# ============================================================================

EXPANSION_START_MIN = 30
"""Minimum valid expansion start index (must be in prediction window)."""

EXPANSION_START_MAX = 74
"""Maximum valid expansion start index."""

EXPANSION_END_MIN = 30
"""Minimum valid expansion end index."""

EXPANSION_END_MAX = 74
"""Maximum valid expansion end index."""

# OHLC price validation (for detecting malformed data)
PRICE_MIN = 0.001  # Prices shouldn't be negative or zero
PRICE_MAX = 10000.0  # Upper bound for typical price ranges

PRICE_CHANGE_THRESHOLD = 2.0
"""Max % change in price per bar (2x is reasonable, >10x is suspect)."""


# ============================================================================
# LABEL VALIDATION
# ============================================================================

VALID_LABELS = ["consolidation", "retracement", "expansion"]
"""Allowed class labels. Updated dynamically during training."""

MIN_SAMPLES_PER_CLASS = 2
"""Minimum samples per class. SMOTE fails with <2."""

MIN_SAMPLES_TOTAL = 20
"""Minimum total samples for meaningful training."""


# ============================================================================
# DATA CHECKSUMS (FOR INTEGRITY)
# ============================================================================

# Dictionary to store known dataset checksums for corruption detection
# Format: {dataset_name: "sha256_hash_of_features"}
KNOWN_DATASET_CHECKSUMS: dict[str, str] = {
    # Will be populated during initial data ingestion
    # Example: "train_v1.parquet": "abc123def456..."
}


def compute_checksum(data, algorithm: str = "sha256") -> str:
    """Compute checksum of data for integrity verification.

    Args:
        data: Input data (numpy array, bytes, or file path)
        algorithm: Hash algorithm ('sha256', 'md5')

    Returns:
        Hex string of checksum
    """
    import numpy as np

    hasher = hashlib.new(algorithm)

    if isinstance(data, np.ndarray):
        hasher.update(data.tobytes())
    elif isinstance(data, (str, Path)):
        with open(data, "rb") as f:
            hasher.update(f.read())
    elif isinstance(data, bytes):
        hasher.update(data)
    else:
        raise TypeError(f"Cannot compute checksum for type {type(data)}")

    return hasher.hexdigest()


# ============================================================================
# DATA SHAPE SPECIFICATIONS
# ============================================================================


class DataShapeSpec:
    """Specification for expected data shapes."""

    # Time series (deep learning models)
    TIMESERIES_3D_OHLC = {
        "description": "[N_samples, 105, 4] OHLC time series",
        "dimensions": 3,
        "expected_shape": (None, 105, 4),  # None = variable N
        "dtype": "float32",
    }

    # Tabular (traditional ML models)
    TABULAR_2D = {
        "description": "[N_samples, N_features] tabular data",
        "dimensions": 2,
        "expected_shape": (None, None),  # Variable N and features
        "dtype": "float32",
    }

    # Labels
    LABELS_1D = {
        "description": "[N_samples] class labels",
        "dimensions": 1,
        "expected_shape": (None,),
        "dtype": "object",
    }

    # Expansion indices
    INDICES_1D = {
        "description": "[N_samples] expansion start/end indices",
        "dimensions": 1,
        "expected_shape": (None,),
        "dtype": "int64",
    }


# ============================================================================
# DATA PIPELINE STAGES
# ============================================================================


class PipelineStages:
    """Specifications for each data pipeline stage."""

    RAW = {
        "name": "Raw Data",
        "format": "parquet",
        "location": "data/raw/",
        "expected_columns": ["symbol", "timestamp", "open", "high", "low", "close"],
    }

    PROCESSED = {
        "name": "Processed Data",
        "format": "parquet",
        "location": "data/processed/train.parquet",
        "expected_columns": ["window_id", "label", "features", "expansion_start", "expansion_end"],
    }

    OOF = {
        "name": "Out-of-Fold Predictions",
        "format": "npy",
        "location": "artifacts/oof/{model_name}/v1/seed_{seed}.npy",
        "expected_shape": "[N_samples, N_classes]",
    }

    PREDICTIONS = {
        "name": "Final Predictions",
        "format": "csv",
        "location": "artifacts/predictions/test.csv",
        "expected_columns": ["window_id", "prediction", "prob_class_0", "prob_class_1", ...],
    }


# ============================================================================
# QUALITY METRICS
# ============================================================================


class QualityMetrics:
    """Data quality thresholds and checks."""

    # Missing data
    MAX_MISSING_PERCENT = 1.0  # Allow up to 1% missing values
    MAX_MISSING_PER_COLUMN = 0.05  # Column: max 5% missing

    # Outliers
    OUTLIER_ZSCORE_THRESHOLD = 5.0  # Flag values >5 std from mean
    OUTLIER_IQR_MULTIPLIER = 3.0  # Flag values >3*IQR from quartiles

    # Class balance
    MIN_CLASS_RATIO = 0.1  # No class should be <10% of total
    MAX_CLASS_RATIO = 0.9  # No class should be >90% of total

    # Label distribution
    EXPECTED_CLASS_DISTRIBUTION = {
        "consolidation": (0.35, 0.45),  # Expected range: 35-45%
        "retracement": (0.30, 0.40),  # Expected range: 30-40%
        "expansion": (0.15, 0.25),  # Expected range: 15-25%
    }

    # Feature statistics
    EXPECTED_PRICE_CORRELATION = 0.7  # Close price should be ~0.7 correlated with open


# ============================================================================
# DATA LOGGING & DIAGNOSTICS
# ============================================================================


class DiagnosticThresholds:
    """Thresholds for diagnostic messages."""

    # Logging levels
    WARN_LOW_SAMPLES = 50  # Warn if <50 total samples
    WARN_CLASS_IMBALANCE = 3.0  # Warn if max/min class ratio > 3x
    WARN_HIGH_MISSING = 0.5  # Warn if >0.5% missing

    # Performance
    EXPECT_MIN_ACCURACY = 0.33  # Random baseline for 3 classes
    WARN_IF_BELOW_BASELINE = 0.35  # Warn if accuracy near random
    ALERT_CLASS_COLLAPSE = 0.10  # Alert if any class accuracy <10%


# ============================================================================
# EXPORT ALL SPECIFICATIONS
# ============================================================================

__all__ = [
    # Format specs
    "EXPECTED_FEATURES_PER_WINDOW",
    "EXPECTED_WINDOW_LENGTH",
    "PAST_WINDOW_START",
    "PAST_WINDOW_END",
    "PREDICTION_WINDOW_START",
    "PREDICTION_WINDOW_END",
    "FUTURE_WINDOW_START",
    "FUTURE_WINDOW_END",
    # Validation ranges
    "EXPANSION_START_MIN",
    "EXPANSION_START_MAX",
    "EXPANSION_END_MIN",
    "EXPANSION_END_MAX",
    "PRICE_MIN",
    "PRICE_MAX",
    "PRICE_CHANGE_THRESHOLD",
    # Labels
    "VALID_LABELS",
    "MIN_SAMPLES_PER_CLASS",
    "MIN_SAMPLES_TOTAL",
    # Checksums
    "KNOWN_DATASET_CHECKSUMS",
    "compute_checksum",
    # Shape specs
    "DataShapeSpec",
    "PipelineStages",
    # Quality
    "QualityMetrics",
    "DiagnosticThresholds",
]
