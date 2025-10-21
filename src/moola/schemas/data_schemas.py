"""Production-grade Pydantic schemas for financial time-series data validation.

Defines strict data contracts for:
- Raw OHLC market data
- Time-series windows (105 timesteps x 4 features)
- Labeled training samples
- Model predictions and metadata

These schemas enforce data integrity across the entire ML pipeline.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

# ============================================================================
# ENUMS AND CONSTANTS
# ============================================================================


class PatternLabel(str, Enum):
    """Valid pattern classification labels."""

    CONSOLIDATION = "consolidation"
    RETRACEMENT = "retracement"
    EXPANSION = "expansion"


class DataStage(str, Enum):
    """Data pipeline stages."""

    RAW = "raw"
    PROCESSED = "processed"
    VALIDATED = "validated"
    AUGMENTED = "augmented"
    TRAINING = "training"


class DataFormat(str, Enum):
    """Supported data formats."""

    PARQUET = "parquet"
    NPY = "npy"
    PICKLE = "pkl"
    CSV = "csv"


# ============================================================================
# TIME-SERIES DATA SCHEMAS
# ============================================================================


class OHLCBar(BaseModel):
    """Single OHLC candlestick bar with validation."""

    open: float = Field(..., ge=0.001, description="Opening price")
    high: float = Field(..., ge=0.001, description="Highest price")
    low: float = Field(..., ge=0.001, description="Lowest price")
    close: float = Field(..., ge=0.001, description="Closing price")
    timestamp: Optional[datetime] = Field(None, description="Bar timestamp")

    @model_validator(mode="after")
    def validate_ohlc_logic(self) -> "OHLCBar":
        """Validate OHLC logical constraints: high >= low, etc."""
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) cannot be less than Low ({self.low})")
        if self.high < self.close:
            raise ValueError(f"High ({self.high}) cannot be less than Close ({self.close})")
        if self.high < self.open:
            raise ValueError(f"High ({self.high}) cannot be less than Open ({self.open})")
        if self.low > self.close:
            raise ValueError(f"Low ({self.low}) cannot be greater than Close ({self.close})")
        if self.low > self.open:
            raise ValueError(f"Low ({self.low}) cannot be greater than Open ({self.open})")

        # Check for unrealistic price jumps (>200% in one bar)
        price_range = self.high - self.low
        avg_price = (self.high + self.low) / 2
        if price_range / avg_price > 2.0:
            raise ValueError(
                f"Unrealistic price range: {price_range / avg_price:.2%} "
                f"(high={self.high}, low={self.low})"
            )

        return self

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [open, high, low, close]."""
        return np.array([self.open, self.high, self.low, self.close], dtype=np.float32)


class TimeSeriesWindow(BaseModel):
    """105-timestep OHLC window with strict validation.

    Structure:
    - 30 past bars (context)
    - 45 prediction window (where patterns emerge)
    - 30 future bars (outcome)
    """

    window_id: str = Field(..., description="Unique window identifier")
    features: List[List[float]] = Field(
        ...,
        min_length=105,
        max_length=105,
        description="105 OHLC bars, each [open, high, low, close]",
    )
    symbol: Optional[str] = Field(None, description="Trading symbol")
    start_timestamp: Optional[datetime] = Field(None, description="Window start time")
    end_timestamp: Optional[datetime] = Field(None, description="Window end time")

    @field_validator("features")
    @classmethod
    def validate_features_shape(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate each timestep has exactly 4 features (OHLC)."""
        if len(v) != 105:
            raise ValueError(f"Expected 105 timesteps, got {len(v)}")

        for i, bar in enumerate(v):
            if len(bar) != 4:
                raise ValueError(f"Timestep {i} has {len(bar)} features, expected 4 (OHLC)")

            # Validate OHLC constraints
            open_p, high, low, close = bar
            if high < low:
                raise ValueError(f"Timestep {i}: high ({high}) < low ({low})")
            if high < max(open_p, close):
                raise ValueError(f"Timestep {i}: high ({high}) < max(open, close)")
            if low > min(open_p, close):
                raise ValueError(f"Timestep {i}: low ({low}) > min(open, close)")

        return v

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array [105, 4]."""
        return np.array(self.features, dtype=np.float32)

    @classmethod
    def from_numpy(cls, arr: np.ndarray, window_id: str) -> "TimeSeriesWindow":
        """Create from numpy array [105, 4]."""
        if arr.shape != (105, 4):
            raise ValueError(f"Expected shape (105, 4), got {arr.shape}")
        return cls(window_id=window_id, features=arr.tolist())


class LabeledWindow(TimeSeriesWindow):
    """Time-series window with pattern label and expansion indices."""

    label: PatternLabel = Field(..., description="Pattern classification label")
    expansion_start: int = Field(
        ..., ge=30, le=74, description="Expansion start index (within prediction window [30:75])"
    )
    expansion_end: int = Field(
        ..., ge=30, le=74, description="Expansion end index (within prediction window [30:75])"
    )
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Label confidence score")

    @model_validator(mode="after")
    def validate_expansion_indices(self) -> "LabeledWindow":
        """Ensure expansion_start <= expansion_end."""
        if self.expansion_start > self.expansion_end:
            raise ValueError(
                f"expansion_start ({self.expansion_start}) cannot be greater than "
                f"expansion_end ({self.expansion_end})"
            )
        return self


# ============================================================================
# DATASET SCHEMAS
# ============================================================================


class UnlabeledDataset(BaseModel):
    """Collection of unlabeled time-series windows for pre-training."""

    windows: List[TimeSeriesWindow] = Field(..., min_length=1)
    total_samples: int = Field(..., gt=0)
    data_stage: DataStage = Field(default=DataStage.RAW)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sample_count(self) -> "UnlabeledDataset":
        """Ensure total_samples matches windows length."""
        if len(self.windows) != self.total_samples:
            raise ValueError(
                f"total_samples ({self.total_samples}) doesn't match "
                f"actual windows count ({len(self.windows)})"
            )
        return self

    def to_numpy(self) -> np.ndarray:
        """Convert all windows to numpy array [N, 105, 4]."""
        return np.array([w.to_numpy() for w in self.windows], dtype=np.float32)


class LabeledDataset(BaseModel):
    """Collection of labeled time-series windows for training/validation."""

    windows: List[LabeledWindow] = Field(..., min_length=2)
    total_samples: int = Field(..., gt=0)
    label_distribution: Dict[str, int] = Field(...)
    data_stage: DataStage = Field(default=DataStage.PROCESSED)
    split_type: Optional[Literal["train", "val", "test"]] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset_quality(self) -> "LabeledDataset":
        """Validate dataset meets quality requirements."""
        # Check sample count
        if len(self.windows) != self.total_samples:
            raise ValueError(
                f"total_samples ({self.total_samples}) doesn't match "
                f"actual windows count ({len(self.windows)})"
            )

        # Validate label distribution
        actual_distribution = {}
        for window in self.windows:
            label = window.label.value
            actual_distribution[label] = actual_distribution.get(label, 0) + 1

        if actual_distribution != self.label_distribution:
            raise ValueError(
                f"Label distribution mismatch. "
                f"Declared: {self.label_distribution}, "
                f"Actual: {actual_distribution}"
            )

        # Check minimum samples per class
        min_samples = min(self.label_distribution.values())
        if min_samples < 2:
            raise ValueError(
                f"Insufficient samples for class with {min_samples} samples. "
                f"Minimum required: 2"
            )

        # Check for extreme class imbalance
        max_samples = max(self.label_distribution.values())
        imbalance_ratio = max_samples / min_samples
        if imbalance_ratio > 10.0:
            raise ValueError(
                f"Severe class imbalance detected: {imbalance_ratio:.1f}x. "
                f"Distribution: {self.label_distribution}"
            )

        return self

    def to_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to numpy arrays (X, y)."""
        X = np.array([w.to_numpy() for w in self.windows], dtype=np.float32)
        y = np.array([w.label.value for w in self.windows], dtype=object)
        return X, y


# ============================================================================
# DATA QUALITY METRICS
# ============================================================================


class DataQualityReport(BaseModel):
    """Data quality assessment report."""

    dataset_name: str = Field(...)
    total_samples: int = Field(...)
    features_shape: tuple[int, ...] = Field(...)

    # Completeness
    missing_values_count: int = Field(default=0)
    missing_percentage: float = Field(default=0.0, ge=0.0, le=100.0)

    # Statistical properties
    price_mean: float = Field(...)
    price_std: float = Field(...)
    price_min: float = Field(...)
    price_max: float = Field(...)

    # Anomalies
    outlier_count: int = Field(default=0)
    outlier_percentage: float = Field(default=0.0, ge=0.0, le=100.0)

    # Temporal consistency
    has_gaps: bool = Field(default=False)
    duplicate_timestamps: int = Field(default=0)

    # Quality score
    quality_score: float = Field(..., ge=0.0, le=100.0)
    passed_validation: bool = Field(...)

    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def compute_quality_score(self) -> "DataQualityReport":
        """Compute overall quality score based on metrics."""
        score = 100.0

        # Penalize missing values
        score -= min(self.missing_percentage * 2, 20)

        # Penalize outliers
        score -= min(self.outlier_percentage, 20)

        # Penalize temporal issues
        if self.has_gaps:
            score -= 10
        if self.duplicate_timestamps > 0:
            score -= 15

        # Penalize validation errors
        score -= min(len(self.validation_errors) * 5, 30)

        self.quality_score = max(score, 0.0)
        return self


# ============================================================================
# DATA LINEAGE
# ============================================================================


class DataLineage(BaseModel):
    """Track data transformation lineage."""

    dataset_id: str = Field(...)
    parent_datasets: List[str] = Field(default_factory=list)
    transformation_type: str = Field(...)
    transformation_params: Dict[str, Any] = Field(default_factory=dict)

    input_path: Optional[Path] = Field(None)
    output_path: Optional[Path] = Field(None)

    rows_in: int = Field(...)
    rows_out: int = Field(...)

    checksum_in: Optional[str] = Field(None)
    checksum_out: Optional[str] = Field(None)

    executed_by: str = Field(default="system")
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    execution_time_seconds: float = Field(...)

    metadata: Dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# DATA VERSIONING
# ============================================================================


class DataVersion(BaseModel):
    """Data version metadata for DVC integration."""

    version_id: str = Field(..., description="Semantic version (e.g., v1.0.0)")
    dataset_name: str = Field(...)
    file_path: Path = Field(...)

    # DVC metadata
    dvc_hash: Optional[str] = Field(None, description="DVC file hash (MD5)")
    dvc_size_bytes: Optional[int] = Field(None, ge=0)

    # Dataset properties
    num_samples: int = Field(..., gt=0)
    feature_shape: tuple[int, ...] = Field(...)
    label_distribution: Optional[Dict[str, int]] = Field(None)

    # Quality metrics
    quality_score: float = Field(..., ge=0.0, le=100.0)
    validation_passed: bool = Field(...)

    # Lineage
    parent_version: Optional[str] = Field(None)
    transformation_applied: Optional[str] = Field(None)

    # Metadata
    created_by: str = Field(default="automated_pipeline")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = Field(None)

    @model_validator(mode="after")
    def validate_versioning(self) -> "DataVersion":
        """Validate version semantics."""
        # Ensure version follows semver
        parts = self.version_id.lstrip("v").split(".")
        if len(parts) != 3:
            raise ValueError(f"Version must follow semver format (v1.0.0), got {self.version_id}")

        try:
            major, minor, patch = map(int, parts)
        except ValueError:
            raise ValueError(f"Invalid version format: {self.version_id}")

        return self


# ============================================================================
# EXPORT ALL SCHEMAS
# ============================================================================

__all__ = [
    # Enums
    "PatternLabel",
    "DataStage",
    "DataFormat",
    # Time-series schemas
    "OHLCBar",
    "TimeSeriesWindow",
    "LabeledWindow",
    # Dataset schemas
    "UnlabeledDataset",
    "LabeledDataset",
    # Quality & lineage
    "DataQualityReport",
    "DataLineage",
    "DataVersion",
]
