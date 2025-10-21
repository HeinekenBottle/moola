"""Enhanced schemas for 11-dimensional feature validation.

Extends existing OHLC schemas to support 11-dimensional relative features
while maintaining backward compatibility with 4D OHLC data.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator

from .schemas import DataFormat, DataStage, PatternLabel


class FeatureDimension(str, Enum):
    """Supported feature dimensions."""

    OHLC_4D = "ohlc_4d"
    RELATIVE_11D = "relative_11d"
    DUAL_INPUT = "dual_input"


class EnhancedOHLCBar(BaseModel):
    """Enhanced OHLC bar with optional 11D relative features."""

    # Core OHLC data (4D)
    open: float = Field(..., ge=0.001, description="Opening price")
    high: float = Field(..., ge=0.001, description="Highest price")
    low: float = Field(..., ge=0.001, description="Lowest price")
    close: float = Field(..., ge=0.001, description="Closing price")
    timestamp: Optional[datetime] = Field(None, description="Bar timestamp")

    # Optional 11D relative features
    relative_features: Optional[List[float]] = Field(
        None,
        min_length=11,
        max_length=11,
        description="11 relative features: [4 log_returns, 3 candle_ratios, 4 z_scores]",
    )

    @model_validator(mode="after")
    def validate_ohlc_logic(self) -> "EnhancedOHLCBar":
        """Validate OHLC logical constraints."""
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

        # Check for unrealistic price jumps
        price_range = self.high - self.low
        avg_price = (self.high + self.low) / 2
        if price_range / avg_price > 2.0:
            raise ValueError(
                f"Unrealistic price range: {price_range / avg_price:.2%} "
                f"(high={self.high}, low={self.low})"
            )
        return self

    def to_ohlc_array(self) -> np.ndarray:
        """Convert to OHLC numpy array [open, high, low, close]."""
        return np.array([self.open, self.high, self.low, self.close], dtype=np.float32)

    def to_relative_array(self) -> Optional[np.ndarray]:
        """Convert to relative features numpy array [11]."""
        if self.relative_features is None:
            return None
        return np.array(self.relative_features, dtype=np.float32)

    def to_enhanced_array(self) -> np.ndarray:
        """Convert to enhanced array [15] = [4 OHLC + 11 relative]."""
        ohlc = self.to_ohlc_array()
        relative = self.to_relative_array()
        if relative is not None:
            return np.concatenate([ohlc, relative])
        else:
            # Pad with zeros if relative features not available
            return np.concatenate([ohlc, np.zeros(11, dtype=np.float32)])


class EnhancedTimeSeriesWindow(BaseModel):
    """Enhanced 105-timestep window with 4D OHLC and optional 11D relative features."""

    window_id: str = Field(..., description="Unique window identifier")
    feature_dimension: FeatureDimension = Field(..., description="Feature dimension type")

    # Core OHLC data (always present)
    ohlc_features: List[List[float]] = Field(
        ...,
        min_length=105,
        max_length=105,
        description="105 OHLC bars, each [open, high, low, close]",
    )

    # Optional 11D relative features
    relative_features: Optional[List[List[float]]] = Field(
        None,
        min_length=105,
        max_length=105,
        description="105 timesteps of 11 relative features each",
    )

    symbol: Optional[str] = Field(None, description="Trading symbol")
    start_timestamp: Optional[datetime] = Field(None, description="Window start time")
    end_timestamp: Optional[datetime] = Field(None, description="Window end time")

    @field_validator("ohlc_features")
    @classmethod
    def validate_ohlc_shape(cls, v: List[List[float]]) -> List[List[float]]:
        """Validate each timestep has exactly 4 OHLC features."""
        if len(v) != 105:
            raise ValueError(f"Expected 105 timesteps, got {len(v)}")

        for i, bar in enumerate(v):
            if len(bar) != 4:
                raise ValueError(f"Timestep {i} has {len(bar)} OHLC features, expected 4 (OHLC)")

            # Validate OHLC constraints
            open_p, high, low, close = bar
            if high < low:
                raise ValueError(f"Timestep {i}: high ({high}) < low ({low})")
            if high < max(open_p, close):
                raise ValueError(f"Timestep {i}: high ({high}) < max(open, close)")
            if low > min(open_p, close):
                raise ValueError(f"Timestep {i}: low ({low}) > min(open, close)")

        return v

    @field_validator("relative_features")
    @classmethod
    def validate_relative_shape(cls, v: Optional[List[List[float]]]) -> Optional[List[List[float]]]:
        """Validate relative features have exactly 11 dimensions."""
        if v is None:
            return None

        if len(v) != 105:
            raise ValueError(f"Expected 105 timesteps, got {len(v)}")

        for i, features in enumerate(v):
            if len(features) != 11:
                raise ValueError(f"Timestep {i} has {len(features)} relative features, expected 11")

        return v

    @model_validator(mode="after")
    def validate_feature_consistency(self) -> "EnhancedTimeSeriesWindow":
        """Ensure feature dimensions are consistent."""
        if (
            self.feature_dimension == FeatureDimension.RELATIVE_11D
            and self.relative_features is None
        ):
            raise ValueError("11D features specified but relative_features data is missing")

        if (
            self.feature_dimension == FeatureDimension.OHLC_4D
            and self.relative_features is not None
        ):
            # Allow 11D data even when marked as 4D (backward compatibility)
            self.feature_dimension = FeatureDimension.DUAL_INPUT

        return self

    def to_ohlc_numpy(self) -> np.ndarray:
        """Convert to OHLC numpy array [105, 4]."""
        return np.array(self.ohlc_features, dtype=np.float32)

    def to_relative_numpy(self) -> Optional[np.ndarray]:
        """Convert to relative features numpy array [105, 11]."""
        if self.relative_features is None:
            return None
        return np.array(self.relative_features, dtype=np.float32)

    def to_enhanced_numpy(self) -> np.ndarray:
        """Convert to enhanced numpy array [105, 15] = [4 OHLC + 11 relative]."""
        ohlc = self.to_ohlc_numpy()
        relative = self.to_relative_numpy()

        if relative is not None:
            return np.concatenate([ohlc, relative], axis=1)
        else:
            # Pad with zeros if relative features not available
            padding = np.zeros((105, 11), dtype=np.float32)
            return np.concatenate([ohlc, padding], axis=1)

    @classmethod
    def from_ohlc_numpy(
        cls, arr: np.ndarray, window_id: str, relative_arr: Optional[np.ndarray] = None
    ) -> "EnhancedTimeSeriesWindow":
        """Create from numpy arrays."""
        if arr.shape != (105, 4):
            raise ValueError(f"Expected OHLC shape (105, 4), got {arr.shape}")

        # Determine feature dimension
        if relative_arr is not None:
            if relative_arr.shape != (105, 11):
                raise ValueError(f"Expected relative shape (105, 11), got {relative_arr.shape}")
            feature_dim = FeatureDimension.DUAL_INPUT
            relative_features = relative_arr.tolist()
        else:
            feature_dim = FeatureDimension.OHLC_4D
            relative_features = None

        return cls(
            window_id=window_id,
            feature_dimension=feature_dim,
            ohlc_features=arr.tolist(),
            relative_features=relative_features,
            symbol=None,
            start_timestamp=None,
            end_timestamp=None,
        )


class EnhancedLabeledWindow(EnhancedTimeSeriesWindow):
    """Enhanced time-series window with pattern label and expansion indices."""

    label: PatternLabel = Field(..., description="Pattern classification label")
    expansion_start: int = Field(
        ..., ge=30, le=74, description="Expansion start index (within prediction window [30:75])"
    )
    expansion_end: int = Field(
        ..., ge=30, le=74, description="Expansion end index (within prediction window [30:75])"
    )
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Label confidence score")

    @model_validator(mode="after")
    def validate_expansion_indices(self) -> "EnhancedLabeledWindow":
        """Ensure expansion_start <= expansion_end."""
        if self.expansion_start > self.expansion_end:
            raise ValueError(
                f"expansion_start ({self.expansion_start}) cannot be greater than "
                f"expansion_end ({self.expansion_end})"
            )
        return self


class EnhancedUnlabeledDataset(BaseModel):
    """Enhanced dataset with support for 11D features."""

    windows: List[EnhancedTimeSeriesWindow] = Field(..., min_length=1)
    total_samples: int = Field(..., gt=0)
    feature_dimension: FeatureDimension = Field(...)
    data_stage: DataStage = Field(default=DataStage.RAW)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_sample_count(self) -> "EnhancedUnlabeledDataset":
        """Ensure total_samples matches windows length."""
        if len(self.windows) != self.total_samples:
            raise ValueError(
                f"total_samples ({self.total_samples}) doesn't match "
                f"actual windows count ({len(self.windows)})"
            )
        return self

    def to_ohlc_numpy(self) -> np.ndarray:
        """Convert all windows to OHLC numpy array [N, 105, 4]."""
        return np.array([w.to_ohlc_numpy() for w in self.windows], dtype=np.float32)

    def to_relative_numpy(self) -> Optional[np.ndarray]:
        """Convert all windows to relative features numpy array [N, 105, 11]."""
        relative_arrays = []
        for w in self.windows:
            rel = w.to_relative_numpy()
            if rel is not None:
                relative_arrays.append(rel)
            else:
                # Generate relative features on-the-fly if not present
                from moola.features.relative_transform import RelativeFeatureTransform

                transformer = RelativeFeatureTransform()
                ohlc = w.to_ohlc_numpy().reshape(1, 105, 4)
                rel = transformer.transform(ohlc).squeeze(0)
                relative_arrays.append(rel)

        return np.array(relative_arrays, dtype=np.float32) if relative_arrays else None

    def to_enhanced_numpy(self) -> np.ndarray:
        """Convert all windows to enhanced numpy array [N, 105, 15]."""
        return np.array([w.to_enhanced_numpy() for w in self.windows], dtype=np.float32)


class EnhancedLabeledDataset(BaseModel):
    """Enhanced labeled dataset with support for 11D features."""

    windows: List[EnhancedLabeledWindow] = Field(..., min_length=2)
    total_samples: int = Field(..., gt=0)
    label_distribution: Dict[str, int] = Field(...)
    feature_dimension: FeatureDimension = Field(...)
    data_stage: DataStage = Field(default=DataStage.PROCESSED)
    split_type: Optional[Literal["train", "val", "test"]] = Field(None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_dataset_quality(self) -> "EnhancedLabeledDataset":
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

    def to_ohlc_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to OHLC numpy arrays (X, y)."""
        X = np.array([w.to_ohlc_numpy() for w in self.windows], dtype=np.float32)
        y = np.array([w.label.value for w in self.windows], dtype=object)
        return X, y

    def to_relative_numpy(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Convert to relative features numpy arrays (X, y)."""
        X_relative = []
        for w in self.windows:
            rel = w.to_relative_numpy()
            if rel is not None:
                X_relative.append(rel)
            else:
                # Generate relative features on-the-fly
                from moola.features.relative_transform import RelativeFeatureTransform

                transformer = RelativeFeatureTransform()
                ohlc = w.to_ohlc_numpy().reshape(1, 105, 4)
                rel = transformer.transform(ohlc).squeeze(0)
                X_relative.append(rel)

        if X_relative:
            X = np.array(X_relative, dtype=np.float32)
            y = np.array([w.label.value for w in self.windows], dtype=object)
            return X, y
        return None

    def to_enhanced_numpy(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to enhanced numpy arrays (X, y) with 15 features."""
        X = np.array([w.to_enhanced_numpy() for w in self.windows], dtype=np.float32)
        y = np.array([w.label.value for w in self.windows], dtype=object)
        return X, y


# Export all enhanced schemas
__all__ = [
    "FeatureDimension",
    "EnhancedOHLCBar",
    "EnhancedTimeSeriesWindow",
    "EnhancedLabeledWindow",
    "EnhancedUnlabeledDataset",
    "EnhancedLabeledDataset",
]
