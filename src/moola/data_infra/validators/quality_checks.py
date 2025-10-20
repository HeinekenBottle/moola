"""Data quality validation checks for financial time-series data.

Implements comprehensive quality gates using Great Expectations patterns
and custom financial data validators.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

from ..schemas import DataQualityReport


# ============================================================================
# QUALITY CHECK THRESHOLDS
# ============================================================================

@dataclass
class QualityThresholds:
    """Configurable quality check thresholds."""

    # Completeness
    max_missing_percent: float = 1.0  # Max 1% missing values
    max_missing_per_column: float = 5.0  # Max 5% missing per column

    # Outliers
    outlier_zscore: float = 5.0  # Flag values >5σ from mean
    outlier_iqr_multiplier: float = 3.0  # Flag values >3*IQR from quartiles

    # Price validation (financial data specific)
    min_price: float = 0.001  # Prices must be positive
    max_price: float = 1_000_000.0  # Upper bound for sanity
    max_price_jump_percent: float = 200.0  # Max 200% jump between bars

    # Temporal consistency
    allow_duplicate_timestamps: bool = False
    max_time_gap_minutes: Optional[int] = None  # None = no check

    # OHLC specific
    check_ohlc_logic: bool = True  # Validate high>=low, etc.


# ============================================================================
# CORE VALIDATORS
# ============================================================================

class TimeSeriesQualityValidator:
    """Validate time-series data quality with financial domain knowledge."""

    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        self.thresholds = thresholds or QualityThresholds()

    def validate_dataset(
        self,
        df: pd.DataFrame,
        dataset_name: str = "unknown"
    ) -> DataQualityReport:
        """Run full quality validation suite on dataset.

        Args:
            df: DataFrame with time-series data
            dataset_name: Name for reporting

        Returns:
            DataQualityReport with validation results
        """
        logger.info(f"Validating dataset: {dataset_name}")
        logger.info(f"Shape: {df.shape}")

        errors = []
        warnings = []

        # 1. Completeness checks
        missing_count = int(df.isnull().sum().sum())
        total_values = df.shape[0] * df.shape[1]
        missing_pct = (missing_count / total_values * 100) if total_values > 0 else 0

        if missing_pct > self.thresholds.max_missing_percent:
            errors.append(
                f"Missing values: {missing_pct:.2f}% exceeds threshold "
                f"{self.thresholds.max_missing_percent}%"
            )

        # 2. Statistical properties
        if 'features' in df.columns:
            prices = self._extract_prices(df)
            price_stats = self._compute_price_stats(prices)
        else:
            price_stats = {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0
            }

        # 3. Outlier detection
        outlier_count, outlier_pct = self._detect_outliers(df, price_stats)

        if outlier_pct > 5.0:
            warnings.append(
                f"High outlier rate: {outlier_pct:.2f}% of samples"
            )

        # 4. Temporal consistency
        has_gaps, duplicate_count = self._check_temporal_consistency(df)

        if has_gaps:
            warnings.append("Temporal gaps detected in time-series")

        if duplicate_count > 0 and not self.thresholds.allow_duplicate_timestamps:
            errors.append(f"Found {duplicate_count} duplicate timestamps")

        # 5. OHLC validation (if applicable)
        if self.thresholds.check_ohlc_logic and 'features' in df.columns:
            ohlc_errors = self._validate_ohlc_logic(df)
            errors.extend(ohlc_errors)

        # 6. Price jump validation
        if 'features' in df.columns:
            jump_errors = self._validate_price_jumps(df)
            errors.extend(jump_errors)

        # Determine pass/fail
        passed = len(errors) == 0

        return DataQualityReport(
            dataset_name=dataset_name,
            total_samples=len(df),
            features_shape=tuple(df.shape),
            missing_values_count=missing_count,
            missing_percentage=missing_pct,
            price_mean=price_stats['mean'],
            price_std=price_stats['std'],
            price_min=price_stats['min'],
            price_max=price_stats['max'],
            outlier_count=outlier_count,
            outlier_percentage=outlier_pct,
            has_gaps=has_gaps,
            duplicate_timestamps=duplicate_count,
            quality_score=0.0,  # Computed by model_validator
            passed_validation=passed,
            validation_errors=errors,
            warnings=warnings,
        )

    def _extract_prices(self, df: pd.DataFrame) -> np.ndarray:
        """Extract price data from features column."""
        try:
            # Handle different feature formats
            if df['features'].dtype == object:
                # Array of arrays format
                prices = []
                for features in df['features']:
                    if isinstance(features, (list, np.ndarray)):
                        arr = np.array(features)
                        if arr.ndim == 2 and arr.shape[1] == 4:
                            # OHLC format: extract close prices
                            prices.append(arr[:, 3])  # Close is column 3
                        elif arr.ndim == 1:
                            prices.append(arr)
                return np.concatenate(prices) if prices else np.array([])
            else:
                return df['features'].values
        except Exception as e:
            logger.warning(f"Could not extract prices: {e}")
            return np.array([])

    def _compute_price_stats(self, prices: np.ndarray) -> Dict[str, float]:
        """Compute statistical properties of prices."""
        if len(prices) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}

        return {
            'mean': float(np.mean(prices)),
            'std': float(np.std(prices)),
            'min': float(np.min(prices)),
            'max': float(np.max(prices))
        }

    def _detect_outliers(
        self,
        df: pd.DataFrame,
        price_stats: Dict[str, float]
    ) -> Tuple[int, float]:
        """Detect outliers using z-score and IQR methods."""
        if 'features' not in df.columns:
            return 0, 0.0

        prices = self._extract_prices(df)
        if len(prices) == 0:
            return 0, 0.0

        # Z-score method
        z_scores = np.abs(stats.zscore(prices, nan_policy='omit'))
        outliers_zscore = np.sum(z_scores > self.thresholds.outlier_zscore)

        # IQR method
        q1, q3 = np.percentile(prices, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - self.thresholds.outlier_iqr_multiplier * iqr
        upper_bound = q3 + self.thresholds.outlier_iqr_multiplier * iqr
        outliers_iqr = np.sum((prices < lower_bound) | (prices > upper_bound))

        # Use more conservative estimate
        outlier_count = max(outliers_zscore, outliers_iqr)
        outlier_pct = (outlier_count / len(prices) * 100) if len(prices) > 0 else 0

        return int(outlier_count), float(outlier_pct)

    def _check_temporal_consistency(self, df: pd.DataFrame) -> Tuple[bool, int]:
        """Check for temporal gaps and duplicate timestamps."""
        has_gaps = False
        duplicate_count = 0

        # Check if timestamp column exists
        timestamp_col = None
        for col in ['timestamp', 'start_timestamp', 'datetime']:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None:
            return False, 0

        timestamps = pd.to_datetime(df[timestamp_col], errors='coerce')
        timestamps = timestamps.dropna().sort_values()

        # Check duplicates
        duplicate_count = int(timestamps.duplicated().sum())

        # Check gaps (if threshold specified)
        if self.thresholds.max_time_gap_minutes is not None and len(timestamps) > 1:
            time_diffs = timestamps.diff()[1:]  # Skip first NaT
            max_gap_minutes = time_diffs.max().total_seconds() / 60
            if max_gap_minutes > self.thresholds.max_time_gap_minutes:
                has_gaps = True

        return has_gaps, duplicate_count

    def _validate_ohlc_logic(self, df: pd.DataFrame) -> List[str]:
        """Validate OHLC logical constraints: high >= low, etc."""
        errors = []

        for idx, features in enumerate(df['features']):
            if not isinstance(features, (list, np.ndarray)):
                continue

            arr = np.array(features)
            if arr.ndim != 2 or arr.shape[1] != 4:
                continue

            # Check each timestep
            for t, bar in enumerate(arr):
                open_p, high, low, close = bar

                if high < low:
                    errors.append(
                        f"Sample {idx}, timestep {t}: high ({high}) < low ({low})"
                    )

                if high < max(open_p, close):
                    errors.append(
                        f"Sample {idx}, timestep {t}: "
                        f"high ({high}) < max(open={open_p}, close={close})"
                    )

                if low > min(open_p, close):
                    errors.append(
                        f"Sample {idx}, timestep {t}: "
                        f"low ({low}) > min(open={open_p}, close={close})"
                    )

        return errors[:10]  # Limit to first 10 errors for readability

    def _validate_price_jumps(self, df: pd.DataFrame) -> List[str]:
        """Validate that prices don't have unrealistic jumps."""
        errors = []
        max_jump_pct = self.thresholds.max_price_jump_percent / 100

        for idx, features in enumerate(df['features']):
            if not isinstance(features, (list, np.ndarray)):
                continue

            arr = np.array(features)
            if arr.ndim != 2 or arr.shape[1] != 4:
                continue

            # Check close-to-close jumps
            closes = arr[:, 3]
            for t in range(1, len(closes)):
                prev_close = closes[t-1]
                curr_close = closes[t]

                if prev_close > 0:
                    jump = abs(curr_close - prev_close) / prev_close
                    if jump > max_jump_pct:
                        errors.append(
                            f"Sample {idx}, timestep {t}: "
                            f"price jump {jump*100:.1f}% exceeds threshold "
                            f"{max_jump_pct*100:.1f}%"
                        )

        return errors[:10]  # Limit to first 10 errors


# ============================================================================
# FINANCIAL-SPECIFIC VALIDATORS
# ============================================================================

class FinancialDataValidator:
    """Financial market data specific validation."""

    @staticmethod
    def validate_price_ranges(df: pd.DataFrame) -> List[str]:
        """Validate price ranges are realistic."""
        errors = []

        if 'features' not in df.columns:
            return errors

        for idx, features in enumerate(df['features']):
            if not isinstance(features, (list, np.ndarray)):
                continue

            arr = np.array(features)
            if arr.ndim == 2 and arr.shape[1] == 4:
                prices = arr.flatten()
            elif arr.ndim == 1:
                prices = arr
            else:
                continue

            # Check for negative or zero prices
            if np.any(prices <= 0):
                errors.append(f"Sample {idx}: contains non-positive prices")

            # Check for extremely high prices (likely data errors)
            if np.any(prices > 10_000_000):
                errors.append(f"Sample {idx}: contains unrealistically high prices")

        return errors[:10]

    @staticmethod
    def validate_volume_if_present(df: pd.DataFrame) -> List[str]:
        """Validate volume data if present."""
        errors = []

        if 'volume' not in df.columns:
            return errors

        volumes = df['volume'].values

        # Check for negative volumes
        if np.any(volumes < 0):
            errors.append("Found negative volume values")

        # Check for suspiciously consistent volumes (may indicate fake data)
        volume_std = np.std(volumes)
        volume_mean = np.mean(volumes)
        if volume_mean > 0 and volume_std / volume_mean < 0.01:
            errors.append("Volume data has suspiciously low variance")

        return errors

    @staticmethod
    def check_market_hours(df: pd.DataFrame, symbol: str) -> List[str]:
        """Check if timestamps align with market hours for the symbol."""
        warnings = []

        # This is a placeholder - would implement market hours checking
        # based on symbol (crypto=24/7, stocks=market hours, etc.)

        return warnings


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "QualityThresholds",
    "TimeSeriesQualityValidator",
    "FinancialDataValidator",
]
