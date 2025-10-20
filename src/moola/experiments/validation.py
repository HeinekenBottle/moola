"""Great Expectations-style validation utilities for experiment data quality.

Provides comprehensive validation rules for:
- Data shape and type checking
- OHLC relationship validation
- Statistical distribution checks
- Data completeness verification
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from moola.config.training_config import OHLC_DIMS, WINDOW_SIZE

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    passed: bool
    message: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO
    details: dict | None = None

    def __str__(self) -> str:
        """Format validation result for logging."""
        status = "✓" if self.passed else "✗"
        return f"{status} [{self.severity}] {self.check_name}: {self.message}"


class DataValidator:
    """Validates experiment data against expected schema and quality rules."""

    def __init__(self, expected_samples: int | None = None):
        """Initialize validator.

        Args:
            expected_samples: Expected number of samples (for count validation)
        """
        self.expected_samples = expected_samples
        self.results: list[ValidationResult] = []

    def validate_all(self, data: np.ndarray, stage: str = "unknown") -> tuple[bool, list[ValidationResult]]:
        """Run all validation checks.

        Args:
            data: Data array to validate [N, 105, 4]
            stage: Stage name for logging (e.g., "pre-augmentation", "post-augmentation")

        Returns:
            Tuple of (all_passed, list of validation results)
        """
        self.results = []

        logger.info(f"Running validation for {stage} stage...")

        # Structure validation
        self._validate_shape(data)
        self._validate_dtype(data)

        # Completeness validation
        self._validate_no_nans(data)
        self._validate_no_infs(data)
        self._validate_sample_count(data)

        # OHLC relationship validation
        self._validate_ohlc_high(data)
        self._validate_ohlc_low(data)
        self._validate_ohlc_spread(data)

        # Statistical validation
        self._validate_price_range(data)
        self._validate_volatility_distribution(data)

        # Summary
        all_passed = all(r.passed for r in self.results)
        errors = [r for r in self.results if not r.passed and r.severity == "ERROR"]
        warnings = [r for r in self.results if not r.passed and r.severity == "WARNING"]

        logger.info(f"\nValidation summary for {stage}:")
        logger.info(f"  Total checks: {len(self.results)}")
        logger.info(f"  Passed: {sum(r.passed for r in self.results)}")
        logger.info(f"  Errors: {len(errors)}")
        logger.info(f"  Warnings: {len(warnings)}")

        for result in self.results:
            if not result.passed:
                logger.log(
                    logging.ERROR if result.severity == "ERROR" else logging.WARNING,
                    f"  {result}"
                )

        return all_passed, self.results

    def _add_result(self, check_name: str, passed: bool, message: str, severity: str = "ERROR", details: dict | None = None):
        """Add validation result."""
        result = ValidationResult(
            check_name=check_name,
            passed=passed,
            message=message,
            severity=severity,
            details=details,
        )
        self.results.append(result)

    def _validate_shape(self, data: np.ndarray):
        """Validate array shape."""
        expected_shape = (None, WINDOW_SIZE, OHLC_DIMS)
        passed = (
            data.ndim == 3
            and data.shape[1] == WINDOW_SIZE
            and data.shape[2] == OHLC_DIMS
        )

        if passed:
            message = f"Shape {data.shape} matches expected [N, {WINDOW_SIZE}, {OHLC_DIMS}]"
        else:
            message = f"Shape {data.shape} does not match expected [N, {WINDOW_SIZE}, {OHLC_DIMS}]"

        self._add_result("shape_validation", passed, message)

    def _validate_dtype(self, data: np.ndarray):
        """Validate data type."""
        passed = np.issubdtype(data.dtype, np.floating)

        if passed:
            message = f"Dtype {data.dtype} is valid floating point"
        else:
            message = f"Dtype {data.dtype} is not floating point"

        self._add_result("dtype_validation", passed, message)

    def _validate_no_nans(self, data: np.ndarray):
        """Check for NaN values."""
        nan_count = np.sum(np.isnan(data))
        passed = nan_count == 0

        if passed:
            message = "No NaN values detected"
        else:
            nan_pct = 100 * nan_count / data.size
            message = f"Found {nan_count} NaN values ({nan_pct:.3f}% of data)"

        self._add_result("nan_check", passed, message, details={"nan_count": int(nan_count)})

    def _validate_no_infs(self, data: np.ndarray):
        """Check for infinite values."""
        inf_count = np.sum(np.isinf(data))
        passed = inf_count == 0

        if passed:
            message = "No infinite values detected"
        else:
            inf_pct = 100 * inf_count / data.size
            message = f"Found {inf_count} infinite values ({inf_pct:.3f}% of data)"

        self._add_result("inf_check", passed, message, details={"inf_count": int(inf_count)})

    def _validate_sample_count(self, data: np.ndarray):
        """Validate sample count."""
        if self.expected_samples is None:
            self._add_result(
                "sample_count",
                True,
                f"Sample count: {len(data)} (no expected count specified)",
                severity="INFO"
            )
            return

        passed = len(data) == self.expected_samples

        if passed:
            message = f"Sample count {len(data)} matches expected {self.expected_samples}"
        else:
            message = f"Sample count {len(data)} does not match expected {self.expected_samples}"

        severity = "WARNING" if not passed else "INFO"
        self._add_result("sample_count", passed, message, severity=severity, details={"actual": len(data), "expected": self.expected_samples})

    def _validate_ohlc_high(self, data: np.ndarray, tolerance: float = 1e-6):
        """Validate High >= max(Open, Close)."""
        if data.shape[2] != 4:
            return

        open_prices = data[:, :, 0]
        high_prices = data[:, :, 1]
        close_prices = data[:, :, 3]

        max_oc = np.maximum(open_prices, close_prices)
        violations = np.sum(high_prices < (max_oc - tolerance))
        passed = violations == 0

        if passed:
            message = "All High prices satisfy H >= max(O,C)"
        else:
            violation_pct = 100 * violations / data.size
            message = f"Found {violations} violations ({violation_pct:.3f}%) where H < max(O,C)"

        self._add_result("ohlc_high_validation", passed, message, details={"violations": int(violations)})

    def _validate_ohlc_low(self, data: np.ndarray, tolerance: float = 1e-6):
        """Validate Low <= min(Open, Close)."""
        if data.shape[2] != 4:
            return

        open_prices = data[:, :, 0]
        low_prices = data[:, :, 2]
        close_prices = data[:, :, 3]

        min_oc = np.minimum(open_prices, close_prices)
        violations = np.sum(low_prices > (min_oc + tolerance))
        passed = violations == 0

        if passed:
            message = "All Low prices satisfy L <= min(O,C)"
        else:
            violation_pct = 100 * violations / data.size
            message = f"Found {violations} violations ({violation_pct:.3f}%) where L > min(O,C)"

        self._add_result("ohlc_low_validation", passed, message, details={"violations": int(violations)})

    def _validate_ohlc_spread(self, data: np.ndarray):
        """Validate High-Low spread is positive."""
        if data.shape[2] != 4:
            return

        high_prices = data[:, :, 1]
        low_prices = data[:, :, 2]

        spread = high_prices - low_prices
        negative_spreads = np.sum(spread < 0)
        passed = negative_spreads == 0

        if passed:
            message = "All H-L spreads are non-negative"
        else:
            violation_pct = 100 * negative_spreads / (data.shape[0] * data.shape[1])
            message = f"Found {negative_spreads} negative spreads ({violation_pct:.3f}%)"

        self._add_result("hl_spread_validation", passed, message, details={"negative_spreads": int(negative_spreads)})

    def _validate_price_range(self, data: np.ndarray):
        """Validate price values are in reasonable range."""
        # Assuming normalized/standardized prices should be roughly in [-5, 5] range
        # Adjust thresholds based on your data preprocessing

        min_val = np.min(data)
        max_val = np.max(data)

        # Reasonable range for standardized financial data
        reasonable_min = -10.0
        reasonable_max = 10.0

        passed = (min_val >= reasonable_min) and (max_val <= reasonable_max)

        if passed:
            message = f"Price range [{min_val:.3f}, {max_val:.3f}] is within reasonable bounds"
        else:
            message = f"Price range [{min_val:.3f}, {max_val:.3f}] exceeds reasonable bounds [{reasonable_min}, {reasonable_max}]"

        severity = "WARNING" if not passed else "INFO"
        self._add_result(
            "price_range_validation",
            passed,
            message,
            severity=severity,
            details={"min": float(min_val), "max": float(max_val)}
        )

    def _validate_volatility_distribution(self, data: np.ndarray):
        """Validate volatility (H-L spread) distribution."""
        if data.shape[2] != 4:
            return

        high_prices = data[:, :, 1]
        low_prices = data[:, :, 2]

        volatility = high_prices - low_prices
        mean_vol = np.mean(volatility)
        std_vol = np.std(volatility)

        # Check if volatility is reasonable (not all zeros, not too extreme)
        passed = (mean_vol > 1e-6) and (std_vol < 100 * mean_vol)

        if passed:
            message = f"Volatility distribution is reasonable (mean={mean_vol:.4f}, std={std_vol:.4f})"
        else:
            message = f"Volatility distribution is suspicious (mean={mean_vol:.4f}, std={std_vol:.4f})"

        severity = "WARNING" if not passed else "INFO"
        self._add_result(
            "volatility_distribution",
            passed,
            message,
            severity=severity,
            details={"mean": float(mean_vol), "std": float(std_vol)}
        )


def validate_experiment_data(
    data: np.ndarray,
    stage: str = "unknown",
    expected_samples: int | None = None,
) -> tuple[bool, list[ValidationResult]]:
    """Convenience function for validating experiment data.

    Args:
        data: Data array to validate [N, 105, 4]
        stage: Stage name for logging
        expected_samples: Expected number of samples

    Returns:
        Tuple of (all_passed, validation results)
    """
    validator = DataValidator(expected_samples=expected_samples)
    return validator.validate_all(data, stage=stage)
