#!/usr/bin/env python3
"""Data drift detection for monitoring production data quality.

Detects statistical drift between training data and production data
using Kolmogorov-Smirnov test, Population Stability Index, and other methods.

Usage:
    python -m moola.data_infra.monitoring.drift_detector \
        --baseline data/processed/train.parquet \
        --current data/processed/new_data.parquet \
        --output data/monitoring/drift_report.json
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats

# ============================================================================
# DRIFT DETECTION METHODS
# ============================================================================


@dataclass
class DriftResult:
    """Result of drift detection for a single feature."""

    feature_name: str
    drift_score: float
    p_value: Optional[float]
    drift_detected: bool
    method: str
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]


class DriftDetector:
    """Statistical drift detection for time-series features."""

    def __init__(
        self, method: Literal["ks_test", "psi", "wasserstein"] = "ks_test", threshold: float = 0.05
    ):
        """Initialize drift detector.

        Args:
            method: Detection method
                - ks_test: Kolmogorov-Smirnov test (p-value threshold)
                - psi: Population Stability Index (PSI threshold)
                - wasserstein: Wasserstein distance (distance threshold)
            threshold: Threshold for drift detection
        """
        self.method = method
        self.threshold = threshold

    def detect_drift(
        self, baseline_data: np.ndarray, current_data: np.ndarray, feature_name: str = "feature"
    ) -> DriftResult:
        """Detect drift between baseline and current data.

        Args:
            baseline_data: Reference/training data
            current_data: New/production data
            feature_name: Name for reporting

        Returns:
            DriftResult with drift detection results
        """
        # Compute statistics
        baseline_stats = self._compute_stats(baseline_data)
        current_stats = self._compute_stats(current_data)

        # Run drift test
        if self.method == "ks_test":
            score, p_value, drift = self._ks_test(baseline_data, current_data)
        elif self.method == "psi":
            score, p_value, drift = self._psi(baseline_data, current_data)
        elif self.method == "wasserstein":
            score, p_value, drift = self._wasserstein(baseline_data, current_data)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return DriftResult(
            feature_name=feature_name,
            drift_score=score,
            p_value=p_value,
            drift_detected=drift,
            method=self.method,
            baseline_stats=baseline_stats,
            current_stats=current_stats,
        )

    def _compute_stats(self, data: np.ndarray) -> Dict[str, float]:
        """Compute statistical properties."""
        return {
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "median": float(np.median(data)),
            "q25": float(np.percentile(data, 25)),
            "q75": float(np.percentile(data, 75)),
        }

    def _ks_test(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, float, bool]:
        """Kolmogorov-Smirnov two-sample test."""
        statistic, p_value = stats.ks_2samp(baseline, current)
        drift_detected = p_value < self.threshold
        return float(statistic), float(p_value), drift_detected

    def _psi(
        self, baseline: np.ndarray, current: np.ndarray, num_bins: int = 10
    ) -> Tuple[float, None, bool]:
        """Population Stability Index (PSI).

        PSI interpretation:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change (drift)
        """
        # Create bins based on baseline
        bins = np.percentile(baseline, np.linspace(0, 100, num_bins + 1))

        # Ensure unique bins
        bins = np.unique(bins)
        if len(bins) <= 1:
            return 0.0, None, False

        # Compute distributions
        baseline_counts, _ = np.histogram(baseline, bins=bins)
        current_counts, _ = np.histogram(current, bins=bins)

        # Normalize to percentages
        baseline_pct = baseline_counts / len(baseline)
        current_pct = current_counts / len(current)

        # Avoid division by zero
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)

        # Compute PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        drift_detected = psi >= self.threshold
        return float(psi), None, drift_detected

    def _wasserstein(self, baseline: np.ndarray, current: np.ndarray) -> Tuple[float, None, bool]:
        """Wasserstein distance (Earth Mover's Distance)."""
        distance = stats.wasserstein_distance(baseline, current)

        # Normalize by baseline range
        baseline_range = np.max(baseline) - np.min(baseline)
        if baseline_range > 0:
            normalized_distance = distance / baseline_range
        else:
            normalized_distance = distance

        drift_detected = normalized_distance >= self.threshold
        return float(normalized_distance), None, drift_detected


# ============================================================================
# TIME-SERIES DRIFT MONITORING
# ============================================================================


class TimeSeriesDriftMonitor:
    """Monitor drift in time-series OHLC features."""

    def __init__(
        self, method: Literal["ks_test", "psi", "wasserstein"] = "ks_test", threshold: float = 0.05
    ):
        self.detector = DriftDetector(method=method, threshold=threshold)

    def monitor_dataset_drift(
        self, baseline_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> Dict[str, DriftResult]:
        """Monitor drift across entire dataset.

        Args:
            baseline_df: Reference dataset (training data)
            current_df: Current dataset (production data)

        Returns:
            Dict mapping feature names to DriftResults
        """
        logger.info("Monitoring data drift...")
        logger.info(f"Baseline samples: {len(baseline_df):,}")
        logger.info(f"Current samples: {len(current_df):,}")

        results = {}

        # Extract price features from OHLC windows
        baseline_prices = self._extract_all_prices(baseline_df)
        current_prices = self._extract_all_prices(current_df)

        # Check drift for each OHLC component
        for i, feature_name in enumerate(["open", "high", "low", "close"]):
            baseline_feat = baseline_prices[:, i]
            current_feat = current_prices[:, i]

            result = self.detector.detect_drift(
                baseline_feat, current_feat, feature_name=feature_name
            )
            results[feature_name] = result

            logger.info(
                f"{feature_name}: drift_score={result.drift_score:.4f}, "
                f"drift={result.drift_detected}"
            )

        # Check overall price distribution drift
        baseline_all = baseline_prices.flatten()
        current_all = current_prices.flatten()

        result = self.detector.detect_drift(baseline_all, current_all, feature_name="all_prices")
        results["all_prices"] = result

        return results

    def _extract_all_prices(self, df: pd.DataFrame) -> np.ndarray:
        """Extract all OHLC prices from features column.

        Returns:
            Array of shape (N_samples * 105, 4) with OHLC prices
        """
        all_prices = []

        for features in df["features"]:
            if isinstance(features, (list, np.ndarray)):
                arr = (
                    np.vstack(features) if isinstance(features[0], (list, np.ndarray)) else features
                )
                all_prices.append(arr)

        if not all_prices:
            return np.array([]).reshape(0, 4)

        stacked = np.vstack(all_prices)
        return stacked

    def generate_drift_report(self, results: Dict[str, DriftResult], output_path: Path):
        """Generate comprehensive drift report.

        Args:
            results: Drift detection results
            output_path: Path to save report JSON
        """
        # Summarize drift
        total_features = len(results)
        drifted_features = sum(1 for r in results.values() if r.drift_detected)
        drift_percentage = (drifted_features / total_features * 100) if total_features > 0 else 0

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_features": total_features,
                "drifted_features": drifted_features,
                "drift_percentage": drift_percentage,
                "overall_drift_detected": drift_percentage > 50,
            },
            "method": self.detector.method,
            "threshold": self.detector.threshold,
            "features": {},
        }

        # Add per-feature results
        for feature_name, result in results.items():
            report["features"][feature_name] = {
                "drift_score": result.drift_score,
                "p_value": result.p_value,
                "drift_detected": result.drift_detected,
                "baseline_stats": result.baseline_stats,
                "current_stats": result.current_stats,
            }

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Drift report saved: {output_path}")

        # Log summary
        logger.info("\n" + "=" * 70)
        logger.info("DRIFT DETECTION SUMMARY")
        logger.info("=" * 70)
        logger.info(
            f"Drift detected in {drifted_features}/{total_features} features ({drift_percentage:.1f}%)"
        )

        if report["summary"]["overall_drift_detected"]:
            logger.warning("⚠️  SIGNIFICANT DRIFT DETECTED - Consider retraining model")
        else:
            logger.success("✓ No significant drift detected")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Data Drift Detection")
    parser.add_argument(
        "--baseline", type=Path, required=True, help="Baseline dataset (training data)"
    )
    parser.add_argument(
        "--current", type=Path, required=True, help="Current dataset (production data)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/monitoring/drift_report.json"),
        help="Output path for drift report",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="ks_test",
        choices=["ks_test", "psi", "wasserstein"],
        help="Drift detection method",
    )
    parser.add_argument("--threshold", type=float, default=0.05, help="Drift detection threshold")

    args = parser.parse_args()

    # Load datasets
    logger.info(f"Loading baseline from: {args.baseline}")
    baseline_df = pd.read_parquet(args.baseline)

    logger.info(f"Loading current from: {args.current}")
    current_df = pd.read_parquet(args.current)

    # Initialize monitor
    monitor = TimeSeriesDriftMonitor(method=args.method, threshold=args.threshold)

    # Detect drift
    results = monitor.monitor_dataset_drift(baseline_df, current_df)

    # Generate report
    monitor.generate_drift_report(results, args.output)


if __name__ == "__main__":
    main()
