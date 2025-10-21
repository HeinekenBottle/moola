#!/usr/bin/env python3
"""Automated data validation pipeline with quality gates.

Runs comprehensive validation checks and generates quality reports.

Usage:
    python -m moola.data_infra.pipelines.validate --stage raw
    python -m moola.data_infra.pipelines.validate --stage windows
    python -m moola.data_infra.pipelines.validate --stage labeled
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import pandas as pd
from loguru import logger

from moola.paths import DATA_DIR

from ..schemas import (
    DataQualityReport,
    LabeledDataset,
    TimeSeriesWindow,
    UnlabeledDataset,
)
from ..validators import (
    FinancialDataValidator,
    QualityThresholds,
    TimeSeriesQualityValidator,
)

# ============================================================================
# VALIDATION STAGES
# ============================================================================


class ValidationPipeline:
    """Orchestrates data validation across pipeline stages."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize validators
        self.quality_validator = TimeSeriesQualityValidator(
            thresholds=QualityThresholds(
                max_missing_percent=1.0,
                outlier_zscore=5.0,
                check_ohlc_logic=True,
                max_price_jump_percent=200.0,
            )
        )
        self.financial_validator = FinancialDataValidator()

    def validate_raw(self, data_path: Path) -> DataQualityReport:
        """Validate raw market data.

        Args:
            data_path: Path to raw parquet file

        Returns:
            DataQualityReport with validation results
        """
        logger.info("=" * 70)
        logger.info("VALIDATING RAW MARKET DATA")
        logger.info("=" * 70)

        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df):,} rows from {data_path}")

        # Run core quality checks
        report = self.quality_validator.validate_dataset(df, dataset_name=data_path.stem)

        # Run financial-specific checks
        price_errors = self.financial_validator.validate_price_ranges(df)
        report.validation_errors.extend(price_errors)

        volume_errors = self.financial_validator.validate_volume_if_present(df)
        report.validation_errors.extend(volume_errors)

        # Save report
        self._save_report(report, "raw_validation_report.json")

        return report

    def validate_windows(self, windows_path: Path) -> DataQualityReport:
        """Validate time-series windows.

        Args:
            windows_path: Path to unlabeled_windows.parquet

        Returns:
            DataQualityReport with validation results
        """
        logger.info("=" * 70)
        logger.info("VALIDATING TIME-SERIES WINDOWS")
        logger.info("=" * 70)

        df = pd.read_parquet(windows_path)
        logger.info(f"Loaded {len(df):,} windows from {windows_path}")

        # Validate window structure
        validation_errors = []

        for idx, row in df.iterrows():
            try:
                # Validate using Pydantic schema
                window = TimeSeriesWindow(
                    window_id=str(row["window_id"]),
                    features=self._convert_features_to_list(row["features"]),
                )
            except Exception as e:
                validation_errors.append(f"Window {idx} failed validation: {str(e)}")

            if len(validation_errors) >= 10:
                break  # Limit to first 10 errors

        # Run quality checks
        report = self.quality_validator.validate_dataset(df, dataset_name=windows_path.stem)

        # Add schema validation errors
        report.validation_errors.extend(validation_errors)

        # Check window consistency
        if "features" in df.columns:
            self._validate_window_shapes(df, report)

        # Save report
        self._save_report(report, "window_validation_report.json")

        return report

    def validate_labeled(self, labeled_path: Path) -> DataQualityReport:
        """Validate labeled training data.

        Args:
            labeled_path: Path to train.parquet

        Returns:
            DataQualityReport with validation results
        """
        logger.info("=" * 70)
        logger.info("VALIDATING LABELED TRAINING DATA")
        logger.info("=" * 70)

        df = pd.read_parquet(labeled_path)
        logger.info(f"Loaded {len(df):,} labeled samples from {labeled_path}")

        # Check required columns
        required_cols = ["window_id", "label", "features"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Run quality checks
        report = self.quality_validator.validate_dataset(df, dataset_name=labeled_path.stem)

        # Validate labels
        label_errors = self._validate_labels(df)
        report.validation_errors.extend(label_errors)

        # Check class balance
        label_dist = df["label"].value_counts().to_dict()
        logger.info(f"Label distribution: {label_dist}")

        min_count = min(label_dist.values())
        max_count = max(label_dist.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        if imbalance_ratio > 10.0:
            report.warnings.append(
                f"Severe class imbalance: {imbalance_ratio:.1f}x " f"(distribution: {label_dist})"
            )

        # Validate expansion indices if present
        if "expansion_start" in df.columns and "expansion_end" in df.columns:
            expansion_errors = self._validate_expansion_indices(df)
            report.validation_errors.extend(expansion_errors)

        # Save report
        self._save_report(report, "labeled_validation_report.json")

        # Save metrics
        metrics = {
            "total_samples": len(df),
            "label_distribution": label_dist,
            "class_imbalance_ratio": float(imbalance_ratio),
            "quality_score": report.quality_score,
            "validation_passed": report.passed_validation,
        }
        self._save_metrics(metrics, "labeled_quality_metrics.json")

        return report

    def _convert_features_to_list(self, features) -> list:
        """Convert features to list format for validation."""
        if isinstance(features, list):
            return features
        elif isinstance(features, np.ndarray):
            # Handle array of arrays
            return [arr.tolist() if hasattr(arr, "tolist") else arr for arr in features]
        else:
            raise ValueError(f"Unsupported features type: {type(features)}")

    def _validate_window_shapes(self, df: pd.DataFrame, report: DataQualityReport):
        """Validate that all windows have consistent shape (105, 4)."""
        import numpy as np

        for idx, features in enumerate(df["features"]):
            try:
                arr = (
                    np.vstack(features) if isinstance(features[0], (list, np.ndarray)) else features
                )
                if arr.shape != (105, 4):
                    report.validation_errors.append(
                        f"Window {idx} has shape {arr.shape}, expected (105, 4)"
                    )
            except Exception as e:
                report.validation_errors.append(f"Window {idx} shape validation failed: {str(e)}")

            if len(report.validation_errors) >= 10:
                break

    def _validate_labels(self, df: pd.DataFrame) -> list[str]:
        """Validate label values and distribution."""
        errors = []

        valid_labels = {"consolidation", "retracement", "expansion"}
        actual_labels = set(df["label"].unique())

        invalid_labels = actual_labels - valid_labels
        if invalid_labels:
            errors.append(
                f"Found invalid labels: {invalid_labels}. " f"Valid labels: {valid_labels}"
            )

        # Check minimum samples per class
        label_counts = df["label"].value_counts()
        for label, count in label_counts.items():
            if count < 2:
                errors.append(
                    f"Label '{label}' has only {count} sample(s). " f"Minimum required: 2"
                )

        return errors

    def _validate_expansion_indices(self, df: pd.DataFrame) -> list[str]:
        """Validate expansion start/end indices."""
        errors = []

        for idx, row in df.iterrows():
            start = row.get("expansion_start")
            end = row.get("expansion_end")

            if pd.isna(start) or pd.isna(end):
                continue

            # Check range: must be in prediction window [30, 75)
            if not (30 <= start < 75):
                errors.append(f"Sample {idx}: expansion_start={start} out of range [30, 75)")

            if not (30 <= end < 75):
                errors.append(f"Sample {idx}: expansion_end={end} out of range [30, 75)")

            # Check ordering
            if start > end:
                errors.append(f"Sample {idx}: expansion_start ({start}) > expansion_end ({end})")

            if len(errors) >= 10:
                break

        return errors

    def _save_report(self, report: DataQualityReport, filename: str):
        """Save validation report to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(report.model_dump(), f, indent=2, default=str)
        logger.info(f"Validation report saved: {output_path}")

    def _save_metrics(self, metrics: dict, filename: str):
        """Save metrics to JSON."""
        output_path = self.output_dir / filename
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Metrics saved: {output_path}")


# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Data Validation Pipeline")
    parser.add_argument(
        "--stage",
        type=str,
        required=True,
        choices=["raw", "windows", "labeled"],
        help="Validation stage",
    )
    parser.add_argument(
        "--data-path", type=Path, help="Path to data file (auto-detected if not specified)"
    )
    parser.add_argument("--output-dir", type=Path, help="Output directory for reports")

    args = parser.parse_args()

    # Auto-detect paths
    if args.data_path is None:
        if args.stage == "raw":
            args.data_path = DATA_DIR / "raw" / "unlabeled_windows.parquet"
        elif args.stage == "windows":
            args.data_path = DATA_DIR / "raw" / "unlabeled_windows.parquet"
        elif args.stage == "labeled":
            args.data_path = DATA_DIR / "processed" / "train.parquet"

    if args.output_dir is None:
        args.output_dir = args.data_path.parent

    # Initialize pipeline
    pipeline = ValidationPipeline(output_dir=args.output_dir)

    # Run validation
    if args.stage == "raw":
        report = pipeline.validate_raw(args.data_path)
    elif args.stage == "windows":
        report = pipeline.validate_windows(args.data_path)
    elif args.stage == "labeled":
        report = pipeline.validate_labeled(args.data_path)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Quality Score: {report.quality_score:.1f}/100")
    logger.info(f"Validation Status: {'PASSED' if report.passed_validation else 'FAILED'}")
    logger.info(f"Errors: {len(report.validation_errors)}")
    logger.info(f"Warnings: {len(report.warnings)}")

    if report.validation_errors:
        logger.error("\nValidation Errors:")
        for error in report.validation_errors[:10]:
            logger.error(f"  - {error}")

    if report.warnings:
        logger.warning("\nWarnings:")
        for warning in report.warnings[:10]:
            logger.warning(f"  - {warning}")

    # Exit with appropriate code
    if not report.passed_validation:
        logger.error("\nValidation FAILED - data quality issues detected")
        exit(1)
    else:
        logger.success("\nValidation PASSED - data quality is acceptable")
        exit(0)


if __name__ == "__main__":
    main()
