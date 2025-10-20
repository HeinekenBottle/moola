#!/usr/bin/env python3
"""Integration tests for data infrastructure.

Tests the complete data pipeline from validation to monitoring.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moola.data_infra.lineage import DataVersionControl, LineageTracker
from moola.data_infra.monitoring import DriftDetector, TimeSeriesDriftMonitor
from moola.data_infra.schemas import (
    DataQualityReport,
    LabeledDataset,
    LabeledWindow,
    OHLCBar,
    PatternLabel,
    TimeSeriesWindow,
    UnlabeledDataset,
)
from moola.data_infra.validators import QualityThresholds, TimeSeriesQualityValidator


class TestSchemas:
    """Test Pydantic schemas."""

    def test_ohlc_bar_valid(self):
        """Test valid OHLC bar."""
        bar = OHLCBar(open=100, high=105, low=95, close=102)
        assert bar.high >= bar.low
        assert bar.high >= bar.close
        assert bar.low <= bar.close

    def test_ohlc_bar_invalid_high_low(self):
        """Test invalid OHLC (high < low)."""
        with pytest.raises(ValueError, match="High.*cannot be less than Low"):
            OHLCBar(open=100, high=95, low=100, close=98)

    def test_ohlc_bar_unrealistic_jump(self):
        """Test unrealistic price jump detection."""
        with pytest.raises(ValueError, match="Unrealistic price range"):
            OHLCBar(open=100, high=500, low=50, close=300)

    def test_timeseries_window_valid(self):
        """Test valid time-series window."""
        features = [
            [100 + i, 105 + i, 95 + i, 102 + i]
            for i in range(105)
        ]
        window = TimeSeriesWindow(
            window_id="test_0",
            features=features
        )
        assert len(window.features) == 105
        assert all(len(bar) == 4 for bar in window.features)

        # Test conversion
        arr = window.to_numpy()
        assert arr.shape == (105, 4)

    def test_timeseries_window_invalid_length(self):
        """Test invalid window length."""
        features = [[100, 105, 95, 102] for _ in range(50)]  # Only 50 timesteps
        with pytest.raises(ValueError, match="Expected 105 timesteps"):
            TimeSeriesWindow(window_id="test_0", features=features)

    def test_timeseries_window_invalid_ohlc(self):
        """Test OHLC validation in window."""
        features = [
            [100, 95, 100, 98]  # high < low - INVALID
        ] + [[100, 105, 95, 102]] * 104

        with pytest.raises(ValueError, match="high.*< low"):
            TimeSeriesWindow(window_id="test_0", features=features)

    def test_labeled_window_valid(self):
        """Test valid labeled window."""
        features = [[100 + i, 105 + i, 95 + i, 102 + i] for i in range(105)]
        window = LabeledWindow(
            window_id="labeled_0",
            features=features,
            label=PatternLabel.CONSOLIDATION,
            expansion_start=35,
            expansion_end=60
        )
        assert window.expansion_start <= window.expansion_end

    def test_labeled_window_invalid_expansion(self):
        """Test invalid expansion indices."""
        features = [[100, 105, 95, 102]] * 105

        # Start > end
        with pytest.raises(ValueError, match="expansion_start.*cannot be greater"):
            LabeledWindow(
                window_id="labeled_0",
                features=features,
                label=PatternLabel.CONSOLIDATION,
                expansion_start=60,
                expansion_end=35
            )

        # Out of range
        with pytest.raises(ValueError):
            LabeledWindow(
                window_id="labeled_0",
                features=features,
                label=PatternLabel.CONSOLIDATION,
                expansion_start=20,  # < 30
                expansion_end=50
            )

    def test_unlabeled_dataset(self):
        """Test unlabeled dataset validation."""
        windows = [
            TimeSeriesWindow(
                window_id=f"window_{i}",
                features=[[100, 105, 95, 102]] * 105
            )
            for i in range(10)
        ]

        dataset = UnlabeledDataset(
            windows=windows,
            total_samples=10
        )

        assert len(dataset.windows) == dataset.total_samples

        # Test conversion
        X = dataset.to_numpy()
        assert X.shape == (10, 105, 4)

    def test_unlabeled_dataset_count_mismatch(self):
        """Test dataset with mismatched sample count."""
        windows = [
            TimeSeriesWindow(
                window_id=f"window_{i}",
                features=[[100, 105, 95, 102]] * 105
            )
            for i in range(10)
        ]

        with pytest.raises(ValueError, match="total_samples.*doesn't match"):
            UnlabeledDataset(windows=windows, total_samples=5)

    def test_labeled_dataset(self):
        """Test labeled dataset validation."""
        windows = [
            LabeledWindow(
                window_id=f"window_{i}",
                features=[[100, 105, 95, 102]] * 105,
                label=PatternLabel.CONSOLIDATION if i < 5 else PatternLabel.RETRACEMENT,
                expansion_start=35,
                expansion_end=60
            )
            for i in range(10)
        ]

        dataset = LabeledDataset(
            windows=windows,
            total_samples=10,
            label_distribution={"consolidation": 5, "retracement": 5}
        )

        # Test conversion
        X, y = dataset.to_numpy()
        assert X.shape == (10, 105, 4)
        assert y.shape == (10,)

    def test_labeled_dataset_class_imbalance(self):
        """Test severe class imbalance detection."""
        windows = [
            LabeledWindow(
                window_id=f"window_{i}",
                features=[[100, 105, 95, 102]] * 105,
                label=PatternLabel.CONSOLIDATION if i < 90 else PatternLabel.RETRACEMENT,
                expansion_start=35,
                expansion_end=60
            )
            for i in range(100)
        ]

        with pytest.raises(ValueError, match="Severe class imbalance"):
            LabeledDataset(
                windows=windows,
                total_samples=100,
                label_distribution={"consolidation": 90, "retracement": 10}
            )


class TestValidators:
    """Test quality validators."""

    def test_quality_validator_valid_data(self):
        """Test validation with valid data."""
        # Create valid OHLC data
        features = [
            np.array([[100 + i, 105 + i, 95 + i, 102 + i] for _ in range(105)])
            for i in range(100)
        ]

        df = pd.DataFrame({
            'window_id': [f'window_{i}' for i in range(100)],
            'features': features
        })

        validator = TimeSeriesQualityValidator()
        report = validator.validate_dataset(df, "test_dataset")

        assert report.passed_validation
        assert report.quality_score > 90
        assert len(report.validation_errors) == 0

    def test_quality_validator_missing_data(self):
        """Test detection of missing values."""
        df = pd.DataFrame({
            'window_id': [f'window_{i}' for i in range(10)],
            'features': [None] * 5 + [np.array([[100, 105, 95, 102]] * 105)] * 5
        })

        validator = TimeSeriesQualityValidator(
            thresholds=QualityThresholds(max_missing_percent=40.0)
        )
        report = validator.validate_dataset(df, "test_dataset")

        assert report.missing_percentage > 0

    def test_quality_validator_price_jumps(self):
        """Test detection of unrealistic price jumps."""
        # Create data with price jump
        normal = np.array([[100, 105, 95, 102]] * 100)
        jump = np.array([[100, 105, 95, 102]] * 4 + [[500, 505, 495, 502]] * 1)  # 5x jump

        features = [normal] * 98 + [jump] * 2

        df = pd.DataFrame({
            'window_id': [f'window_{i}' for i in range(100)],
            'features': features
        })

        validator = TimeSeriesQualityValidator(
            thresholds=QualityThresholds(max_price_jump_percent=200.0)
        )
        report = validator.validate_dataset(df, "test_dataset")

        # Should detect price jumps
        assert any("price jump" in err.lower() for err in report.validation_errors)


class TestLineageTracking:
    """Test lineage tracking."""

    def test_lineage_tracker(self):
        """Test lineage tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create dummy files
            input_path = tmpdir / "input.parquet"
            output_path = tmpdir / "output.parquet"

            pd.DataFrame({'a': [1, 2, 3]}).to_parquet(input_path)
            pd.DataFrame({'b': [4, 5]}).to_parquet(output_path)

            # Track transformation
            tracker = LineageTracker(lineage_dir=tmpdir / "lineage")
            lineage = tracker.log_transformation(
                dataset_id="test_dataset",
                transformation_type="test_transform",
                input_path=input_path,
                output_path=output_path,
                rows_in=3,
                rows_out=2,
                params={"test_param": "value"}
            )

            assert lineage.dataset_id == "test_dataset"
            assert lineage.rows_in == 3
            assert lineage.rows_out == 2
            assert lineage.checksum_in is not None
            assert lineage.checksum_out is not None

            # Retrieve lineage
            retrieved = tracker.get_lineage("test_dataset")
            assert retrieved.dataset_id == lineage.dataset_id

    def test_lineage_ancestors_descendants(self):
        """Test lineage ancestry tracking."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = LineageTracker(lineage_dir=Path(tmpdir) / "lineage")

            # Create lineage chain: A -> B -> C
            tracker.log_transformation(
                dataset_id="dataset_b",
                transformation_type="step1",
                input_path=None,
                output_path=None,
                rows_in=100,
                rows_out=50,
                parent_datasets=["dataset_a"]
            )

            tracker.log_transformation(
                dataset_id="dataset_c",
                transformation_type="step2",
                input_path=None,
                output_path=None,
                rows_in=50,
                rows_out=25,
                parent_datasets=["dataset_b"]
            )

            # Check ancestry
            ancestors = tracker.get_ancestors("dataset_c")
            assert "dataset_b" in ancestors
            assert "dataset_a" in ancestors

            # Check descendants
            descendants = tracker.get_descendants("dataset_a")
            assert "dataset_b" in descendants
            assert "dataset_c" in descendants


class TestVersionControl:
    """Test data version control."""

    def test_version_creation(self):
        """Test creating data versions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test dataset
            data_path = tmpdir / "train.parquet"
            df = pd.DataFrame({
                'window_id': ['w0', 'w1', 'w2'],
                'features': [np.array([[100, 105, 95, 102]] * 105)] * 3
            })
            df.to_parquet(data_path)

            # Create version
            vc = DataVersionControl(versions_dir=tmpdir / "versions")
            version = vc.create_version(
                dataset_name="train",
                file_path=data_path,
                version_id="v1.0.0",
                notes="Test version"
            )

            assert version.version_id == "v1.0.0"
            assert version.num_samples == 3
            assert version.dvc_hash is not None

            # Retrieve version
            retrieved = vc.get_version("train", "v1.0.0")
            assert retrieved.version_id == version.version_id

    def test_version_validation(self):
        """Test version ID validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            data_path = tmpdir / "test.parquet"
            pd.DataFrame({'a': [1]}).to_parquet(data_path)

            vc = DataVersionControl(versions_dir=tmpdir / "versions")

            # Invalid version format
            with pytest.raises(ValueError, match="Version must follow semver"):
                vc.create_version(
                    dataset_name="test",
                    file_path=data_path,
                    version_id="1.0"  # Missing patch version
                )


class TestDriftDetection:
    """Test drift detection."""

    def test_ks_test_no_drift(self):
        """Test KS test with no drift."""
        baseline = np.random.normal(100, 10, 1000)
        current = np.random.normal(100, 10, 1000)

        detector = DriftDetector(method="ks_test", threshold=0.05)
        result = detector.detect_drift(baseline, current, "test_feature")

        # Should not detect drift for same distribution
        assert result.p_value > 0.05
        assert not result.drift_detected

    def test_ks_test_with_drift(self):
        """Test KS test with drift."""
        baseline = np.random.normal(100, 10, 1000)
        current = np.random.normal(150, 10, 1000)  # Different mean

        detector = DriftDetector(method="ks_test", threshold=0.05)
        result = detector.detect_drift(baseline, current, "test_feature")

        # Should detect drift
        assert result.p_value < 0.05
        assert result.drift_detected

    def test_psi_drift(self):
        """Test PSI drift detection."""
        baseline = np.random.normal(100, 10, 1000)
        current = np.random.normal(120, 10, 1000)  # Shifted distribution

        detector = DriftDetector(method="psi", threshold=0.2)
        result = detector.detect_drift(baseline, current, "test_feature")

        # PSI should detect shift
        assert result.drift_score > 0
        # May or may not cross threshold depending on shift magnitude

    def test_wasserstein_drift(self):
        """Test Wasserstein distance drift detection."""
        baseline = np.random.uniform(0, 100, 1000)
        current = np.random.uniform(50, 150, 1000)  # Shifted range

        detector = DriftDetector(method="wasserstein", threshold=0.1)
        result = detector.detect_drift(baseline, current, "test_feature")

        assert result.drift_score > 0
        assert result.p_value is None  # Wasserstein doesn't provide p-value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
