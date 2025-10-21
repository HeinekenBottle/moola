"""Data pipeline integration tests.

Tests the complete data flow from raw OHLC data through feature engineering
to model training, ensuring compatibility and data integrity.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moola.cli import ingest
from moola.features import AdvancedFeatureEngineer, FeatureConfig
from moola.models import get_model
from moola.utils.seeds import set_seed


class TestDataPipelineIntegration:
    """Test data pipeline components integration."""

    @pytest.fixture
    def test_data(self):
        """Generate synthetic OHLC test data."""
        np.random.seed(42)
        N = 50  # Small test dataset
        T = 105  # Time steps

        # Generate synthetic OHLC data
        X_ohlc = np.random.randn(N, T, 4)

        # Simple labels for testing
        labels = ["class_A", "class_B"]
        y = np.random.choice(labels, N)

        # Create DataFrame
        rows = []
        for i in range(N):
            engineer = AdvancedFeatureEngineer(FeatureConfig())
            X_engineered = engineer.transform(X_ohlc[i : i + 1])

            row_data = {
                "window_id": i,
                "label": y[i],
                "features": X_engineered[0].tolist(),
            }
            rows.append(row_data)

        df = pd.DataFrame(rows)
        return df

    def test_ohlc_to_features_transformation(self, test_data):
        """Test OHLC to feature engineering transformation."""
        # Test raw OHLC transformation
        engineer = AdvancedFeatureEngineer(FeatureConfig())

        # Extract raw features from test data
        raw_features = np.array([np.array(f) for f in test_data["features"]])

        # Transform raw OHLC back (for validation)
        X_ohlc = np.random.randn(len(test_data), 105, 4)
        transformed_features = engineer.transform(X_ohlc)

        # Validate transformations
        assert transformed_features.shape[0] == len(test_data)
        assert transformed_features.shape[1] == 105  # Time steps
        assert transformed_features.shape[2] >= 20  # Minimum features
        assert transformed_features.shape[2] <= 50  # Maximum features

        # Test feature importance tracking
        feature_names = engineer.feature_names
        assert len(feature_names) > 0
        assert isinstance(feature_names, list)

    def test_data_schema_validation(self, test_data):
        """Test data schema validation."""
        # Test data schema requirements
        required_columns = ["window_id", "label", "features"]
        missing_columns = [col for col in required_columns if col not in test_data.columns]

        assert len(missing_columns) == 0, f"Missing required columns: {missing_columns}"

        # Validate feature structure
        for idx, row in test_data.iterrows():
            features = row["features"]
            assert isinstance(features, list), f"Features at index {idx} not a list"
            assert len(features) == 105, f"Features at index {idx} have wrong length"

            # Each time step should have 4+ features
            for t, time_step in enumerate(features):
                assert isinstance(time_step, list), f"Time step {t} at index {idx} not a list"
                assert (
                    len(time_step) >= 4
                ), f"Time step {t} at index {idx} has insufficient features"

    def test_model_training_integration(self, test_data):
        """Test model training with engineered features."""
        # Save test data
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "test_data.parquet"
            test_data.to_parquet(data_path)

            # Test data ingestion
            ingest(str(Path("configs")), [], str(data_path))

            # Test model training
            model = get_model("simple_lstm", device="cpu")

            # Extract features for training
            X = np.array([np.array(f) for f in test_data["features"]])
            y = test_data["label"].values

            # Train model
            model.fit(X, y)

            # Test predictions
            X_test = X[:5]  # First 5 samples for testing
            predictions = model.predict(X_test)
            probabilities = model.predict_proba(X_test)

            # Validate predictions
            assert len(predictions) == 5
            assert probabilities.shape == (5, 2)  # 2 classes

            # Validate prediction types
            assert all(isinstance(pred, str) for pred in predictions)
            assert all(isinstance(prob, (float, np.float32)) for prob in probabilities[0])

    def test_data_augmentation_integration(self, test_data):
        """Test data augmentation pipeline integration."""
        from moola.utils.augmentation import mixup_criterion, mixup_cutmix

        # Extract test features
        X = np.array([np.array(f) for f in test_data["features"]])
        y = test_data["label"].values

        # Convert to indices for training
        label_to_idx = {"class_A": 0, "class_B": 1}
        y_indices = np.array([label_to_idx[label] for label in y])

        # Test mixup/cutmix augmentation
        X_aug, y_a, y_b, lam = mixup_cutmix(X, y_indices, mixup_alpha=0.4, cutmix_prob=0.5)

        # Validate augmentation
        assert X_aug.shape == X.shape
        assert len(y_a) == len(y)
        assert len(y_b) == len(y)
        assert 0.0 <= lam <= 1.0

        # Test temporal augmentation
        from moola.utils.temporal_augmentation import TemporalAugmentation

        temporal_aug = TemporalAugmentation(
            jitter_prob=0.5,
            jitter_sigma=0.05,
            scaling_prob=0.3,
            scaling_sigma=0.1,
            permutation_prob=0.0,
            time_warp_prob=0.0,
            rotation_prob=0.0,
        )

        X_temp_aug = temporal_aug.apply_augmentation(X_aug)
        assert X_temp_aug.shape == X.shape

    def test_data_splitting_consistency(self, test_data):
        """Test data splitting consistency across folds."""
        from sklearn.model_selection import StratifiedKFold

        X = np.array([np.array(f) for f in test_data["features"]])
        y = test_data["label"].values

        # Test stratified K-fold splitting
        k = 3
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Validate split ratios
            assert len(X_train) + len(X_val) == len(X)
            assert len(y_train) + len(y_val) == len(y)

            # Validate stratification
            train_classes = np.unique(y_train)
            val_classes = np.unique(y_val)

            assert (
                len(train_classes) == 2
            ), f"Fold {fold_idx}: Training has only {len(train_classes)} classes"
            assert (
                len(val_classes) == 2
            ), f"Fold {fold_idx}: Validation has only {len(val_classes)} classes"

    def test_memory_efficiency(self, test_data):
        """Test memory efficiency with small datasets."""
        import torch

        # Test with small batch sizes
        model = get_model("simple_lstm", device="cpu", batch_size=16)

        X = np.array([np.array(f) for f in test_data["features"]])
        y = test_data["label"].values

        # Check memory usage during training
        set_seed(42)
        model.fit(X, y, n_epochs=2)  # Short training for memory test

        # Verify model can handle the dataset
        predictions = model.predict(X[:10])
        assert len(predictions) == 10

        # Test model size
        param_count = sum(p.numel() for p in model.model.parameters())
        assert param_count < 100000, f"Model too large: {param_count:,} parameters"

    def test_error_handling(self, test_data):
        """Test error handling in data pipeline."""
        # Test invalid input shapes
        with pytest.raises(ValueError):
            engineer = AdvancedFeatureEngineer()
            engineer.transform(np.random.randn(10, 105, 3))  # Wrong feature count

        # Test model with invalid input
        model = get_model("simple_lstm", device="cpu")

        with pytest.raises(ValueError):
            model.predict(np.random.randn(5, 100, 4))  # Wrong time steps

        # Test model without fitting
        with pytest.raises(ValueError):
            model.predict(np.random.randn(5, 105, 4))

    def test_data_quality_metrics(self, test_data):
        """Test data quality and metrics calculation."""
        # Test label distribution
        label_counts = test_data["label"].value_counts()
        assert len(label_counts) == 2, "Should have 2 classes"

        # Check for reasonable class balance
        min_count = label_counts.min()
        max_count = label_counts.max()
        balance_ratio = max_count / min_count

        assert balance_ratio < 3.0, f"Class imbalance too severe: {balance_ratio:.2f}"

        # Test feature quality
        X = np.array([np.array(f) for f in test_data["features"]])

        # Check for NaN values
        assert not np.isnan(X).any(), "Features contain NaN values"

        # Check for infinite values
        assert not np.isinf(X).any(), "Features contain infinite values"

        # Check reasonable feature ranges
        feature_ranges = np.max(X, axis=(0, 1)) - np.min(X, axis=(0, 1))
        assert np.all(feature_ranges > 0), "Some features have zero variance"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
