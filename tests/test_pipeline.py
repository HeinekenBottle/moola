"""Tests for moola pipeline: ingest, train, evaluate."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_schema_validation():
    """Test TrainingDataRow schema validates correctly."""
    from moola.schema import TrainingDataRow

    # Valid row
    valid_data = {"window_id": 1, "label": "class_A", "features": [0.1, 0.2, 0.3]}
    row = TrainingDataRow(**valid_data)
    assert row.window_id == 1
    assert row.label == "class_A"
    assert row.features == [0.1, 0.2, 0.3]

    # Test with numpy array (should convert to list)
    valid_data_np = {"window_id": 2, "label": "class_B", "features": np.array([1.0, 2.0])}
    row_np = TrainingDataRow(**valid_data_np)
    assert isinstance(row_np.features, list)

    # Invalid: non-numeric features
    with pytest.raises(ValueError):
        TrainingDataRow(window_id=3, label="class_A", features=["not", "numeric"])


def test_train_saves_model(tmp_path):
    """Test that train command saves model.bin."""
    import os
    import pickle
    import subprocess

    # Set up temp directories
    data_dir = tmp_path / "data"
    artifacts_dir = data_dir / "artifacts"
    logs_dir = data_dir / "logs"
    processed_dir = data_dir / "processed"

    for d in [artifacts_dir, logs_dir, processed_dir]:
        d.mkdir(parents=True)

    # Create dummy train.parquet
    df = pd.DataFrame(
        {
            "window_id": list(range(100)),
            "label": ["class_A" if i % 2 == 0 else "class_B" for i in range(100)],
            "features": [np.random.randn(5).tolist() for _ in range(100)],
        }
    )
    train_path = processed_dir / "train.parquet"
    df.to_parquet(train_path, index=False)

    # Set env vars to use temp dirs
    env = os.environ.copy()
    env["MOOLA_DATA_DIR"] = str(data_dir)
    env["MOOLA_ARTIFACTS_DIR"] = str(artifacts_dir)
    env["MOOLA_LOG_DIR"] = str(logs_dir)

    # Run train command
    result = subprocess.run(["moola", "train"], env=env, capture_output=True, text=True)

    # Check model.bin was created
    model_path = artifacts_dir / "model.bin"
    assert model_path.exists(), f"model.bin not found at {model_path}"

    # Verify it's a valid pickle file
    with open(model_path, "rb") as f:
        model = pickle.load(f)
        assert hasattr(model, "predict"), "Saved model should have predict method"


def test_evaluate_writes_metrics(tmp_path):
    """Test that evaluate writes metrics.json with required keys."""
    import os
    import pickle
    import subprocess

    from sklearn.linear_model import LogisticRegression

    # Set up temp directories
    data_dir = tmp_path / "data"
    artifacts_dir = data_dir / "artifacts"
    logs_dir = data_dir / "logs"
    processed_dir = data_dir / "processed"

    for d in [artifacts_dir, logs_dir, processed_dir]:
        d.mkdir(parents=True)

    # Create dummy train.parquet
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "window_id": list(range(100)),
            "label": ["class_A" if i % 2 == 0 else "class_B" for i in range(100)],
            "features": [np.random.randn(5).tolist() for _ in range(100)],
        }
    )
    train_path = processed_dir / "train.parquet"
    df.to_parquet(train_path, index=False)

    # Create dummy model.bin
    X = np.random.randn(100, 5)
    y = np.array(["class_A" if i % 2 == 0 else "class_B" for i in range(100)])
    model = LogisticRegression(random_state=42, max_iter=100)
    model.fit(X, y)

    model_path = artifacts_dir / "model.bin"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # Set env vars
    env = os.environ.copy()
    env["MOOLA_DATA_DIR"] = str(data_dir)
    env["MOOLA_ARTIFACTS_DIR"] = str(artifacts_dir)
    env["MOOLA_LOG_DIR"] = str(logs_dir)

    # Run evaluate command
    result = subprocess.run(["moola", "evaluate"], env=env, capture_output=True, text=True)

    # Check metrics.json exists
    metrics_path = artifacts_dir / "metrics.json"
    assert metrics_path.exists(), f"metrics.json not found at {metrics_path}"

    # Verify required keys
    with open(metrics_path) as f:
        metrics = json.load(f)

    required_keys = {"accuracy", "f1", "timestamp"}
    for key in required_keys:
        assert key in metrics, f"Missing required key '{key}' in metrics.json"

    # Verify metrics are valid numbers
    assert 0.0 <= metrics["accuracy"] <= 1.0, "accuracy should be between 0 and 1"
    assert 0.0 <= metrics["f1"] <= 1.0, "f1 should be between 0 and 1"
    assert isinstance(metrics["timestamp"], str), "timestamp should be a string"
