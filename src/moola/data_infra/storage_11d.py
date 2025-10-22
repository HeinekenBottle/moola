"""Minimal Stones-only data loading helpers for 11×T parquet datasets."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch


def _decode_feature(sample: Any) -> np.ndarray:
    """Convert stored feature payloads into a float32 array."""
    if isinstance(sample, bytes):
        sample = pickle.loads(sample)
    array = np.asarray(sample, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D feature array, got shape {array.shape}")
    return array


def load_dataset(parquet_path: str | Path) -> Dict[str, np.ndarray]:
    """Load Stones training data from parquet.

    The parquet file is expected to contain:
    - ``features``: serialized (T, 11) arrays (bytes or nested lists)
    - ``label``: integer class labels
    - ``ptr_start`` / ``ptr_end`` (or legacy ``expansion_start`` / ``expansion_end``)
    """
    table = pq.read_table(parquet_path)
    data = table.to_pydict()

    if "features" not in data:
        raise ValueError("Parquet file must contain a 'features' column")

    features = [_decode_feature(sample) for sample in data["features"]]
    X = np.stack(features, axis=0).astype(np.float32)

    if "label" not in data:
        raise ValueError("Parquet file must contain a 'label' column")
    y = np.asarray(data["label"], dtype=np.int64)

    start_key = "ptr_start" if "ptr_start" in data else "expansion_start"
    end_key = "ptr_end" if "ptr_end" in data else "expansion_end"
    if start_key not in data or end_key not in data:
        raise ValueError("Parquet file must provide pointer boundaries")

    ptr_start = np.asarray(data[start_key], dtype=np.int64)
    ptr_end = np.asarray(data[end_key], dtype=np.int64)

    return {"X": X, "y": y, "ptr_start": ptr_start, "ptr_end": ptr_end}


def prepare_inputs(batch: Dict[str, np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a loaded Stones batch into tensors."""
    X = torch.from_numpy(batch["X"]).contiguous()
    y = torch.from_numpy(batch["y"]).long()
    pointers = torch.from_numpy(
        np.stack([batch["ptr_start"], batch["ptr_end"]], axis=1).astype(np.int64)
    )
    return X, y, pointers


class StonesDataProcessor:
    """Compatibility wrapper that mimics the legacy dual-input processor."""

    def process_training_data(
        self, df: pd.DataFrame, enable_engineered_features: bool = False
    ) -> Dict[str, Any]:
        if "features" not in df.columns:
            raise ValueError("DataFrame must contain a 'features' column")

        features = [_decode_feature(sample) for sample in df["features"]]
        X = np.stack(features, axis=0).astype(np.float32)

        labels_array = df["label"].to_numpy()
        if np.issubdtype(labels_array.dtype, np.number):
            y = labels_array.astype(np.int64)
        else:
            codes, _ = pd.factorize(df["label"], sort=True)
            y = codes.astype(np.int64)

        start_col = "ptr_start" if "ptr_start" in df.columns else "expansion_start"
        end_col = "ptr_end" if "ptr_end" in df.columns else "expansion_end"
        if start_col not in df.columns or end_col not in df.columns:
            raise ValueError("DataFrame must include pointer boundaries")

        expansion_start = df[start_col].to_numpy(dtype=np.int64)
        expansion_end = df[end_col].to_numpy(dtype=np.int64)

        return {
            "X": X,
            "X_ohlc": X,
            "X_engineered": None,
            "feature_names": [],
            "expansion_start": expansion_start,
            "expansion_end": expansion_end,
            "y": y,
            "metadata": {"augmentation_metadata": {}},
        }

    @staticmethod
    def get_feature_statistics(_features: np.ndarray | None):
        return None


def create_dual_input_processor(**_: Any) -> StonesDataProcessor:
    """Return a minimal processor compatible with the old CLI."""
    return StonesDataProcessor()


def prepare_model_inputs(
    processed_data: Dict[str, Any],
    model_type: str | None = None,
    use_engineered_features: bool = False,
) -> Dict[str, Any]:
    """Shim that exposes the subset of keys required by the CLI."""
    return {
        "X": processed_data["X"],
        "y": processed_data["y"],
        "expansion_start": processed_data.get("expansion_start"),
        "expansion_end": processed_data.get("expansion_end"),
    }
