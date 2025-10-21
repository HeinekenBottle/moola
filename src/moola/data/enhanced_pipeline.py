"""Enhanced data pipeline for 11-dimensional feature integration.

Provides backward-compatible data loading and transformation for both
4D OHLC and 11D relative features with automatic feature generation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from moola.data_infra.schemas_11d import (
    DataStage,
    EnhancedLabeledDataset,
    EnhancedLabeledWindow,
    EnhancedTimeSeriesWindow,
    EnhancedUnlabeledDataset,
    FeatureDimension,
)
from moola.features.relative_transform import RelativeFeatureTransform


class EnhancedDataPipeline:
    """Enhanced data pipeline supporting 4D OHLC and 11D relative features."""

    def __init__(self, eps: float = 1e-8):
        """Initialize the enhanced data pipeline.

        Args:
            eps: Small constant for numerical stability in relative transforms
        """
        self.relative_transform = RelativeFeatureTransform(eps=eps)
        self.eps = eps

    def load_raw_data(self, data_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load raw data and detect feature dimensionality.

        Args:
            data_path: Path to parquet file containing window data

        Returns:
            Tuple of (ohlc_data, relative_data) where:
            - ohlc_data: [N, 105, 4] OHLC data (always present)
            - relative_data: [N, 105, 11] relative features (optional)
        """
        logger.info(f"Loading data from {data_path}")

        # Load parquet data
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} samples with columns: {list(df.columns)}")

        # Extract OHLC features (always present)
        if "features" in df.columns:
            # Features stored as nested arrays
            features_list = df["features"].tolist()

            # Check if features are 4D OHLC or 11D enhanced
            first_sample = features_list[0]
            if isinstance(first_sample, np.ndarray):
                if first_sample.shape == (105, 4):
                    # Pure 4D OHLC data
                    ohlc_data = np.array(features_list, dtype=np.float32)
                    relative_data = None
                    feature_dim = FeatureDimension.OHLC_4D
                elif first_sample.shape == (105, 11):
                    # Pure 11D relative features (need to extract OHLC)
                    relative_data = np.array(features_list, dtype=np.float32)
                    ohlc_data = self._reconstruct_ohlc_from_relative(relative_data)
                    feature_dim = FeatureDimension.RELATIVE_11D
                elif first_sample.shape == (105, 15):
                    # Enhanced 15D data (4 OHLC + 11 relative)
                    enhanced_data = np.array(features_list, dtype=np.float32)
                    ohlc_data = enhanced_data[:, :, :4]
                    relative_data = enhanced_data[:, :, 4:]
                    feature_dim = FeatureDimension.DUAL_INPUT
                else:
                    raise ValueError(f"Unexpected feature shape: {first_sample.shape}")
            else:
                # List of lists format
                if len(first_sample[0]) == 4:
                    # 4D OHLC data
                    ohlc_data = np.array(features_list, dtype=np.float32)
                    relative_data = None
                    feature_dim = FeatureDimension.OHLC_4D
                elif len(first_sample[0]) == 11:
                    # 11D relative features
                    relative_data = np.array(features_list, dtype=np.float32)
                    ohlc_data = self._reconstruct_ohlc_from_relative(relative_data)
                    feature_dim = FeatureDimension.RELATIVE_11D
                else:
                    raise ValueError(f"Unexpected feature dimension: {len(first_sample[0])}")
        else:
            raise ValueError("Data must contain 'features' column")

        logger.info(f"Detected feature dimension: {feature_dim}")
        logger.info(f"OHLC data shape: {ohlc_data.shape}")
        if relative_data is not None:
            logger.info(f"Relative data shape: {relative_data.shape}")

        return ohlc_data, relative_data

    def _reconstruct_ohlc_from_relative(self, relative_data: np.ndarray) -> np.ndarray:
        """Reconstruct approximate OHLC data from relative features.

        This is a lossy reconstruction used when only relative features are available.
        For production use, store original OHLC data alongside relative features.

        Args:
            relative_data: [N, 105, 11] relative features

        Returns:
            Approximate OHLC data [N, 105, 4]
        """
        logger.warning("Reconstructing OHLC from relative features - this is approximate")

        N, T, F = relative_data.shape
        ohlc_data = np.zeros((N, T, 4), dtype=np.float32)

        for i in range(N):
            for t in range(T):
                rel_features = relative_data[i, t]

                # Extract log returns (first 4 features)
                log_returns = rel_features[:4]

                # Reconstruct approximate prices starting from 100
                if t == 0:
                    base_price = 100.0
                    ohlc_data[i, t] = [base_price] * 4
                else:
                    prev_ohlc = ohlc_data[i, t - 1]
                    # Apply log returns to previous close
                    new_close = prev_ohlc[3] * np.exp(log_returns[3])

                    # Approximate other OHLC values based on typical patterns
                    # This is a simplified reconstruction
                    body_size = abs(new_close - prev_ohlc[3])
                    ohlc_data[i, t] = [
                        prev_ohlc[3],  # open = previous close
                        max(new_close, prev_ohlc[3]) + body_size * 0.1,  # high
                        min(new_close, prev_ohlc[3]) - body_size * 0.1,  # low
                        new_close,  # close
                    ]

        return ohlc_data

    def ensure_11d_features(
        self, ohlc_data: np.ndarray, relative_data: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Ensure 11D relative features are available.

        Args:
            ohlc_data: [N, 105, 4] OHLC data
            relative_data: Optional [N, 105, 11] relative features

        Returns:
            [N, 105, 11] relative features
        """
        if relative_data is not None:
            logger.info("Using provided 11D relative features")
            return relative_data

        logger.info("Generating 11D relative features from OHLC data")
        relative_data = self.relative_transform.transform(ohlc_data)
        logger.info(f"Generated relative features: {relative_data.shape}")

        return relative_data

    def create_enhanced_dataset(
        self,
        data_path: Path,
        labels: Optional[np.ndarray] = None,
        expansion_start: Optional[np.ndarray] = None,
        expansion_end: Optional[np.ndarray] = None,
        window_ids: Optional[list] = None,
    ) -> Union[EnhancedUnlabeledDataset, EnhancedLabeledDataset]:
        """Create enhanced dataset with automatic feature dimension detection.

        Args:
            data_path: Path to parquet data file
            labels: Optional label array [N]
            expansion_start: Optional expansion start indices [N]
            expansion_end: Optional expansion end indices [N]
            window_ids: Optional window ID list [N]

        Returns:
            Enhanced dataset (unlabeled or labeled based on inputs)
        """
        # Load raw data
        ohlc_data, relative_data = self.load_raw_data(data_path)

        # Ensure 11D features are available
        relative_data = self.ensure_11d_features(ohlc_data, relative_data)

        # Generate window IDs if not provided
        N = ohlc_data.shape[0]
        if window_ids is None:
            window_ids = [f"window_{i:06d}" for i in range(N)]

        # Determine feature dimension
        if relative_data is not None:
            feature_dim = FeatureDimension.DUAL_INPUT
        else:
            feature_dim = FeatureDimension.OHLC_4D

        # Create windows
        if labels is not None:
            # Labeled dataset
            if expansion_start is None or expansion_end is None:
                raise ValueError("expansion_start and expansion_end required for labeled dataset")

            windows = []
            label_distribution = {}

            for i in range(N):
                window = EnhancedLabeledWindow(
                    window_id=window_ids[i],
                    feature_dimension=feature_dim,
                    ohlc_features=ohlc_data[i].tolist(),
                    relative_features=(
                        relative_data[i].tolist() if relative_data is not None else None
                    ),
                    label=labels[i],
                    expansion_start=int(expansion_start[i]),
                    expansion_end=int(expansion_end[i]),
                    symbol="NQ",
                    start_timestamp=None,
                    end_timestamp=None,
                )
                windows.append(window)

                # Track label distribution
                label_str = str(labels[i])
                label_distribution[label_str] = label_distribution.get(label_str, 0) + 1

            dataset = EnhancedLabeledDataset(
                windows=windows,
                total_samples=N,
                label_distribution=label_distribution,
                feature_dimension=feature_dim,
                data_stage=DataStage.PROCESSED,
                metadata={"source_file": str(data_path)},
            )

        else:
            # Unlabeled dataset
            windows = []
            for i in range(N):
                window = EnhancedTimeSeriesWindow(
                    window_id=window_ids[i],
                    feature_dimension=feature_dim,
                    ohlc_features=ohlc_data[i].tolist(),
                    relative_features=(
                        relative_data[i].tolist() if relative_data is not None else None
                    ),
                    symbol="NQ",
                    start_timestamp=None,
                    end_timestamp=None,
                )
                windows.append(window)

            dataset = EnhancedUnlabeledDataset(
                windows=windows,
                total_samples=N,
                feature_dimension=feature_dim,
                data_stage=DataStage.PROCESSED,
                metadata={"source_file": str(data_path)},
            )

        logger.info(f"Created enhanced dataset: {type(dataset).__name__} with {N} samples")
        return dataset

    def save_enhanced_dataset(
        self, dataset: Union[EnhancedUnlabeledDataset, EnhancedLabeledDataset], output_path: Path
    ) -> None:
        """Save enhanced dataset to parquet format.

        Args:
            dataset: Enhanced dataset to save
            output_path: Output parquet file path
        """
        logger.info(f"Saving enhanced dataset to {output_path}")

        # Convert to pandas DataFrame
        rows = []
        for window in dataset.windows:
            row = {
                "window_id": window.window_id,
                "feature_dimension": window.feature_dimension.value,
                "ohlc_features": window.ohlc_features,
                "relative_features": window.relative_features,
                "symbol": window.symbol,
                "start_timestamp": window.start_timestamp,
                "end_timestamp": window.end_timestamp,
            }

            # Add label information if available
            if isinstance(window, EnhancedLabeledWindow):
                row.update(
                    {
                        "label": window.label.value,
                        "expansion_start": window.expansion_start,
                        "expansion_end": window.expansion_end,
                        "confidence": window.confidence,
                    }
                )

            rows.append(row)

        df = pd.DataFrame(rows)

        # Save to parquet
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine="pyarrow")

        logger.info(f"Saved {len(df)} samples to {output_path}")

    def get_feature_statistics(self, dataset: EnhancedUnlabeledDataset) -> Dict[str, Any]:
        """Compute comprehensive feature statistics for quality monitoring.

        Args:
            dataset: Enhanced dataset

        Returns:
            Dictionary of feature statistics
        """
        ohlc_data = dataset.to_ohlc_numpy()
        relative_data = dataset.to_relative_numpy()

        stats = {
            "total_samples": len(dataset.windows),
            "feature_dimension": dataset.feature_dimension.value,
            "ohlc_stats": {
                "shape": ohlc_data.shape,
                "mean": float(ohlc_data.mean()),
                "std": float(ohlc_data.std()),
                "min": float(ohlc_data.min()),
                "max": float(ohlc_data.max()),
                "missing_values": int(np.isnan(ohlc_data).sum()),
                "inf_values": int(np.isinf(ohlc_data).sum()),
            },
        }

        if relative_data is not None:
            stats["relative_stats"] = {
                "shape": relative_data.shape,
                "mean": float(relative_data.mean()),
                "std": float(relative_data.std()),
                "min": float(relative_data.min()),
                "max": float(relative_data.max()),
                "missing_values": int(np.isnan(relative_data).sum()),
                "inf_values": int(np.isinf(relative_data).sum()),
            }

        return stats


# Convenience function for backward compatibility
def load_enhanced_data(
    data_path: Path, ensure_11d: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load data with optional 11D feature generation.

    Args:
        data_path: Path to parquet file
        ensure_11d: Whether to generate 11D features if not present

    Returns:
        Tuple of (ohlc_data, relative_data)
    """
    pipeline = EnhancedDataPipeline()
    ohlc_data, relative_data = pipeline.load_raw_data(data_path)

    if ensure_11d and relative_data is None:
        relative_data = pipeline.ensure_11d_features(ohlc_data)

    return ohlc_data, relative_data
