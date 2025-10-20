"""Data management for LSTM pre-training optimization experiments.

This module provides:
- Experiment-specific augmented dataset generation
- DVC-based data versioning
- Great Expectations quality gates
- RTX 4090-optimized data loading
- OHLC relationship preservation validation

Usage:
    >>> from moola.experiments.data_manager import ExperimentDataManager
    >>> manager = ExperimentDataManager(experiment_id="exp_phase1_timewarp_0.10")
    >>> manager.prepare_experiment(time_warp_sigma=0.10)
    >>> dataloader = manager.get_dataloader(batch_size=512)
"""

import hashlib
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, TensorDataset

from moola.config.training_config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PIN_MEMORY,
    MASKED_LSTM_AUG_JITTER_SIGMA,
    MASKED_LSTM_AUG_VOLATILITY_RANGE,
    OHLC_DIMS,
    WINDOW_SIZE,
)

logger = logging.getLogger(__name__)


@dataclass
class AugmentationConfig:
    """Configuration for temporal augmentation parameters."""

    time_warp_sigma: float
    jitter_sigma: float = MASKED_LSTM_AUG_JITTER_SIGMA
    volatility_range: tuple[float, float] = MASKED_LSTM_AUG_VOLATILITY_RANGE
    num_versions: int = 4  # Number of augmented versions per sample (1 original + 4 aug = 5x)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "time_warp_sigma": self.time_warp_sigma,
            "jitter_sigma": self.jitter_sigma,
            "volatility_range": list(self.volatility_range),
            "num_versions": self.num_versions,
        }

    def get_hash(self) -> str:
        """Generate deterministic hash for cache invalidation."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


@dataclass
class ExperimentMetadata:
    """Metadata for experiment tracking and reproducibility."""

    experiment_id: str
    augmentation_config: AugmentationConfig
    dataset_hash: str
    sample_count: int
    raw_sample_count: int
    created_at: str
    data_version: str = "v1"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "experiment_id": self.experiment_id,
            "augmentation_config": self.augmentation_config.to_dict(),
            "dataset_hash": self.dataset_hash,
            "sample_count": self.sample_count,
            "raw_sample_count": self.raw_sample_count,
            "created_at": self.created_at,
            "data_version": self.data_version,
        }


class OHLCValidator:
    """Validates OHLC relationships and data quality."""

    @staticmethod
    def validate_ohlc_relationships(data: np.ndarray, tolerance: float = 1e-6) -> tuple[bool, list[str]]:
        """Validate OHLC relationships: H >= max(O,C), L <= min(O,C).

        Args:
            data: Array of shape [N, 105, 4] with OHLC features
            tolerance: Numerical tolerance for floating point comparisons

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        # Check shape
        if data.ndim != 3:
            errors.append(f"Expected 3D array, got shape {data.shape}")
            return False, errors

        if data.shape[1] != WINDOW_SIZE:
            errors.append(f"Expected window size {WINDOW_SIZE}, got {data.shape[1]}")

        if data.shape[2] != OHLC_DIMS:
            errors.append(f"Expected {OHLC_DIMS} features (OHLC), got {data.shape[2]}")

        if errors:
            return False, errors

        # Extract OHLC
        open_prices = data[:, :, 0]
        high_prices = data[:, :, 1]
        low_prices = data[:, :, 2]
        close_prices = data[:, :, 3]

        # Check for NaNs
        if np.any(np.isnan(data)):
            nan_count = np.sum(np.isnan(data))
            errors.append(f"Found {nan_count} NaN values in data")

        # Check for Infs
        if np.any(np.isinf(data)):
            inf_count = np.sum(np.isinf(data))
            errors.append(f"Found {inf_count} Inf values in data")

        # Validate: High >= max(Open, Close)
        max_oc = np.maximum(open_prices, close_prices)
        high_violations = np.sum(high_prices < (max_oc - tolerance))
        if high_violations > 0:
            violation_pct = 100 * high_violations / data.size
            errors.append(
                f"High price violations: {high_violations} points "
                f"({violation_pct:.3f}%) where H < max(O,C)"
            )

        # Validate: Low <= min(Open, Close)
        min_oc = np.minimum(open_prices, close_prices)
        low_violations = np.sum(low_prices > (min_oc + tolerance))
        if low_violations > 0:
            violation_pct = 100 * low_violations / data.size
            errors.append(
                f"Low price violations: {low_violations} points "
                f"({violation_pct:.3f}%) where L > min(O,C)"
            )

        is_valid = len(errors) == 0
        return is_valid, errors

    @staticmethod
    def log_validation_results(is_valid: bool, errors: list[str], context: str = ""):
        """Log validation results with appropriate severity."""
        prefix = f"[{context}] " if context else ""

        if is_valid:
            logger.info(f"{prefix}✓ OHLC validation passed")
        else:
            logger.error(f"{prefix}✗ OHLC validation FAILED:")
            for error in errors:
                logger.error(f"{prefix}  - {error}")


class TemporalAugmentor:
    """Applies temporal augmentation while preserving OHLC relationships."""

    def __init__(self, config: AugmentationConfig, seed: int = 42):
        """Initialize augmentor with configuration.

        Args:
            config: Augmentation configuration
            seed: Random seed for reproducibility
        """
        self.config = config
        self.rng = np.random.RandomState(seed)

    def time_warp(self, x: np.ndarray) -> np.ndarray:
        """Apply time warping with configurable sigma.

        Args:
            x: Array of shape [N, 105, 4]

        Returns:
            Time-warped array of same shape
        """
        N, T, F = x.shape
        warped = np.zeros_like(x)

        for i in range(N):
            # Generate smooth warping path
            warp_steps = np.linspace(0, T - 1, num=T)
            warp_noise = self.rng.randn(T) * self.config.time_warp_sigma * T

            # Apply cumulative sum for smooth warping
            warp_noise = np.cumsum(warp_noise)
            warp_noise = warp_noise - np.linspace(warp_noise[0], warp_noise[-1], T)

            warped_indices = warp_steps + warp_noise
            warped_indices = np.clip(warped_indices, 0, T - 1)

            # Interpolate all features
            for f in range(F):
                warped[i, :, f] = np.interp(warp_steps, warped_indices, x[i, :, f])

        return warped

    def add_jitter(self, x: np.ndarray) -> np.ndarray:
        """Add Gaussian jitter noise.

        Args:
            x: Array of shape [N, 105, 4]

        Returns:
            Jittered array of same shape
        """
        noise = self.rng.randn(*x.shape) * self.config.jitter_sigma
        return x + noise

    def scale_volatility(self, x: np.ndarray) -> np.ndarray:
        """Scale volatility (H-L spread) while preserving OHLC relationships.

        Args:
            x: Array of shape [N, 105, 4] with OHLC

        Returns:
            Scaled array with preserved OHLC relationships
        """
        N, T, F = x.shape
        assert F == 4, "Expected OHLC features"

        scaled = x.copy()

        for i in range(N):
            # Sample scaling factor
            scale = self.rng.uniform(*self.config.volatility_range)

            # Compute midpoint for each bar
            midpoint = (x[i, :, 1] + x[i, :, 2]) / 2  # (H + L) / 2

            # Scale deviations from midpoint
            scaled[i, :, 0] = midpoint + (x[i, :, 0] - midpoint) * scale  # Open
            scaled[i, :, 1] = midpoint + (x[i, :, 1] - midpoint) * scale  # High
            scaled[i, :, 2] = midpoint + (x[i, :, 2] - midpoint) * scale  # Low
            scaled[i, :, 3] = midpoint + (x[i, :, 3] - midpoint) * scale  # Close

        return scaled

    def augment(self, x: np.ndarray, apply_time_warp: bool = True) -> np.ndarray:
        """Apply full augmentation pipeline.

        Args:
            x: Array of shape [N, 105, 4]
            apply_time_warp: Whether to apply time warping

        Returns:
            Augmented array of same shape
        """
        # Always apply jitter (50% prob handled by caller)
        augmented = self.add_jitter(x)

        # Apply time warp if enabled (30% prob handled by caller)
        if apply_time_warp:
            augmented = self.time_warp(augmented)

        # Apply volatility scaling (30% prob handled by caller)
        if self.rng.rand() < 0.3:
            augmented = self.scale_volatility(augmented)

        return augmented


class ExperimentDataManager:
    """Manages data lifecycle for LSTM pre-training experiments."""

    def __init__(
        self,
        experiment_id: str,
        project_root: Path | None = None,
        use_dvc: bool = True,
    ):
        """Initialize data manager for experiment.

        Args:
            experiment_id: Unique experiment identifier (e.g., "exp_phase1_timewarp_0.10")
            project_root: Project root directory (defaults to moola repo root)
            use_dvc: Enable DVC tracking for versioning
        """
        self.experiment_id = experiment_id
        self.project_root = project_root or Path(__file__).parents[3]
        self.use_dvc = use_dvc

        # Setup directory structure
        self.data_root = self.project_root / "data"
        self.raw_dir = self.data_root / "raw"
        self.experiments_dir = self.data_root / "experiments"
        self.artifacts_dir = self.data_root / "artifacts" / "pretrained"

        self.experiment_dir = self.experiments_dir / experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Paths for data files
        self.raw_data_path = self.raw_dir / "unlabeled_windows.parquet"
        self.augmented_data_path = self.experiment_dir / "augmented.parquet"
        self.config_path = self.experiment_dir / "config.yaml"
        self.metadata_path = self.experiment_dir / "metadata.json"

        logger.info(f"Initialized experiment: {experiment_id}")
        logger.info(f"  Experiment dir: {self.experiment_dir}")

    def load_raw_data(self) -> np.ndarray:
        """Load raw unlabeled data.

        Returns:
            Array of shape [11873, 105, 4] with OHLC features
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(
                f"Raw data not found: {self.raw_data_path}\n"
                f"Expected unlabeled_windows.parquet with 11,873 samples"
            )

        logger.info(f"Loading raw data from {self.raw_data_path}")
        df = pd.read_parquet(self.raw_data_path)

        # Extract features - stored as nested arrays [[O,H,L,C], [O,H,L,C], ...]
        # Each row has 105 timesteps, each with 4 OHLC values
        features_list = []
        for feature in df["features"]:
            if isinstance(feature[0], np.ndarray):
                # Nested structure: stack to get [105, 4]
                stacked = np.stack(feature)
                features_list.append(stacked)
            else:
                # Already stacked or single dimension
                features_list.append(feature)

        features = np.array(features_list, dtype=np.float32)
        logger.info(f"Loaded {len(features)} raw samples, shape: {features.shape}")

        return features

    def reshape_to_ohlc(self, features: np.ndarray) -> np.ndarray:
        """Reshape features to OHLC format (if needed).

        Args:
            features: Array from load_raw_data()

        Returns:
            Array of shape [N, 105, 4] with OHLC features

        Note:
            Data is already loaded in OHLC format [N, 105, 4] from load_raw_data(),
            so this method just validates the shape.
        """
        if features.ndim == 3 and features.shape[1:] == (WINDOW_SIZE, OHLC_DIMS):
            # Already in correct OHLC format [N, 105, 4]
            logger.info(f"Data already in OHLC format: {features.shape}")
            return features
        elif features.ndim == 2 and features.shape[1] == WINDOW_SIZE * OHLC_DIMS:
            # Flattened OHLC: reshape [N, 420] → [N, 105, 4]
            N = features.shape[0]
            reshaped = features.reshape(N, WINDOW_SIZE, OHLC_DIMS)
            logger.info(f"Reshaped flattened OHLC: {features.shape} → {reshaped.shape}")
            return reshaped
        else:
            raise ValueError(
                f"Unexpected features shape: {features.shape}. "
                f"Expected [N, {WINDOW_SIZE}, {OHLC_DIMS}] or [N, {WINDOW_SIZE * OHLC_DIMS}]"
            )

    def prepare_experiment(
        self,
        time_warp_sigma: float,
        jitter_sigma: float = MASKED_LSTM_AUG_JITTER_SIGMA,
        volatility_range: tuple[float, float] = MASKED_LSTM_AUG_VOLATILITY_RANGE,
        num_versions: int = 4,
        force_regenerate: bool = False,
        seed: int = 42,
    ) -> ExperimentMetadata:
        """Prepare augmented dataset for experiment.

        Args:
            time_warp_sigma: Time warping magnitude (0.10-0.30)
            jitter_sigma: Jitter noise magnitude
            volatility_range: Volatility scaling range
            num_versions: Number of augmented versions per sample
            force_regenerate: Regenerate even if cached data exists
            seed: Random seed for reproducibility

        Returns:
            Experiment metadata
        """
        # Create augmentation config
        aug_config = AugmentationConfig(
            time_warp_sigma=time_warp_sigma,
            jitter_sigma=jitter_sigma,
            volatility_range=volatility_range,
            num_versions=num_versions,
        )

        # Check if data already exists
        if self.augmented_data_path.exists() and not force_regenerate:
            logger.info(f"Loading cached augmented data from {self.augmented_data_path}")
            df = pd.read_parquet(self.augmented_data_path)
            metadata = self._load_metadata()

            # Verify config matches
            cached_config = AugmentationConfig(**metadata["augmentation_config"])
            if cached_config.to_dict() != aug_config.to_dict():
                logger.warning(
                    "Cached data config mismatch - regenerating with new parameters"
                )
            else:
                logger.info(f"Using cached data: {len(df)} samples")
                return ExperimentMetadata(**metadata)

        # Generate augmented data
        logger.info("Generating augmented dataset...")
        start_time = time.time()

        # Load raw data
        raw_features = self.load_raw_data()
        raw_count = len(raw_features)

        # Reshape to OHLC format
        raw_ohlc = self.reshape_to_ohlc(raw_features)

        # Pre-augmentation validation
        is_valid, errors = OHLCValidator.validate_ohlc_relationships(raw_ohlc)
        OHLCValidator.log_validation_results(is_valid, errors, "PRE-AUGMENTATION")

        if not is_valid and not force_regenerate:
            raise ValueError("Raw data failed OHLC validation. Use force_regenerate=True to proceed anyway.")

        # Apply augmentation
        augmentor = TemporalAugmentor(aug_config, seed=seed)
        augmented_samples = [raw_ohlc]  # Include original data

        for version_idx in range(num_versions):
            logger.info(f"  Generating augmentation version {version_idx + 1}/{num_versions}")

            # Apply augmentation with probabilities
            apply_time_warp = np.random.rand() < 0.3  # 30% probability
            augmented = augmentor.augment(raw_ohlc, apply_time_warp=apply_time_warp)
            augmented_samples.append(augmented)

        # Combine all versions
        all_data = np.concatenate(augmented_samples, axis=0)
        total_samples = len(all_data)

        logger.info(f"Generated {total_samples} total samples ({raw_count} × {num_versions + 1})")

        # Post-augmentation validation
        is_valid, errors = OHLCValidator.validate_ohlc_relationships(all_data)
        OHLCValidator.log_validation_results(is_valid, errors, "POST-AUGMENTATION")

        # Save augmented data
        self._save_augmented_data(all_data)

        # Create metadata
        dataset_hash = self._compute_dataset_hash(all_data)
        metadata = ExperimentMetadata(
            experiment_id=self.experiment_id,
            augmentation_config=aug_config,
            dataset_hash=dataset_hash,
            sample_count=total_samples,
            raw_sample_count=raw_count,
            created_at=pd.Timestamp.now().isoformat(),
        )

        # Save metadata and config
        self._save_metadata(metadata)
        self._save_config(aug_config)

        elapsed = time.time() - start_time
        logger.info(f"✓ Dataset preparation complete in {elapsed:.1f}s")
        logger.info(f"  Samples: {total_samples} ({raw_count} → {total_samples})")
        logger.info(f"  Shape: {all_data.shape}")
        logger.info(f"  Hash: {dataset_hash}")

        # Track with DVC if enabled
        if self.use_dvc:
            self._track_with_dvc()

        return metadata

    def get_dataloader(
        self,
        batch_size: int = DEFAULT_BATCH_SIZE,
        num_workers: int = DEFAULT_NUM_WORKERS,
        pin_memory: bool = DEFAULT_PIN_MEMORY,
        shuffle: bool = True,
    ) -> DataLoader:
        """Get optimized DataLoader for RTX 4090.

        Args:
            batch_size: Batch size (default: 512 for 24GB VRAM)
            num_workers: Number of worker threads (default: 8)
            pin_memory: Enable pinned memory for faster GPU transfer
            shuffle: Shuffle data

        Returns:
            PyTorch DataLoader optimized for RTX 4090
        """
        if not self.augmented_data_path.exists():
            raise FileNotFoundError(
                f"Augmented data not found: {self.augmented_data_path}\n"
                f"Run prepare_experiment() first"
            )

        logger.info(f"Creating DataLoader for {self.experiment_id}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Num workers: {num_workers}")
        logger.info(f"  Pin memory: {pin_memory}")

        # Load augmented data
        df = pd.read_parquet(self.augmented_data_path)
        features = np.array(df["features"].tolist())

        # Convert to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)

        # Create dataset
        dataset = TensorDataset(features_tensor)

        # Create dataloader with RTX 4090 optimizations
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=num_workers > 0,  # Keep workers alive
            prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
        )

        logger.info(f"✓ DataLoader ready: {len(dataset)} samples, {len(dataloader)} batches")

        return dataloader

    def benchmark_loading(self, num_iterations: int = 10) -> dict[str, float]:
        """Benchmark data loading performance.

        Args:
            num_iterations: Number of iterations to average

        Returns:
            Dictionary with timing metrics
        """
        logger.info(f"Benchmarking data loading ({num_iterations} iterations)...")

        dataloader = self.get_dataloader()

        # Warmup
        for _ in dataloader:
            break

        # Benchmark full epoch
        epoch_times = []
        for iteration in range(num_iterations):
            start = time.time()
            for batch in dataloader:
                pass  # Just iterate
            epoch_times.append(time.time() - start)

        results = {
            "mean_epoch_time": np.mean(epoch_times),
            "std_epoch_time": np.std(epoch_times),
            "min_epoch_time": np.min(epoch_times),
            "samples_per_second": len(dataloader.dataset) / np.mean(epoch_times),
        }

        logger.info("Benchmark results:")
        logger.info(f"  Mean epoch time: {results['mean_epoch_time']:.3f}s")
        logger.info(f"  Throughput: {results['samples_per_second']:.0f} samples/s")

        return results

    def _save_augmented_data(self, data: np.ndarray):
        """Save augmented data to parquet."""
        # Flatten to [N, 105*4] for storage
        N, T, F = data.shape
        flattened = data.reshape(N, T * F)

        df = pd.DataFrame({
            "window_id": np.arange(N),
            "features": list(flattened),
        })

        df.to_parquet(self.augmented_data_path, index=False)
        logger.info(f"Saved augmented data: {self.augmented_data_path}")

    def _save_config(self, config: AugmentationConfig):
        """Save augmentation config to YAML."""
        with open(self.config_path, "w") as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
        logger.info(f"Saved config: {self.config_path}")

    def _save_metadata(self, metadata: ExperimentMetadata):
        """Save experiment metadata to JSON."""
        with open(self.metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        logger.info(f"Saved metadata: {self.metadata_path}")

    def _load_metadata(self) -> dict:
        """Load experiment metadata."""
        with open(self.metadata_path) as f:
            return json.load(f)

    def _compute_dataset_hash(self, data: np.ndarray) -> str:
        """Compute deterministic hash for dataset."""
        # Use first 1000 samples for speed
        sample_data = data[:1000] if len(data) > 1000 else data
        data_bytes = sample_data.tobytes()
        return hashlib.sha256(data_bytes).hexdigest()[:16]

    def _track_with_dvc(self):
        """Track augmented data with DVC."""
        try:
            import subprocess

            # Check if DVC is initialized
            dvc_dir = self.project_root / ".dvc"
            if not dvc_dir.exists():
                logger.warning("DVC not initialized - skipping tracking")
                return

            # Add to DVC
            subprocess.run(
                ["dvc", "add", str(self.augmented_data_path)],
                cwd=self.project_root,
                check=True,
                capture_output=True,
            )
            logger.info(f"✓ Tracked with DVC: {self.augmented_data_path}")

        except (ImportError, subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.warning(f"DVC tracking failed: {e}")
