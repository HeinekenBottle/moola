"""Integration of pseudo-sample generation with the existing Moola training pipeline.

This module provides seamless integration of advanced pseudo-sample generation
strategies with the existing training infrastructure, enabling automatic dataset
augmentation during model training.

Key features:
1. Automatic pseudo-sample generation during training
2. Quality-controlled sample integration
3. Dynamic dataset expansion
4. Performance monitoring and adaptation
5. Memory-efficient batch processing
"""

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .pseudo_sample_generation import PseudoSampleGenerationPipeline
from .pseudo_sample_validation import FinancialDataValidator, ValidationReport


@dataclass
class AugmentationConfig:
    """Configuration for pseudo-sample augmentation during training."""

    enable_augmentation: bool = True
    augmentation_ratio: float = 2.0  # Target ratio of total/original samples
    quality_threshold: float = 0.7
    validation_frequency: int = 100  # Validate every N batches
    max_memory_usage_gb: float = 4.0
    strategy_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "temporal_augmentation": 0.25,
            "pattern_synthesis": 0.25,
            "statistical_simulation": 0.2,
            "market_condition": 0.3,
        }
    )
    enable_self_supervised: bool = False
    self_supervised_confidence: float = 0.95


class AugmentedDataset(Dataset):
    """Dataset wrapper that includes pseudo-samples with original data."""

    def __init__(
        self,
        original_data: np.ndarray,
        original_labels: np.ndarray,
        pseudo_data: Optional[np.ndarray] = None,
        pseudo_labels: Optional[np.ndarray] = None,
        mix_ratio: float = 0.5,
    ):
        """Initialize augmented dataset.

        Args:
            original_data: Original OHLC data [N, T, 4]
            original_labels: Original labels [N]
            pseudo_data: Generated pseudo-samples [M, T, 4]
            pseudo_labels: Pseudo-sample labels [M]
            mix_ratio: Ratio of pseudo to original samples in each batch
        """
        self.original_data = original_data
        self.original_labels = original_labels
        self.pseudo_data = (
            pseudo_data
            if pseudo_data is not None
            else np.array([]).reshape(0, original_data.shape[1], 4)
        )
        self.pseudo_labels = pseudo_labels if pseudo_labels is not None else np.array([])
        self.mix_ratio = mix_ratio

        self.total_original = len(original_data)
        self.total_pseudo = len(self.pseudo_data)

        # Create indices for both datasets
        self.original_indices = np.arange(self.total_original)
        self.pseudo_indices = np.arange(self.total_pseudo)

    def __len__(self) -> int:
        """Return total dataset size."""
        return self.total_original + self.total_pseudo

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset.

        Args:
            idx: Sample index

        Returns:
            Tuple of (data, label)
        """
        if idx < self.total_original:
            # Return original sample
            return (
                torch.FloatTensor(self.original_data[idx]),
                torch.tensor(self.original_labels[idx], dtype=torch.long),
            )
        else:
            # Return pseudo-sample
            pseudo_idx = idx - self.total_original
            return (
                torch.FloatTensor(self.pseudo_data[pseudo_idx]),
                torch.tensor(self.pseudo_labels[pseudo_idx], dtype=torch.long),
            )

    def update_pseudo_samples(self, pseudo_data: np.ndarray, pseudo_labels: np.ndarray):
        """Update pseudo-samples in the dataset.

        Args:
            pseudo_data: New pseudo-samples
            pseudo_labels: New pseudo-labels
        """
        self.pseudo_data = pseudo_data
        self.pseudo_labels = pseudo_labels
        self.total_pseudo = len(pseudo_data)
        self.pseudo_indices = np.arange(self.total_pseudo)

    def get_original_subset(self) -> Dataset:
        """Get dataset with only original samples."""
        return torch.utils.data.TensorDataset(
            torch.FloatTensor(self.original_data),
            torch.tensor(self.original_labels, dtype=torch.long),
        )


class DynamicAugmentedDataset(Dataset):
    """Dynamic dataset that generates pseudo-samples on-the-fly."""

    def __init__(
        self,
        original_data: np.ndarray,
        original_labels: np.ndarray,
        generator: PseudoSampleGenerationPipeline,
        config: AugmentationConfig,
        cache_size: int = 1000,
    ):
        """Initialize dynamic augmented dataset.

        Args:
            original_data: Original OHLC data [N, T, 4]
            original_labels: Original labels [N]
            generator: Pseudo-sample generation pipeline
            config: Augmentation configuration
            cache_size: Number of pseudo-samples to cache
        """
        self.original_data = original_data
        self.original_labels = original_labels
        self.generator = generator
        self.config = config
        self.cache_size = cache_size

        self.total_original = len(original_data)
        self.original_indices = np.arange(self.total_original)

        # Cache for generated samples
        self.pseudo_cache = {
            "data": np.array([]).reshape(0, original_data.shape[1], 4),
            "labels": np.array([]),
            "generation_count": 0,
        }

        # Statistics
        self.stats = {"total_generated": 0, "total_used": 0, "cache_hits": 0, "cache_misses": 0}

    def __len__(self) -> int:
        """Return effective dataset size."""
        return int(self.total_original * (1 + self.config.augmentation_ratio))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item from dataset with dynamic generation.

        Args:
            idx: Sample index

        Returns:
            Tuple of (data, label)
        """
        if idx < self.total_original:
            # Return original sample
            return (
                torch.FloatTensor(self.original_data[idx]),
                torch.tensor(self.original_labels[idx], dtype=torch.long),
            )
        else:
            # Generate or return cached pseudo-sample
            pseudo_idx = idx % max(self.cache_size, 1)

            if pseudo_idx < len(self.pseudo_cache["data"]):
                # Return cached sample
                self.stats["cache_hits"] += 1
                data = self.pseudo_cache["data"][pseudo_idx]
                label = self.pseudo_cache["labels"][pseudo_idx]
                self.stats["total_used"] += 1
            else:
                # Generate new sample
                self.stats["cache_misses"] += 1
                new_data, new_labels, _ = self.generator.generate_samples(
                    self.original_data, self.original_labels, 1
                )

                if len(new_data) > 0:
                    # Add to cache
                    if len(self.pseudo_cache["data"]) == 0:
                        self.pseudo_cache["data"] = new_data
                        self.pseudo_cache["labels"] = new_labels
                    else:
                        self.pseudo_cache["data"] = np.vstack([self.pseudo_cache["data"], new_data])
                        self.pseudo_cache["labels"] = np.hstack(
                            [self.pseudo_cache["labels"], new_labels]
                        )

                    data = new_data[0]
                    label = new_labels[0]
                    self.stats["total_generated"] += 1
                    self.stats["total_used"] += 1
                else:
                    # Fallback to original sample
                    orig_idx = np.random.randint(0, self.total_original)
                    return (
                        torch.FloatTensor(self.original_data[orig_idx]),
                        torch.tensor(self.original_labels[orig_idx], dtype=torch.long),
                    )

            return (torch.FloatTensor(data), torch.tensor(label, dtype=torch.long))

    def get_statistics(self) -> Dict[str, Any]:
        """Get generation statistics.

        Returns:
            Dictionary of statistics
        """
        return self.stats.copy()

    def clear_cache(self):
        """Clear the pseudo-sample cache."""
        self.pseudo_cache = {
            "data": np.array([]).reshape(0, self.original_data.shape[1], 4),
            "labels": np.array([]),
            "generation_count": 0,
        }


class TrainingPipelineIntegrator:
    """Integrates pseudo-sample generation with the training pipeline."""

    def __init__(self, config: AugmentationConfig, logger: Optional[logging.Logger] = None):
        """Initialize training pipeline integrator.

        Args:
            config: Augmentation configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

        # Initialize components
        self.generator = PseudoSampleGenerationPipeline(
            strategy_weights=config.strategy_weights, validation_threshold=config.quality_threshold
        )
        self.validator = FinancialDataValidator()

        # Training state
        self.training_state = {
            "epoch": 0,
            "batch": 0,
            "total_samples_generated": 0,
            "quality_scores": [],
            "generation_times": [],
        }

    def prepare_augmented_dataloader(
        self,
        original_data: np.ndarray,
        original_labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        dynamic_generation: bool = True,
    ) -> DataLoader:
        """Prepare augmented dataloader for training.

        Args:
            original_data: Original OHLC data [N, T, 4]
            original_labels: Original labels [N]
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of workers for data loading
            dynamic_generation: Whether to use dynamic generation

        Returns:
            Configured DataLoader
        """
        if not self.config.enable_augmentation:
            # Return original dataset
            dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(original_data), torch.tensor(original_labels, dtype=torch.long)
            )
            return DataLoader(
                dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
            )

        if dynamic_generation:
            # Use dynamic generation
            dataset = DynamicAugmentedDataset(
                original_data, original_labels, self.generator, self.config
            )
        else:
            # Pre-generate pseudo-samples
            n_pseudo_samples = int(len(original_data) * self.config.augmentation_ratio)
            pseudo_data, pseudo_labels, _ = self.generator.generate_samples(
                original_data, original_labels, n_pseudo_samples
            )

            dataset = AugmentedDataset(original_data, original_labels, pseudo_data, pseudo_labels)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def set_encoder_model(self, encoder_model: torch.nn.Module):
        """Set encoder model for self-supervised pseudo-labeling.

        Args:
            encoder_model: Pre-trained encoder model
        """
        self.generator.set_encoder_model(encoder_model)

        if self.config.enable_self_supervised:
            self.generator.enable_self_supervised(self.config.self_supervised_confidence)
            self.logger.info("Self-supervised pseudo-labeling enabled")

    def validate_batch_quality(
        self, original_batch: torch.Tensor, generated_batch: torch.Tensor
    ) -> ValidationReport:
        """Validate quality of generated batch.

        Args:
            original_batch: Original data batch [B, T, 4]
            generated_batch: Generated data batch [B, T, 4]

        Returns:
            Validation report
        """
        original_np = original_batch.cpu().numpy()
        generated_np = generated_batch.cpu().numpy()

        report = self.validator.validate_pseudo_samples(original_np, generated_np)

        # Store quality score
        self.training_state["quality_scores"].append(report.overall_quality_score)

        return report

    def adaptive_generation(
        self,
        original_data: np.ndarray,
        original_labels: np.ndarray,
        current_performance: float,
        target_performance: float,
    ) -> int:
        """Adaptively adjust pseudo-sample generation based on performance.

        Args:
            original_data: Original OHLC data
            original_labels: Original labels
            current_performance: Current model performance metric
            target_performance: Target performance metric

        Returns:
            Number of samples to generate
        """
        performance_gap = target_performance - current_performance

        if performance_gap <= 0:
            # Good performance, generate fewer samples
            n_samples = int(len(original_data) * 0.5)
        elif performance_gap > 0.2:
            # Poor performance, generate more samples
            n_samples = int(len(original_data) * 3.0)
        else:
            # Moderate performance, standard generation
            n_samples = int(len(original_data) * self.config.augmentation_ratio)

        return n_samples

    def monitor_memory_usage(self) -> float:
        """Monitor current memory usage.

        Returns:
            Memory usage in GB
        """
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            memory_gb = memory_info.rss / (1024**3)
            return memory_gb
        except ImportError:
            self.logger.warning("psutil not available, cannot monitor memory usage")
            return 0.0

    def check_memory_constraints(self) -> bool:
        """Check if memory usage is within constraints.

        Returns:
            True if memory usage is acceptable
        """
        current_memory = self.monitor_memory_usage()
        return current_memory <= self.config.max_memory_usage_gb

    def generate_with_constraints(
        self, original_data: np.ndarray, original_labels: np.ndarray, n_samples: int
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate samples with memory and quality constraints.

        Args:
            original_data: Original OHLC data
            original_labels: Original labels
            n_samples: Number of samples to generate

        Returns:
            Tuple of (generated_data, generated_labels, metadata)
        """
        if not self.check_memory_constraints():
            self.logger.warning("Memory usage high, reducing generation batch size")
            n_samples = min(n_samples, n_samples // 2)

        start_time = time.time()

        generated_data, generated_labels, metadata = self.generator.generate_samples(
            original_data, original_labels, n_samples
        )

        generation_time = time.time() - start_time
        self.training_state["generation_times"].append(generation_time)

        # Update statistics
        self.training_state["total_samples_generated"] += len(generated_data)

        metadata.update(
            {
                "generation_time": generation_time,
                "memory_usage_gb": self.monitor_memory_usage(),
                "samples_per_second": (
                    len(generated_data) / generation_time if generation_time > 0 else 0
                ),
            }
        )

        return generated_data, generated_labels, metadata

    def get_training_report(self) -> Dict[str, Any]:
        """Get comprehensive training report.

        Returns:
            Dictionary with training statistics and metrics
        """
        report = {
            "training_state": self.training_state.copy(),
            "generation_performance": self.generator.get_generation_report(),
            "memory_usage_gb": self.monitor_memory_usage(),
            "config": self.config.__dict__,
        }

        if self.training_state["quality_scores"]:
            report["average_quality"] = np.mean(self.training_state["quality_scores"])
            report["quality_trend"] = (
                np.mean(self.training_state["quality_scores"][-10:])
                - np.mean(self.training_state["quality_scores"][:10])
                if len(self.training_state["quality_scores"]) > 20
                else 0
            )

        if self.training_state["generation_times"]:
            report["average_generation_time"] = np.mean(self.training_state["generation_times"])

        return report

    def save_training_state(self, filepath: Union[str, Path]):
        """Save training state to file.

        Args:
            filepath: Path to save state
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state_data = {
            "training_state": self.training_state,
            "config": self.config.__dict__,
            "generation_report": self.generator.get_generation_report(),
        }

        with open(filepath, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

        self.logger.info(f"Training state saved to {filepath}")

    def load_training_state(self, filepath: Union[str, Path]):
        """Load training state from file.

        Args:
            filepath: Path to load state from
        """
        filepath = Path(filepath)

        if not filepath.exists():
            self.logger.warning(f"Training state file not found: {filepath}")
            return

        with open(filepath, "r") as f:
            state_data = json.load(f)

        self.training_state = state_data["training_state"]
        self.logger.info(f"Training state loaded from {filepath}")


class PseudoSampleTrainingCallback:
    """Callback for training loop integration."""

    def __init__(self, integrator: TrainingPipelineIntegrator):
        """Initialize callback.

        Args:
            integrator: Training pipeline integrator
        """
        self.integrator = integrator

    def on_epoch_start(self, epoch: int):
        """Called at the start of each epoch.

        Args:
            epoch: Current epoch number
        """
        self.integrator.training_state["epoch"] = epoch

    def on_batch_start(self, batch: int):
        """Called at the start of each batch.

        Args:
            batch: Current batch number
        """
        self.integrator.training_state["batch"] = batch

    def on_batch_end(
        self,
        batch: int,
        original_batch: torch.Tensor,
        generated_batch: Optional[torch.Tensor] = None,
    ):
        """Called at the end of each batch.

        Args:
            batch: Current batch number
            original_batch: Original data batch
            generated_batch: Generated data batch (if any)
        """
        if generated_batch is not None:
            # Validate quality periodically
            if batch % self.integrator.config.validation_frequency == 0:
                report = self.integrator.validate_batch_quality(original_batch, generated_batch)

                # Log quality metrics
                if hasattr(self.integrator, "logger"):
                    self.integrator.logger.info(
                        f"Batch {batch} - Quality Score: {report.overall_quality_score:.3f}"
                    )

                # Adjust generation parameters if quality is poor
                if report.overall_quality_score < 0.6:
                    if hasattr(self.integrator, "logger"):
                        self.integrator.logger.warning(
                            f"Low quality score ({report.overall_quality_score:.3f}) "
                            f"at batch {batch}"
                        )

    def on_epoch_end(self, epoch: int, performance_metric: float):
        """Called at the end of each epoch.

        Args:
            epoch: Current epoch number
            performance_metric: Performance metric for the epoch
        """
        # Save training state periodically
        if epoch % 10 == 0:
            self.integrator.save_training_state(f"training_state_epoch_{epoch}.json")


if __name__ == "__main__":
    # Example usage
    print("Training Pipeline Integration Module")
    print("This module provides seamless integration of pseudo-sample generation")
    print("with the existing Moola training infrastructure.")
    print("Available components:")
    print("1. AugmentedDataset - Dataset wrapper with pseudo-samples")
    print("2. DynamicAugmentedDataset - Dynamic on-the-fly generation")
    print("3. TrainingPipelineIntegrator - Main integration class")
    print("4. PseudoSampleTrainingCallback - Training loop integration")
