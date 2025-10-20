"""Benchmarking utilities for data loading and augmentation performance.

Provides tools to measure and optimize data pipeline performance for RTX 4090.
"""

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a data loading benchmark."""

    experiment_id: str
    batch_size: int
    num_workers: int
    pin_memory: bool

    # Timing metrics
    mean_epoch_time: float
    std_epoch_time: float
    min_epoch_time: float
    max_epoch_time: float

    # Throughput metrics
    samples_per_second: float
    batches_per_second: float

    # Memory metrics (optional)
    peak_memory_mb: float | None = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "mean_epoch_time": self.mean_epoch_time,
            "std_epoch_time": self.std_epoch_time,
            "min_epoch_time": self.min_epoch_time,
            "max_epoch_time": self.max_epoch_time,
            "samples_per_second": self.samples_per_second,
            "batches_per_second": self.batches_per_second,
            "peak_memory_mb": self.peak_memory_mb,
        }

    def __str__(self) -> str:
        """Format benchmark result for display."""
        lines = [
            f"Benchmark Results - {self.experiment_id}",
            "=" * 60,
            f"Configuration:",
            f"  Batch size: {self.batch_size}",
            f"  Num workers: {self.num_workers}",
            f"  Pin memory: {self.pin_memory}",
            "",
            f"Performance:",
            f"  Mean epoch time: {self.mean_epoch_time:.3f}s ± {self.std_epoch_time:.3f}s",
            f"  Min epoch time: {self.min_epoch_time:.3f}s",
            f"  Max epoch time: {self.max_epoch_time:.3f}s",
            "",
            f"Throughput:",
            f"  Samples/second: {self.samples_per_second:,.0f}",
            f"  Batches/second: {self.batches_per_second:.1f}",
        ]

        if self.peak_memory_mb is not None:
            lines.append("")
            lines.append(f"Memory:")
            lines.append(f"  Peak memory: {self.peak_memory_mb:.1f} MB")

        return "\n".join(lines)


class DataLoaderBenchmark:
    """Benchmark data loading performance."""

    def __init__(self, experiment_id: str):
        """Initialize benchmark.

        Args:
            experiment_id: Experiment identifier for tracking
        """
        self.experiment_id = experiment_id

    def benchmark_dataloader(
        self,
        dataloader: DataLoader,
        num_iterations: int = 10,
        warmup_iterations: int = 2,
        measure_memory: bool = False,
    ) -> BenchmarkResult:
        """Benchmark a DataLoader.

        Args:
            dataloader: DataLoader to benchmark
            num_iterations: Number of iterations to average
            warmup_iterations: Number of warmup iterations (not counted)
            measure_memory: Track peak GPU memory usage

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking DataLoader for {self.experiment_id}")
        logger.info(f"  Dataset size: {len(dataloader.dataset)}")
        logger.info(f"  Batch size: {dataloader.batch_size}")
        logger.info(f"  Num batches: {len(dataloader)}")
        logger.info(f"  Num workers: {dataloader.num_workers}")

        # Warmup iterations
        logger.info(f"Running {warmup_iterations} warmup iterations...")
        for i in range(warmup_iterations):
            for batch in dataloader:
                pass

        # Benchmark iterations
        logger.info(f"Running {num_iterations} benchmark iterations...")
        epoch_times = []
        peak_memory = None

        for iteration in range(num_iterations):
            if measure_memory and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            start_time = time.time()

            for batch in dataloader:
                pass  # Just iterate, don't process

            elapsed = time.time() - start_time
            epoch_times.append(elapsed)

            if measure_memory and torch.cuda.is_available():
                peak_memory_bytes = torch.cuda.max_memory_allocated()
                peak_memory = peak_memory_bytes / 1024**2  # Convert to MB

            logger.info(f"  Iteration {iteration + 1}/{num_iterations}: {elapsed:.3f}s")

        # Compute statistics
        mean_epoch_time = np.mean(epoch_times)
        std_epoch_time = np.std(epoch_times)
        min_epoch_time = np.min(epoch_times)
        max_epoch_time = np.max(epoch_times)

        samples_per_second = len(dataloader.dataset) / mean_epoch_time
        batches_per_second = len(dataloader) / mean_epoch_time

        result = BenchmarkResult(
            experiment_id=self.experiment_id,
            batch_size=dataloader.batch_size,
            num_workers=dataloader.num_workers,
            pin_memory=dataloader.pin_memory,
            mean_epoch_time=mean_epoch_time,
            std_epoch_time=std_epoch_time,
            min_epoch_time=min_epoch_time,
            max_epoch_time=max_epoch_time,
            samples_per_second=samples_per_second,
            batches_per_second=batches_per_second,
            peak_memory_mb=peak_memory,
        )

        logger.info("\n" + str(result))

        return result

    def grid_search(
        self,
        dataset: torch.utils.data.Dataset,
        batch_sizes: list[int] = [128, 256, 512, 1024],
        num_workers_options: list[int] = [0, 4, 8, 16],
        pin_memory_options: list[bool] = [True, False],
        num_iterations: int = 5,
    ) -> list[BenchmarkResult]:
        """Grid search over DataLoader configurations.

        Args:
            dataset: Dataset to benchmark
            batch_sizes: List of batch sizes to test
            num_workers_options: List of num_workers to test
            pin_memory_options: List of pin_memory settings
            num_iterations: Iterations per configuration

        Returns:
            List of benchmark results sorted by throughput
        """
        logger.info("Starting grid search for DataLoader optimization...")
        results = []

        total_configs = len(batch_sizes) * len(num_workers_options) * len(pin_memory_options)
        current_config = 0

        for batch_size in batch_sizes:
            for num_workers in num_workers_options:
                for pin_memory in pin_memory_options:
                    current_config += 1
                    logger.info(
                        f"\nTesting configuration {current_config}/{total_configs}: "
                        f"batch_size={batch_size}, num_workers={num_workers}, "
                        f"pin_memory={pin_memory}"
                    )

                    dataloader = DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        persistent_workers=num_workers > 0,
                        prefetch_factor=2 if num_workers > 0 else None,
                    )

                    result = self.benchmark_dataloader(
                        dataloader,
                        num_iterations=num_iterations,
                        warmup_iterations=1,
                    )

                    results.append(result)

        # Sort by throughput (descending)
        results.sort(key=lambda r: r.samples_per_second, reverse=True)

        logger.info("\n" + "=" * 60)
        logger.info("Grid Search Complete - Top 5 Configurations:")
        logger.info("=" * 60)

        for i, result in enumerate(results[:5], 1):
            logger.info(
                f"\n{i}. batch_size={result.batch_size}, "
                f"num_workers={result.num_workers}, "
                f"pin_memory={result.pin_memory}"
            )
            logger.info(f"   Throughput: {result.samples_per_second:,.0f} samples/s")
            logger.info(f"   Epoch time: {result.mean_epoch_time:.3f}s")

        return results


def benchmark_augmentation_speed(
    augmentor,
    data: np.ndarray,
    num_iterations: int = 100,
) -> dict[str, float]:
    """Benchmark augmentation performance.

    Args:
        augmentor: Augmentor instance (e.g., TemporalAugmentor)
        data: Sample data to augment [N, 105, 4]
        num_iterations: Number of iterations

    Returns:
        Dictionary with timing metrics
    """
    logger.info(f"Benchmarking augmentation speed ({num_iterations} iterations)...")
    logger.info(f"  Data shape: {data.shape}")

    times = []

    for i in range(num_iterations):
        start = time.time()
        _ = augmentor.augment(data, apply_time_warp=True)
        times.append(time.time() - start)

    results = {
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "min_time": np.min(times),
        "samples_per_second": len(data) / np.mean(times),
    }

    logger.info("Augmentation benchmark results:")
    logger.info(f"  Mean time: {results['mean_time']:.4f}s")
    logger.info(f"  Throughput: {results['samples_per_second']:,.0f} samples/s")

    return results
