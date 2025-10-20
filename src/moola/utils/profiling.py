"""PyTorch profiling utilities for performance analysis.

Tools for identifying training bottlenecks and optimizing GPU utilization.
Provides wrappers for PyTorch Profiler and custom performance monitors.

Usage:
    >>> from moola.utils.profiling import ProfilerContext, GPUMonitor
    >>>
    >>> # Profile training loop
    >>> with ProfilerContext(enabled=True, warmup=5, active=10) as prof:
    ...     for epoch in range(50):
    ...         train_one_epoch(model, dataloader)
    ...         prof.step()  # Record epoch
    >>>
    >>> # Monitor GPU utilization
    >>> monitor = GPUMonitor(device=0)
    >>> monitor.start()
    >>> train_model(...)
    >>> stats = monitor.stop()
    >>> print(f"Average GPU util: {stats['avg_utilization']:.1f}%")
"""

import time
from contextlib import contextmanager
from pathlib import Path

import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


class ProfilerContext:
    """Context manager for PyTorch profiling.

    Wraps torch.profiler.profile with convenient defaults for
    identifying performance bottlenecks.

    Args:
        enabled: Enable profiling (False = no overhead)
        warmup: Number of warmup steps before profiling
        active: Number of active profiling steps
        output_dir: Directory to save profiler traces
        record_shapes: Record tensor shapes
        profile_memory: Profile memory usage
        with_stack: Record Python stack traces

    Example:
        >>> with ProfilerContext(enabled=True, warmup=5, active=10) as prof:
        ...     for i, batch in enumerate(dataloader):
        ...         output = model(batch)
        ...         loss.backward()
        ...         optimizer.step()
        ...         prof.step()  # Must call after each iteration
        >>> prof.print_summary()
    """

    def __init__(
        self,
        enabled: bool = True,
        warmup: int = 5,
        active: int = 10,
        output_dir: Path | None = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
    ):
        self.enabled = enabled and torch.cuda.is_available()
        self.warmup = warmup
        self.active = active
        self.output_dir = Path(output_dir) if output_dir else Path("profiling")
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.profiler = None

    def __enter__(self):
        if not self.enabled:
            return self

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Configure profiler
        schedule = torch.profiler.schedule(
            wait=1,
            warmup=self.warmup,
            active=self.active,
            repeat=1,
        )

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(self.output_dir)),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
        )

        self.profiler.__enter__()
        print(f"[PROFILER] Started profiling (warmup={self.warmup}, active={self.active})")
        print(f"[PROFILER] Traces will be saved to: {self.output_dir}")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler:
            self.profiler.__exit__(exc_type, exc_val, exc_tb)
            print(f"[PROFILER] Profiling complete")

    def step(self):
        """Advance profiler to next step (call after each iteration)."""
        if self.profiler:
            self.profiler.step()

    def print_summary(self, sort_by: str = "cuda_time_total", row_limit: int = 10):
        """Print profiler summary table.

        Args:
            sort_by: Sort key ('cuda_time_total', 'cpu_time_total', 'cpu_memory_usage')
            row_limit: Number of rows to display
        """
        if not self.profiler:
            return

        print("\n" + "=" * 80)
        print("PROFILER SUMMARY - Top Operations by CUDA Time")
        print("=" * 80)
        print(self.profiler.key_averages().table(
            sort_by=sort_by,
            row_limit=row_limit
        ))


class GPUMonitor:
    """Real-time GPU utilization and memory monitor.

    Uses NVIDIA Management Library (NVML) to track GPU metrics during training.
    Requires `nvidia-ml-py` package (pip install nvidia-ml-py).

    Args:
        device: CUDA device index (0 for first GPU)
        sampling_interval: Sampling interval in seconds

    Example:
        >>> monitor = GPUMonitor(device=0)
        >>> monitor.start()
        >>> train_model(...)
        >>> stats = monitor.stop()
        >>> print(f"Peak memory: {stats['peak_memory_mb']:.0f} MB")
        >>> print(f"Avg utilization: {stats['avg_utilization']:.1f}%")
    """

    def __init__(self, device: int = 0, sampling_interval: float = 1.0):
        if not PYNVML_AVAILABLE:
            raise ImportError(
                "nvidia-ml-py not installed. Install with: pip install nvidia-ml-py"
            )

        self.device = device
        self.sampling_interval = sampling_interval
        self.samples = []
        self.monitoring = False
        self.start_time = None

        # Initialize NVML
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(device)

    def start(self):
        """Start monitoring GPU metrics."""
        self.samples = []
        self.monitoring = True
        self.start_time = time.time()
        print(f"[GPU MONITOR] Started monitoring GPU {self.device}")

    def sample(self) -> dict[str, float]:
        """Take a single GPU measurement.

        Returns:
            Dictionary with GPU metrics
        """
        if not self.monitoring:
            return {}

        # Get GPU utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)

        # Get memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)

        sample = {
            'timestamp': time.time() - self.start_time,
            'gpu_utilization': util.gpu,  # 0-100%
            'memory_utilization': util.memory,  # 0-100%
            'memory_used_mb': mem_info.used / 1024**2,  # MB
            'memory_total_mb': mem_info.total / 1024**2,  # MB
        }

        self.samples.append(sample)
        return sample

    def stop(self) -> dict[str, float]:
        """Stop monitoring and return summary statistics.

        Returns:
            Dictionary with aggregated statistics
        """
        self.monitoring = False

        if not self.samples:
            return {}

        # Aggregate statistics
        gpu_utils = [s['gpu_utilization'] for s in self.samples]
        mem_utils = [s['memory_utilization'] for s in self.samples]
        mem_used = [s['memory_used_mb'] for s in self.samples]

        stats = {
            'avg_utilization': sum(gpu_utils) / len(gpu_utils),
            'max_utilization': max(gpu_utils),
            'min_utilization': min(gpu_utils),
            'avg_memory_utilization': sum(mem_utils) / len(mem_utils),
            'peak_memory_mb': max(mem_used),
            'avg_memory_mb': sum(mem_used) / len(mem_used),
            'total_memory_mb': self.samples[0]['memory_total_mb'],
            'num_samples': len(self.samples),
            'duration_sec': time.time() - self.start_time,
        }

        print(f"\n[GPU MONITOR] Monitoring stopped - Summary:")
        print(f"  Duration: {stats['duration_sec']:.1f}s")
        print(f"  Avg GPU utilization: {stats['avg_utilization']:.1f}%")
        print(f"  Peak GPU utilization: {stats['max_utilization']:.1f}%")
        print(f"  Peak memory: {stats['peak_memory_mb']:.0f} MB / {stats['total_memory_mb']:.0f} MB")

        return stats

    def __del__(self):
        """Cleanup NVML."""
        try:
            pynvml.nvmlShutdown()
        except:
            pass


@contextmanager
def profile_training(
    enabled: bool = True,
    warmup: int = 5,
    active: int = 10,
    output_dir: Path | None = None,
):
    """Convenient context manager for profiling training loops.

    Args:
        enabled: Enable profiling
        warmup: Warmup steps
        active: Active profiling steps
        output_dir: Output directory for traces

    Example:
        >>> with profile_training(enabled=True, warmup=5, active=10):
        ...     for epoch in range(50):
        ...         train_one_epoch(model, dataloader)
    """
    profiler = ProfilerContext(
        enabled=enabled,
        warmup=warmup,
        active=active,
        output_dir=output_dir,
    )

    with profiler:
        yield profiler


def log_gpu_stats(device: int = 0, prefix: str = ""):
    """Log current GPU utilization and memory to console.

    Args:
        device: CUDA device index
        prefix: Prefix for log messages

    Example:
        >>> log_gpu_stats(device=0, prefix="[EPOCH 10]")
        [EPOCH 10] GPU Util: 87% | Memory: 18432 MB / 24576 MB (75%)
    """
    if not torch.cuda.is_available():
        return

    # PyTorch memory stats
    mem_allocated = torch.cuda.memory_allocated(device) / 1024**2
    mem_reserved = torch.cuda.memory_reserved(device) / 1024**2
    mem_max = torch.cuda.get_device_properties(device).total_memory / 1024**2

    print(f"{prefix} GPU Memory: {mem_allocated:.0f} MB allocated, "
          f"{mem_reserved:.0f} MB reserved, "
          f"{mem_max:.0f} MB total")

    # NVML stats if available
    if PYNVML_AVAILABLE:
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"{prefix} GPU Utilization: {util.gpu}%")
            pynvml.nvmlShutdown()
        except:
            pass


def estimate_training_time(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    seconds_per_batch: float,
) -> dict[str, float]:
    """Estimate total training time based on batch timing.

    Args:
        num_samples: Total number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        seconds_per_batch: Measured time per batch (seconds)

    Returns:
        Dictionary with time estimates

    Example:
        >>> # Measure first 10 batches
        >>> times = []
        >>> for i, batch in enumerate(dataloader):
        ...     start = time.time()
        ...     train_step(batch)
        ...     times.append(time.time() - start)
        ...     if i >= 10:
        ...         break
        >>> avg_time = sum(times) / len(times)
        >>> est = estimate_training_time(59365, 512, 50, avg_time)
        >>> print(f"Estimated training time: {est['total_minutes']:.1f} min")
    """
    batches_per_epoch = (num_samples + batch_size - 1) // batch_size
    total_batches = batches_per_epoch * num_epochs

    total_seconds = total_batches * seconds_per_batch

    return {
        'batches_per_epoch': batches_per_epoch,
        'total_batches': total_batches,
        'seconds_per_batch': seconds_per_batch,
        'seconds_per_epoch': batches_per_epoch * seconds_per_batch,
        'total_seconds': total_seconds,
        'total_minutes': total_seconds / 60,
        'total_hours': total_seconds / 3600,
    }


# ============================================================================
# EXPORT ALL
# ============================================================================

__all__ = [
    'ProfilerContext',
    'GPUMonitor',
    'profile_training',
    'log_gpu_stats',
    'estimate_training_time',
]
