#!/usr/bin/env python3
"""Benchmark LSTM training performance on RTX 4090.

Comprehensive benchmarking suite for measuring the impact of performance optimizations:
    1. Baseline (no optimizations)
    2. AMP only
    3. DataLoader optimization only
    4. All optimizations combined

Generates detailed performance reports with speedup analysis.

Usage:
    python scripts/benchmark_lstm_performance.py --device cuda --output benchmarks/
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from moola.config.performance_config import (
    apply_performance_optimizations,
    get_amp_scaler,
    get_optimized_dataloader_kwargs,
)
from moola.config.training_config import (
    DEFAULT_SEED,
    MASKED_LSTM_HIDDEN_DIM,
    MASKED_LSTM_NUM_LAYERS,
)
from moola.models.bilstm_masked_autoencoder import (
    BiLSTMMaskedAutoencoder,
    apply_masking,
)
from moola.utils.seeds import get_device, set_seed

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"


class BenchmarkConfig:
    """Benchmark configuration."""

    def __init__(
        self,
        name: str,
        use_amp: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        prefetch_factor: Optional[int] = None,
        persistent_workers: bool = False,
    ):
        self.name = name
        self.use_amp = use_amp
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers


def benchmark_training_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    num_steps: int = 50,
) -> Dict[str, float]:
    """Benchmark a single training configuration.

    Args:
        model: Model to train
        dataloader: Training dataloader
        device: Device to use
        use_amp: Enable automatic mixed precision
        num_steps: Number of steps to benchmark

    Returns:
        Dictionary with benchmark metrics
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = get_amp_scaler() if use_amp else None

    # Warmup (ignore timing)
    print("  Warmup...", end=" ", flush=True)
    for i, (batch_X,) in enumerate(dataloader):
        if i >= 5:
            break

        batch_X = batch_X.to(device, non_blocking=True)
        x_masked, mask = apply_masking(batch_X, model.mask_token, mask_ratio=0.15)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                reconstruction = model(x_masked)
                loss, _ = model.compute_loss(reconstruction, batch_X, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstruction = model(x_masked)
            loss, _ = model.compute_loss(reconstruction, batch_X, mask)
            loss.backward()
            optimizer.step()

    print("done")

    # Benchmark
    print(f"  Benchmarking {num_steps} steps...", end=" ", flush=True)
    step_times = []
    data_times = []

    if device.type == "cuda":
        torch.cuda.synchronize()

    for i, (batch_X,) in enumerate(dataloader):
        if i >= num_steps:
            break

        data_start = time.time()
        batch_X = batch_X.to(device, non_blocking=True)
        data_time = time.time() - data_start

        x_masked, mask = apply_masking(batch_X, model.mask_token, mask_ratio=0.15)

        step_start = time.time()
        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                reconstruction = model(x_masked)
                loss, _ = model.compute_loss(reconstruction, batch_X, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            reconstruction = model(x_masked)
            loss, _ = model.compute_loss(reconstruction, batch_X, mask)
            loss.backward()
            optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()

        step_time = time.time() - step_start
        step_times.append(step_time)
        data_times.append(data_time)

    print("done")

    # Calculate statistics
    avg_step_time = np.mean(step_times[5:])  # Exclude first 5 for stability
    std_step_time = np.std(step_times[5:])
    avg_data_time = np.mean(data_times[5:])

    return {
        'avg_step_time': avg_step_time,
        'std_step_time': std_step_time,
        'avg_data_time': avg_data_time,
        'min_step_time': np.min(step_times),
        'max_step_time': np.max(step_times),
        'throughput': len(batch_X) / avg_step_time,  # samples/sec
        'data_overhead_pct': (avg_data_time / avg_step_time) * 100,
    }


def run_benchmark_suite(
    X_data: np.ndarray,
    device: str = "cuda",
    batch_size: int = 512,
    num_steps: int = 50,
) -> pd.DataFrame:
    """Run comprehensive benchmark suite.

    Args:
        X_data: Training data [N, T, F]
        device: Device to use
        batch_size: Batch size
        num_steps: Number of steps per benchmark

    Returns:
        DataFrame with benchmark results
    """
    device_obj = get_device(device)
    set_seed(DEFAULT_SEED)

    # Benchmark configurations
    configs = [
        BenchmarkConfig(
            name="Baseline (No Optimizations)",
            use_amp=False,
            num_workers=0,
            pin_memory=False,
        ),
        BenchmarkConfig(
            name="AMP Only",
            use_amp=True,
            num_workers=0,
            pin_memory=False,
        ),
        BenchmarkConfig(
            name="DataLoader Optimized",
            use_amp=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        ),
        BenchmarkConfig(
            name="All Optimizations",
            use_amp=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
        ),
    ]

    results = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Benchmarking: {config.name}")
        print(f"{'='*60}")

        # Create model
        model = BiLSTMMaskedAutoencoder(
            input_dim=4,
            hidden_dim=MASKED_LSTM_HIDDEN_DIM,
            num_layers=MASKED_LSTM_NUM_LAYERS,
            dropout=0.2,
        ).to(device_obj)

        # Create dataloader
        dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_data))
        dataloader_kwargs = {
            'num_workers': config.num_workers,
            'pin_memory': config.pin_memory,
        }
        if config.prefetch_factor:
            dataloader_kwargs['prefetch_factor'] = config.prefetch_factor
        if config.persistent_workers and config.num_workers > 0:
            dataloader_kwargs['persistent_workers'] = True

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            **dataloader_kwargs
        )

        # Benchmark
        metrics = benchmark_training_step(
            model,
            dataloader,
            device_obj,
            use_amp=config.use_amp,
            num_steps=num_steps,
        )

        # Add config info
        metrics['config'] = config.name
        metrics['use_amp'] = config.use_amp
        metrics['num_workers'] = config.num_workers
        metrics['pin_memory'] = config.pin_memory

        results.append(metrics)

        # Print summary
        print(f"\n  Results:")
        print(f"    Step time: {metrics['avg_step_time']*1000:.1f} ± {metrics['std_step_time']*1000:.1f} ms")
        print(f"    Throughput: {metrics['throughput']:.0f} samples/sec")
        print(f"    Data loading overhead: {metrics['data_overhead_pct']:.1f}%")

        # Cleanup
        del model
        del dataloader
        if device == "cuda":
            torch.cuda.empty_cache()

    return pd.DataFrame(results)


def calculate_speedups(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate speedup relative to baseline.

    Args:
        df: Benchmark results

    Returns:
        DataFrame with speedup columns added
    """
    baseline_time = df[df['config'] == 'Baseline (No Optimizations)']['avg_step_time'].values[0]

    df['speedup'] = baseline_time / df['avg_step_time']
    df['time_reduction_pct'] = (1 - df['avg_step_time'] / baseline_time) * 100

    return df


def estimate_epoch_time(
    step_time: float,
    num_samples: int,
    batch_size: int,
) -> float:
    """Estimate epoch time from step time.

    Args:
        step_time: Average time per step (seconds)
        num_samples: Total number of samples
        batch_size: Batch size

    Returns:
        Estimated epoch time (seconds)
    """
    num_steps = (num_samples + batch_size - 1) // batch_size
    return step_time * num_steps


def generate_report(
    df: pd.DataFrame,
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    output_dir: Path,
):
    """Generate comprehensive benchmark report.

    Args:
        df: Benchmark results
        num_samples: Number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        output_dir: Output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate epoch and training times
    df['epoch_time_sec'] = df['avg_step_time'].apply(
        lambda x: estimate_epoch_time(x, num_samples, batch_size)
    )
    df['epoch_time_min'] = df['epoch_time_sec'] / 60
    df['total_training_min'] = df['epoch_time_min'] * num_epochs

    # Save detailed results
    csv_path = output_dir / "benchmark_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[SAVE] Detailed results saved to: {csv_path}")

    # Generate summary report
    report_path = output_dir / "performance_summary.md"

    with open(report_path, 'w') as f:
        f.write("# LSTM Performance Benchmark - RTX 4090\n\n")
        f.write(f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Dataset**: {num_samples:,} samples\n")
        f.write(f"**Batch size**: {batch_size}\n")
        f.write(f"**Epochs**: {num_epochs}\n\n")

        f.write("## Benchmark Results\n\n")
        f.write("| Configuration | Step Time (ms) | Throughput (samples/sec) | Speedup | Epoch Time (min) | Total Time (min) |\n")
        f.write("|---------------|----------------|-------------------------|---------|------------------|------------------|\n")

        for _, row in df.iterrows():
            f.write(f"| {row['config']} | {row['avg_step_time']*1000:.1f} | {row['throughput']:.0f} | "
                   f"{row['speedup']:.2f}× | {row['epoch_time_min']:.1f} | {row['total_training_min']:.1f} |\n")

        f.write("\n## Performance Improvements\n\n")

        baseline_time = df[df['config'] == 'Baseline (No Optimizations)']['total_training_min'].values[0]
        optimized_time = df[df['config'] == 'All Optimizations']['total_training_min'].values[0]
        total_speedup = baseline_time / optimized_time
        time_saved = baseline_time - optimized_time

        f.write(f"- **Baseline training time**: {baseline_time:.1f} minutes\n")
        f.write(f"- **Optimized training time**: {optimized_time:.1f} minutes\n")
        f.write(f"- **Total speedup**: {total_speedup:.2f}×\n")
        f.write(f"- **Time saved**: {time_saved:.1f} minutes ({time_saved/baseline_time*100:.1f}%)\n\n")

        f.write("## Individual Optimization Impact\n\n")

        amp_speedup = df[df['config'] == 'AMP Only']['speedup'].values[0]
        dataloader_speedup = df[df['config'] == 'DataLoader Optimized']['speedup'].values[0]

        f.write(f"1. **AMP (Automatic Mixed Precision)**\n")
        f.write(f"   - Speedup: {amp_speedup:.2f}×\n")
        f.write(f"   - Time reduction: {(1-1/amp_speedup)*100:.1f}%\n\n")

        f.write(f"2. **DataLoader Optimization**\n")
        f.write(f"   - Speedup: {dataloader_speedup:.2f}×\n")
        f.write(f"   - Time reduction: {(1-1/dataloader_speedup)*100:.1f}%\n\n")

        f.write("## Recommendations\n\n")

        if optimized_time < 22:
            f.write(f"✅ **Target achieved!** Optimized training completes in {optimized_time:.1f} minutes (target: 18-22 min)\n\n")
        elif optimized_time < 30:
            f.write(f"⚠️ **Improved but not at target.** Optimized training: {optimized_time:.1f} minutes (target: 18-22 min)\n\n")
            f.write("Additional optimizations to consider:\n")
            f.write("- Enable pre-augmentation caching (eliminates 5-10% CPU overhead)\n")
            f.write("- Reduce early stopping patience (saves 3-5 minutes)\n")
            f.write("- Use `torch.compile()` if PyTorch 2.0+ (experimental)\n\n")
        else:
            f.write(f"❌ **Target not met.** Optimized training: {optimized_time:.1f} minutes (target: 18-22 min)\n\n")
            f.write("Troubleshooting:\n")
            f.write("- Verify GPU is RTX 4090 with proper drivers\n")
            f.write("- Check GPU utilization (should be >85%)\n")
            f.write("- Ensure no background processes competing for GPU\n\n")

    print(f"[SAVE] Performance summary saved to: {report_path}")

    # Generate visualization
    plot_path = output_dir / "benchmark_comparison.png"
    plot_benchmark_results(df, plot_path)
    print(f"[SAVE] Visualization saved to: {plot_path}")


def plot_benchmark_results(df: pd.DataFrame, output_path: Path):
    """Generate benchmark comparison plots.

    Args:
        df: Benchmark results
        output_path: Output image path
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Step time comparison
    ax1 = axes[0]
    configs = df['config'].values
    step_times = df['avg_step_time'].values * 1000  # Convert to ms

    bars = ax1.bar(range(len(configs)), step_times, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Step Time (ms)')
    ax1.set_title('Training Step Time')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([c.split()[0] for c in configs], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bar, val in zip(bars, step_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    # Plot 2: Speedup comparison
    ax2 = axes[1]
    speedups = df['speedup'].values

    bars = ax2.bar(range(len(configs)), speedups, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Speedup (×)')
    ax2.set_title('Training Speedup vs Baseline')
    ax2.set_xticks(range(len(configs)))
    ax2.set_xticklabels([c.split()[0] for c in configs], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()

    # Add values on bars
    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=9)

    # Plot 3: Total training time
    ax3 = axes[2]
    total_times = df['total_training_min'].values

    bars = ax3.bar(range(len(configs)), total_times, color=['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4'])
    ax3.axhline(y=22, color='green', linestyle='--', alpha=0.5, label='Target (22 min)')
    ax3.axhline(y=35, color='red', linestyle='--', alpha=0.5, label='Baseline (~35 min)')
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Total Training Time (min)')
    ax3.set_title(f'Total Training Time ({df["epoch_time_min"].values[0]*50:.0f} epochs)')
    ax3.set_xticks(range(len(configs)))
    ax3.set_xticklabels([c.split()[0] for c in configs], rotation=45, ha='right')
    ax3.grid(axis='y', alpha=0.3)
    ax3.legend()

    # Add values on bars
    for bar, val in zip(bars, total_times):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Generated plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark LSTM training performance")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--samples", type=int, default=10000, help="Number of samples for benchmark")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=50, help="Number of steps per benchmark")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for time estimation")
    parser.add_argument("--output", type=Path, default=BENCHMARK_DIR / "rtx4090")

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return

    print(f"\n{'='*70}")
    print(f"LSTM PERFORMANCE BENCHMARK - RTX 4090")
    print(f"{'='*70}")
    print(f"  Device: {args.device}")
    print(f"  Samples: {args.samples:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps per config: {args.steps}")
    print(f"  Estimated epochs: {args.epochs}")
    print(f"{'='*70}\n")

    # Generate synthetic data
    print("[DATA] Generating synthetic benchmark data...")
    X_data = np.random.randn(args.samples, 105, 4).astype(np.float32)

    # Apply backend optimizations
    if args.device == "cuda":
        apply_performance_optimizations(device=args.device)

    # Run benchmarks
    df_results = run_benchmark_suite(
        X_data,
        device=args.device,
        batch_size=args.batch_size,
        num_steps=args.steps,
    )

    # Calculate speedups
    df_results = calculate_speedups(df_results)

    # Generate report
    generate_report(
        df_results,
        num_samples=args.samples,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir=args.output,
    )

    print(f"\n{'='*70}")
    print(f"BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"  Results saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
