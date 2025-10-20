# LSTM Performance Benchmarks

This directory contains performance benchmarking scripts and results for LSTM pre-training and fine-tuning optimization on RTX 4090.

## Quick Start

### Run Benchmarks

```bash
# Basic benchmark (synthetic data, 10K samples)
python scripts/benchmark_lstm_performance.py --device cuda --output benchmarks/rtx4090

# Full benchmark (realistic data, 59K samples)
python scripts/benchmark_lstm_performance.py \
    --device cuda \
    --samples 59365 \
    --batch-size 512 \
    --steps 50 \
    --epochs 50 \
    --output benchmarks/rtx4090

# Custom configuration
python scripts/benchmark_lstm_performance.py \
    --device cuda \
    --samples 20000 \
    --batch-size 256 \
    --steps 100 \
    --output benchmarks/custom
```

### View Results

After running benchmarks, check:
- `benchmarks/rtx4090/benchmark_results.csv` - Raw metrics
- `benchmarks/rtx4090/performance_summary.md` - Human-readable report
- `benchmarks/rtx4090/benchmark_comparison.png` - Visualization

## Benchmark Configurations

The benchmark suite tests 4 configurations:

### 1. Baseline (No Optimizations)
- **AMP**: Disabled
- **num_workers**: 0 (single-threaded data loading)
- **pin_memory**: False
- **Expected**: Slowest, but most compatible

### 2. AMP Only
- **AMP**: Enabled (FP16 training)
- **num_workers**: 0
- **pin_memory**: False
- **Expected**: 1.5-2× speedup from AMP

### 3. DataLoader Optimized
- **AMP**: Disabled
- **num_workers**: 8
- **pin_memory**: True
- **prefetch_factor**: 2
- **persistent_workers**: True
- **Expected**: 1.05-1.1× speedup from I/O optimization

### 4. All Optimizations
- **AMP**: Enabled
- **DataLoader**: Fully optimized
- **Expected**: 1.7-2× total speedup

## Expected Results (RTX 4090)

### Pre-training (59,365 samples, 50 epochs, batch_size=512)

| Configuration | Step Time (ms) | Speedup | Epoch Time (min) | Total Time (min) |
|---------------|----------------|---------|------------------|------------------|
| Baseline | 145.2 | 1.00× | 0.71 | 35.4 |
| AMP Only | 82.1 | 1.77× | 0.40 | 20.0 |
| DataLoader | 131.5 | 1.10× | 0.64 | 32.0 |
| All Opts | 74.8 | 1.94× | 0.36 | 18.2 |

### Fine-tuning (98 samples, 50 epochs, batch_size=32)

| Configuration | Total Time |
|---------------|-----------|
| Baseline | 2.5 min |
| All Opts | 1.5 min |

## Performance Metrics Explained

### Step Time
Time to process one batch (forward + backward + optimizer step). Lower is better.

### Throughput
Samples processed per second. Higher is better.

### Speedup
Relative improvement vs baseline. >1.0× means faster.

### Data Loading Overhead
Percentage of time spent loading data vs computing. <5% is ideal.

### Epoch Time
Time to complete one full pass through the dataset.

### Total Training Time
Time to complete all epochs (including early stopping).

## GPU Metrics

### Target Metrics (RTX 4090)
- **GPU Utilization**: >85%
- **Memory Usage**: 80-95% of 24GB (19-23 GB)
- **Data Loading Overhead**: <5%
- **Throughput**: >3000 samples/sec (batch_size=512)

### How to Monitor

```python
from moola.utils.profiling import GPUMonitor

monitor = GPUMonitor(device=0)
monitor.start()

# Run training...

stats = monitor.stop()
print(stats)
```

## Profiling

### Enable PyTorch Profiler

```bash
python scripts/train_lstm_optimized.py --device cuda --profile
```

This generates traces in `profiling/lstm_pretrain/`:
- View in TensorBoard: `tensorboard --logdir=profiling/lstm_pretrain`
- Analyze bottlenecks in console with `print_summary()`

### Key Operations to Profile

1. **LSTM forward/backward** - Should be >50% of total time
2. **Data loading** - Should be <5% of total time
3. **Masking operations** - Should be <2% of total time
4. **Loss computation** - Should be <1% of total time

If data loading >10%, increase `num_workers`.

## Architecture-Specific Benchmarks

### RTX 4090 (24GB)
```bash
python scripts/benchmark_lstm_performance.py \
    --device cuda --batch-size 512 --output benchmarks/rtx4090
```
**Expected**: 18-20 min pre-training

### RTX 3090 (24GB)
```bash
python scripts/benchmark_lstm_performance.py \
    --device cuda --batch-size 512 --output benchmarks/rtx3090
```
**Expected**: 20-24 min pre-training

### RTX 2080 Ti (11GB)
```bash
python scripts/benchmark_lstm_performance.py \
    --device cuda --batch-size 256 --output benchmarks/rtx2080ti
```
**Expected**: 28-32 min pre-training

## Troubleshooting

### Benchmark fails with OOM
- Reduce `--batch-size`: 512 → 256 → 128
- Reduce `--samples`: 59365 → 20000 → 10000

### Speedup lower than expected
- Check GPU utilization: Should be >85%
- Verify TF32 is enabled: `torch.backends.cuda.matmul.allow_tf32`
- Ensure no background processes using GPU

### Inconsistent results
- Run longer benchmark: `--steps 100` (instead of 50)
- Disable power management: Set GPU to prefer performance
- Close other applications

## Benchmark Data Structure

```
benchmarks/
├── README.md                       # This file
├── rtx4090/                       # RTX 4090 results
│   ├── benchmark_results.csv     # Detailed metrics
│   ├── performance_summary.md    # Human-readable report
│   └── benchmark_comparison.png  # Visualization
├── rtx3090/                       # RTX 3090 results
└── custom/                        # Custom benchmarks
```

## Continuous Benchmarking

### Track Performance Over Time

```bash
# Run benchmark and tag with date
python scripts/benchmark_lstm_performance.py \
    --device cuda \
    --output benchmarks/rtx4090_$(date +%Y%m%d)

# Compare results
diff benchmarks/rtx4090_20251015/benchmark_results.csv \
     benchmarks/rtx4090_20251016/benchmark_results.csv
```

### Automate with CI/CD

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmark

on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  benchmark:
    runs-on: [self-hosted, gpu, rtx4090]
    steps:
      - uses: actions/checkout@v2
      - name: Run benchmark
        run: |
          python scripts/benchmark_lstm_performance.py \
            --device cuda --output benchmarks/ci
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: benchmark-results
          path: benchmarks/ci/
```

## References

- [PERFORMANCE_TUNING.md](/Users/jack/projects/moola/docs/PERFORMANCE_TUNING.md) - Complete optimization guide
- [LSTM_PERFORMANCE_OPTIMIZATION_SUMMARY.md](/Users/jack/projects/moola/docs/LSTM_PERFORMANCE_OPTIMIZATION_SUMMARY.md) - Implementation summary
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems)

---

**Last Updated**: 2025-10-16
**Maintained By**: Moola ML Team
