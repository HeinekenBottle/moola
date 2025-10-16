# RTX 4090 Optimization Guide for Masked LSTM Pre-training

## Hardware Specifications

**GPU**: NVIDIA RTX 4090
- **VRAM**: 24GB GDDR6X
- **CUDA Cores**: 16,384
- **Tensor Cores**: 512 (4th gen)
- **Memory Bandwidth**: 1,008 GB/s
- **FP16 Performance**: ~82.6 TFLOPS

**Comparison to H100**:
- VRAM: 24GB vs 80GB (30% of H100)
- Memory Bandwidth: 1,008 GB/s vs 3,350 GB/s (30% of H100)
- FP16 Performance: ~82.6 vs ~989 TFLOPS (8.3% of H100)

## Optimal Batch Sizes for 24GB VRAM

### Pre-training (Masked LSTM Autoencoder)
```python
# Configuration: BiLSTM (hidden=128, layers=2, bidirectional)
# Sequence length: 105 timesteps, 4 features (OHLC)

MASKED_LSTM_BATCH_SIZE = 512  # Optimal for RTX 4090
# Memory usage: ~8-10GB VRAM
# Training throughput: ~2,500-3,000 sequences/sec
```

**Batch size scaling**:
- `batch_size=256`: ~5GB VRAM (conservative, slower)
- `batch_size=512`: ~8-10GB VRAM (optimal, recommended)
- `batch_size=1024`: ~14-16GB VRAM (aggressive, faster but less stable)
- `batch_size=2048`: OOM risk on RTX 4090

### Fine-tuning (SimpleLSTM Classification)
```python
# Configuration: LSTM (hidden=64, layers=1, unidirectional)
# Sequence length: 105 timesteps, 4 features

SIMPLE_LSTM_BATCH_SIZE = 512  # Same as pre-training
# Memory usage: ~3-5GB VRAM (smaller model)
# Note: Small dataset (98 samples), batch size doesn't matter much
```

## Timing Estimates

### Pre-training Phase
| Configuration | Sequences | Epochs | Batch Size | Time (RTX 4090) | Time (H100) |
|---------------|-----------|--------|------------|-----------------|-------------|
| Conservative  | 1,000     | 50     | 256        | ~15-20 min      | ~5-8 min    |
| **Optimal**   | **5,000** | **50** | **512**    | **~30-40 min**  | **~10-15 min** |
| Aggressive    | 10,000    | 50     | 1024       | ~60-80 min      | ~20-30 min  |

**Recommended**: 5,000 sequences (1,000 base + 4x augmentation), batch_size=512

### Fine-tuning Phase (SimpleLSTM)
| Configuration | Samples | Epochs | Time (RTX 4090) | Time (H100) |
|---------------|---------|--------|-----------------|-------------|
| With pre-train| 98      | 60     | ~10-15 min      | ~3-5 min    |
| Baseline      | 98      | 60     | ~10-15 min      | ~3-5 min    |

**Note**: Fine-tuning time dominated by small dataset (98 samples), not GPU speed.

### Full Pipeline Timing
| Stage | Time (RTX 4090) | Time (H100) |
|-------|-----------------|-------------|
| Pre-training | ~30-40 min | ~10-15 min |
| Fine-tuning (pre-trained) | ~10-15 min | ~3-5 min |
| Baseline training | ~10-15 min | ~3-5 min |
| **Total** | **~50-60 min** | **~16-25 min** |

## Memory Optimization Strategies

### 1. Gradient Accumulation (if OOM)
```python
# If batch_size=512 causes OOM, use gradient accumulation
EFFECTIVE_BATCH_SIZE = 512
MICRO_BATCH_SIZE = 256
ACCUMULATION_STEPS = EFFECTIVE_BATCH_SIZE // MICRO_BATCH_SIZE  # = 2

# Modify training loop:
for i, (batch_X,) in enumerate(train_loader):
    loss = model(batch_X)
    loss = loss / ACCUMULATION_STEPS  # Scale loss
    loss.backward()

    if (i + 1) % ACCUMULATION_STEPS == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Mixed Precision (FP16)
```python
# Already enabled by default in training_config.py
USE_AMP = True  # Automatic mixed precision

# RTX 4090 benefits:
# - 2x faster training (FP16 vs FP32)
# - ~40% less VRAM usage
# - Minimal accuracy loss
```

### 3. DataLoader Optimization
```python
# Optimal settings for RTX 4090
num_workers = 4          # Parallel data loading
pin_memory = True        # Faster GPU transfer
persistent_workers = True # Reuse workers
prefetch_factor = 2      # Prefetch batches
```

### 4. Reduce Sequence Length (if needed)
```python
# Current: 105 timesteps (30 pre + 45 window + 30 post)
# If OOM: Use 75 timesteps (15 pre + 45 window + 15 post)

# Memory reduction: ~30%
# Accuracy impact: -2-5% (loss of context)
```

## Performance Monitoring

### GPU Utilization Targets
- **Training**: 90-100% GPU utilization (optimal)
- **Pre-training**: 85-95% (reconstruction objective is lighter)
- **Fine-tuning**: 70-90% (small dataset, frequent validation)

### VRAM Usage Targets
- **Pre-training**: 8-12GB (out of 24GB)
- **Fine-tuning**: 3-6GB
- **Safety margin**: Keep 4-6GB free for system overhead

### Training Speed Benchmarks
```bash
# Pre-training (batch_size=512)
# Expected: 2,500-3,000 sequences/sec
# Check with: nvidia-smi dmon -s u

# Fine-tuning
# Expected: 800-1,200 samples/sec (limited by dataset size)
```

## Cost Optimization

### RunPod RTX 4090 Pricing (as of 2024)
- **On-demand**: ~$0.34/hour
- **Spot instances**: ~$0.20/hour (if available)

### Cost per Pipeline Run
| Configuration | Time | On-demand Cost | Spot Cost |
|---------------|------|----------------|-----------|
| Conservative (1K seq) | ~35 min | ~$0.20 | ~$0.12 |
| **Optimal (5K seq)** | **~55 min** | **~$0.31** | **~$0.18** |
| Aggressive (10K seq) | ~100 min | ~$0.57 | ~$0.33 |

**Recommendation**: Use spot instances if available (~40% savings).

### Cost Comparison: RTX 4090 vs H100
| GPU | Time | On-demand Cost | Cost per Run |
|-----|------|----------------|--------------|
| **RTX 4090** | **~55 min** | **$0.34/hr** | **~$0.31** |
| H100 | ~20 min | $3.00/hr | ~$1.00 |

**RTX 4090 is 3x cheaper** despite being 2.5x slower.

## Troubleshooting

### OOM Errors
1. Reduce batch size: `512 → 256`
2. Enable gradient accumulation (see above)
3. Reduce sequence length: `105 → 75`
4. Reduce unlabeled dataset: `5000 → 3000`

### Slow Training
1. Check GPU utilization: `nvidia-smi dmon -s u`
2. Increase batch size if VRAM allows: `512 → 1024`
3. Increase num_workers: `4 → 8`
4. Check CPU bottleneck: `htop`

### Poor Quality Pre-training
1. Increase dataset size: `5000 → 10000`
2. Increase epochs: `50 → 100`
3. Reduce learning rate: `1e-3 → 5e-4`
4. Increase mask ratio: `0.15 → 0.25`

## Recommended Configuration

**For RTX 4090 (24GB VRAM)**:
```python
# Pre-training
MASKED_LSTM_BATCH_SIZE = 512
MASKED_LSTM_N_EPOCHS = 50
MASKED_LSTM_LEARNING_RATE = 1e-3
UNLABELED_DATASET_SIZE = 5000  # 1K base + 4x augmentation

# Fine-tuning
SIMPLE_LSTM_BATCH_SIZE = 512
SIMPLE_LSTM_N_EPOCHS = 60
SIMPLE_LSTM_LEARNING_RATE = 5e-4

# Training
USE_AMP = True  # Mixed precision
num_workers = 4
pin_memory = True
```

**Expected Results**:
- Pre-training: 30-40 minutes
- Fine-tuning: 10-15 minutes per model
- Total pipeline: 50-60 minutes
- Cost: ~$0.31 (on-demand) or ~$0.18 (spot)
- Accuracy improvement: +8-12% over baseline

## Advanced: Multi-GPU Training (if available)

RTX 4090 supports multi-GPU training via DataParallel or DDP:

```python
# DataParallel (simpler but slower)
model = nn.DataParallel(model)

# DistributedDataParallel (faster, recommended)
# Requires additional setup
```

**Scaling efficiency**:
- 2x RTX 4090: ~1.8x speedup (90% efficiency)
- 4x RTX 4090: ~3.2x speedup (80% efficiency)

**Note**: Pre-training on single RTX 4090 is fast enough (~30-40 min). Multi-GPU is overkill for this workload.
