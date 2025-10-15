# RunPod GPU Testing Commands

Quick reference for testing GPU fixes on RunPod.

## 1. Pre-Training Checks

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check GPU info
nvidia-smi

# Verify GPU from Python
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('ERROR: No CUDA available!')
"
```

## 2. Test Single Model Training

```bash
# Test CNN-Transformer with GPU
moola train --model cnn_transformer --device cuda

# Test RWKV-TS with GPU
moola train --model rwkv_ts --device cuda

# Test XGBoost with GPU
moola train --model xgb --device cuda
```

## 3. Monitor GPU During Training

Open a second terminal and run:
```bash
# Live monitoring (updates every 1 second)
watch -n 1 nvidia-smi

# Or one-time check
nvidia-smi
```

**Success Criteria:**
- GPU Utilization: **80-95%** ✓
- GPU Memory: **2-6 GB** allocated
- Power Usage: **150-250W** (for A100)

## 4. Test Full OOF Pipeline

```bash
# Generate OOF for all deep learning models
moola oof --model cnn_transformer --device cuda
moola oof --model rwkv_ts --device cuda

# Should be 10-20x faster than CPU!
```

## 5. Benchmark Comparison

### CPU Baseline (before fixes):
```bash
# Run on CPU to compare
time moola train --model cnn_transformer --device cpu
# Expected: 15-30 minutes
```

### GPU Test (after fixes):
```bash
# Run on GPU
time moola train --model cnn_transformer --device cuda
# Expected: 1-3 minutes (10-20x faster)
```

## 6. Check Training Logs

Look for these lines in output:
```
[GPU] Training on: NVIDIA A100-SXM4-40GB
[GPU] Memory allocated: 0.05 GB
[GPU] Mixed precision (FP16): True
Epoch [2/10] Loss: 0.8234 Acc: 0.6500 GPU: 2.34GB
```

If you see `[WARNING] CUDA requested but not available`, something is wrong!

## 7. Troubleshooting

### If GPU shows 0% utilization:
```bash
# Check if PyTorch sees GPU
python -c "import torch; print(torch.cuda.is_available())"

# Check NVIDIA drivers
nvidia-smi

# Try reinstalling PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### If "CUDA out of memory":
```bash
# Reduce batch size in configs/default.yaml or model initialization
# Or set environment variable:
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### If models still slow:
```bash
# Check you're actually using GPU device:
moola train --model cnn_transformer --device cuda  # NOT --device cpu!
```

## 8. Full Pipeline Test

```bash
# Complete pipeline from scratch
moola ingest
moola oof --model logreg --device cpu
moola oof --model rf --device cpu
moola oof --model xgb --device cuda
moola oof --model rwkv_ts --device cuda
moola oof --model cnn_transformer --device cuda
moola stack-train
moola audit --section all
```

## Expected Timings (on A100)

| Model | CPU Time | GPU Time | Speedup |
|-------|----------|----------|---------|
| CNN-Transformer | 15-30 min | 1-3 min | 10-15x |
| RWKV-TS | 15-30 min | 1-3 min | 10-15x |
| XGBoost | 3-5 min | 30-60 sec | 3-5x |
| **Total OOF** | 60-120 min | 10-20 min | 5-10x |

## Success Checklist

After running tests:

- [x] `nvidia-smi` shows GPU utilization 80-95%
- [x] Training logs show `[GPU] Training on: <GPU_NAME>`
- [x] Mixed precision enabled: `[GPU] Mixed precision (FP16): True`
- [x] Training is 10-50x faster than CPU baseline
- [x] GPU memory usage is 2-6 GB (not 0 GB)
- [x] No "CUDA out of memory" errors with default batch size

If all checked ✓ → GPU fixes are working! 🎉
