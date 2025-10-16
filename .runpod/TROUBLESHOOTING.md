# Moola RunPod Troubleshooting Guide

Solutions for common issues encountered during setup, training, and deployment.

---

## Table of Contents
1. [Pod Issues](#pod-issues)
2. [Training Issues](#training-issues)
3. [Data & Model Issues](#data--model-issues)
4. [Performance Issues](#performance-issues)
5. [Deployment & Sync Issues](#deployment--sync-issues)

---

## Pod Issues

### Pod Not Starting

**Symptom:** Pod status stays "Starting" for >5 minutes

**Root Causes:**
- Network volume not accessible
- Container disk too small
- GPU out of stock in region

**Solutions:**
1. **Check RunPod dashboard**: Look for error messages in pod details
2. **Verify network volume**: Ensure "moola" volume is available in selected region
3. **Try different GPU/region**: Switch to different GPU type or region (eu-ro-2 if eu-ro-1 unavailable)
4. **Increase container disk**: Set to minimum 50 GB (100 GB recommended)
5. **Terminate and retry**: Kill pod and start fresh

```bash
# Check network volume availability
aws s3 ls \
    --region eu-ro-1 \
    --endpoint-url https://s3api-eu-ro-1.runpod.io \
    s3://hg878tp14w/
```

---

### Pod Keeps Crashing / Out of Memory

**Symptom:** Pod starts then crashes/stops after minutes

**Root Causes:**
- Insufficient GPU VRAM for model
- Training process consuming too much RAM
- Network storage sync causing issues

**Solutions:**
1. **Check GPU VRAM**: Use higher-end GPU (RTX 4090 = 24 GB vs RTX 3090 = 24 GB)
2. **Reduce batch size**: Edit `configs/default.yaml`:
   ```yaml
   cnn_transformer:
     batch_size: 256  # Reduce from 512
   ```
3. **Disable mixed precision**: If instability seen:
   ```yaml
   training:
     use_amp: false  # Disable FP16
   ```
4. **Check pod logs**: `tail -f /workspace/logs/moola.log` for OOM errors
5. **Reduce num_workers**: Decrease DataLoader workers:
   ```yaml
   training:
     num_workers: 8  # Reduce from 16
   ```

```bash
# Monitor memory during training
watch -n 1 "nvidia-smi; free -h"
```

---

### SSH Connection Timeout

**Symptom:** SSH hangs or times out connecting to pod

**Root Causes:**
- Pod IP not fully initialized
- Firewall blocking SSH port 22
- Network latency

**Solutions:**
1. **Wait longer**: Sometimes pods take 30-60 seconds to fully initialize
2. **Use web terminal**: Log into RunPod dashboard, click "Connect" → "Web Terminal"
3. **Check SSH key**: Verify `~/.ssh/id_rsa` has correct permissions:
   ```bash
   chmod 600 ~/.ssh/id_rsa
   ```
4. **Try different network**: Switch WiFi or use VPN if behind corporate firewall
5. **Verify pod IP**: Copy IP from RunPod dashboard, not from cached terminal

---

## Training Issues

### OOF Predictions All Zeros

**Symptom:** OOF metrics show accuracy=0 or all zeros in `.npy` file

**Root Causes:**
- Data not loaded correctly
- Model not predicting (frozen weights)
- Fold iteration issue

**Solutions:**
1. **Verify data format**:
   ```bash
   python -c "
   import numpy as np, pandas as pd
   df = pd.read_parquet('data/processed/train.parquet')
   X = np.stack([np.stack(f) for f in df['features']])
   print(f'Data shape: {X.shape}')
   print(f'Data range: [{X.min():.3f}, {X.max():.3f}]')
   print(f'Has NaN: {np.isnan(X).any()}')
   "
   ```

2. **Check fold data splits**:
   ```bash
   moola audit --section oof
   ```

3. **Enable verbose logging**:
   ```bash
   moola oof --model cnn_transformer --over logging.level=DEBUG
   ```

4. **Check per-fold accuracies** in logs:
   - Look for `Fold N accuracy: X.XXX`
   - If any fold shows 0%, check that fold's data

---

### Model Accuracy Very Low / Unchanged Per Epoch

**Symptom:** All folds have accuracy ~0.33 (random guessing), no improvement

**Root Causes:**
- Wrong data shape
- Model not training (frozen layers)
- Class collapse (model predicts single class)

**Solutions:**
1. **Verify input shape**: Must be [N, 105, 4] for OHLC
   ```python
   from moola.validation.data_validator import validate_data_shape
   validate_data_shape(X, y)
   ```

2. **Check class distribution**:
   ```bash
   moola ingest --cfg-dir configs --input data/processed/train.parquet
   # Should show class counts
   ```

3. **Verify model is training**:
   - Check logs for loss values (should decrease)
   - Look for `[CLASS BALANCE]` output
   - Verify learning rate > 0

4. **Detect class collapse**:
   ```python
   import numpy as np
   oof = np.load('artifacts/oof/cnn_transformer/v1/seed_1337.npy')
   preds = np.argmax(oof, axis=1)
   unique, counts = np.unique(preds, return_counts=True)
   print(f"Predicted class distribution: {dict(zip(unique, counts))}")
   # Should be roughly balanced, not all one class
   ```

---

### CNN-Transformer Specific: Encoder Not Loading

**Symptom:** Logs show `[SSL] WARNING: Key not found in model` or weights unchanged

**Root Causes:**
- Pre-trained encoder not found
- Architecture mismatch between encoder and model
- Weights not actually transferred

**Solutions:**
1. **Verify encoder path exists**:
   ```bash
   ls -lh artifacts/models/ts_tcc/pretrained_encoder.pt
   ```

2. **Check encoder architecture matches**:
   - Encoder must have same: `cnn_channels=[64,128,128]`, `cnn_kernels=[3,5,9]`
   - View encoder hyperparams: `grep -r "cnn_channels" artifacts/`

3. **Check weight transfer in logs**:
   ```bash
   tail -200 /workspace/logs/moola.log | grep "\[SSL\]"
   ```
   Should show:
   ```
   [SSL] Loading pre-trained encoder from ...
   [SSL] Loaded 74 pre-trained layers
   [SSL] Encoder pre-training complete - ready for fine-tuning
   ```

4. **Manually verify weights changed**:
   ```python
   import torch
   checkpoint = torch.load('artifacts/models/cnn_transformer/model.pkl')
   state = checkpoint['model_state_dict']

   # Check CNN blocks were updated
   for key in state:
       if 'cnn_blocks' in key:
           print(f"✓ {key} loaded")
   ```

---

### Memory Error During Training

**Symptom:** `RuntimeError: CUDA out of memory` or similar

**Root Causes:**
- Batch size too large for available VRAM
- Memory accumulation (no garbage collection)
- Mixed precision not working

**Solutions:**
1. **Reduce batch size**:
   ```yaml
   # configs/default.yaml
   cnn_transformer:
     batch_size: 128  # From 512
   ```

2. **Clear GPU cache**:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

3. **Disable mixed precision**:
   ```yaml
   training:
     use_amp: false
   ```

4. **Reduce num_workers**:
   ```yaml
   training:
     num_workers: 0  # Disable parallel loading
   ```

5. **Use smaller data subset** (testing):
   ```bash
   # Limit training to first N samples
   moola train --over data.max_samples=50
   ```

---

## Data & Model Issues

### Invalid Expansion Indices (7 samples removed)

**Symptom:** Logs show `[DATA CLEAN] Removed 7/115 invalid samples`

**Root Causes:**
- Expansion start >= expansion end
- Indices outside valid range [30, 74]
- Malformed data from preprocessing

**Solutions:**
1. **Check data quality** before loading:
   ```bash
   moola audit --section base
   ```

2. **View invalid samples**:
   ```python
   import pandas as pd
   df = pd.read_parquet('data/processed/train.parquet')

   # Find invalid rows
   invalid = (
       (df['expansion_start'] >= df['expansion_end']) |
       (df['expansion_start'] < 30) |
       (df['expansion_end'] > 74)
   )
   print(df[invalid][['window_id', 'expansion_start', 'expansion_end']])
   ```

3. **Fix source data**: Regenerate from raw data with proper validation

---

### Data Leakage Between Train/Val Splits

**Symptom:** Validation accuracy suspiciously close to training accuracy

**Root Causes:**
- Improper stratification
- Data duplication during augmentation
- Incorrect split indices

**Solutions:**
1. **Verify split integrity**:
   ```bash
   moola audit --section all
   ```

2. **Check data checksums**:
   ```python
   from moola.validation.data_validator import compute_data_checksum

   X_train, X_val = ...
   cs_train = compute_data_checksum(X_train)
   cs_val = compute_data_checksum(X_val)

   # Should be completely different
   print(f"Train checksum: {cs_train[:16]}...")
   print(f"Val checksum: {cs_val[:16]}...")
   ```

3. **Use stratified split**:
   ```yaml
   training:
     stratified_split: true  # Ensure class balance preserved
   ```

---

## Performance Issues

### Training Very Slow

**Symptom:** Training takes >5 minutes per fold on RTX 4090

**Root Causes:**
- CPU bottleneck (I/O limited)
- num_workers too high (context switching)
- Mixed precision not working

**Solutions:**
1. **Check GPU utilization**:
   ```bash
   nvidia-smi dmon
   # Should show ~90%+ utilization
   ```

2. **Optimize DataLoader**:
   ```yaml
   training:
     batch_size: 512      # Larger batches
     num_workers: 8       # Not too high (2-8 is optimal)
     pin_memory: true
     persistent_workers: true
   ```

3. **Enable mixed precision**:
   ```yaml
   training:
     use_amp: true  # FP16 for faster compute
   ```

4. **Profile code**:
   ```bash
   python -m torch.utils.bottleneck moola train --model cnn_transformer
   ```

---

### OOF Generation Takes Too Long

**Symptom:** Each fold takes >2 minutes

**Root Causes:**
- Model too large
- Validation set too large
- Unnecessary augmentation

**Solutions:**
1. **Reduce model complexity**:
   ```yaml
   cnn_transformer:
     transformer_layers: 2  # From 3
     transformer_heads: 2   # From 4
   ```

2. **Reduce val_split** if testing:
   ```yaml
   cnn_transformer:
     val_split: 0.05  # From 0.15 (for testing only)
   ```

3. **Skip augmentation during val**:
   ```python
   # In oof.py: don't augment validation data
   # (already handled, but verify in logs)
   ```

---

## Deployment & Sync Issues

### Network Storage Sync Fails / Times Out

**Symptom:** `./sync-to-storage.sh` fails with timeout or connection error

**Root Causes:**
- Wrong AWS credentials
- Network connectivity issues
- S3 endpoint unreachable

**Solutions:**
1. **Verify credentials**:
   ```bash
   # Check environment variables are set
   env | grep AWS

   # Test S3 access
   aws s3 ls \
       --region eu-ro-1 \
       --endpoint-url https://s3api-eu-ro-1.runpod.io \
       s3://hg878tp14w/
   ```

2. **Check network-storage.env**:
   ```bash
   cat /Users/jack/projects/moola/.runpod/network-storage.env
   # Verify all variables set correctly
   ```

3. **Test connectivity**:
   ```bash
   curl -v https://s3api-eu-ro-1.runpod.io
   ```

4. **Retry with explicit flags**:
   ```bash
   aws s3 sync \
       --region eu-ro-1 \
       --endpoint-url https://s3api-eu-ro-1.runpod.io \
       local/path/ \
       s3://hg878tp14w/path/
   ```

---

### Can't Download Results from Pod

**Symptom:** `./sync-from-storage.sh` shows no files or fails

**Root Causes:**
- Training didn't upload artifacts
- Wrong S3 path
- Pod still training (files locked)

**Solutions:**
1. **Check training completed**:
   ```bash
   # SSH to pod
   ls -lh /workspace/artifacts/

   # Check for expected files
   ls /workspace/artifacts/oof/*/v1/seed_*.npy
   ```

2. **Manually upload**:
   ```bash
   # From pod SSH
   aws s3 cp \
       --recursive \
       /workspace/artifacts/ \
       s3://hg878tp14w/artifacts/ \
       --region eu-ro-1 \
       --endpoint-url https://s3api-eu-ro-1.runpod.io
   ```

3. **Check local sync**:
   ```bash
   cd /Users/jack/projects/moola/.runpod
   source network-storage.env
   aws s3 ls s3://hg878tp14w/artifacts/ --region eu-ro-1 --endpoint-url $S3_ENDPOINT
   ```

---

## GPU-Specific Issues

### GPU Shows No Available Memory But nvidia-smi Says Free

**Symptom:** PyTorch says out of memory, but `nvidia-smi` shows free memory

**Root Causes:**
- Memory fragmentation
- GPU process not released
- Multiple training runs in background

**Solutions:**
1. **Kill hanging processes**:
   ```bash
   pkill -f python
   pkill -f moola
   pkill -f torch
   ```

2. **Clear GPU cache**:
   ```bash
   python -c "import torch; torch.cuda.empty_cache(); print('Cache cleared')"
   ```

3. **Check what's using GPU**:
   ```bash
   nvidia-smi | grep python
   # Kill specific PID
   kill -9 <PID>
   ```

4. **Reset GPU**:
   ```bash
   # Requires sudo on some systems
   nvidia-smi -pm 1  # Enable persistence mode
   nvidia-smi -pm 0  # Disable if issues persist
   ```

---

## Logs & Debugging

### Enable Verbose Logging

```bash
# In configs/default.yaml
logging:
  level: DEBUG
  format: "[%(levelname)s] %(name)s | %(message)s"

# Or via CLI
moola oof --model cnn_transformer --over logging.level=DEBUG
```

### Check Specific Log Sections

```bash
# Show only SSL (encoder loading) messages
grep "\[SSL\]" /workspace/logs/moola.log

# Show only data validation messages
grep "\[DATA\|CLASS\|LOSS\]" /workspace/logs/moola.log

# Show only training metrics
grep "Epoch\|accuracy\|loss" /workspace/logs/moola.log

# Real-time monitoring
tail -f /workspace/logs/moola.log | grep "\[SSL\]"
```

### Collect Diagnostic Information

```bash
# Generate system diagnostic
{
  echo "=== Python ==="
  python --version
  echo "=== PyTorch ==="
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
  echo "=== GPU ==="
  nvidia-smi
  echo "=== Disk ==="
  df -h /workspace
  echo "=== Memory ==="
  free -h
  echo "=== Moola ==="
  python -c "import moola; print(f'Moola installed')"
  echo "=== Config ==="
  cat configs/default.yaml
} | tee diagnostic_report.txt
```

---

## Still Having Issues?

1. **Collect diagnostics**: Run the diagnostic script above
2. **Check logs**: Look for timestamps around the error
3. **Search error message**: Grep logs for exact error text
4. **Review recent changes**: Check git history for config changes
5. **Test with defaults**: Run with default config to isolate issue
6. **Ask for help**: Include diagnostic output and recent logs

---

*Last Updated: 2025-10-16*
*Compiled from real deployment issues and solutions*
