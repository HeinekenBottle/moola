# Full Real Training Pipeline Deployment Guide

## Overview
The `full_training_real.py` script is ready for deployment to RunPod. This implements the complete training pipeline with:
- Real Jade model training with pointer-favoring (1.82x weight ratio)
- Enhanced hit@1 and hit@3 metrics with ±3 timestep tolerance
- Comprehensive model analysis and uncertainty monitoring
- Production-ready checkpointing and detailed logging

## Quick Deployment Steps

### 1. Sync to RunPod
```bash
# Replace YOUR_RUNPOD_IP with your actual RunPod IP address
./scripts/sync_to_runpod.sh YOUR_RUNPOD_IP ubuntu
```

### 2. SSH into RunPod
```bash
ssh -i ~/.ssh/runpod_key ubuntu@YOUR_RUNPOD_IP
cd /workspace/moola
```

### 3. Run the Full Training Pipeline
```bash
# Option A: Direct execution
python3 full_training_real.py

# Option B: With nohup (for persistent training)
nohup python3 full_training_real.py > full_training.log 2>&1 &

# Monitor progress
tail -f full_training.log
```

### 4. Expected Outputs
The script will generate:
- `checkpoints/stage1_encoder.pth` - Simulated pretrained encoder
- `checkpoints/stage2_jade_best_hit3.pth` - Best model by Hit@3 performance
- `checkpoints/stage2_jade_best_loss.pth` - Best model by loss
- `checkpoints/stage2_jade_metrics.json` - Detailed metrics history
- `checkpoints/comprehensive_training_report.json` - Full training report
- `checkpoints/model_performance_analysis.json` - Performance analysis

## What the Script Does

### Stage 1: Simulated Pretraining (2-3 minutes)
- Simulates masked LSTM encoder pretraining
- Creates dummy encoder checkpoint for compatibility
- Logs pretraining progress

### Stage 2: Full Jade Training (15-30 minutes on GPU)
- **Dataset**: 174 samples, 105 timesteps, 11 features, 2 classes
- **Pointer-Favoring**: σ_ptr=0.55, σ_type=1.00 (1.82x ratio)
- **Training**: 60 epochs, batch size 29, early stopping
- **Metrics**: Hit@1 (exact), Hit@3 (±3 tolerance), MAE, type accuracy
- **Optimization**: AdamW, LR scheduling, gradient clipping

### Performance Targets
- **Excellent**: Hit@3 > 70%
- **Good**: Hit@3 > 50%
- **Needs Improvement**: Hit@3 < 50%

## Monitoring Progress

### Key Metrics to Watch
```bash
# Watch the log for these patterns:
grep "Hit@3" full_training.log
grep "Pointer-favoring" full_training.log
grep "Best validation Hit@3" full_training.log
```

### Expected Log Output
```
Epoch 1/60:
  Train: Loss=1.2345, Acc=0.650, Hit@1=0.120, Hit@3=0.340
  Val:   Loss=1.1892, Acc=0.680, Hit@1=0.140, Hit@3=0.380
  σ_ptr=0.550, σ_type=1.000, Ratio=1.82x, LR=3.00e-04, Time=2.1s
  MAE: Start=12.34, End=13.56, Center=8.45, Length=15.23
```

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size in the script
2. **Data loading errors**: Check data file exists at `data/processed/labeled/train_latest_11d.parquet`
3. **Import errors**: Run `python3 scripts/runpod/verify_runpod_env.py`

### Manual Sync Check
```bash
# Verify files synced correctly
ls -la full_training_real.py
ls -la data/processed/labeled/train_latest_11d.parquet
```

## Post-Training Analysis

### Retrieve Results
```bash
# On your Mac (new terminal):
scp -i ~/.ssh/runpod_key ubuntu@YOUR_IP:/workspace/moola/checkpoints/*.json ./
scp -i ~/.ssh/runpod_key ubuntu@YOUR_IP:/workspace/moola/checkpoints/*.pth ./
```

### Key Files to Examine
1. `comprehensive_training_report.json` - Full training summary
2. `model_performance_analysis.json` - Performance metrics
3. `stage2_jade_best_hit3.pth` - Best performing model

### Performance Assessment
The script will automatically assess performance:
- 🎯 EXCELLENT: Hit@3 > 70%
- ✅ GOOD: Hit@3 > 50%  
- ⚠️ NEEDS IMPROVEMENT: Hit@3 < 50%

## Next Steps After Training

1. **Review Results**: Examine the comprehensive report
2. **Model Analysis**: Check uncertainty parameters and convergence
3. **Performance Validation**: Compare Hit@3 against baseline
4. **Further Training**: If needed, adjust hyperparameters and rerun

## Technical Details

- **Random Seed**: 1337 (for reproducibility)
- **Device**: CUDA (automatically falls back to CPU)
- **Early Stopping**: 20 epochs patience
- **LR Scheduling**: ReduceLROnPlateau with factor 0.5
- **Gradient Clipping**: Max norm 2.0
- **Pointer Encoding**: Center+length normalized to [0,1]

The script is production-ready and includes comprehensive error handling, logging, and performance analysis.