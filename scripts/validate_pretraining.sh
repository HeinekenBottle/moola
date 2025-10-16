#!/bin/bash
# Validation script to check if pre-training is running correctly

POD_HOST="213.173.110.220"
POD_PORT="36832"
POD_SSH="root@${POD_HOST}"

echo "=========================================="
echo "PRE-TRAINING VALIDATION CHECK"
echo "=========================================="
echo ""

# Check 1: Data file exists
echo "[CHECK 1] Verifying unlabeled_windows.parquet exists..."
if ssh -p ${POD_PORT} ${POD_SSH} "[ -f /workspace/data/raw/unlabeled_windows.parquet ]"; then
    echo "✅ File exists"

    # Get file size
    FILE_SIZE=$(ssh -p ${POD_PORT} ${POD_SSH} "du -h /workspace/data/raw/unlabeled_windows.parquet | cut -f1")
    echo "   Size: ${FILE_SIZE} (expected: ~2.2M)"

    # Count samples
    SAMPLE_COUNT=$(ssh -p ${POD_PORT} ${POD_SSH} "cd /workspace && python3 -c 'import pandas as pd; df=pd.read_parquet(\"data/raw/unlabeled_windows.parquet\"); print(len(df))'")
    echo "   Samples: ${SAMPLE_COUNT} (expected: 11873)"

    if [ "$SAMPLE_COUNT" == "11873" ]; then
        echo "   ✅ Correct sample count"
    else
        echo "   ❌ WRONG sample count!"
    fi
else
    echo "❌ File NOT found - run FIX_PRETRAINING_NOW.sh first!"
    exit 1
fi
echo ""

# Check 2: Process running
echo "[CHECK 2] Checking if pre-training process is running..."
PROCESS_COUNT=$(ssh -p ${POD_PORT} ${POD_SSH} "ps aux | grep -c 'pretrain_tcc_unlabeled' || true")
if [ "$PROCESS_COUNT" -gt "1" ]; then
    echo "✅ Pre-training process is running"
else
    echo "⚠️  Process not found - may have completed or not started"
fi
echo ""

# Check 3: Log file exists and shows progress
echo "[CHECK 3] Checking log file..."
if ssh -p ${POD_PORT} ${POD_SSH} "[ -f /workspace/pretrain_correct.log ]"; then
    echo "✅ Log file exists"

    # Check for key indicators
    echo ""
    echo "Last 15 lines of log:"
    echo "----------------------------------------"
    ssh -p ${POD_PORT} ${POD_SSH} "tail -15 /workspace/pretrain_correct.log"
    echo "----------------------------------------"
    echo ""

    # Extract sample count from log
    LOGGED_SAMPLES=$(ssh -p ${POD_PORT} ${POD_SSH} "grep -oP 'Loaded \K[0-9,]+(?= unlabeled samples)' /workspace/pretrain_correct.log | head -1 | tr -d ','")
    if [ ! -z "$LOGGED_SAMPLES" ]; then
        echo "   Samples loaded in training: ${LOGGED_SAMPLES}"
        if [ "$LOGGED_SAMPLES" == "11873" ]; then
            echo "   ✅ CORRECT sample count in training"
        else
            echo "   ❌ WRONG sample count in training!"
        fi
    fi
else
    echo "⚠️  Log file not found - training may not have started"
fi
echo ""

# Check 4: GPU utilization
echo "[CHECK 4] Checking GPU utilization..."
GPU_UTIL=$(ssh -p ${POD_PORT} ${POD_SSH} "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits")
GPU_MEM=$(ssh -p ${POD_PORT} ${POD_SSH} "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits")
echo "   GPU Utilization: ${GPU_UTIL}%"
echo "   GPU Memory Used: ${GPU_MEM} MiB"

if [ "$GPU_UTIL" -gt "50" ]; then
    echo "   ✅ GPU actively training"
elif [ "$GPU_UTIL" -gt "0" ]; then
    echo "   ⚠️  Low GPU usage - may be between epochs"
else
    echo "   ❌ GPU idle - training not running or completed"
fi
echo ""

# Check 5: Model checkpoint
echo "[CHECK 5] Checking for model checkpoint..."
if ssh -p ${POD_PORT} ${POD_SSH} "[ -f /workspace/models/ts_tcc/pretrained_encoder.pt ]"; then
    CHECKPOINT_SIZE=$(ssh -p ${POD_PORT} ${POD_SSH} "du -h /workspace/models/ts_tcc/pretrained_encoder.pt | cut -f1")
    echo "✅ Checkpoint exists: ${CHECKPOINT_SIZE}"

    if [[ "$CHECKPOINT_SIZE" =~ M$ ]]; then
        echo "   ✅ Reasonable size (expected: 10-20 MB)"
    else
        echo "   ⚠️  Small checkpoint - may be incomplete"
    fi
else
    echo "⚠️  Checkpoint not found - training in progress or failed"
fi
echo ""

echo "=========================================="
echo "VALIDATION SUMMARY"
echo "=========================================="
echo ""
echo "To monitor in real-time:"
echo "  ssh -p ${POD_PORT} ${POD_SSH} 'tail -f /workspace/pretrain_correct.log'"
echo ""
echo "To check GPU usage:"
echo "  ssh -p ${POD_PORT} ${POD_SSH} 'nvidia-smi'"
echo ""
echo "Expected training time: 20-40 minutes"
echo "=========================================="
