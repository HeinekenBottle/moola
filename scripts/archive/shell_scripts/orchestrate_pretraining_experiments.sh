#!/bin/bash
# Parallel Pre-training Experiment Orchestration
# Deploys 5 BiLSTM masked autoencoder experiments with different hyperparameters
# Usage: ./orchestrate_pretraining_experiments.sh <ssh_host> <ssh_port> <ssh_key>

set -e

SSH_HOST="${1:-root@213.173.108.43}"
SSH_PORT="${2:-15395}"
SSH_KEY="${3:-~/.ssh/id_ed25519}"

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║         MOOLA PRE-TRAINING EXPERIMENT ORCHESTRATION                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Target: $SSH_HOST:$SSH_PORT"
echo "SSH Key: $SSH_KEY"
echo ""

# ============================================================================
# EXPERIMENT CONFIGURATION MATRIX
# ============================================================================
# Each experiment tests a different critical hyperparameter combination
# Based on MLOps audit findings and specifications
# ============================================================================

declare -A EXPERIMENTS=(
    # Experiment 1: BASELINE (recommended spec)
    ["exp1_baseline"]="--epochs 75 --mask-strategy patch --time-warp-sigma 0.12 --num-augmentations 4"

    # Experiment 2: CONSERVATIVE (lower epochs, safer)
    ["exp2_conservative"]="--epochs 50 --mask-strategy patch --time-warp-sigma 0.10 --num-augmentations 3"

    # Experiment 3: AGGRESSIVE (more epochs, stronger aug)
    ["exp3_aggressive"]="--epochs 100 --mask-strategy patch --time-warp-sigma 0.15 --num-augmentations 5"

    # Experiment 4: BLOCK MASKING (different mask strategy)
    ["exp4_block_mask"]="--epochs 75 --mask-strategy block --time-warp-sigma 0.12 --num-augmentations 4"

    # Experiment 5: RANDOM MASKING (baseline strategy)
    ["exp5_random_mask"]="--epochs 75 --mask-strategy random --time-warp-sigma 0.12 --num-augmentations 4"
)

declare -A EXP_DESCRIPTIONS=(
    ["exp1_baseline"]="Baseline (recommended): 75 epochs, patch mask, sigma=0.12, 4x aug"
    ["exp2_conservative"]="Conservative: 50 epochs, patch mask, sigma=0.10, 3x aug"
    ["exp3_aggressive"]="Aggressive: 100 epochs, patch mask, sigma=0.15, 5x aug"
    ["exp4_block_mask"]="Block masking: 75 epochs, block mask, sigma=0.12, 4x aug"
    ["exp5_random_mask"]="Random masking: 75 epochs, random mask, sigma=0.12, 4x aug"
)

# Expected training duration (GPU RTX 4090)
declare -A EXP_DURATION=(
    ["exp1_baseline"]="35-40 min"
    ["exp2_conservative"]="25-30 min"
    ["exp3_aggressive"]="45-50 min"
    ["exp4_block_mask"]="35-40 min"
    ["exp5_random_mask"]="35-40 min"
)

# ============================================================================
# PHASE 1: VERIFY SSH CONNECTION
# ============================================================================
echo "Phase 1: Verifying SSH connection..."
if ! ssh -o ConnectTimeout=10 -o StrictHostKeyChecking=no -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "echo '✓ SSH connection successful'"; then
    echo "✗ SSH connection failed. Waiting for RunPod to come online..."
    echo "Please ensure the RunPod instance is running and try again."
    exit 1
fi
echo ""

# ============================================================================
# PHASE 2: UPLOAD UNLABELED DATA
# ============================================================================
echo "Phase 2: Uploading unlabeled data..."
echo "Creating directory structure..."
ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "mkdir -p /workspace/data/raw /workspace/data/artifacts/pretrained /workspace/logs/pretraining"

echo "Checking if data already exists..."
if ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "test -f /workspace/data/raw/unlabeled_windows.parquet"; then
    EXISTING_SIZE=$(ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "ls -lh /workspace/data/raw/unlabeled_windows.parquet | awk '{print \$5}'")
    echo "✓ Unlabeled data already exists on RunPod ($EXISTING_SIZE)"

    # Verify sample count
    SAMPLE_COUNT=$(ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "python3 -c 'import pandas as pd; df = pd.read_parquet(\"/workspace/data/raw/unlabeled_windows.parquet\"); print(len(df))'")
    echo "  Samples: $SAMPLE_COUNT"

    if [ "$SAMPLE_COUNT" != "11873" ]; then
        echo "  ⚠ WARNING: Expected 11,873 samples, got $SAMPLE_COUNT"
        echo "  Re-uploading data..."
        scp -P "$SSH_PORT" -i "$SSH_KEY" data/raw/unlabeled_windows.parquet "$SSH_HOST:/workspace/data/raw/"
    fi
else
    echo "Uploading unlabeled_windows.parquet (2.2 MB, 11,873 samples)..."
    scp -P "$SSH_PORT" -i "$SSH_KEY" data/raw/unlabeled_windows.parquet "$SSH_HOST:/workspace/data/raw/"
    echo "✓ Upload complete"
fi
echo ""

# ============================================================================
# PHASE 3: UPLOAD UPDATED CODEBASE
# ============================================================================
echo "Phase 3: Syncing updated codebase (configuration fixes)..."
echo "Uploading SimpleLSTM with new hyperparameters (hidden_size=128, time_warp_sigma=0.12)..."

# Use rsync for efficient sync
rsync -avz --progress -e "ssh -p $SSH_PORT -i $SSH_KEY" \
    --include='src/***' \
    --include='scripts/***' \
    --exclude='data/' \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    . "$SSH_HOST:/workspace/moola/"

echo "✓ Codebase synced"
echo ""

# ============================================================================
# PHASE 4: LAUNCH PARALLEL EXPERIMENTS
# ============================================================================
echo "Phase 4: Launching parallel pre-training experiments..."
echo ""
echo "Deploying 5 experiments in parallel:"
for exp_name in "${!EXPERIMENTS[@]}"; do
    echo "  • $exp_name: ${EXP_DESCRIPTIONS[$exp_name]}"
    echo "    Expected duration: ${EXP_DURATION[$exp_name]}"
done
echo ""

# Launch experiments in parallel
PIDS=()
for exp_name in exp1_baseline exp2_conservative exp3_aggressive exp4_block_mask exp5_random_mask; do
    ARGS="${EXPERIMENTS[$exp_name]}"
    LOG_FILE="/workspace/logs/pretraining/${exp_name}.log"
    OUTPUT_PATH="/workspace/data/artifacts/pretrained/${exp_name}_encoder.pt"

    echo "Launching $exp_name..."
    ssh -p "$SSH_PORT" -i "$SSH_KEY" "$SSH_HOST" "
        cd /workspace/moola && \
        PYTHONPATH=/workspace/moola/src:/workspace/moola:\$PYTHONPATH \
        nohup python3 -m moola.cli pretrain-bilstm \
            --input data/raw/unlabeled_windows.parquet \
            --output $OUTPUT_PATH \
            --device cuda \
            --augment \
            $ARGS \
        > $LOG_FILE 2>&1 &
        echo \$!
    " &

    PIDS+=($!)
    echo "  ✓ Launched (SSH session PID: ${PIDS[-1]})"
    sleep 2  # Stagger launches to avoid GPU contention
done

echo ""
echo "✓ All 5 experiments launched in parallel"
echo ""

# ============================================================================
# PHASE 5: MONITORING INSTRUCTIONS
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                      MONITORING INSTRUCTIONS                                 ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "To monitor experiment progress:"
echo ""
echo "1. View all experiment logs:"
echo "   ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST 'tail -f /workspace/logs/pretraining/*.log'"
echo ""
echo "2. Monitor specific experiment:"
for exp_name in exp1_baseline exp2_conservative exp3_aggressive exp4_block_mask exp5_random_mask; do
    echo "   ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST 'tail -f /workspace/logs/pretraining/${exp_name}.log'"
done
echo ""
echo "3. Check GPU usage:"
echo "   ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST 'watch -n 1 nvidia-smi'"
echo ""
echo "4. List completed checkpoints:"
echo "   ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST 'ls -lh /workspace/data/artifacts/pretrained/*.pt'"
echo ""
echo "Expected completion time: 45-50 minutes (when exp3_aggressive finishes)"
echo ""

# ============================================================================
# PHASE 6: VALIDATION CHECKLIST
# ============================================================================
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                     VALIDATION CHECKLIST                                     ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Once experiments complete, verify:"
echo ""
echo "  [ ] Dataset: 11,873 samples loaded (not 89!)"
echo "  [ ] Batches per epoch: ~24 (not 1)"
echo "  [ ] Training duration: 25-50 minutes (not 2-3)"
echo "  [ ] Final val loss: < 0.01 (converged reconstruction)"
echo "  [ ] Encoder file size: 10-20 MB (not tiny)"
echo "  [ ] Log shows 'Loaded 11,873 unlabeled samples'"
echo ""
echo "After validation, select best encoder and run fine-tuning:"
echo ""
echo "  ssh -p $SSH_PORT -i $SSH_KEY $SSH_HOST \\"
echo "    'cd /workspace/moola && python3 -m moola.cli oof \\"
echo "      --model simple_lstm --device cuda --seed 1337 \\"
echo "      --load-pretrained-encoder data/artifacts/pretrained/exp1_baseline_encoder.pt'"
echo ""
echo "Expected results:"
echo "  Overall accuracy: 64-72% (currently 57%)"
echo "  Class 1 accuracy: 40-60% (currently 0%)"
echo ""
echo "🚀 Orchestration complete! Experiments running in parallel on GPU cluster."
echo ""
