#!/bin/bash
# Monitor RunPod training progress

RUNPOD_HOST="root@203.57.40.224"
RUNPOD_PORT="10129"
SSH_KEY="~/.ssh/id_ed25519"

echo "🔍 Moola Baseline Training Monitor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ssh -p $RUNPOD_PORT -i $SSH_KEY $RUNPOD_HOST << 'EOF'
cd /root/moola

# Check if process is running
PROC_STATUS=$(ps aux | grep "train_baseline_100ep" | grep -v grep | wc -l)

if [ "$PROC_STATUS" -eq 0 ]; then
  echo "❌ Training process NOT running"
  echo ""
  echo "Check for results:"
  ls -lh artifacts/baseline_100ep/*.csv 2>/dev/null || echo "  No CSV files yet"
  exit 1
fi

echo "✅ Training process running"
echo ""

# Get runtime
RUNTIME=$(ps -p $(pgrep -f "train_baseline_100ep") -o etime= | xargs)
echo "⏱️  Runtime: $RUNTIME"
echo ""

# Get current epoch from best model
echo "📊 Progress:"
python3 << 'PYEOF'
import torch
try:
    checkpoint = torch.load("/root/moola/artifacts/baseline_100ep/best_model.pt", map_location="cpu")
    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('val_loss', 0.0)
    print(f"  • Best model: Epoch {epoch}, Val Loss = {val_loss:.4f}")
except Exception as e:
    print(f"  • Unable to load checkpoint: {e}")
PYEOF

echo ""

# Check for checkpoint files
CHECKPOINTS=$(ls artifacts/baseline_100ep/checkpoint_*.pt 2>/dev/null | wc -l)
echo "  • Checkpoints saved: $CHECKPOINTS"

# GPU status
echo ""
echo "🖥️  GPU:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader | \
  awk '{print "  • Utilization: " $1 ", VRAM: " $2 ", Temp: " $3}'

# Memory
echo ""
echo "💾 System Memory:"
free -h | awk 'NR==2{printf "  • Used: %s / %s (%.0f%%)\n", $3,$2,($3/$2)*100}'

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Training saves metrics CSVs at END (after epoch 100)"
echo "   Expected total time: ~15-20 minutes"
echo ""
echo "Run this script again to check progress, or use:"
echo "   watch -n 10 ./scripts/monitor_runpod_training.sh"
EOF
