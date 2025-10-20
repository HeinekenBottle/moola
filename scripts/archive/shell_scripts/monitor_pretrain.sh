#!/bin/bash
# Monitor TS-TCC pre-training on RunPod

SSH_CMD="ssh -i ~/.ssh/id_ed25519 -p 27424 root@213.173.102.99"

echo "===================="
echo "TS-TCC Pre-training Monitor"
echo "===================="
echo ""

echo "Process Status:"
$SSH_CMD "ps aux | grep 'pretrain_tcc_unlabeled' | grep -v grep | head -1" || echo "Not running"
echo ""

echo "GPU Status:"
$SSH_CMD "nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader"
echo ""

echo "Latest Training Output:"
$SSH_CMD "tail -20 /workspace/moola/logs/pretrain_tcc_unlabeled.log"
echo ""

echo "Encoder File:"
$SSH_CMD "ls -lh /workspace/moola/models/ts_tcc/pretrained_encoder.pt 2>/dev/null" || echo "Not yet created"
