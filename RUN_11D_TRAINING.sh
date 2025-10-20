#!/bin/bash
# 11D RelativeTransform Training Pipeline
# Based on ChatGPT's trajectory with all fixes applied
# Date: 2025-10-20

set -e  # Exit on error

echo "========================================================================"
echo "11D RELATIVETRANSFORM TRAINING PIPELINE"
echo "========================================================================"
echo ""
echo "This script executes the complete 11D training trajectory:"
echo "  1. Pretrain BiLSTM encoder on 11D features (masked autoencoder)"
echo "  2. Fine-tune EnhancedSimpleLSTM with frozen encoder (5 epochs)"
echo "  3. Fine-tune EnhancedSimpleLSTM with unfrozen encoder (25 epochs)"
echo "  4. Evaluate and gate for promotion"
echo ""
echo "Pitfalls avoided:"
echo "  ✓ Feature-dim mismatch: --input-dim 11 everywhere"
echo "  ✓ Hidden seeds: --seed 17 end-to-end"
echo "  ✓ Aug in val/test: --augment-data false"
echo ""
echo "========================================================================"
echo ""

# Check if CUDA is available
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "⚠️  WARNING: CUDA not available. This will be VERY slow on CPU."
    echo "Press Ctrl+C to cancel, or Enter to continue anyway..."
    read
fi

# Step 1: Pretrain 11D BiLSTM Encoder
echo ""
echo "========================================================================"
echo "STEP 1: PRETRAIN 11D BILSTM ENCODER"
echo "========================================================================"
echo ""
echo "Training masked autoencoder on 11D RelativeTransform features..."
echo "Expected time: ~20-30 minutes on GPU"
echo ""

python3 -m moola.cli pretrain-bilstm \
  --input data/processed/labeled/train_latest_relative.parquet \
  --input-dim 11 \
  --hidden-dim 128 \
  --num-layers 2 \
  --mask-ratio 0.15 \
  --mask-strategy patch \
  --epochs 50 \
  --batch-size 256 \
  --device cuda \
  --output artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --seed 17

echo ""
echo "✅ Step 1 complete: Encoder saved to artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt"
echo ""

# Step 2: Fine-tune with Frozen Encoder
echo ""
echo "========================================================================"
echo "STEP 2: FINE-TUNE WITH FROZEN ENCODER"
echo "========================================================================"
echo ""
echo "Training classifier head only (encoder frozen)..."
echo "Expected time: ~5-10 minutes on GPU"
echo ""

python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --freeze-encoder \
  --input-dim 11 \
  --epochs 5 \
  --augment-data false \
  --device cuda \
  --seed 17 \
  --save-run true

echo ""
echo "✅ Step 2 complete: Frozen encoder fine-tuning done"
echo ""

# Step 3: Fine-tune with Unfrozen Encoder
echo ""
echo "========================================================================"
echo "STEP 3: FINE-TUNE WITH UNFROZEN ENCODER"
echo "========================================================================"
echo ""
echo "Fine-tuning entire model (encoder unfrozen)..."
echo "Expected time: ~15-20 minutes on GPU"
echo ""

python3 -m moola.cli train \
  --model enhanced_simple_lstm \
  --data data/processed/labeled/train_latest_relative.parquet \
  --split data/splits/fwd_chain_v3.json \
  --pretrained-encoder artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt \
  --no-freeze-encoder \
  --input-dim 11 \
  --epochs 25 \
  --augment-data false \
  --device cuda \
  --seed 17 \
  --save-run true

echo ""
echo "✅ Step 3 complete: Full model fine-tuning done"
echo ""

# Step 4: Evaluation Instructions
echo ""
echo "========================================================================"
echo "STEP 4: EVALUATION AND GATING"
echo "========================================================================"
echo ""
echo "To evaluate the trained model, find the RUN_ID from the output above,"
echo "then run:"
echo ""
echo "  python3 -m moola.cli eval \\"
echo "    --run artifacts/runs/<RUN_ID> \\"
echo "    --metrics pr_auc brier ece accuracy f1_macro f1_per_class \\"
echo "    --event-metrics hit_at_pm3 lead_lag pointer_f1 \\"
echo "    --save-reliability-plot artifacts/plots/reliability_enh_rel.png \\"
echo "    --save-metrics artifacts/metrics/enh_rel.json"
echo ""
echo "Promotion rule:"
echo "  ✓ PR-AUC ↑ (higher than baseline)"
echo "  ✓ Brier ↓ (lower than baseline)"
echo "  ✓ ECE ≤ baseline + 0.02"
echo ""
echo "If gates pass, promote with:"
echo "  cp artifacts/runs/<RUN_ID>/model.pkl \\"
echo "     artifacts/models/supervised/enhanced_baseline_v2_relative.pt"
echo ""
echo "========================================================================"
echo "TRAINING PIPELINE COMPLETE"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Evaluate the model (see instructions above)"
echo "  2. Check metrics against baseline"
echo "  3. Promote if gates pass"
echo "  4. Work toward CPSA 0.25 → 0.5"
echo ""
echo "What we're working toward:"
echo "  - Short term: EnhancedSimpleLSTM + RelativeTransform + MAE encoder"
echo "  - Then: CPSA 0.25 → 0.5 (val/test real)"
echo "  - Later: TS2Vec pretrain for stronger adapter"
echo ""

