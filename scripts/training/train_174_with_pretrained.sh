#!/bin/bash

# Train Enhanced SimpleLSTM on 174 annotated samples with pretrained encoder
# This script is designed for RunPod GPU training

echo "🚀 Training Enhanced SimpleLSTM on 174 annotated samples"
echo "📅 Date: $(date)"
echo "🖥️  Host: $(hostname)"

# Set paths
DATA_PATH="/workspace/moola/data/processed/train_combined_174.parquet"
SPLIT_PATH="/workspace/moola/data/artifacts/splits/train_174_temporal.json"
PRETRAINED_PATH="/workspace/moola/artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt"

# Check if data exists
if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Training data not found: $DATA_PATH"
    echo "Please ensure the 174-sample training data is available"
    exit 1
fi

if [ ! -f "$SPLIT_PATH" ]; then
    echo "❌ Split file not found: $SPLIT_PATH"
    exit 1
fi

# Check if pretrained encoder exists
if [ ! -f "$PRETRAINED_PATH" ]; then
    echo "⚠️  Pretrained encoder not found: $PRETRAINED_PATH"
    echo "Training without pretrained encoder..."
    PRETRAINED_FLAG=""
else
    echo "✅ Using pretrained encoder: $PRETRAINED_PATH"
    PRETRAINED_FLAG="--pretrained-encoder $PRETRAINED_PATH --freeze-encoder"
fi

# Run training
echo "🏃‍♂️ Starting training..."
cd /workspace/moola

python3 -m moola.cli train \
    --model enhanced_simple_lstm \
    --data "$DATA_PATH" \
    --split "$SPLIT_PATH" \
    --device cuda \
    --use-engineered-features \
    --max-engineered-features 20 \
    $PRETRAINED_FLAG \
    --log-pretrained-stats

echo "✅ Training completed!"
echo "📊 Check results in experiment_results.jsonl"

# Show latest results
if [ -f "experiment_results.jsonl" ]; then
    echo "📈 Latest results:"
    tail -n 1 experiment_results.jsonl | python3 -m json.tool
fi