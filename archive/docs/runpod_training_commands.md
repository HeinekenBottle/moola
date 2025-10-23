# Pointer-Favoring Patch Training Commands
# Execute these commands sequentially on RunPod after SSH connection

# 1. Change to workspace directory
cd /workspace/moola

# 2. Verify environment (optional but recommended)
python3 scripts/runpod/verify_runpod_env.py

# 3. Train pretrained encoder with specified config
python3 -m moola.cli pretrain-bilstm \
    --input data/raw/unlabeled_windows.parquet \
    --output artifacts/encoders/pretrained/stones_encoder_mae.pt \
    --device cuda \
    --epochs 100 \
    --batch-size 64 \
    --mask-ratio 0.4 \
    --input-dim 11 \
    --mask-strategy patch \
    --seed 1337

# 4. Train Jade model with pointer-favoring patch
python3 -m moola.cli train \
    --cfg-dir configs \
    --model jade \
    --data data/processed/train.parquet \
    --split artifacts/splits/v1/fold_0_temporal.json \
    --device cuda \
    --epochs 60 \
    --batch-size 29 \
    --predict-pointers \
    --save-run

# 5. Train Sapphire model with pretrained encoder
python3 -m moola.cli train \
    --cfg-dir configs \
    --model sapphire \
    --data data/processed/train.parquet \
    --split artifacts/splits/v1/fold_0_temporal.json \
    --device cuda \
    --epochs 40 \
    --batch-size 29 \
    --predict-pointers \
    --pretrained-encoder artifacts/encoders/pretrained/stones_encoder_mae.pt \
    --freeze-encoder \
    --save-run

# 6. Train Opal model with adaptive fine-tuning
python3 -m moola.cli train \
    --cfg-dir configs \
    --model opal \
    --data data/processed/train.parquet \
    --split artifacts/splits/v1/fold_0_temporal.json \
    --device cuda \
    --epochs 40 \
    --batch-size 29 \
    --predict-pointers \
    --pretrained-encoder artifacts/encoders/pretrained/stones_encoder_mae.pt \
    --no-freeze-encoder \
    --save-run

# 7. Check results
ls -la artifacts/runs/
ls -la artifacts/encoders/pretrained/

# Expected outputs:
# - Learned log_sigmas printed each epoch (σ_ptr ≈ 0.74, σ_type = 1.00)
# - Hit@±3, F1, ECE metrics reported
# - Hit@±3 should improve vs prior runs
# - Model checkpoints saved in artifacts/runs/