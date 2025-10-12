#!/bin/bash
# Precise Training Script with python3 + dependency verification
# Usage: bash precise-train.sh

set -e

echo "⚡ PRECISE 2-CLASS TRAINING (python3 + dependency verification)"
echo "=========================================================="

# Environment
cd /workspace/moola
source /workspace/venv/bin/activate

# Verify environment
echo "🔍 Environment verification..."
echo "Python: $(python3 --version)"
echo "Pip: $(python3 -m pip --version)"

# Check critical dependencies
echo "📋 Critical dependency check..."
python3 -c "
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

packages = {
    'numpy': 'np',
    'torch': 'torch',
    'pandas': 'pd',
    'sklearn': 'sklearn',
    'xgboost': 'xgb',
    'loguru': None,
    'click': None,
    'rich': None
}

print('Dependency Status:')
for pkg, alias in packages.items():
    try:
        if alias:
            exec(f'import {pkg} as {alias}')
        else:
            exec(f'import {pkg}')
        module = sys.modules[pkg] if not alias else sys.modules[alias.split('.')[0]]
        print(f'  ✅ {pkg:12s}: {module.__version__}')
    except ImportError as e:
        print(f'  ❌ {pkg:12s}: {e}')

# Check CUDA
try:
    import torch
    if torch.cuda.is_available():
        print(f'  ✅ CUDA: {torch.version.cuda}')
        print(f'  ✅ GPU: {torch.cuda.get_device_name(0)}')
    else:
        print(f'  ⚠️  CUDA: Not available')
except:
    print(f'  ❌ CUDA: Check failed')
"

echo ""

# Verify data
echo "📊 Data verification..."
python3 -c "
import pandas as pd
import numpy as np

df = pd.read_parquet('/workspace/data/processed/train.parquet')
print(f'Dataset: {df.shape[0]} samples, {df.shape[1]} columns')
print(f'Classes: {sorted(df[\"label\"].unique())}')
print(f'Distribution: {df[\"label\"].value_counts().to_dict()}')
print(f'Data types: {df.dtypes.value_counts().to_dict()}')
print('✅ Data verification passed')
"

echo ""

# Phase 1: Baseline Models (deterministic)
echo "📊 Phase 1: Baseline Models (CPU)"
echo "================================="

baseline_models=("logreg" "rf" "xgb")
for model in "${baseline_models[@]}"; do
    echo "Training $model..."
    python3 -m moola.cli oof --model "$model" --device cpu --seed 1337

    # Verify OOF predictions created
    oof_path="/workspace/artifacts/oof/$model/v1/seed_1337.npy"
    if [[ -f "$oof_path" ]]; then
        echo "  ✅ OOF saved: $(python3 -c "import numpy as np; print(np.load('$oof_path').shape)")"
    else
        echo "  ❌ OOF not found!"
        exit 1
    fi
done

echo ""

# Phase 2: Deep Learning (GPU)
echo "🧠 Phase 2: Deep Learning (GPU)"
echo "=============================="

deep_models=("rwkv_ts" "cnn_transformer")
epochs=25

for model in "${deep_models[@]}"; do
    echo "Training $model ($epochs epochs)..."

    # Pre-GPU check
    python3 -c "
import torch
if not torch.cuda.is_available():
    print('❌ GPU not available for deep learning!')
    exit(1)
print(f'✅ GPU available: {torch.cuda.get_device_name(0)}')
"

    python3 -m moola.cli oof --model "$model" --device cuda --seed 1337 --epochs "$epochs"

    # Verify OOF predictions
    oof_path="/workspace/artifacts/oof/$model/v1/seed_1337.npy"
    if [[ -f "$oof_path" ]]; then
        echo "  ✅ OOF saved: $(python3 -c "import numpy as np; print(np.load('$oof_path').shape)")"
    else
        echo "  ❌ OOF not found!"
        exit 1
    fi
done

echo ""

# Phase 3: Stacking
echo "🎯 Phase 3: Stacking"
echo "===================="

python3 -m moola.cli stack-train --seed 1337

# Verify stack model
stack_path="/workspace/artifacts/models/stack/stack.pkl"
if [[ -f "$stack_path" ]]; then
    echo "  ✅ Stack model saved: $(ls -lh "$stack_path" | awk '{print $5}')"
else
    echo "  ❌ Stack model not found!"
    exit 1
fi

echo ""

# Results Summary
echo "📈 PRECISE RESULTS SUMMARY"
echo "=========================="

python3 -c "
import json
import numpy as np
import pandas as pd
from pathlib import Path

print('📊 Final Results:')
print('=' * 50)

# Stack results
try:
    with open('/workspace/artifacts/models/stack/metrics.json') as f:
        stack = json.load(f)
    print(f'Stack Model:')
    print(f'  Accuracy: {stack[\"accuracy\"]:.3f}')
    print(f'  F1 Score: {stack[\"f1\"]:.3f}')
    print(f'  ECE:      {stack[\"ece\"]:.3f}')
    print(f'  Log Loss: {stack[\"logloss\"]:.3f}')
except:
    print('❌ Stack results not found')

print()

# Base models
print('Base Models:')
print('-' * 20)
models = ['logreg', 'rf', 'xgb', 'rwkv_ts', 'cnn_transformer']
for model in models:
    metrics_file = f'/workspace/artifacts/oof/{model}/v1/metrics.json'
    try:
        with open(metrics_file) as f:
            m = json.load(f)
        print(f'{model:15s}: acc={m[\"accuracy\"]:.3f}, f1={m[\"f1\"]:.3f}')
    except:
        print(f'{model:15s}: Not available')

print()

# File summary
print('📁 Generated Files:')
print('-' * 20)
for root, dirs, files in Path('/workspace/artifacts').walk():
    level = root.replace('/workspace/artifacts', '').count('/')
    indent = '  ' * level
    print(f'{indent}{root.name}/')
    subindent = '  ' * (level + 1)
    for file in files:
        if file.endswith(('.npy', '.pkl', '.json')):
            print(f'{subindent}{file}')
"

echo ""

# Performance validation
echo "🎯 Performance Validation"
echo "======================="

python3 -c "
import json
import numpy as np

# Expected performance targets for 2-class problem
targets = {
    'baseline_accuracy': 0.55,  # Above random
    'target_accuracy': 0.65,    # Good performance
    'target_f1': 0.60           # Good F1
}

try:
    with open('/workspace/artifacts/models/stack/metrics.json') as f:
        results = json.load(f)

    acc = results['accuracy']
    f1 = results['f1']

    print(f'Performance Assessment:')
    print(f'  Accuracy: {acc:.3f} (target: >{targets[\"target_accuracy\"]:.2f})')
    print(f'  F1 Score: {f1:.3f} (target: >{targets[\"target_f1\"]:.2f})')

    if acc > targets['target_accuracy'] and f1 > targets['target_f1']:
        print('  ✅ EXCELLENT: Exceeded targets!')
    elif acc > targets['baseline_accuracy']:
        print('  ⚠️  GOOD: Above baseline, room for improvement')
    else:
        print('  ❌ POOR: Below baseline - check training')

except Exception as e:
    print(f'❌ Could not validate performance: {e}')
"

echo ""
echo "🎉 PRECISE TRAINING COMPLETE!"
echo ""
echo "📊 Results Summary:"
echo "   All artifacts saved to: /workspace/artifacts/"
echo "   Dependency tree: pip-tree"
echo "   Reproducible env: requirements.txt"
echo ""
echo "📥 Download to local:"
echo "   bash .runpod/sync-from-storage.sh artifacts"
echo ""
echo "🔍 Verify on local:"
echo "   python3 -c 'import joblib; m=joblib.load(\"data/artifacts/models/stack/stack.pkl\")'"