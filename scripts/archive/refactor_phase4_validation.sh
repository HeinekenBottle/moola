#!/bin/bash

# Phase 4: Validation
# Run tests, check performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Phase 4: Validation - Starting..."

# Safety check
if [ ! -f "pyproject.toml" ] || [ ! -d "src/moola" ]; then
    echo "Error: Not in project root"
    exit 1
fi

# Create backup (though validation shouldn't change files)
BACKUP_FILE=".backup_phase4_$(date +%Y%m%d_%H%M%S).tar.gz"
echo "Creating backup: $BACKUP_FILE"
tar czf "$BACKUP_FILE" --exclude='artifacts' --exclude='data' --exclude='logs' --exclude='.git' .

# Run linting
echo "Running linting..."
if command -v make &> /dev/null; then
    make lint
else
    echo "Make not found, running ruff directly..."
    python3 -m ruff check src/ tests/ scripts/
fi

# Run tests
echo "Running tests..."
if command -v make &> /dev/null; then
    make test
else
    echo "Make not found, running pytest directly..."
    python3 -m pytest tests/ -v --tb=short
fi

# Check imports
echo "Checking imports..."
python3 -c "import moola; print('Import successful')"

# Test feature building
echo "Testing feature building..."
if [ -f "data/sample_ohlcv.parquet" ]; then
    python3 -m moola.features.relativity --config configs/features/relativity.yaml --in data/sample_ohlcv.parquet --out /tmp/test_relativity.parquet
    python3 -m moola.features.zigzag --config configs/features/zigzag.yaml --in data/sample_ohlcv.parquet --out /tmp/test_zigzag.parquet
    echo "Feature building successful"
else
    echo "Sample data not found, skipping feature tests"
fi

# Test model loading
echo "Testing model loading..."
python3 -c "
from moola.models.jade_core import JadeCompact
import torch
model = JadeCompact(input_size=10, hidden_size=96, num_layers=1)
x = torch.randn(2, 105, 10)
out = model(x)
print('Model forward pass successful, output shape:', out['logits'].shape)
"

# Check performance (simple timing)
echo "Checking performance..."
python3 -c "
import time
from moola.models.jade_core import JadeCompact
import torch

model = JadeCompact(input_size=10, hidden_size=96, num_layers=1)
x = torch.randn(32, 105, 10)

start = time.time()
for _ in range(10):
    _ = model(x)
end = time.time()

print(f'10 forward passes took {end - start:.3f} seconds')
print(f'Average: {(end - start)/10:.3f} seconds per pass')
"

echo "Phase 4 completed successfully. Backup: $BACKUP_FILE"
touch .refactor_phase4_done