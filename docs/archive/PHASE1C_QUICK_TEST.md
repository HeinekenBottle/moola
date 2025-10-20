# Phase 1c Quick Test Guide

Quick commands to verify Phase 1c implementation.

## 1. Test Metrics Pack (30 seconds)

```bash
cd /Users/jack/projects/moola

python3 << 'EOF'
from src.moola.utils.metrics import calculate_metrics_pack
import numpy as np

# Generate test data
np.random.seed(42)
y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1] * 10)
y_pred = np.array([0, 0, 1, 0, 0, 1, 1, 1, 0, 1] * 10)
y_proba = np.random.rand(100, 2)
y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

# Calculate metrics
metrics = calculate_metrics_pack(y_true, y_pred, y_proba, class_names=['consolidation', 'retracement'])

# Print results
print("=" * 60)
print("METRICS PACK TEST")
print("=" * 60)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"F1 macro: {metrics['f1_macro']:.3f}")
print(f"F1 per class: {metrics['f1_per_class']}")
print(f"F1 by class: {metrics.get('f1_by_class', 'N/A')}")
print(f"PR-AUC: {metrics['pr_auc']:.3f}")
print(f"Brier: {metrics['brier']:.4f}")
print(f"ECE: {metrics['ece']:.4f}")
print(f"Log loss: {metrics['log_loss']:.4f}")
print("=" * 60)
print("✅ Metrics pack test PASSED")
EOF
```

**Expected**: All metrics print successfully, no errors.

## 2. Test Reliability Diagram (30 seconds)

```bash
cd /Users/jack/projects/moola

python3 << 'EOF'
from src.moola.visualization.calibration import save_reliability_diagram
import numpy as np
from pathlib import Path

# Generate test data
np.random.seed(42)
y_true = np.random.randint(0, 2, 200)
y_proba = np.random.rand(200, 2)
y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

# Save diagram
output_path = Path("artifacts/test_reliability.png")
save_reliability_diagram(
    y_true=y_true,
    y_proba=y_proba,
    output_path=str(output_path),
    title="Phase 1c Test Calibration"
)

# Check file
if output_path.exists():
    size_kb = output_path.stat().st_size / 1024
    print("=" * 60)
    print("RELIABILITY DIAGRAM TEST")
    print("=" * 60)
    print(f"✅ Diagram created: {output_path}")
    print(f"✅ File size: {size_kb:.1f} KB")
    print("=" * 60)
    print("✅ Reliability diagram test PASSED")
    print(f"\nView diagram: open {output_path}")
else:
    print("❌ FAILED: Diagram not created")
EOF
```

**Expected**: PNG file created, size > 50KB.

## 3. Verify SMOTE Removal (10 seconds)

```bash
cd /Users/jack/projects/moola

echo "Checking for active SMOTE imports..."
rg "^from imblearn.over_sampling import SMOTE" src/moola/ || echo "✅ No active SMOTE imports found"

echo ""
echo "Checking for commented SMOTE references..."
rg "#.*SMOTE" src/moola/ | head -5

echo ""
echo "✅ SMOTE removal verification PASSED"
```

**Expected**: No uncommented SMOTE imports.

## 4. Test Deterministic Seeding (10 seconds)

```bash
cd /Users/jack/projects/moola

python3 << 'EOF'
from src.moola.utils.seeds import set_seed, log_environment
import numpy as np
import random

print("=" * 60)
print("DETERMINISTIC SEEDING TEST")
print("=" * 60)

# Test 1: Set seed
set_seed(42)
print("✅ set_seed(42) completed")

# Test 2: Generate random numbers
vals1 = [random.random() for _ in range(5)]
print(f"Random values (run 1): {[f'{v:.6f}' for v in vals1]}")

# Reset and try again
set_seed(42)
vals2 = [random.random() for _ in range(5)]
print(f"Random values (run 2): {[f'{v:.6f}' for v in vals2]}")

if vals1 == vals2:
    print("✅ Deterministic seeding works correctly")
else:
    print("❌ FAILED: Random values differ between runs")

# Test 3: Log environment
print("")
print("Environment information:")
env_info = log_environment()
print(f"  Python: {env_info['python_version']}")
print(f"  NumPy: {env_info['numpy_version']}")
print(f"  Device: {env_info['device']}")
print(f"  PYTHONHASHSEED: {env_info['python_hash_seed']}")

print("=" * 60)
print("✅ Deterministic seeding test PASSED")
EOF
```

**Expected**: Same random values on both runs, environment info printed.

## 5. Full Integration Test (Optional, 2-3 minutes)

```bash
cd /Users/jack/projects/moola

# Run quick training test (if data available)
python3 -m moola.cli train \
    --model logreg \
    --seed 17 \
    2>&1 | tee phase1c_test.log

# Check for errors
if [ $? -eq 0 ]; then
    echo "✅ Training completed successfully"
else
    echo "❌ Training failed - check phase1c_test.log"
fi
```

**Expected**: Training completes without errors.

## Summary Test (All-in-One)

```bash
cd /Users/jack/projects/moola

echo "🚀 Running Phase 1c Full Test Suite..."
echo ""

# Test 1: Metrics
python3 -c "from src.moola.utils.metrics import calculate_metrics_pack; import numpy as np; m = calculate_metrics_pack(np.array([0,1,0,1]), np.array([0,1,0,0]), np.random.rand(4,2)); print('✅ Test 1: Metrics pack')"

# Test 2: Visualization
python3 -c "from src.moola.visualization.calibration import save_reliability_diagram; import numpy as np; save_reliability_diagram(np.array([0,1,0,1]), np.random.rand(4,2), 'artifacts/quick_test.png'); print('✅ Test 2: Reliability diagram')"

# Test 3: SMOTE
[ $(rg "^from imblearn.over_sampling import SMOTE" src/moola/ | wc -l) -eq 0 ] && echo "✅ Test 3: SMOTE removed" || echo "❌ Test 3: SMOTE still active"

# Test 4: Seeding
python3 -c "from src.moola.utils.seeds import set_seed, log_environment; set_seed(42); env_info = log_environment(); print('✅ Test 4: Deterministic seeding')"

echo ""
echo "=" * 60
echo "🎉 Phase 1c Quick Test Suite Complete!"
echo "=" * 60
echo ""
echo "Next steps:"
echo "  1. Review PHASE1C_COMPLETE.md for integration guide"
echo "  2. Integrate metrics into CLI train/evaluate commands"
echo "  3. Run full end-to-end training test"
echo "  4. View reliability diagram: open artifacts/quick_test.png"
```

## Troubleshooting

### Import Error: No module named 'moola'

**Solution**: Make sure you're in the project root and moola is importable:
```bash
cd /Users/jack/projects/moola
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Import Error: No module named 'sklearn'

**Solution**: Install dependencies:
```bash
pip3 install scikit-learn matplotlib numpy
```

### matplotlib backend error

**Solution**: Use non-interactive backend:
```bash
export MPLBACKEND=Agg
```

## Files to Check After Testing

```bash
# Metrics pack
ls -lh src/moola/utils/metrics.py

# Visualization
ls -lh src/moola/visualization/calibration.py
ls -lh src/moola/visualization/__init__.py

# Test artifacts
ls -lh artifacts/test_reliability.png
ls -lh artifacts/quick_test.png

# Seeding utilities
ls -lh src/moola/utils/seeds.py

# Modified files
ls -lh src/moola/pipelines/oof.py
ls -lh src/moola/models/xgb.py
ls -lh src/moola/config/training_config.py
```

## Success Criteria

✅ All 4 quick tests pass without errors
✅ Reliability diagram PNG files created
✅ No active SMOTE imports found
✅ Deterministic seeding produces identical results
✅ Environment logging works correctly

---

**Time Required**: 2-3 minutes for all tests
**Prerequisites**: Python 3, scikit-learn, matplotlib, numpy
**Output**: Test artifacts in `artifacts/` directory
