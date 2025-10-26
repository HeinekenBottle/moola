# Quick Feature Pipeline Diagnostic

**One-liner to validate feature pipeline:**

```bash
python3 scripts/diagnose_feature_pipeline.py
```

---

## Expected Output (Healthy Pipeline)

```
✅ PASS: Mask ratio matches expected value (85.71%)
✅ PASS: Swing detection rates are healthy (>20%)
✅ PASS: No NaN values detected
✅ PASS: Candle features have healthy non-zero rates
✅ PASS: Swing features have reasonable zero rates
✅ PASS: All features within expected ranges

🎉 ALL CHECKS PASSED!
```

---

## Quick Manual Check

```bash
# Load 100 bars and verify features exist for all
python3 << 'PYEOF'
import pandas as pd, numpy as np, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from moola.features.relativity import build_features, RelativityConfig

df = pd.read_parquet("data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet").head(200)
X, mask, meta = build_features(df, RelativityConfig())

# Check range_z (should never be zero for valid bars)
range_z = X[:, :, 5]
zero_rate = (range_z == 0).sum() / range_z.size * 100

print(f"Windows: {X.shape[0]}, Features: {X.shape[2]}")
print(f"range_z zero rate: {zero_rate:.2f}% (expect ~0%)")
print(f"Valid mask ratio: {mask.sum() / mask.size * 100:.2f}% (expect ~85%)")
print("✅ OK" if zero_rate < 1 and abs(mask.sum() / mask.size * 100 - 85.71) < 1 else "❌ ISSUE")
PYEOF
```

---

## What to Look For

### Good Signs ✅
- Candle features: 80-100% non-zero
- Swing features: 78-97% non-zero
- Proxy features: 94-97% non-zero
- range_z: 100% non-zero
- Valid mask ratio: ~85.71%
- No NaN values

### Bad Signs ❌
- Any feature with >50% zeros (except bars_since_* which can be ~20%)
- NaN values detected
- range_z with zeros (indicates missing computation)
- Mask ratio != 85.71%
- Features outside expected ranges

---

## Troubleshooting

**If diagnostic fails:**

1. Check if relativity.py was modified recently
2. Verify zigzag.py is unchanged
3. Run: `git diff src/moola/features/`
4. Compare against commit: `2d986ef` (working version)

**If you need to revert:**
```bash
git checkout 2d986ef -- src/moola/features/relativity.py
```

---

**Last Validated:** 2025-10-25 (Commit 2d986ef)
