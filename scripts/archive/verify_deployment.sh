#!/bin/bash
# Verify deployment package before upload to RunPod
# Usage: ./scripts/verify_deployment.sh

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🔍 Verifying RunPod Deployment Package"
echo "======================================="
echo ""

cd "$PROJECT_ROOT"

ERRORS=0
WARNINGS=0

# Check 1: Correct data file exists
echo -n "✓ Checking data file... "
if [[ ! -f "data/processed/train_pivot_134.parquet" ]]; then
    echo -e "${RED}FAIL${NC}"
    echo "  ❌ Missing data/processed/train_pivot_134.parquet"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}PASS${NC}"
    echo "  ✓ train_pivot_134.parquet exists ($(du -h data/processed/train_pivot_134.parquet | cut -f1))"
fi

# Check 2: Symlink is correct
echo -n "✓ Checking symlink... "
if [[ ! -L "data/processed/train.parquet" ]]; then
    echo -e "${RED}FAIL${NC}"
    echo "  ❌ data/processed/train.parquet is not a symlink"
    ERRORS=$((ERRORS + 1))
else
    TARGET=$(readlink data/processed/train.parquet)
    if [[ "$TARGET" == "train_pivot_134.parquet" ]]; then
        echo -e "${GREEN}PASS${NC}"
        echo "  ✓ train.parquet → train_pivot_134.parquet"
    else
        echo -e "${RED}FAIL${NC}"
        echo "  ❌ train.parquet → $TARGET (should be train_pivot_134.parquet)"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Check 3: requirements-runpod.txt exists
echo -n "✓ Checking minimal requirements... "
if [[ ! -f "requirements-runpod.txt" ]]; then
    echo -e "${RED}FAIL${NC}"
    echo "  ❌ Missing requirements-runpod.txt"
    ERRORS=$((ERRORS + 1))
else
    PKG_COUNT=$(grep -c "^[a-zA-Z]" requirements-runpod.txt || true)
    echo -e "${GREEN}PASS${NC}"
    echo "  ✓ requirements-runpod.txt exists ($PKG_COUNT packages)"
fi

# Check 4: No obsolete data files
echo -n "✓ Checking for obsolete files... "
OBSOLETE=()
if [[ -f "data/processed/train_2class.parquet" ]]; then
    OBSOLETE+=("train_2class.parquet")
fi
if [[ -f "data/processed/train_3class.parquet" ]]; then
    OBSOLETE+=("train_3class.parquet")
fi
if [[ -f "data/processed/reversals_archive.parquet" ]]; then
    OBSOLETE+=("reversals_archive.parquet")
fi
if [[ -d "data/processed/archive_corrupted_data" ]]; then
    OBSOLETE+=("archive_corrupted_data/")
fi

if [[ ${#OBSOLETE[@]} -gt 0 ]]; then
    echo -e "${YELLOW}WARN${NC}"
    echo "  ⚠️  Found obsolete files:"
    for file in "${OBSOLETE[@]}"; do
        echo "     - data/processed/$file"
    done
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}PASS${NC}"
    echo "  ✓ No obsolete data files found"
fi

# Check 5: Verify expansion indices in data
echo -n "✓ Verifying expansion indices... "
if python3 -c "
import sys
import pandas as pd
try:
    df = pd.read_parquet('data/processed/train_pivot_134.parquet')
    assert 'expansion_start' in df.columns, 'Missing expansion_start column'
    assert 'expansion_end' in df.columns, 'Missing expansion_end column'
    assert len(df) == 134, f'Expected 134 samples, got {len(df)}'
    assert 'features' in df.columns, 'Missing features column'
    print(f'✓ {len(df)} samples with expansion indices')
    sys.exit(0)
except Exception as e:
    print(f'✗ {str(e)}')
    sys.exit(1)
" 2>&1; then
    echo -e "${GREEN}PASS${NC}"
else
    echo -e "${RED}FAIL${NC}"
    ERRORS=$((ERRORS + 1))
fi

# Check 6: Verify deployment scripts use correct data
echo -n "✓ Checking deployment scripts... "
DEPLOY_ERRORS=()
if grep -q "train_2class.parquet" .runpod/deploy.sh 2>/dev/null; then
    DEPLOY_ERRORS+=("deploy.sh references train_2class.parquet")
fi
if grep -q "train_2class.parquet" .runpod/deploy-fast.sh 2>/dev/null; then
    DEPLOY_ERRORS+=("deploy-fast.sh references train_2class.parquet")
fi

if [[ ${#DEPLOY_ERRORS[@]} -gt 0 ]]; then
    echo -e "${RED}FAIL${NC}"
    for err in "${DEPLOY_ERRORS[@]}"; do
        echo "  ❌ $err"
    done
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}PASS${NC}"
    echo "  ✓ Deployment scripts use train_pivot_134.parquet"
fi

# Check 7: Cache cleanup
echo -n "✓ Checking for cache files... "
CACHE_COUNT=$(find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ruff_cache" \) 2>/dev/null | wc -l)
if [[ $CACHE_COUNT -gt 0 ]]; then
    echo -e "${YELLOW}WARN${NC}"
    echo "  ⚠️  Found $CACHE_COUNT cache directories (run: find . -name '__pycache__' -exec rm -rf {} +)"
    WARNINGS=$((WARNINGS + 1))
else
    echo -e "${GREEN}PASS${NC}"
    echo "  ✓ No cache files found"
fi

# Summary
echo ""
echo "======================================="
echo "Verification Summary"
echo "======================================="
if [[ $ERRORS -eq 0 ]] && [[ $WARNINGS -eq 0 ]]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Deployment package is clean and ready for RunPod."
    echo ""
    echo "Next steps:"
    echo "  1. Deploy: .runpod/deploy-fast.sh"
    echo "  2. SSH to RunPod: ssh runpod"
    echo "  3. Run training: bash scripts/start.sh --train"
    exit 0
elif [[ $ERRORS -eq 0 ]]; then
    echo -e "${YELLOW}⚠️  $WARNINGS warnings (non-critical)${NC}"
    echo ""
    echo "Deployment package is usable but has minor issues."
    echo "Consider fixing warnings before deploying."
    exit 0
else
    echo -e "${RED}❌ $ERRORS critical errors, $WARNINGS warnings${NC}"
    echo ""
    echo "Deployment package has issues that must be fixed!"
    echo "Fix errors before deploying to RunPod."
    exit 1
fi
