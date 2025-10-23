# Scratch Directory - Temporary Artifacts

**AGENTS.md Section 4: Temporary artifacts policy**

## Purpose
Temporary work directory for experiments, exploratory analysis, and WIP artifacts that should **not** be imported from `src/`.

## Directory Structure
```
scratch/
├── README.md                 # This file - master index
├── jade/                     # Jade model experiments
├── features/                 # Feature engineering experiments  
└── cleanup/                  # Automated cleanup scripts
```

## Rules (Non-negotiable)

1. **Filenames must include component + intent**  
   Example: `scratch/jade/ablate_drop_wicks_v0.ipynb`
   Example: `scratch/features/feat_relativity_test_v1.py`

2. **7-day expiry policy** - Delete or promote within 7 days  
   - Add expiry date in file header comment
   - Weekly cleanup runs every Friday at 5 PM

3. **No imports from scratch/** - Never import scratch code in src/

4. **Track ownership and purpose** - Every file must include:
   ```python
   # Owner: <name>
   # Purpose: <one-line description>
   # Created: 2025-MM-DD
   # Expires: 2025-MM-DD  (7 days from creation)
   # Status: <wip|testing|obsolete>
   ```

## Current Artifacts

### Jade Experiments
*None currently*

### Feature Experiments  
*None currently*

## Cleanup Commands

```bash
# Find expired files
find scratch/ -name "*.py" -o -name "*.ipynb" | while read f; do
    if grep -q "Expires:" "$f"; then
        expiry=$(grep "Expires:" "$f" | cut -d: -f2 | xargs)
        if [[ $(date -d "$expiry" +%s) -lt $(date +%s) ]]; then
            echo "EXPIRED: $f (expired $expiry)"
        fi
    fi
done

# Manual cleanup (run with caution)
# ./scratch/cleanup/cleanup_expired.sh
```

## Promotion Guidelines

**When to promote to src/:**
- ✅ Code is tested and has unit tests
- ✅ Follows AGENTS.md naming conventions  
- ✅ Has proper documentation
- ✅ Passes linting and type checking
- ✅ Added to appropriate test suite

**When to keep in scratch/:**
- ❌ Exploratory analysis
- ❌ One-off experiments
- ❌ Debugging scripts
- ❌ Performance benchmarks
- ❌ Temporary data processing

## Git Integration

```bash
# scratch/ is gitignored - never commit artifacts
# However, commit this README.md for documentation
git add scratch/README.md
git commit -m "docs: add scratch directory guidelines"
```

## Emergency Cleanup

If disk space is critical:
```bash
# Clear everything older than 7 days
find scratch/ -type f -mtime +7 -delete
echo "Emergency cleanup completed $(date)"
```

---

**Last updated:** 2025-10-22  
**Cleanup schedule:** Every Friday 5:00 PM  
**Contact:** @moola-team for policy questions
