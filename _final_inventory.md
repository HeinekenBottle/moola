# Final Cleanup Inventory - 2025-10-21

## 1. Duplicate Directories Analysis

| Directory Pair | Location | Status | Action |
|---------------|----------|--------|--------|
| `config/` vs `configs/` | `src/moola/config/` vs `src/moola/configs/` | DUPLICATE | KEEP `configs/`, MERGE Python configs from `config/` to `utils/`, ARCHIVE `config/` |
| Root `configs/` | `/configs/` | PRIMARY | KEEP - Contains Stones YAML configs |
| `data/artifacts/` vs `artifacts/` | Root level | DUPLICATE | KEEP `artifacts/`, `data/artifacts/` appears empty/redundant |

**Decision:**
- **KEEP:** `src/moola/configs/` (YAML configs), root `configs/` (Stones configs)
- **MERGE:** `src/moola/config/*.py` → `src/moola/utils/config.py` or keep as separate module
- **ARCHIVE:** `src/moola/config/` after extracting Python config classes
- **CHECK:** `data/artifacts/` - if empty, remove

## 2. Stray Documentation Files

### Root-Level Markdown Files
| File | Size | Keep/Archive | Reason |
|------|------|--------------|--------|
| `README.md` | 5.7K | KEEP | Primary project documentation |
| `CLAUDE.md` | 21K | KEEP | AI assistant context (critical) |
| `CLEANUP_SESSION_2025-10-21.md` | 7.7K | ARCHIVE | Recent cleanup summary - move to archive |
| `README_CLEANUP.txt` | 1.2K | ARCHIVE | Previous cleanup summary - move to archive |

### Hidden/Meta Directories
| Directory | Status | Action |
|-----------|--------|--------|
| `.claude/` | KEEP | AI assistant skills and agents |
| `.factory/` | EVALUATE | Contains droids and docs - may be archivable |
| `.benchmarks/` | CHECK | May be empty or obsolete |
| `.dvc/` | CHECK | DVC not used (dvc.yaml already archived) |

**Recommendation:**
- Archive `CLEANUP_SESSION_2025-10-21.md` and `README_CLEANUP.txt` to `~/moola_archive/cleanup_docs/`
- Evaluate `.factory/` - if not actively used, archive to `~/moola_archive/factory_system/`
- Remove `.dvc/` if DVC not initialized (dvc.yaml already archived)
- Check `.benchmarks/` - remove if empty

## 3. Heavy Tracked Artifacts

### Git-Tracked Status
```bash
git ls-files | grep -E "\.(pt|pth|pkl|parquet|npy)$"
# Result: EMPTY (no heavy files tracked) ✅
```

**Status:** ✅ All heavy artifacts properly git-ignored

### Large Files Present (Git-Ignored)
Located in `artifacts/` and `data/` directories:
- `*.pt` files: 9 encoder/model files (100K-5MB each)
- `*.pkl` files: 20+ model checkpoints (100K-10MB each)
- `*.parquet` files: Data files in `data/` (git-ignored)
- `*.npy` files: OOF predictions in `data/oof/` (git-ignored)

**Action:** None needed - already properly git-ignored

## 4. Python Cache Files

| Type | Count | Action |
|------|-------|--------|
| `__pycache__/` directories | 30 | REMOVE ALL |
| `*.pyc` files | Unknown | REMOVE ALL |
| `.pytest_cache/` | 1 | REMOVE |
| `.ruff_cache/` | 1 | REMOVE |

**Command:**
```bash
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
rm -rf .pytest_cache .ruff_cache
```

## 5. Configuration Files Audit

### Root `configs/` Directory
```
configs/
├── __init__.py
├── default.yaml
└── model/
    ├── jade.yaml      ✅ KEEP (Stones)
    ├── opal.yaml      ✅ KEEP (Stones)
    └── sapphire.yaml  ✅ KEEP (Stones)
```

**Status:** ✅ Only Stones configs present

### `src/moola/configs/` Directory
```
src/moola/configs/
├── __init__.py
├── model/
│   ├── __init__.py
│   └── enhanced_simple_lstm.yaml  ⚠️ EVALUATE (duplicate of Stones configs?)
└── train/
    ├── __init__.py
    └── multitask.yaml  ✅ KEEP (training config)
```

**Action:**
- Check if `enhanced_simple_lstm.yaml` duplicates Stones configs
- If duplicate, remove; if unique training config, keep

## 6. Scripts Audit

### `scripts/` Directory
```
scripts/
├── __pycache__/           ❌ REMOVE
├── cleanlab/
│   └── run_cleanlab.py    ✅ KEEP (main entrypoint)
├── demo_bootstrap_ci.py   ⚠️ EVALUATE (is this used?)
├── generate_report.py     ⚠️ EVALUATE (is this used?)
└── runpod/
    ├── README.md          ✅ KEEP
    ├── dependency_audit.py ✅ KEEP (validation utility)
    └── verify_runpod_env.py ✅ KEEP (validation utility)
```

**Action:**
- Remove `scripts/__pycache__/`
- Check if `demo_bootstrap_ci.py` and `generate_report.py` are referenced in Makefile or code
- If not used, archive to `~/moola_archive/scripts_extras/`

## 7. Summary Statistics

### Current State
- **Total directories:** ~50+ (including subdirectories)
- **Duplicate directories:** 2 confirmed (`config/` vs `configs/`)
- **Stray docs:** 2 cleanup summaries (archivable)
- **Heavy files tracked:** 0 ✅
- **Cache directories:** 30+ `__pycache__/` + 2 tool caches
- **Stones configs:** 3 (jade, sapphire, opal) ✅

### Actions Required
1. **Deduplication:** Merge `src/moola/config/` → `src/moola/utils/` or keep separate
2. **Archive docs:** Move 2 cleanup summaries to archive
3. **Remove caches:** Delete all `__pycache__/`, `.pytest_cache`, `.ruff_cache`
4. **Evaluate scripts:** Check usage of `demo_bootstrap_ci.py`, `generate_report.py`
5. **Check directories:** `.factory/`, `.benchmarks/`, `.dvc/`, `data/artifacts/`
6. **Verify configs:** Ensure no duplicate YAML configs

### Expected Cleanup Impact
- **Directories removed:** 2-5 (config/, caches, possibly .factory/, .dvc/)
- **Files archived:** 2-4 (cleanup docs, possibly unused scripts)
- **Cache files removed:** 100+ (all __pycache__ and tool caches)
- **Functionality impact:** NONE (all changes preserve working code)

