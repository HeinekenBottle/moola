# Ambiguous Files & Directories Audit
**Date:** 2025-10-20  
**Purpose:** Identify unclear/ambiguous files and directories at project root

---

## ✅ **KEEP - Standard Python/Dev Tools**

### `.venv/` - Python Virtual Environment
**Purpose:** Python virtual environment (created by `python3 -m venv .venv`)  
**Status:** ✅ **KEEP** - Essential for Python development  
**Why:** Contains all Python dependencies isolated from system Python  
**Size:** ~256 MB (typical for ML projects)  
**Action:** Already in .gitignore (correct)

### `.benchmarks/` - Pytest Benchmarks
**Purpose:** Pytest benchmark results  
**Status:** ✅ **KEEP** - Performance testing data  
**Contents:** README.md with benchmark documentation  
**Action:** Already in .gitignore (correct)

### `.dvc/` - Data Version Control
**Purpose:** DVC (Data Version Control) configuration  
**Status:** ✅ **KEEP** - Essential for data versioning  
**Action:** Already tracked by git (correct)

### `.claude/` - Claude Code Cache
**Purpose:** Local Claude Code cache (auto-generated)  
**Status:** ✅ **KEEP** - Auto-managed by Claude Code  
**Action:** Already in .gitignore (correct)

### `.factory/` - Factory Cache
**Purpose:** Local Factory cache (auto-generated)  
**Status:** ✅ **KEEP** - Auto-managed by Factory  
**Action:** Already in .gitignore (correct)

---

## ⚠️ **AMBIGUOUS - Needs Review**

### `candlesticks/` - Separate Git Repository
**Purpose:** Annotation interface project (separate repo)  
**Status:** ⚠️ **AMBIGUOUS** - This is a separate project!  
**Contents:** Full git repo with .git/, backend/, frontend/, scripts/  
**Issue:** This appears to be the `/Users/jack/projects/candlesticks` repo duplicated here  
**Recommendation:** 
- **DELETE** this directory (it's a duplicate)
- The real Candlesticks project is at `/Users/jack/projects/candlesticks`
- Integration should be via data files in `data/corrections/candlesticks_annotations/`
- **Action:** `rm -rf candlesticks/` (after confirming no unique data)

### `archived/` - Old Cleanup Artifacts
**Purpose:** Archived files from 2025-10-19 cleanup  
**Status:** ⚠️ **AMBIGUOUS** - Temporary archive  
**Contents:** 
- `2025-10-19-cleanup/_tm_commands/`
- `2025-10-19-cleanup/examples/`
- `2025-10-19-cleanup/moola-sdk/`
- `2025-10-19-cleanup/terraform/`
**Recommendation:**
- **DELETE** if no longer needed (appears to be old infrastructure code)
- These look like old Terraform/SDK files that were removed
- **Action:** `rm -rf archived/` (after confirming not needed)

### `benchmarks/` - Empty Except README
**Purpose:** Performance benchmarking (pytest-benchmark)  
**Status:** ⚠️ **AMBIGUOUS** - Only contains README, no actual benchmarks  
**Contents:** Just README.md (6.7 KB)  
**Recommendation:**
- **KEEP** directory structure (for future benchmarks)
- Already in .gitignore (correct)
- **Action:** No action needed

### `experiments/` - Experiment Framework
**Purpose:** Experiment orchestration framework  
**Status:** ⚠️ **AMBIGUOUS** - Overlaps with existing workflow  
**Contents:**
- `run_experiment.py`
- `run_parallel_phase1.sh`
- `test_pipeline.py`
- `verify_setup.py`
- `phase1_configs.yaml`
- READMEs and documentation
**Issue:** 
- Overlaps with `scripts/` directory
- Overlaps with `src/moola/cli.py` functionality
- May be an old experiment framework
**Recommendation:**
- **CONSOLIDATE** or **DELETE**
- If still used, move to `scripts/experiments/`
- If not used, delete
- **Action:** Review with user - likely can be deleted or moved

### `hooks/` - Git/Tool Hooks
**Purpose:** Pre-commit and tool hooks  
**Status:** ⚠️ **AMBIGUOUS** - Mix of git hooks and custom scripts  
**Contents:**
- `context_reminder.py`
- `posttooluse.sh`
- `pretooluse.py`
- `userpromptsubmit.sh`
**Issue:**
- Not standard git hooks (those are in `.git/hooks/`)
- Appears to be custom tool hooks (possibly for AI tools?)
- Unclear if actively used
**Recommendation:**
- **REVIEW** - Check if these are actively used
- If for AI tools, consider moving to `~/.config/` or `~/dotfiles/`
- If not used, delete
- **Action:** Review with user

### `logs/` - Application Logs
**Purpose:** Application logs (generate_structure_labels)  
**Status:** ⚠️ **AMBIGUOUS** - Should be in `data/logs/` or `artifacts/logs/`  
**Contents:**
- 6 log files from 2025-10-17
- `runpod_backup/` subdirectory
**Recommendation:**
- **MOVE** to `artifacts/logs/` (more appropriate location)
- Or **DELETE** if old logs not needed
- **Action:** `mv logs/ artifacts/logs/` or `rm -rf logs/`

### `monitoring/` - Production Monitoring
**Purpose:** Production monitoring infrastructure (Prometheus, Grafana)  
**Status:** ⚠️ **AMBIGUOUS** - Production infrastructure in dev repo  
**Contents:**
- `ab_testing_framework.py`
- `automated_response.py`
- `business_metrics_bridge.py`
- `deploy_monitoring.sh`
- `grafana/` configs
- `prometheus.yml`
- `performance_tracker.py`
**Issue:**
- This is production infrastructure code
- Doesn't belong in ML training repo
- Should be in separate infrastructure repo
**Recommendation:**
- **MOVE** to separate repo (e.g., `moola-infrastructure`)
- Or **ARCHIVE** if not actively used
- **Action:** Review with user - likely should be separate repo

### `results/` - Experiment Results
**Purpose:** Experiment results (JSON lines)  
**Status:** ⚠️ **AMBIGUOUS** - Overlaps with `artifacts/` and `data/`  
**Contents:**
- `experiment_results.jsonl` (818 bytes)
- `gated_workflow_results_final.jsonl` (1.7 KB)
- `gated_workflow_results_latest.jsonl` (1.9 KB)
**Recommendation:**
- **MOVE** to `artifacts/results/` (more appropriate location)
- Consolidate with existing results tracking
- **Action:** `mv results/ artifacts/results/`

### `configs/` - Model Configurations
**Purpose:** Hydra configuration files for models  
**Status:** ⚠️ **AMBIGUOUS** - Overlaps with `src/moola/config/`  
**Contents:**
- `cnn_transformer.yaml`
- `default.yaml`
- `hardware/` configs
- `simple_lstm.yaml`
- `ssl.yaml`
**Issue:**
- Overlaps with `src/moola/config/` directory
- Unclear which is canonical
**Recommendation:**
- **CONSOLIDATE** - Move to `src/moola/config/` (canonical location)
- Or keep at root if used by CLI directly
- **Action:** Review with user - likely move to `src/moola/config/`

### `models/` - Empty Model Directories
**Purpose:** Model storage (but mostly empty)  
**Status:** ⚠️ **AMBIGUOUS** - Overlaps with `artifacts/models/`  
**Contents:**
- `pretrained/` (empty)
- `stack/` (empty)
- `ts_tcc/` (has 1 file)
**Issue:**
- We just created `artifacts/models/` and `artifacts/encoders/`
- This `models/` directory is now redundant
- Already moved encoder from `models/pretrained/` to `artifacts/encoders/pretrained/`
**Recommendation:**
- **DELETE** `models/pretrained/` and `models/stack/` (empty)
- **MOVE** `models/ts_tcc/` to `artifacts/models/` if needed
- **Action:** Clean up empty directories

---

## 📋 **Recommended Actions Summary**

### **HIGH PRIORITY - Delete/Move**

1. **`candlesticks/`** - DELETE (duplicate of separate repo)
   ```bash
   rm -rf candlesticks/
   ```

2. **`archived/`** - DELETE (old cleanup artifacts)
   ```bash
   rm -rf archived/
   ```

3. **`models/pretrained/`** - DELETE (empty, replaced by artifacts/encoders/)
   ```bash
   rm -rf models/pretrained/
   ```

4. **`models/stack/`** - DELETE (empty, replaced by artifacts/models/ensemble/)
   ```bash
   rm -rf models/stack/
   ```

5. **`results/`** - MOVE to artifacts/
   ```bash
   mv results/ artifacts/results/
   ```

6. **`logs/`** - MOVE to artifacts/
   ```bash
   mv logs/ artifacts/logs/
   ```

### **MEDIUM PRIORITY - Review with User**

7. **`experiments/`** - Review if still needed
   - If yes: Move to `scripts/experiments/`
   - If no: Delete

8. **`monitoring/`** - Review if actively used
   - If yes: Move to separate infrastructure repo
   - If no: Archive or delete

9. **`hooks/`** - Review if actively used
   - If yes: Document purpose
   - If no: Delete

10. **`configs/`** - Consolidate with `src/moola/config/`
    - Move to `src/moola/config/` if not used by CLI directly

### **LOW PRIORITY - Keep**

11. **`benchmarks/`** - Keep (for future benchmarks)
12. **`.venv/`** - Keep (Python virtual environment)
13. **`.dvc/`** - Keep (data version control)
14. **`.claude/`** - Keep (auto-managed)
15. **`.factory/`** - Keep (auto-managed)

---

## 📊 **Impact Analysis**

### **If All High Priority Actions Taken:**

**Directories Deleted:** 4
- `candlesticks/` (~1 MB)
- `archived/` (~500 KB)
- `models/pretrained/` (empty)
- `models/stack/` (empty)

**Directories Moved:** 2
- `results/` → `artifacts/results/`
- `logs/` → `artifacts/logs/`

**Root Directory Reduction:**
- Before: 16 directories at root
- After: 10 directories at root (37.5% reduction)

**Cleaner Structure:**
```
moola/
├── .venv/              # Python virtual environment
├── artifacts/          # All artifacts (models, encoders, results, logs)
├── configs/            # Configuration files (or move to src/moola/config/)
├── data/               # All data (raw, processed, oof, batches)
├── docs/               # Documentation
├── experiments/        # (Review - possibly delete or move)
├── hooks/              # (Review - possibly delete)
├── monitoring/         # (Review - possibly move to separate repo)
├── scripts/            # Utility scripts
├── src/                # Source code
└── tests/              # Test suite
```

---

## 🎯 **Next Steps**

1. **Review this audit** - Confirm recommendations
2. **Execute high priority actions** - Delete/move obvious duplicates
3. **Review medium priority items** - Decide on experiments/, monitoring/, hooks/
4. **Update .gitignore** - Ensure new structure is properly ignored
5. **Document decisions** - Update CLAUDE.md with final structure

---

## ❓ **Questions for User**

1. **`candlesticks/`** - Can we delete this? (It's a duplicate of the separate repo)
2. **`experiments/`** - Is this still used? Or can we delete/consolidate?
3. **`monitoring/`** - Is this production code? Should it be in a separate repo?
4. **`hooks/`** - What are these hooks for? Are they actively used?
5. **`configs/`** - Should these be in `src/moola/config/` instead?

