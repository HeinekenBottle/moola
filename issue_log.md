# Issue Log - Tmux Pane 3 Analysis
**Created:** 2025-10-20
**Source:** Tmux pane 3 capture during RunPod training session

## Issues Identified

### 1. Missing `configs/` Directory Error
**Severity:** Medium
**Status:** Workaround applied

**Problem:**
- Error: `configs: No such file or directory`
- RunPod expected a `configs/` directory that didn't exist locally or on remote

**Root Cause:**
- Directory not tracked in git (likely in .gitignore or just never created)
- SSH/SCP workflow transferred code but not empty directories

**Workaround Applied:**
- Created empty `configs/` directory on RunPod: `mkdir -p configs`
- Pre-training command has `--cfg-dir` option that can override default

**Permanent Fix Needed:**
- [ ] Decide if configs/ should be tracked in git (with .gitkeep)
- [ ] Or make code handle missing configs/ gracefully
- [ ] Document configs/ requirement in deployment docs

---

### 2. GitHub Sync Issues During RunPod Deployment
**Severity:** Medium
**Status:** Fixed in session, pattern recurring

**Problem:**
- Had to "fix GitHub sync issue - push latest code and re-pull on RunPod"
- Suggests local code was ahead of GitHub remote

**Root Cause:**
- Working on local changes without pushing to GitHub first
- RunPod pulls from GitHub, creating version mismatch

**Impact:**
- Wasted time: re-transfer data, reinstall dependencies
- Multiple round-trips to get environment correct

**Pattern:**
```
Mac (latest code) → GitHub (stale) → RunPod (stale clone)
                                          ↓
                                     Doesn't work!
                                          ↓
                              Push to GitHub, re-clone, retry
```

**Recommendations:**
- [ ] Add pre-deployment check: "Git status clean? Pushed to origin?"
- [ ] Create a deployment checklist/script
- [ ] Consider: git pre-push hook to remind about RunPod sync

---

### 3. Data Re-transfer Required After Code Sync
**Severity:** Low
**Status:** Completed, but inefficient

**Problem:**
- "Re-transfer data and reinstall dependencies"
- Data transfer is expensive (11D dataset is large)

**Root Cause:**
- Code sync issue forced clean slate approach
- No incremental fix possible

**Optimization Opportunities:**
- [ ] Keep data persistent on RunPod between sessions
- [ ] Separate code updates from data updates
- [ ] Use rsync instead of SCP for incremental transfers

---

### 4. Background Task Management
**Severity:** Low
**Status:** Working as designed

**Observation:**
- BiLSTM pre-training running in background
- Todo list shows progress: 3/6 tasks completed
- Status bar shows "1 background task"

**Not a problem, but note:**
- Background bash task 3404b3 (git branch creation) was killed before completing push
- May have orphaned branch `runpod-ready` locally

---

## Workflow Pain Points Summary

### High-Friction Areas:
1. **Git sync discipline** - Easy to forget to push before deploying to RunPod
2. **Directory assumptions** - Code expects certain dirs to exist
3. **All-or-nothing deploys** - Small code fix requires full re-setup

### Current Workarounds:
✅ Manual directory creation on RunPod
✅ Multiple SSH sessions to fix issues
✅ Re-transfer data when uncertain

### Suggested Improvements:
1. **Pre-deployment script** (`scripts/pre_deploy_check.sh`):
   ```bash
   # Check git status
   # Verify required directories
   # Confirm push to GitHub
   # Show checklist
   ```

2. **Graceful directory handling** in Python:
   ```python
   # In cli.py or wherever configs/ is used
   if not Path("configs").exists():
       logger.warning("configs/ not found, using defaults")
       Path("configs").mkdir(exist_ok=True)
   ```

3. **RunPod deployment guide** with common pitfalls:
   - [ ] Always push to GitHub first
   - [ ] Check .gitignore doesn't exclude critical dirs
   - [ ] Use rsync for data updates

---

## Action Items

### Immediate (This Session):
- [x] Document configs/ directory issue
- [ ] Check if BiLSTM pre-training completed successfully
- [ ] Verify encoder weights saved correctly

### Short Term (Next Session):
- [ ] Add configs/ to git with .gitkeep OR make code handle missing dir
- [ ] Create pre-deployment checklist
- [ ] Document "GitHub-first" deployment pattern in WORKFLOW_SSH_SCP_GUIDE.md

### Long Term (Next 2-4 Weeks):
- [ ] Build deployment helper script
- [ ] Investigate persistent RunPod storage for data
- [ ] Add git hooks for deployment reminders

---

## Notes
- Background task running: BiLSTM pre-training on GPU
- No critical blockers, workflow proceeding
- Issues are process/workflow related, not code bugs
