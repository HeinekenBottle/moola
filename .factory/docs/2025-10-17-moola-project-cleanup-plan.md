## Project Cleanup Plan

### **Phase 1: Requirements Consolidation**
1. Merge and deduplicate the 4 requirements files into 2 focused files:
   - `requirements.txt` - Core dependencies only (cleaned up)
   - `requirements-runpod.txt` - Streamlined RunPod packages
2. Update `pyproject.toml` to be the single source of truth
3. Remove redundant requirements files

### **Phase 2: Script Consolidation** 
1. Consolidate deployment/orchestration into a single Python script
2. Archive shell scripts since you're moving away from them
3. Keep `deploy_to_fresh_pod.py` as the base, merge useful functionality from others

### **Phase 3: Documentation Cleanup**
1. Keep `WORKFLOW_SSH_SCP_GUIDE.md` as primary workflow doc
2. Archive or merge overlapping orchestration guides
3. Remove outdated documentation

### **Phase 4: File Structure Cleanup**
1. Archive old shell scripts and experiments  
2. Clean up artifacts directory
3. Remove duplicate configs and old deployment files

**Result:** Cleaner project structure with eliminated duplication, focused on SSH/SCP workflow without shell scripts.