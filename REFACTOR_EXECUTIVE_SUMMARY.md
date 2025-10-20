# Moola Architecture Refactor - Executive Summary
**Date:** 2025-10-20  
**Status:** Planning Phase  
**Estimated Effort:** 6 hours  
**Risk Level:** Medium (requires code changes + testing)

---

## 🎯 Problem Statement

The Moola project has accumulated **root-level clutter** and **ambiguous naming** that makes it difficult to:
1. Understand which files are essential vs temporary
2. Distinguish between encoders, models, and weights
3. Identify which datasets are 4D OHLC vs 11D RelativeTransform
4. Separate unlabeled (pretraining) from labeled (supervised) data
5. Avoid duplicating global AI tool configurations

**Current Issues:**
- **Root directory:** 8 AI agent config files (duplicates of ~/dotfiles)
- **Scattered artifacts:** 7 model/metadata files at root instead of artifacts/
- **Ambiguous data naming:** 6+ versions of training data with unclear differences
- **Confusing model taxonomy:** Mixed terminology for encoders vs models vs weights

---

## 🎯 Solution Overview

### Three-Part Refactor

1. **Root Directory Cleanup**
   - Remove 8 AI agent config files (duplicates of ~/dotfiles)
   - Remove 3 temporary documentation files
   - Move 7 scattered artifacts to proper locations
   - Result: Clean root with only 15 essential files

2. **Data Taxonomy Refactor**
   - Separate unlabeled (2.2M samples) from labeled (174 samples)
   - Separate 4D OHLC from 11D RelativeTransform
   - Archive historical datasets with explanatory README
   - Result: Clear data flow from raw → processed → splits → oof

3. **Model/Encoder Taxonomy Refactor**
   - Distinguish encoders (feature extractors) from models (complete architectures)
   - Separate pretrained encoders from supervised models
   - Consistent naming convention: `{architecture}_{pretraining}_{features}_{size}.ext`
   - Result: Clear distinction between reusable encoders and trained models

---

## 📊 Key Decisions

### Decision 1: AI Tool Configurations
**Question:** Should AI agent configs be in the project or global?

**Answer:** **Global only** (~/dotfiles)

**Rationale:**
- Claude Code, OpenCode, and Factory are already configured globally
- API keys are in ~/dotfiles/.env (loaded by .zshrc)
- MCP servers are in ~/dotfiles/claude/.mcp.json
- Duplicating configs creates maintenance burden and security risk

**Action:**
- Delete `.mcp.json` (empty, duplicates global)
- Delete `.env` (contains unused GLM_API_KEY)
- Delete `claude_code_zai_env.sh` (duplicate of global)
- Delete 8 OpenCode agent/command files (duplicates of global)
- Create `.env.example` (no secrets, project-specific variables only)

### Decision 2: Data Organization
**Question:** How to organize unlabeled vs labeled data?

**Answer:** **Separate by purpose** (pretraining vs supervised)

**Structure:**
```
data/
├── raw/
│   ├── unlabeled/  # 2.2M samples for pretraining
│   └── labeled/    # (future: raw labeled data)
├── processed/
│   ├── unlabeled/  # 4D and 11D versions
│   ├── labeled/    # Current training set (174 samples)
│   └── archived/   # Historical datasets
```

**Rationale:**
- Clear separation of pretraining (unlabeled) vs supervised (labeled) workflows
- Explicit naming of 4D vs 11D features
- Historical datasets archived with README for context

### Decision 3: Encoder vs Model Taxonomy
**Question:** How to distinguish encoders from models?

**Answer:** **Separate directories + naming convention**

**Structure:**
```
artifacts/
├── encoders/
│   ├── pretrained/  # Self-supervised encoders
│   └── supervised/  # Encoders from supervised training
├── models/
│   ├── supervised/  # No pretraining
│   ├── pretrained/  # Fine-tuned from pretrained encoders
│   └── ensemble/    # Stacking ensemble
```

**Naming Convention:**
- Encoders: `{architecture}_{pretraining}_{features}_v{version}.pt`
  - Example: `bilstm_mae_4d_v1.pt`
- Models: `{architecture}_{encoder}_{features}_{size}.pkl`
  - Example: `simple_lstm_bilstm_mae_4d_174.pkl`

**Rationale:**
- Encoders are reusable feature extractors (can be loaded into multiple models)
- Models are complete architectures (encoder + classifier)
- Naming convention makes it clear what each file contains

---

## 📋 Migration Plan (6 Phases)

### Phase 1: Root Cleanup (30 min, Low Risk)
**Goal:** Remove duplicate configs and temporary docs

**Actions:**
- Delete 8 AI agent config files
- Delete 3 temporary documentation files
- Create .env.example (no secrets)

**Risk:** Low (no code changes)

### Phase 2: Move Scattered Artifacts (30 min, Medium Risk)
**Goal:** Consolidate artifacts into proper directories

**Actions:**
- Move 2 model files to artifacts/models/supervised/
- Move 1 metadata file to artifacts/metadata/
- Move 1 OOF file to artifacts/oof/
- Move 3 RunPod bundles to artifacts/runpod_bundles/
- Delete 2 duplicate directories

**Risk:** Medium (may break hardcoded paths)

### Phase 3: Data Taxonomy Refactor (1 hour, High Risk)
**Goal:** Reorganize data with clear taxonomy

**Actions:**
- Create new directory structure
- Move current training dataset to processed/labeled/
- Archive 6 historical datasets with README
- Reorganize OOF predictions by source

**Risk:** High (requires updating data loading code)

### Phase 4: Model/Encoder Taxonomy Refactor (1 hour, High Risk)
**Goal:** Separate encoders from models with clear naming

**Actions:**
- Create new encoder/model directory structure
- Rename existing encoder with new convention
- Move existing models to proper directories
- Update all path references in code

**Risk:** High (requires updating model loading code)

### Phase 5: Code Changes (2 hours, High Risk)
**Goal:** Update all path references in code

**Files to Update:**
- src/moola/paths.py
- src/moola/models/pretrained_utils.py
- src/moola/pretraining/masked_lstm_pretrain.py
- src/moola/cli.py
- scripts/*.py (50+ files)
- tests/*.py

**Risk:** High (must update all references before moving files)

### Phase 6: Documentation Updates (1 hour, Low Risk)
**Goal:** Update all documentation with new structure

**Files to Update:**
- CLAUDE.md
- README.md
- docs/ARCHITECTURE.md
- Create MIGRATION_GUIDE.md
- Create data/processed/archived/README.md
- Create artifacts/encoders/README.md
- Create artifacts/models/README.md

**Risk:** Low (documentation only)

---

## ✅ Success Criteria

### Quantitative
- [ ] Root directory: ≤15 files (currently ~40)
- [ ] No duplicate configs (currently 8)
- [ ] No scattered artifacts (currently 7)
- [ ] All tests pass (currently passing)
- [ ] All CLI commands work (currently working)

### Qualitative
- [ ] Clear separation of unlabeled vs labeled data
- [ ] Clear separation of 4D vs 11D features
- [ ] Clear distinction between encoders and models
- [ ] Consistent naming convention
- [ ] No duplicate directories or files

---

## 🚨 Risks & Mitigation

### High Risk: Breaking Existing Code
**Risk:** Moving files breaks hardcoded paths in 50+ scripts

**Mitigation:**
1. Update all path references in code BEFORE moving files
2. Use `git grep` to find all hardcoded paths
3. Test after each phase
4. Create backup branch before starting

### Medium Risk: Losing Experiment History
**Risk:** Archiving old datasets loses context

**Mitigation:**
1. Create detailed README in archived/ directory
2. Document each dataset's purpose and creation date
3. Keep all archived datasets (don't delete)

### Low Risk: Documentation Out of Sync
**Risk:** Documentation doesn't match new structure

**Mitigation:**
1. Update documentation in same PR as refactor
2. Review all docs before merging
3. Test all examples in documentation

---

## 📅 Recommended Execution Order

1. **Phase 1: Root Cleanup** (30 min)
   - Low risk, immediate benefit
   - No code changes required

2. **Phase 5: Code Changes** (2 hours)
   - Prepare code for migration
   - Update all path references
   - Test thoroughly

3. **Phase 2: Move Scattered Artifacts** (30 min)
   - Medium risk, code already updated
   - Test after moving

4. **Phase 3: Data Taxonomy Refactor** (1 hour)
   - High risk, code already updated
   - Test data loading

5. **Phase 4: Model/Encoder Taxonomy Refactor** (1 hour)
   - High risk, code already updated
   - Test model loading

6. **Phase 6: Documentation Updates** (1 hour)
   - Low risk, final step
   - Review all changes

**Total Time:** 6 hours

---

## 🎯 Next Steps

### Immediate (Before Starting)
1. **Review this plan** - Confirm approach with user
2. **Create backup branch** - `git checkout -b refactor/architecture-cleanup`
3. **Run full test suite** - Establish baseline
4. **Create backup** - `tar -czf moola_backup_$(date +%Y%m%d).tar.gz .`

### Phase 1 (Start Here)
1. **Delete AI agent configs** - Low risk, immediate benefit
2. **Delete temporary docs** - Low risk
3. **Create .env.example** - Low risk
4. **Test** - Verify nothing breaks

### Phase 2-6 (After Phase 1)
1. **Execute phases in recommended order**
2. **Test after each phase**
3. **Commit after each phase**
4. **Create PR when complete**

---

## 📚 Reference Documents

1. **ARCHITECTURE_REFACTOR_PLAN.md** - Detailed migration plan (this document)
2. **REFACTOR_VISUAL_GUIDE.md** - Visual before/after comparison
3. **~/dotfiles/QUICK_HANDOFF.md** - Global AI tool configuration reference
4. **CLAUDE.md** - Current architecture (to be updated)

---

## 🤔 Open Questions

1. **Should we keep WORKFLOW_SSH_SCP_GUIDE.md at root or move to docs/?**
   - Recommendation: Keep at root (frequently referenced)

2. **Should we delete or archive CLEANUP_SUMMARY_2025-10-19.md?**
   - Recommendation: Archive to docs/archive/

3. **Should we create a MIGRATION_GUIDE.md for future reference?**
   - Recommendation: Yes, document the migration process

4. **Should we update .gitignore to prevent future clutter?**
   - Recommendation: Yes, add patterns for common clutter

---

## 💡 Future Improvements (Post-Refactor)

1. **Automated path validation** - Script to check all paths are valid
2. **Dataset registry** - Central registry of all datasets with metadata
3. **Model registry** - Central registry of all models with performance metrics
4. **Artifact versioning** - Semantic versioning for encoders and models
5. **Automated cleanup** - Pre-commit hook to prevent root clutter

---

## 📞 Contact

**Questions or concerns?** Review the detailed plan in:
- `ARCHITECTURE_REFACTOR_PLAN.md` - Full migration details
- `REFACTOR_VISUAL_GUIDE.md` - Visual diagrams and examples

**Ready to start?** Follow the recommended execution order above.

