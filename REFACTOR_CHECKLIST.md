# Moola Architecture Refactor - Execution Checklist
**Date:** 2025-10-20  
**Estimated Time:** 6 hours  
**Status:** Ready to Execute

---

## 🚀 Pre-Flight Checklist

### Before You Start
- [ ] Read `REFACTOR_EXECUTIVE_SUMMARY.md` (5 min)
- [ ] Read `ARCHITECTURE_REFACTOR_PLAN.md` (15 min)
- [ ] Review `REFACTOR_VISUAL_GUIDE.md` (10 min)
- [ ] Confirm you have 6 hours available
- [ ] Confirm you're on `main` branch with clean working directory

### Create Backup
```bash
# 1. Check current branch and status
git status
git branch

# 2. Create backup branch
git checkout -b refactor/architecture-cleanup

# 3. Create local backup (optional but recommended)
cd ..
tar -czf moola_backup_$(date +%Y%m%d_%H%M%S).tar.gz moola/
cd moola

# 4. Run full test suite (establish baseline)
python3 -m pytest tests/ -v

# 5. Verify CLI works
python3 -m moola.cli --help
```

---

## 📋 Phase 1: Root Cleanup (30 min, Low Risk)

### Step 1.1: Delete AI Agent Configs
```bash
# Verify these are duplicates of ~/dotfiles
ls -la .mcp.json .env claude_code_zai_env.sh
cat .mcp.json  # Should be empty or minimal
cat .env       # Should only have GLM_API_KEY (unused)

# Delete AI agent configs
rm .mcp.json
rm .env
rm claude_code_zai_env.sh

# Delete OpenCode agent files
rm code_reviewer_agent.md
rm data_scientist_agent.md
rm ml_engineer_agent.md
rm python_pro_agent.md

# Delete OpenCode command files
rm code_review_command.md
rm data_analysis_command.md
rm ml_pipeline_command.md
rm python_dev_command.md
```

**Checklist:**
- [ ] Deleted .mcp.json
- [ ] Deleted .env
- [ ] Deleted claude_code_zai_env.sh
- [ ] Deleted 4 *_agent.md files
- [ ] Deleted 4 *_command.md files

### Step 1.2: Delete Temporary Documentation
```bash
# Archive or delete temporary docs
rm CLEANUP_SUMMARY_2025-10-19.md
rm WELCOME_BACK.md
rm CLAUDE_DESKTOP_ML_TRAINING_PROMPT.md
```

**Checklist:**
- [ ] Deleted CLEANUP_SUMMARY_2025-10-19.md
- [ ] Deleted WELCOME_BACK.md
- [ ] Deleted CLAUDE_DESKTOP_ML_TRAINING_PROMPT.md

### Step 1.3: Create .env.example
```bash
# Create .env.example (no secrets)
cat > .env.example << 'EOF'
# Moola Project Environment Variables
# Global API keys are configured in ~/dotfiles/.env

# Project-specific variables (examples)
# DATABASE_URL=postgresql://localhost/moola
# REDIS_URL=redis://localhost:6379
# LOG_LEVEL=INFO

# Note: Do NOT add global API keys here
# - ANTHROPIC_API_KEY (in ~/dotfiles/.env)
# - OPENAI_API_KEY (in ~/dotfiles/.env)
# - GLM_API_KEY (in ~/dotfiles/.env)
EOF
```

**Checklist:**
- [ ] Created .env.example
- [ ] Verified no secrets in .env.example

### Step 1.4: Test Phase 1
```bash
# Verify nothing broke
python3 -m moola.cli --help
python3 -m pytest tests/test_import.py -v

# Commit Phase 1
git add -A
git commit -m "refactor: Phase 1 - Remove duplicate AI configs and temp docs"
```

**Checklist:**
- [ ] CLI still works
- [ ] Tests still pass
- [ ] Committed Phase 1

---

## 📋 Phase 2: Move Scattered Artifacts (30 min, Medium Risk)

### Step 2.1: Create New Directories
```bash
# Create artifact directories
mkdir -p artifacts/models/supervised
mkdir -p artifacts/metadata
mkdir -p artifacts/oof
mkdir -p artifacts/runpod_bundles
```

**Checklist:**
- [ ] Created artifacts/models/supervised/
- [ ] Created artifacts/metadata/
- [ ] Created artifacts/oof/
- [ ] Created artifacts/runpod_bundles/

### Step 2.2: Move Model Files
```bash
# Move scattered model files
git mv model_174_baseline.pkl artifacts/models/supervised/simple_lstm_baseline_174.pkl
git mv model_174_pretrained.pkl artifacts/models/supervised/simple_lstm_pretrained_174.pkl
```

**Checklist:**
- [ ] Moved model_174_baseline.pkl
- [ ] Moved model_174_pretrained.pkl

### Step 2.3: Move Metadata Files
```bash
# Move metadata
git mv feature_metadata_174.json artifacts/metadata/
```

**Checklist:**
- [ ] Moved feature_metadata_174.json

### Step 2.4: Move OOF Predictions
```bash
# Move OOF predictions
git mv test_oof.npy artifacts/oof/simple_lstm_174.npy
```

**Checklist:**
- [ ] Moved test_oof.npy

### Step 2.5: Move RunPod Bundles
```bash
# Move RunPod bundles
mv runpod_bundle_*.tar.gz artifacts/runpod_bundles/ 2>/dev/null || true
mv runpod_bundle_20251020_013740 artifacts/runpod_bundles/ 2>/dev/null || true
mv runpod_bundle_build artifacts/runpod_bundles/ 2>/dev/null || true
```

**Checklist:**
- [ ] Moved runpod_bundle_*.tar.gz
- [ ] Moved runpod_bundle_20251020_013740/
- [ ] Moved runpod_bundle_build/

### Step 2.6: Delete Duplicate Directories
```bash
# Delete duplicates
rm -rf test_splits/  # Duplicate of data/splits/
rm -rf runpod_results/  # Duplicate of artifacts/runpod_results/
```

**Checklist:**
- [ ] Deleted test_splits/
- [ ] Deleted runpod_results/

### Step 2.7: Test Phase 2
```bash
# Verify nothing broke
python3 -m moola.cli --help
python3 -m pytest tests/test_import.py -v

# Commit Phase 2
git add -A
git commit -m "refactor: Phase 2 - Move scattered artifacts to proper locations"
```

**Checklist:**
- [ ] CLI still works
- [ ] Tests still pass
- [ ] Committed Phase 2

---

## 📋 Phase 3: Data Taxonomy Refactor (1 hour, High Risk)

### Step 3.1: Create New Data Structure
```bash
# Create new directories
mkdir -p data/raw/unlabeled
mkdir -p data/raw/labeled
mkdir -p data/processed/unlabeled
mkdir -p data/processed/labeled/metadata
mkdir -p data/processed/archived
mkdir -p data/oof/supervised
mkdir -p data/oof/pretrained
```

**Checklist:**
- [ ] Created data/raw/unlabeled/
- [ ] Created data/raw/labeled/
- [ ] Created data/processed/unlabeled/
- [ ] Created data/processed/labeled/metadata/
- [ ] Created data/processed/archived/
- [ ] Created data/oof/supervised/
- [ ] Created data/oof/pretrained/

### Step 3.2: Move Current Training Dataset
```bash
# Copy (don't move yet) current training dataset
cp data/processed/train_latest.parquet data/processed/labeled/train_latest.parquet
```

**Checklist:**
- [ ] Copied train_latest.parquet to labeled/

### Step 3.3: Archive Historical Datasets
```bash
# Move historical datasets to archived/
git mv data/processed/train_clean.parquet data/processed/archived/
git mv data/processed/train_clean_backup.parquet data/processed/archived/ 2>/dev/null || true
git mv data/processed/train_combined_174.parquet data/processed/archived/ 2>/dev/null || true
git mv data/processed/train_combined_175.parquet data/processed/archived/ 2>/dev/null || true
git mv data/processed/train_combined_178.parquet data/processed/archived/ 2>/dev/null || true
git mv data/processed/train_pivot_134.parquet data/processed/archived/ 2>/dev/null || true
git mv data/processed/train_smote_300.parquet data/processed/archived/ 2>/dev/null || true
git mv data/processed/train_3class_backup.parquet data/processed/archived/ 2>/dev/null || true
git mv data/processed/train_clean_phase2.parquet data/processed/archived/ 2>/dev/null || true
```

**Checklist:**
- [ ] Moved train_clean.parquet
- [ ] Moved other historical datasets

### Step 3.4: Create Archived Dataset README
```bash
cat > data/processed/archived/README.md << 'EOF'
# Archived Training Datasets

This directory contains historical training datasets for reference and reproducibility.

## Datasets

### train_clean.parquet (98 samples)
- **Date:** 2025-10-15
- **Purpose:** Training set before batch 200 integration
- **Samples:** 98 labeled windows
- **Source:** Batches 1-199 (keepers only)

### train_combined_174.parquet (174 samples)
- **Date:** 2025-10-18
- **Purpose:** First integration of batch 200 keepers
- **Samples:** 174 labeled windows (98 + 76 from batch 200)
- **Source:** train_clean.parquet + batch_200_clean_keepers.parquet

### train_combined_175.parquet (175 samples)
- **Date:** 2025-10-18
- **Purpose:** Experimental dataset with additional sample
- **Samples:** 175 labeled windows
- **Source:** train_combined_174.parquet + 1 additional sample

### train_combined_178.parquet (178 samples)
- **Date:** 2025-10-19
- **Purpose:** Experimental dataset with additional samples
- **Samples:** 178 labeled windows
- **Source:** train_combined_175.parquet + 3 additional samples

### train_pivot_134.parquet (134 samples)
- **Date:** 2025-10-10
- **Purpose:** Intermediate dataset during batch integration
- **Samples:** 134 labeled windows
- **Source:** Unknown (needs investigation)

### train_smote_300.parquet (300 samples)
- **Date:** 2025-10-12
- **Purpose:** Synthetic augmentation experiment using SMOTE
- **Samples:** 300 windows (174 real + 126 synthetic)
- **Source:** train_combined_174.parquet + SMOTE augmentation
- **Note:** Not used in production (synthetic data quality concerns)

## Current Dataset

The current training dataset is:
- **Location:** `data/processed/labeled/train_latest.parquet`
- **Samples:** 174 labeled windows
- **Date:** 2025-10-20

## Notes

- All archived datasets are kept for reproducibility
- Do NOT delete these files
- If you need to reproduce old experiments, use these datasets
- For new experiments, use `data/processed/labeled/train_latest.parquet`
EOF
```

**Checklist:**
- [ ] Created data/processed/archived/README.md

### Step 3.5: Move OOF Predictions
```bash
# Move OOF predictions to supervised/
mv data/oof/*.npy data/oof/supervised/ 2>/dev/null || true
```

**Checklist:**
- [ ] Moved OOF predictions to supervised/

### Step 3.6: Test Phase 3
```bash
# Test data loading
python3 << 'EOF'
import pandas as pd

# Test current dataset
df = pd.read_parquet("data/processed/labeled/train_latest.parquet")
print(f"Current dataset: {len(df)} samples")

# Test archived dataset
df_archived = pd.read_parquet("data/processed/archived/train_clean.parquet")
print(f"Archived dataset: {len(df_archived)} samples")

print("✅ Data loading works!")
EOF

# Commit Phase 3
git add -A
git commit -m "refactor: Phase 3 - Reorganize data with clear taxonomy"
```

**Checklist:**
- [ ] Data loading works
- [ ] Committed Phase 3

---

## 📋 Phase 4: Model/Encoder Taxonomy Refactor (1 hour, High Risk)

### Step 4.1: Create New Encoder/Model Structure
```bash
# Create directories
mkdir -p artifacts/encoders/pretrained
mkdir -p artifacts/encoders/supervised
mkdir -p artifacts/models/pretrained
mkdir -p artifacts/models/ensemble
```

**Checklist:**
- [ ] Created artifacts/encoders/pretrained/
- [ ] Created artifacts/encoders/supervised/
- [ ] Created artifacts/models/pretrained/
- [ ] Created artifacts/models/ensemble/

### Step 4.2: Move Existing Encoder
```bash
# Move BiLSTM encoder
git mv models/pretrained/bilstm_encoder.pt artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt
```

**Checklist:**
- [ ] Moved bilstm_encoder.pt

### Step 4.3: Move Existing Models
```bash
# Move existing models
git mv artifacts/models/enhanced_simple_lstm/model.pkl artifacts/models/supervised/enhanced_simple_lstm_174.pkl 2>/dev/null || true
git mv artifacts/models/logreg/model.pkl artifacts/models/supervised/logreg_174.pkl 2>/dev/null || true
```

**Checklist:**
- [ ] Moved enhanced_simple_lstm model
- [ ] Moved logreg model

### Step 4.4: Test Phase 4
```bash
# Test encoder loading
python3 << 'EOF'
import torch

# Test encoder loading
encoder_path = "artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt"
try:
    encoder = torch.load(encoder_path, map_location="cpu")
    print(f"✅ Encoder loaded: {encoder_path}")
except Exception as e:
    print(f"❌ Failed to load encoder: {e}")
EOF

# Commit Phase 4
git add -A
git commit -m "refactor: Phase 4 - Separate encoders from models with clear naming"
```

**Checklist:**
- [ ] Encoder loading works
- [ ] Committed Phase 4

---

## 📋 Phase 5: Code Changes (2 hours, High Risk)

### Step 5.1: Update src/moola/paths.py
**Note:** This requires manual editing. Update all artifact paths to match new structure.

**Checklist:**
- [ ] Updated ENCODER_PATH references
- [ ] Updated MODEL_PATH references
- [ ] Updated DATA_PATH references

### Step 5.2: Update src/moola/models/pretrained_utils.py
**Note:** Update encoder loading paths.

**Checklist:**
- [ ] Updated encoder loading paths

### Step 5.3: Update src/moola/pretraining/masked_lstm_pretrain.py
**Note:** Update encoder save paths.

**Checklist:**
- [ ] Updated encoder save paths

### Step 5.4: Update src/moola/cli.py
**Note:** Update default paths.

**Checklist:**
- [ ] Updated default paths

### Step 5.5: Update scripts/*.py
**Note:** This is the most time-consuming step. Use `git grep` to find all hardcoded paths.

```bash
# Find all hardcoded paths
git grep -n "models/pretrained" scripts/
git grep -n "data/processed/train" scripts/
git grep -n "artifacts/models" scripts/
```

**Checklist:**
- [ ] Updated all script paths

### Step 5.6: Test Phase 5
```bash
# Run full test suite
python3 -m pytest tests/ -v

# Test CLI
python3 -m moola.cli --help

# Commit Phase 5
git add -A
git commit -m "refactor: Phase 5 - Update all path references in code"
```

**Checklist:**
- [ ] All tests pass
- [ ] CLI works
- [ ] Committed Phase 5

---

## 📋 Phase 6: Documentation Updates (1 hour, Low Risk)

### Step 6.1: Update CLAUDE.md
**Note:** Update directory structure section.

**Checklist:**
- [ ] Updated directory structure
- [ ] Updated data flow diagram
- [ ] Updated model architecture section

### Step 6.2: Update README.md
**Note:** Update quick start paths.

**Checklist:**
- [ ] Updated quick start paths
- [ ] Updated example commands

### Step 6.3: Update docs/ARCHITECTURE.md
**Note:** Update architecture diagrams.

**Checklist:**
- [ ] Updated architecture diagrams
- [ ] Updated data flow section

### Step 6.4: Test Phase 6
```bash
# Commit Phase 6
git add -A
git commit -m "refactor: Phase 6 - Update documentation with new structure"
```

**Checklist:**
- [ ] Committed Phase 6

---

## ✅ Final Verification

### Run Full Test Suite
```bash
python3 -m pytest tests/ -v
```

**Checklist:**
- [ ] All tests pass

### Verify CLI Commands
```bash
python3 -m moola.cli --help
python3 -m moola.cli doctor
```

**Checklist:**
- [ ] CLI works
- [ ] Doctor command works

### Verify Root Directory
```bash
ls -la | grep -v "^\." | wc -l
# Should be ≤15 files
```

**Checklist:**
- [ ] Root directory has ≤15 files

### Create PR
```bash
# Push to remote
git push origin refactor/architecture-cleanup

# Create PR on GitHub
# Title: "refactor: Clean up directory structure and clarify data/model taxonomy"
# Description: See REFACTOR_EXECUTIVE_SUMMARY.md
```

**Checklist:**
- [ ] Pushed to remote
- [ ] Created PR
- [ ] Requested review

---

## 🎉 Success!

If you've completed all phases, you should have:
- ✅ Clean root directory (≤15 files)
- ✅ Clear data taxonomy (unlabeled/labeled, 4D/11D)
- ✅ Clear model taxonomy (encoders/models/weights)
- ✅ No duplicate configs or artifacts
- ✅ All tests passing
- ✅ All documentation updated

**Next Steps:**
1. Get PR reviewed
2. Merge to main
3. Delete backup branch
4. Celebrate! 🎉

