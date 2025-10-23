# Refactoring Scripts

Automated scripts to execute the project refactoring plan in phases.

## Overview

The refactoring is divided into 4 phases:

1. **Stripping**: Archive non-essential files, remove unused models/features
2. **Fixing**: Update model configs, fix imports and type errors
3. **Consolidation**: Merge duplicate code, unify configs
4. **Validation**: Run tests, check performance

## Usage

### Run All Phases
```bash
./scripts/refactor_master.sh
```

### Run Individual Phases
```bash
./scripts/refactor_phase1_stripping.sh
./scripts/refactor_phase2_fixing.py
./scripts/refactor_phase3_consolidation.py
./scripts/refactor_phase4_validation.sh
```

### Rollback
```bash
# Rollback all phases
./scripts/refactor_rollback.sh

# Rollback specific phase
./scripts/refactor_rollback.sh 2
```

## Safety Features

- **Backups**: Each phase creates a timestamped tar.gz backup
- **Checkpoints**: Phase completion is tracked with `.refactor_phaseX_done` files
- **Rollback**: Restore from backups if needed
- **Safety Checks**: Verify project structure before proceeding

## Phase Details

### Phase 1: Stripping
- Archives old documentation and scripts to `archive/`
- Removes unused model configs (Opal, Sapphire)
- Cleans up `__pycache__`, cache directories, DVC

### Phase 2: Fixing
- Fixes type errors in `zigzag.py` (None checks)
- Comments out missing imports in `cli.py`
- Updates configs to inherit from base

### Phase 3: Consolidation
- Finds and archives duplicate files
- Consolidates requirements files
- Unifies model configs

### Phase 4: Validation
- Runs linting and tests
- Tests feature building and model loading
- Performance benchmarking

## Files Created

- `scripts/refactor_master.sh` - Main orchestration script
- `scripts/refactor_phase[1-4]_*.sh/py` - Individual phase scripts
- `scripts/refactor_rollback.sh` - Rollback utility
- `.backup_phaseX_*.tar.gz` - Backup archives
- `.refactor_phaseX_done` - Completion flags

## Notes

- Scripts assume the project root has `pyproject.toml` and `src/moola/`
- Backups exclude artifacts, data, logs, and .git
- Rollback restores the entire project state from backup