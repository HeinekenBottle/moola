#!/bin/bash

# Master refactoring script
# Runs phases 1-4 in sequence with safety checks

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Starting refactoring process..."

# Phase 1: Stripping
if [ ! -f ".refactor_phase1_done" ]; then
    echo "Running Phase 1: Stripping..."
    bash "$SCRIPT_DIR/refactor_phase1_stripping.sh"
else
    echo "Phase 1 already completed, skipping..."
fi

# Phase 2: Fixing
if [ ! -f ".refactor_phase2_done" ]; then
    echo "Running Phase 2: Fixing..."
    python3 "$SCRIPT_DIR/refactor_phase2_fixing.py"
else
    echo "Phase 2 already completed, skipping..."
fi

# Phase 3: Consolidation
if [ ! -f ".refactor_phase3_done" ]; then
    echo "Running Phase 3: Consolidation..."
    python3 "$SCRIPT_DIR/refactor_phase3_consolidation.py"
else
    echo "Phase 3 already completed, skipping..."
fi

# Phase 4: Validation
if [ ! -f ".refactor_phase4_done" ]; then
    echo "Running Phase 4: Validation..."
    bash "$SCRIPT_DIR/refactor_phase4_validation.sh"
else
    echo "Phase 4 already completed, skipping..."
fi

echo "Refactoring process completed successfully!"