#!/bin/bash

# Rollback refactoring changes
# Usage: ./refactor_rollback.sh [phase_number]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

PHASE="${1:-all}"

echo "Rolling back refactoring changes for phase: $PHASE"

# Find latest backup for each phase
rollback_phase() {
    local phase=$1
    local backup_file=$(ls -t .backup_phase${phase}_*.tar.gz 2>/dev/null | head -1)
    if [ -z "$backup_file" ]; then
        echo "No backup found for phase $phase"
        return 1
    fi

    echo "Rolling back from: $backup_file"
    tar xzf "$backup_file"

    # Remove done flag
    rm -f ".refactor_phase${phase}_done"

    echo "Rolled back phase $phase"
}

if [ "$PHASE" = "all" ]; then
    for p in 4 3 2 1; do
        if [ -f ".refactor_phase${p}_done" ]; then
            rollback_phase $p
        fi
    done
else
    rollback_phase "$PHASE"
fi

echo "Rollback completed"