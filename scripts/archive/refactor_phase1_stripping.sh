#!/bin/bash

# Phase 1: Stripping
# Archive non-essential files, remove unused models/features

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "Phase 1: Stripping - Starting..."

# Safety check: ensure we're in the right directory
if [ ! -f "pyproject.toml" ] || [ ! -d "src/moola" ]; then
    echo "Error: Not in project root directory"
    exit 1
fi

# Create backup
BACKUP_FILE=".backup_phase1_$(date +%Y%m%d_%H%M%S).tar.gz"
echo "Creating backup: $BACKUP_FILE"
tar czf "$BACKUP_FILE" --exclude='artifacts' --exclude='data' --exclude='logs' --exclude='.git' --exclude="$BACKUP_FILE" --exclude='.backup_*' .

# Archive non-essential files
echo "Archiving non-essential files..."

# Move old documentation to archive
mkdir -p archive/docs
for doc in CLEANUP_SESSION_*.md README_CLEANUP.txt; do
    if [ -f "$doc" ]; then
        mv "$doc" archive/docs/
        echo "Archived: $doc"
    fi
done

# Archive old scripts
mkdir -p archive/scripts
for script in scripts/demo_bootstrap_ci.py src/moola/cli_feature_aware.py; do
    if [ -f "$script" ]; then
        mv "$script" archive/scripts/
        echo "Archived: $script"
    fi
done

# Remove unused model configs (Opal and Sapphire are legacy, only Jade implemented)
echo "Removing unused model configs..."
if [ -f "configs/model/opal.yaml" ]; then
    mv configs/model/opal.yaml archive/
    echo "Archived: configs/model/opal.yaml"
fi
if [ -f "configs/model/sapphire.yaml" ]; then
    mv configs/model/sapphire.yaml archive/
    echo "Archived: configs/model/sapphire.yaml"
fi

# Remove archived config from clean structure
if [ -f "src/moola/configs/model/enhanced_simple_lstm.yaml" ]; then
    mv src/moola/configs/model/enhanced_simple_lstm.yaml archive/
    echo "Archived: src/moola/configs/model/enhanced_simple_lstm.yaml"
fi

# Remove any __pycache__ directories
echo "Removing __pycache__ directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Remove other cache directories
for cache in .pytest_cache .ruff_cache .benchmarks; do
    if [ -d "$cache" ]; then
        rm -rf "$cache"
        echo "Removed: $cache"
    fi
done

# Remove DVC if present
if [ -d ".dvc" ]; then
    rm -rf .dvc
    echo "Removed: .dvc"
fi

echo "Phase 1 completed. Backup created: $BACKUP_FILE"
touch .refactor_phase1_done