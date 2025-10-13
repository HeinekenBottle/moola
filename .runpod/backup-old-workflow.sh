#!/bin/bash
# Backup old RunPod workflow before migration
# Creates a backup of all existing scripts for safety

set -e

echo "🔄 Backing up old RunPod workflow..."
echo "==================================="

BACKUP_DIR=".runpod/backup-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "📦 Creating backup at: $BACKUP_DIR"

# Backup all scripts
cp -r scripts "$BACKUP_DIR/"
echo "  ✅ scripts/ backed up"

# Backup main scripts
cp *.sh "$BACKUP_DIR/" 2>/dev/null || true
echo "  ✅ Main scripts backed up"

# Backup config files
cp *.env "$BACKUP_DIR/" 2>/dev/null || true
echo "  ✅ Config files backed up"

# Create backup manifest
cat > "$BACKUP_DIR/BACKUP_MANIFEST.txt" <<EOF
RunPod Workflow Backup
=====================
Created: $(date)
Purpose: Backup of pre-migration workflow

Contents:
- All deployment scripts (15+ files)
- Configuration files
- Environment files

Migration: These files are replaced by the single deploy.sh script

To restore (if needed):
  cp -r ./* ../
  cd ..
  # Then use old workflow commands

New workflow uses:
  deploy.sh - Single command deployment
EOF

echo "  ✅ Backup manifest created"

echo ""
echo "✅ Backup complete!"
echo "📁 Location: $BACKUP_DIR"
echo ""
echo "🔄 You can now safely migrate to the new workflow:"
echo "   1. bash .runpod/deploy.sh deploy"
echo "   2. ssh runpod"
echo "   3. bash .runpod/deploy.sh train"
echo ""
echo "💡 If you need to restore the old workflow:"
echo "   cp -r $BACKUP_DIR/* .runpod/"