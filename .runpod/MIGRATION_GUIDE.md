# RunPod Migration Guide: From Complex to Simple

## Overview
This guide helps you migrate from the complex multi-script workflow to a single-command deployment system.

## Old vs New Workflow

### Old Workflow (Problems)
```bash
# ❌ Complex 6-step process with multiple failure points
bash .runpod/simple-sync.sh              # Often fails with pagination errors
ssh runpod                               # Manual connection
bash /runpod-volume/scripts/network-storage-cleanup.sh  # Manual cleanup
bash /runpod-volume/scripts/network-storage-repopulate.sh  # Manual repopulation
bash /runpod-volume/scripts/robust-setup.sh  # Manual setup (10+ minutes)
python3 -m moola.cli oof --model xgb --device cuda --seed 1337  # Manual training
```

**Problems:**
- Multiple sync scripts with pagination errors
- Manual environment setup each time
- Confusing mount points (/workspace vs /runpod-volume)
- 15+ specialized scripts with overlapping functions
- No idempotency - scripts can't be safely re-run

### New Workflow (Solution)
```bash
# ✅ Simple 3-step process, single command
bash .runpod/deploy.sh deploy    # Deploy everything (2 minutes)
ssh runpod                       # Connect to pod
bash .runpod/deploy.sh train     # Auto-setup and train
```

**Benefits:**
- Single script handles everything
- Automatic environment setup with caching
- Reliable file synchronization
- Idempotent operations (safe to re-run)
- Clear, consistent paths

## Migration Steps

### Step 1: Set AWS Credentials (Once)
```bash
export AWS_ACCESS_KEY_ID="your-runpod-access-key"
export AWS_SECRET_ACCESS_KEY="your-runpod-secret-key"
```

Get these from: https://www.runpod.io/console/user/settings

### Step 2: Deploy Your Project
```bash
# From your local project directory
bash .runpod/deploy.sh deploy
```

This replaces ALL of these old scripts:
- `simple-sync.sh`
- `sync-scripts-robust.sh`
- `sync-to-storage.sh`
- `sync-from-storage.sh`
- All cleanup/repopulate scripts

### Step 3: Start Training
```bash
# SSH to RunPod
ssh runpod

# Start automatic setup and training
bash .runpod/deploy.sh train
```

This replaces ALL of these old scripts:
- `robust-setup.sh`
- `network-storage-cleanup.sh`
- `network-storage-repopulate.sh`
- `runpod-train.sh`
- `precise-train.sh`

### Step 4: Monitor Progress
```bash
# Check status
bash .runpod/deploy.sh status

# View logs
bash .runpod/deploy.sh logs
```

## Old Scripts You Can Remove

After migration, you can safely delete these files:
```
.runpod/
├── simple-sync.sh                 ❌ Delete
├── sync-scripts-robust.sh         ❌ Delete
├── sync-to-storage.sh             ❌ Delete
├── sync-from-storage.sh           ❌ Delete
├── clean-storage.sh               ❌ Delete
├── scripts/
│   ├── robust-setup.sh            ❌ Delete
│   ├── clean-network-storage.sh   ❌ Delete
│   ├── network-storage-repopulate.sh  ❌ Delete
│   ├── runpod-train.sh            ❌ Delete
│   ├── precise-train.sh           ❌ Delete
│   └── [all other scripts]        ❌ Delete
└── network-storage.env            ❌ Delete
```

**Keep only:**
- `.runpod/deploy.sh` (the new unified script)
- `.runpod/MIGRATION_GUIDE.md` (this guide)

## What the New Script Does Automatically

### `deploy.sh deploy`
1. **Packages Essentials**: Only copies necessary files (configs, data, source)
2. **Creates Startup Script**: Generates unified setup+training script
3. **Reliable Upload**: Uses single sync operation (no pagination errors)
4. **Cleans Up**: Removes temporary files automatically

### `deploy.sh train` (on RunPod)
1. **Detects Storage**: Automatically finds /workspace or /runpod-volume
2. **Environment Setup**: Creates venv if needed, installs dependencies
3. **Caching**: Reuses existing venv and dependencies across sessions
4. **Verification**: Tests all imports and GPU availability
5. **Automatic Training**: Runs the complete training pipeline

## Troubleshooting

### If deploy fails:
```bash
# Check credentials
aws s3 ls --region eu-ro-1 --endpoint-url https://s3api-eu-ro-1.runpod.io s3://hg878tp14w

# Re-deploy (it's idempotent)
bash .runpod/deploy.sh deploy
```

### If training fails:
```bash
# Check setup
bash .runpod/deploy.sh status

# View logs
bash .runpod/deploy.sh logs

# Restart training (safe to re-run)
bash .runpod/deploy.sh train
```

### If you need to start fresh:
```bash
# Clean everything and redeploy
bash .runpod/deploy.sh cleanup
bash .runpod/deploy.sh deploy
ssh runpod
bash .runpod/deploy.sh train
```

## Key Differences

| Old Way | New Way |
|---------|---------|
| 15+ specialized scripts | 1 unified script |
| Manual environment setup | Automatic setup with caching |
| Multiple sync methods | Single reliable sync |
| Confusing paths | Automatic path detection |
| Not idempotent | Safe to re-run any command |
| 30+ minute setup | 2 minute deployment |

## Performance Benefits

- **Faster Setup**: Cached virtual environment reduces setup from 10+ minutes to 30 seconds
- **Reliable Sync**: Single upload operation eliminates pagination errors
- **Automatic Recovery**: Safe to re-run if anything fails
- **Less Complexity**: 1 script instead of 15+ reduces failure points

## Need Help?

If you encounter issues during migration:
1. Check that AWS credentials are set correctly
2. Verify network storage is mounted (`ssh runpod` and `ls -la /workspace`)
3. Try the cleanup command and redeploy
4. Check the status and logs for detailed information

The new system is designed to be much more reliable and simpler than the old workflow.