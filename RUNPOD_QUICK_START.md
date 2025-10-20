# RunPod Quick Start - File Transfer Only

**Problem Solved:** No more 15-20 minute `pip install` waits! Training in seconds, not minutes.

## One-Time Setup (On Your Mac)

```bash
# Build the deployment bundle (includes pre-downloaded dependencies)
./scripts/build_runpod_bundle.sh
```

**What this does:**
- Downloads ALL Python packages as wheels (pandas, numpy, sklearn, etc.)
- Packages your code + data
- Creates a self-contained bundle (~50-100 MB)
- **Takes 2-3 minutes, but you only do this once**

**Output:** `runpod_bundle_YYYYMMDD_HHMMSS.tar.gz`

---

## Every Training Run

```bash
# Deploy and train in ONE command
./scripts/runpod_deploy_bundle.sh <RUNPOD_IP> runpod_bundle_*.tar.gz
```

**What this does:**
1. Uploads bundle to RunPod (SCP - no Docker needed)
2. Extracts bundle
3. Installs dependencies from local wheels (**~30 seconds!**)
4. Runs training
5. Downloads results automatically

**Total time:** Upload (1-2 min) + Install (30 sec) + Train (seconds) = **2-3 minutes total**

---

## Quick Commands

```bash
# Full workflow
./scripts/build_runpod_bundle.sh
./scripts/runpod_deploy_bundle.sh 44.201.123.45 runpod_bundle_*.tar.gz

# View results
cat experiment_results.jsonl | tail -1 | python3 -m json.tool
tail -100 artifacts/runs/baseline_*.log

# Cleanup RunPod
ssh -i ~/.ssh/runpod_key ubuntu@<IP> 'rm -rf ~/runpod_workspace ~/runpod_bundle_*.tar.gz'
```

---

## Advanced: Custom Training Args

Edit `runpod_deploy_bundle.sh` line 82 to add custom args:

```bash
# Example: Enable augmentation
bash runpod_bootstrap.sh --augment-data true --augmentation-ratio 0.5

# Example: Use pre-trained encoder
bash runpod_bootstrap.sh --pretrained-encoder models/bilstm_encoder.pt
```

---

## What's Different From Before?

| Old Way | New Way |
|---------|---------|
| SSH in, pip install requirements (15-20 min) | Pre-downloaded wheels (~30 sec) |
| Multiple SSH commands | ONE command (`./scripts/runpod_deploy_bundle.sh`) |
| Manual download of results | Automatic download |
| Hard to reproduce | Bundled dependencies |
| Lost between sessions | Self-contained bundle |

---

## Troubleshooting

**Bundle build fails:**
```bash
# Check requirements.txt exists
ls requirements.txt

# Try manual wheel download
pip3 download -r requirements.txt -d ./test_wheels
```

**Upload fails:**
```bash
# Test SSH connection
ssh -i ~/.ssh/runpod_key ubuntu@<IP> echo "Connected"

# Check SSH key permissions
chmod 600 ~/.ssh/runpod_key
```

**Training fails:**
```bash
# SSH in and check logs manually
ssh -i ~/.ssh/runpod_key ubuntu@<IP>
cd ~/runpod_workspace/moola
cat artifacts/runs/baseline_*.log
```

---

## Files Created

```
scripts/
├── build_runpod_bundle.sh       # Create deployment bundle
├── runpod_deploy_bundle.sh      # Upload + train + download
└── (old SSH scripts - ignore)   # Replaced by bundle workflow

runpod_bundle_YYYYMMDD_HHMMSS.tar.gz  # Self-contained package
├── wheels/                           # Pre-downloaded dependencies
├── moola/                            # Code + data
├── runpod_bootstrap.sh               # Single setup + train script
└── README.txt                        # Bundle documentation
```

---

## Why This Works

**Before:** RunPod base images only have PyTorch + CUDA. You had to `pip install` everything else over the network.

**Now:** We pre-download ALL dependencies as wheels on your Mac, upload them, and install from local files. No network needed = 30x faster.

**Resources Used:**
- WebSearch: RunPod PyTorch templates, dependency management
- GitHub: runpod/containers Dockerfiles, worker templates
- Documentation: RunPod custom templates, requirements.txt handling

---

## Next Steps

1. **Baseline training:**
   ```bash
   ./scripts/build_runpod_bundle.sh
   ./scripts/runpod_deploy_bundle.sh <IP> runpod_bundle_*.tar.gz
   ```

2. **Check results:**
   ```bash
   python3 << 'EOF'
   import json
   results = [json.loads(line) for line in open('experiment_results.jsonl')]
   latest = results[-1]
   print(f"Accuracy: {latest['metrics']['accuracy']:.4f}")
   print(f"PR-AUC: {latest['metrics']['pr_auc']:.4f}")
   print(f"Brier: {latest['metrics']['brier']:.4f}")
   EOF
   ```

3. **Iterate:** Update code, rebuild bundle, redeploy

---

**No Docker Required on Mac** ✅
**No Long pip install Waits** ✅
**File Transfer Only** ✅
**One Command Deployment** ✅
