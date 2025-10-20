# RunPod SSH/SCP Quick Reference

## Connection Details

```bash
Host: 213.173.110.215
Port: 26324
User: root
Key: ~/.ssh/id_ed25519
Path: /workspace/moola
```

## Quick Commands

### SSH Connect
```bash
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519
```

### One-liner: Upload Code + Run Workflow
```bash
# Upload scripts
scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    /Users/jack/projects/moola/scripts/runpod_gated_workflow \
    root@213.173.110.215:/workspace/moola/scripts/ \
&& \
# Upload source
scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    /Users/jack/projects/moola/src \
    root@213.173.110.215:/workspace/moola/ \
&& \
# SSH and run
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519 \
    'cd /workspace/moola && python3 scripts/runpod_gated_workflow/run_all.py'
```

### Download Results
```bash
# Get results JSON
scp -i ~/.ssh/id_ed25519 -P 26324 \
    root@213.173.110.215:/workspace/moola/gated_workflow_results.jsonl \
    /Users/jack/projects/moola/

# Get all models
scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    root@213.173.110.215:/workspace/moola/artifacts/models \
    /Users/jack/projects/moola/artifacts/

# Get pretrained encoder
scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    root@213.173.110.215:/workspace/moola/artifacts/pretrained \
    /Users/jack/projects/moola/artifacts/
```

### Check Status (while running)
```bash
# SSH and tail results
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519 \
    'tail -f /workspace/moola/gated_workflow_results.jsonl'

# Check GPU usage
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519 \
    'watch -n 1 nvidia-smi'
```

### Resume After Disconnection
```bash
# SSH back in
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519

# Check last completed gate
tail -n 5 /workspace/moola/gated_workflow_results.jsonl

# Resume from next gate (example: resume from gate 4)
cd /workspace/moola
python3 scripts/runpod_gated_workflow/run_all.py --start-gate 4
```

## Full Workflow Example

```bash
# Step 1: Upload code (from Mac)
scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    /Users/jack/projects/moola/scripts/runpod_gated_workflow \
    root@213.173.110.215:/workspace/moola/scripts/

scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    /Users/jack/projects/moola/src \
    root@213.173.110.215:/workspace/moola/

# Step 2: SSH to RunPod
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519

# Step 3: Verify environment (on RunPod)
cd /workspace/moola
python3 scripts/runpod_gated_workflow/0_verify_env.py

# Step 4: Run full workflow (on RunPod)
nohup python3 scripts/runpod_gated_workflow/run_all.py > workflow.log 2>&1 &

# Step 5: Monitor progress (on RunPod)
tail -f workflow.log
# OR
tail -f /workspace/moola/gated_workflow_results.jsonl

# Step 6: Exit SSH (safe - workflow continues)
exit

# Step 7: Download results later (from Mac)
scp -i ~/.ssh/id_ed25519 -P 26324 \
    root@213.173.110.215:/workspace/moola/gated_workflow_results.jsonl \
    /Users/jack/projects/moola/

scp -i ~/.ssh/id_ed25519 -r -P 26324 \
    root@213.173.110.215:/workspace/moola/artifacts \
    /Users/jack/projects/moola/
```

## Background Execution (Survive SSH Disconnect)

```bash
# SSH to RunPod
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519

# Run in background with nohup
cd /workspace/moola
nohup python3 scripts/runpod_gated_workflow/run_all.py > workflow.log 2>&1 &

# Get process ID
echo $!

# Exit SSH (workflow continues)
exit

# Check status later (SSH back in)
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519
tail -f /workspace/moola/workflow.log
```

## Individual Gate Execution

```bash
# SSH to RunPod
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519

cd /workspace/moola

# Run specific gate
python3 scripts/runpod_gated_workflow/0_verify_env.py
python3 scripts/runpod_gated_workflow/1_smoke_enhanced.py
python3 scripts/runpod_gated_workflow/2_control_minirocket.py
python3 scripts/runpod_gated_workflow/3_pretrain_bilstm.py
python3 scripts/runpod_gated_workflow/4_finetune_enhanced.py
python3 scripts/runpod_gated_workflow/5_augment_train.py
python3 scripts/runpod_gated_workflow/6_baseline_simplelstm.py
python3 scripts/runpod_gated_workflow/7_ensemble.py
```

## Troubleshooting

### Connection refused
```bash
# Verify port and host
ssh -v root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519
```

### Permission denied (publickey)
```bash
# Check key permissions
chmod 600 ~/.ssh/id_ed25519

# Verify key is correct
ssh-keygen -l -f ~/.ssh/id_ed25519
```

### SCP transfer interrupted
```bash
# Use rsync for resume capability
rsync -avz -e "ssh -i ~/.ssh/id_ed25519 -p 26324" \
    /Users/jack/projects/moola/scripts/runpod_gated_workflow \
    root@213.173.110.215:/workspace/moola/scripts/
```

### Check available disk space
```bash
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519 \
    'df -h /workspace'
```

### Kill hung workflow
```bash
# SSH to RunPod
ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519

# Find process
ps aux | grep run_all.py

# Kill process
kill -9 <PID>
```

## Results Analysis (on Mac)

```bash
# Pretty print all results
cat gated_workflow_results.jsonl | jq .

# Show only metrics
cat gated_workflow_results.jsonl | jq '{gate: .gate, status: .status, metrics: .metrics}'

# Find best F1 score
cat gated_workflow_results.jsonl | jq -s 'max_by(.metrics.val_f1 // 0) | {gate: .gate, f1: .metrics.val_f1}'

# Show failed gates
cat gated_workflow_results.jsonl | jq 'select(.status == "failed")'

# Summary table
cat gated_workflow_results.jsonl | jq -r '[.gate, .status, (.metrics.val_f1 // "N/A")] | @tsv'
```

## Aliases (add to ~/.bashrc or ~/.zshrc)

```bash
# SSH to RunPod
alias runpod='ssh root@213.173.110.215 -p 26324 -i ~/.ssh/id_ed25519'

# Upload code
alias runpod-upload='scp -i ~/.ssh/id_ed25519 -r -P 26324 /Users/jack/projects/moola/scripts/runpod_gated_workflow root@213.173.110.215:/workspace/moola/scripts/ && scp -i ~/.ssh/id_ed25519 -r -P 26324 /Users/jack/projects/moola/src root@213.173.110.215:/workspace/moola/'

# Download results
alias runpod-download='scp -i ~/.ssh/id_ed25519 -P 26324 root@213.173.110.215:/workspace/moola/gated_workflow_results.jsonl /Users/jack/projects/moola/ && scp -i ~/.ssh/id_ed25519 -r -P 26324 root@213.173.110.215:/workspace/moola/artifacts /Users/jack/projects/moola/'

# Usage:
# runpod
# runpod-upload
# runpod-download
```

## Expected Output

### Successful Workflow
```
[2025-10-18T12:00:00] [INFO] GATED TRAINING WORKFLOW - MASTER ORCHESTRATOR
...
[2025-10-18T14:30:00] [SUCCESS] ALL GATES PASSED - WORKFLOW COMPLETE
```

### Failed Workflow
```
[2025-10-18T12:45:00] [ERROR] GATE 3 FAILED with exit code 1
[2025-10-18T12:45:00] [ERROR] WORKFLOW ABORTED: Gate 3 failed
```

---

**Pro Tips**:
- Use `tmux` or `screen` on RunPod for persistent sessions
- Monitor GPU with `watch -n 1 nvidia-smi` in separate window
- Download results incrementally (after each successful gate)
- Keep local backup of `gated_workflow_results.jsonl`
