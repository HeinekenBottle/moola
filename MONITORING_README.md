# Pane 3 Monitoring System

## Overview
Continuous monitoring of tmux pane 3 to capture detailed issue logs during RunPod training sessions.

## Components

### 1. Monitor Script (`scripts/monitor_pane3.sh`)
**Purpose:** Captures tmux pane 3 content at regular intervals

**Usage:**
```bash
# Start monitoring (every 20 seconds)
./scripts/monitor_pane3.sh 20 pane3_running_log.txt &

# Start with different interval (every 60 seconds)
./scripts/monitor_pane3.sh 60 pane3_running_log.txt &
```

**Running in Background:**
```bash
# Check running monitors
ps aux | grep monitor_pane3

# Kill the monitor
kill $(pgrep -f monitor_pane3.sh)

# Or use bash ID if started via Claude Code
# It will show as background task
```

### 2. Analysis Script (`scripts/analyze_pane3_log.py`)
**Purpose:** Parse log file and extract issues, errors, patterns

**Usage:**
```bash
# Analyze the log
python3 scripts/analyze_pane3_log.py pane3_running_log.txt

# Quick analysis
python3 scripts/analyze_pane3_log.py
```

**Output:**
- Total captures
- Error count and details
- Missing modules/files
- SSH failures
- Todo progress tracking
- Background task status

### 3. Log File (`pane3_running_log.txt`)
**Format:**
```
==================================================
Pane 3 Monitor Started: Mon 20 Oct 2025 21:30:06 IST
Interval: 20s
==================================================

──────────────────────────────────────────────────
Capture at: 2025-10-20 21:30:06
──────────────────────────────────────────────────
[pane content here]

──────────────────────────────────────────────────
Capture at: 2025-10-20 21:30:26
──────────────────────────────────────────────────
[pane content here]
```

## Current Session

**Started:** 2025-10-20 21:30:06 IST
**Interval:** 20 seconds
**Log File:** `pane3_running_log.txt`
**Background Task ID:** 52ceaa

## Common Operations

### View Live Log
```bash
# Tail the log in real-time
tail -f pane3_running_log.txt

# Watch for errors
tail -f pane3_running_log.txt | grep -i error

# Count captures so far
grep -c "Capture at:" pane3_running_log.txt
```

### Stop Monitoring
```bash
# Find the background task
ps aux | grep monitor_pane3.sh

# Kill it
kill [PID]

# Or via Claude Code background tasks:
# Use KillShell with ID 52ceaa
```

### Analyze Issues
```bash
# Full report
python3 scripts/analyze_pane3_log.py

# Extract just errors
grep -A 2 "Error\|ERROR\|❌" pane3_running_log.txt

# Show todo progress over time
grep -A 10 "☐\|☒" pane3_running_log.txt
```

## What to Look For

### Critical Issues:
- ❌ Import errors (ModuleNotFoundError)
- ❌ Missing files/directories
- ❌ SSH connection failures
- ❌ GPU errors (CUDA out of memory)
- ❌ Training crashes

### Workflow Issues:
- ⚠️ GitHub sync problems
- ⚠️ Re-transfers/re-installs
- ⚠️ Directory creation workarounds
- ⚠️ Missing configs

### Progress Indicators:
- ✅/☐ Todo list progress
- Background task status
- Training epochs (if visible)

## Integration with issue_log.md

After monitoring session completes:
1. Run analysis script to get structured issues
2. Update `issue_log.md` with findings
3. Archive the log: `mv pane3_running_log.txt logs/pane3_[date].log`
4. Create action items for recurring issues

## Tips

- **Longer sessions:** Use 60s interval to reduce log size
- **Debugging:** Use 10-15s interval for rapid iteration
- **Overnight runs:** 120s interval is sufficient
- **Log rotation:** Archive logs after each major session

## Clean Up

```bash
# Stop monitor
kill $(pgrep -f monitor_pane3.sh)

# Archive log
mkdir -p logs/archived
mv pane3_running_log.txt logs/archived/pane3_$(date +%Y%m%d_%H%M%S).log

# Clear temp files
rm -f /tmp/pane3_*.tmp
```
