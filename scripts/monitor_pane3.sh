#!/bin/bash
# Monitor tmux pane 3 continuously and log output
# Usage: ./monitor_pane3.sh [interval_seconds] [output_file]

INTERVAL="${1:-30}"  # Default: capture every 30 seconds
LOG_FILE="${2:-pane3_monitor.log}"
PANE_ID="3"

echo "==================================================" > "$LOG_FILE"
echo "Pane 3 Monitor Started: $(date)" >> "$LOG_FILE"
echo "Interval: ${INTERVAL}s" >> "$LOG_FILE"
echo "==================================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# Function to capture and log
capture_pane() {
    echo "──────────────────────────────────────────────────" >> "$LOG_FILE"
    echo "Capture at: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    echo "──────────────────────────────────────────────────" >> "$LOG_FILE"

    # Capture pane content
    if tmux capture-pane -t "$PANE_ID" -p >> "$LOG_FILE" 2>&1; then
        echo "" >> "$LOG_FILE"
    else
        echo "ERROR: Failed to capture pane $PANE_ID" >> "$LOG_FILE"
        echo "" >> "$LOG_FILE"
    fi
}

# Main monitoring loop
echo "Starting continuous monitoring of pane $PANE_ID..."
echo "Logging to: $LOG_FILE"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    capture_pane
    sleep "$INTERVAL"
done
