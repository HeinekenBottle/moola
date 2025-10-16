#!/usr/bin/env python3
"""Real-time monitoring for masked LSTM pre-training on RunPod.

Connects to RunPod via SSH and monitors:
- Pre-training loss curves
- Validation metrics
- GPU utilization and memory
- ETA calculations

Usage:
    python scripts/monitor_pretraining.py --host 213.173.110.220 --port 36832
    python scripts/monitor_pretraining.py --watch  # Interactive mode
"""

import argparse
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple


class RunPodMonitor:
    """Monitor RunPod training progress via SSH."""

    def __init__(
        self,
        host: str,
        port: int,
        key_path: str = "~/.ssh/id_ed25519",
    ):
        self.host = host
        self.port = port
        self.key_path = str(Path(key_path).expanduser())

    def ssh_run(self, command: str) -> Tuple[int, str]:
        """Execute SSH command and capture output.

        Args:
            command: Shell command to execute

        Returns:
            Tuple of (exit_code, output)
        """
        ssh_cmd = (
            f"ssh root@{self.host} -p {self.port} -i {self.key_path} "
            f"-o StrictHostKeyChecking=no "
            f"'{command}'"
        )

        try:
            result = subprocess.run(
                ssh_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode, result.stdout + result.stderr
        except subprocess.TimeoutExpired:
            return -1, "Command timed out"

    def check_training_status(self) -> dict:
        """Check if training is currently running.

        Returns:
            Dictionary with status information
        """
        # Check for Python training process
        exit_code, output = self.ssh_run(
            "ps aux | grep 'python.*moola' | grep -v grep"
        )

        is_training = (exit_code == 0) and ("python" in output)

        return {
            'is_training': is_training,
            'process_info': output.strip() if is_training else None,
        }

    def get_gpu_stats(self) -> dict:
        """Get GPU utilization and memory stats.

        Returns:
            Dictionary with GPU metrics
        """
        # Query GPU stats
        exit_code, output = self.ssh_run(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu "
            "--format=csv,noheader,nounits"
        )

        if exit_code == 0:
            parts = output.strip().split(',')
            if len(parts) == 4:
                return {
                    'gpu_util': int(parts[0].strip()),
                    'memory_used': int(parts[1].strip()),
                    'memory_total': int(parts[2].strip()),
                    'temperature': int(parts[3].strip()),
                }

        return {
            'gpu_util': 0,
            'memory_used': 0,
            'memory_total': 24000,  # Default for RTX 4090
            'temperature': 0,
        }

    def get_training_logs(self, lines: int = 50) -> str:
        """Fetch recent training logs.

        Args:
            lines: Number of lines to fetch

        Returns:
            Log output
        """
        # Try to read logs from common locations
        log_commands = [
            f"tail -n {lines} /tmp/training.log 2>/dev/null",
            f"tail -n {lines} /workspace/moola/training.log 2>/dev/null",
            f"journalctl -n {lines} 2>/dev/null | grep -i 'epoch\\|loss\\|accuracy'",
        ]

        for cmd in log_commands:
            exit_code, output = self.ssh_run(cmd)
            if exit_code == 0 and output.strip():
                return output

        return "No logs found"

    def display_status(self):
        """Display current training status."""
        print("="*80)
        print("RUNPOD TRAINING MONITOR")
        print("="*80)
        print(f"Host: {self.host}:{self.port}")
        print()

        # Training status
        status = self.check_training_status()
        print("[TRAINING STATUS]")
        if status['is_training']:
            print("  ✅ Training in progress")
            print(f"  Process: {status['process_info']}")
        else:
            print("  ⚠️  No training detected")
        print()

        # GPU stats
        gpu = self.get_gpu_stats()
        print("[GPU STATS]")
        print(f"  Utilization: {gpu['gpu_util']}%")
        print(f"  VRAM: {gpu['memory_used']}/{gpu['memory_total']} MB ({gpu['memory_used']/gpu['memory_total']*100:.1f}%)")
        print(f"  Temperature: {gpu['temperature']}°C")
        print()

        # Recent logs
        print("[RECENT LOGS]")
        print("-"*80)
        logs = self.get_training_logs(lines=20)
        print(logs)
        print("-"*80)

    def watch(self, interval: int = 5):
        """Continuously monitor training progress.

        Args:
            interval: Refresh interval in seconds
        """
        try:
            while True:
                # Clear screen
                print("\033[2J\033[H")  # ANSI escape codes

                # Display status
                self.display_status()

                # Wait before next refresh
                print(f"\nRefreshing in {interval}s... (Ctrl+C to exit)")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor masked LSTM pre-training on RunPod"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="213.173.110.220",
        help="RunPod host IP",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=36832,
        help="SSH port",
    )
    parser.add_argument(
        "--key",
        type=str,
        default="~/.ssh/id_ed25519",
        help="SSH private key path",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously monitor (refresh every 5s)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (for --watch mode)",
    )

    args = parser.parse_args()

    # Initialize monitor
    monitor = RunPodMonitor(
        host=args.host,
        port=args.port,
        key_path=args.key,
    )

    # Run monitoring
    if args.watch:
        monitor.watch(interval=args.interval)
    else:
        monitor.display_status()


if __name__ == "__main__":
    main()
