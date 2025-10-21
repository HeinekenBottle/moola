"""
RunPod SCP-based training orchestrator.

Allows AI agents to directly interact with RunPod pods via file transfer and command execution.
Enables precise debugging, incremental fix deployment, and real-time monitoring.

Key Features:
- Direct SCP file transfer (upload/download single files or directories)
- SSH command execution with real-time output streaming
- Training pipeline orchestration with artifact collection
- Pre-flight environment verification
- Error detection and automatic recovery
"""

import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np


class RunPodOrchestrator:
    """
    RunPod SCP-based training orchestrator.

    Architecture:
    - SCP for file transfer (precise control over what gets deployed)
    - SSH for command execution (real-time feedback)
    - Streaming output capture (AI can inspect logs immediately)
    - Artifact download (fetch results for validation)

    Example:
        >>> orch = RunPodOrchestrator(
        ...     host="213.173.98.6",
        ...     port=14385,
        ...     key_path="~/.ssh/id_ed25519"
        ... )
        >>> orch.verify_environment()
        >>> orch.upload_file("src/moola/models/cnn_transformer.py",
        ...                  "/workspace/moola/src/moola/models/cnn_transformer.py")
        >>> orch.run_training("cnn_transformer", device="cuda")
        >>> orch.download_results("cnn_transformer", "/tmp/results/")
    """

    def __init__(
        self,
        host: str,
        port: int,
        key_path: str,
        workspace: str = "/workspace/moola",
        timeout: int = 600,
        verbose: bool = True,
    ):
        """
        Initialize RunPod orchestrator.

        Args:
            host: RunPod pod IP address (e.g., "213.173.98.6")
            port: SSH port (e.g., 14385)
            key_path: Path to SSH private key (e.g., "~/.ssh/id_ed25519")
            workspace: Remote workspace directory (default: "/workspace/moola")
            timeout: Default command timeout in seconds (default: 600)
            verbose: Print detailed logs (default: True)
        """
        self.host = host
        self.port = port
        self.key_path = str(Path(key_path).expanduser())
        self.workspace = workspace
        self.timeout = timeout
        self.verbose = verbose

        # Verify SSH key exists
        if not Path(self.key_path).exists():
            raise FileNotFoundError(f"SSH key not found: {self.key_path}")

        if self.verbose:
            print(f"[RUNPOD] Initialized orchestrator")
            print(f"  Host: {self.host}:{self.port}")
            print(f"  Workspace: {self.workspace}")
            print(f"  SSH Key: {self.key_path}")

    def _run_command(
        self,
        command: str,
        capture_output: bool = True,
        stream_output: bool = False,
        timeout: int | None = None,
    ) -> tuple[int, str, str]:
        """
        Execute shell command locally.

        Args:
            command: Shell command to execute
            capture_output: Capture stdout/stderr (default: True)
            stream_output: Print output in real-time (default: False)
            timeout: Command timeout in seconds (default: self.timeout)

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if timeout is None:
            timeout = self.timeout

        if self.verbose and stream_output:
            print(f"[CMD] {command}")

        try:
            if stream_output:
                # Stream output in real-time
                process = subprocess.Popen(
                    command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                stdout_lines = []
                stderr_lines = []

                # Read stdout
                for line in iter(process.stdout.readline, ""):
                    if line:
                        stdout_lines.append(line)
                        if self.verbose:
                            print(line, end="")

                # Wait for completion
                return_code = process.wait(timeout=timeout)

                # Read any remaining stderr
                stderr = process.stderr.read()
                if stderr:
                    stderr_lines.append(stderr)
                    if self.verbose:
                        print(stderr, end="")

                stdout = "".join(stdout_lines)
                stderr = "".join(stderr_lines)

                return return_code, stdout, stderr
            else:
                # Capture output without streaming
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=capture_output,
                    text=True,
                    timeout=timeout,
                )
                return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            print(f"[ERROR] Command timed out after {timeout}s")
            return -1, "", f"Command timed out after {timeout}s"
        except Exception as e:
            print(f"[ERROR] Command failed: {e}")
            return -1, "", str(e)

    def upload_file(self, local_path: str | Path, remote_path: str) -> bool:
        """
        Upload single file via SCP.

        Args:
            local_path: Local file path
            remote_path: Remote file path (absolute)

        Returns:
            True if upload succeeded, False otherwise

        Example:
            >>> orch.upload_file(
            ...     "src/moola/models/cnn_transformer.py",
            ...     "/workspace/moola/src/moola/models/cnn_transformer.py"
            ... )
        """
        local_path = Path(local_path)

        if not local_path.exists():
            print(f"[ERROR] Local file not found: {local_path}")
            return False

        # Create remote directory first
        remote_dir = str(Path(remote_path).parent)
        self.execute_command(f"mkdir -p {remote_dir}")

        cmd = (
            f"scp -P {self.port} -i {self.key_path} "
            f"-o StrictHostKeyChecking=no "
            f"{local_path} root@{self.host}:{remote_path}"
        )

        if self.verbose:
            print(f"[UPLOAD] {local_path.name} → {remote_path}")

        return_code, stdout, stderr = self._run_command(cmd)

        if return_code == 0:
            if self.verbose:
                print(f"  ✓ Upload complete")
            return True
        else:
            print(f"[ERROR] Upload failed: {stderr}")
            return False

    def upload_directory(
        self,
        local_dir: str | Path,
        remote_dir: str,
        exclude_patterns: list[str] | None = None,
    ) -> bool:
        """
        Upload directory recursively via SCP.

        Args:
            local_dir: Local directory path
            remote_dir: Remote directory path (absolute)
            exclude_patterns: List of patterns to exclude (e.g., ["*.pyc", "__pycache__"])

        Returns:
            True if upload succeeded, False otherwise

        Example:
            >>> orch.upload_directory(
            ...     "src/moola/validation/",
            ...     "/workspace/moola/src/moola/validation/",
            ...     exclude_patterns=["*.pyc", "__pycache__"]
            ... )
        """
        local_dir = Path(local_dir)

        if not local_dir.exists():
            print(f"[ERROR] Local directory not found: {local_dir}")
            return False

        # Build exclude arguments for rsync (more robust than scp -r)
        exclude_args = ""
        if exclude_patterns:
            for pattern in exclude_patterns:
                exclude_args += f" --exclude='{pattern}'"

        # Use rsync over SSH for better control
        cmd = (
            f"rsync -avz --progress "
            f"-e 'ssh -p {self.port} -i {self.key_path} -o StrictHostKeyChecking=no' "
            f"{exclude_args} "
            f"{local_dir}/ root@{self.host}:{remote_dir}/"
        )

        if self.verbose:
            print(f"[UPLOAD DIR] {local_dir} → {remote_dir}")

        return_code, stdout, stderr = self._run_command(cmd, stream_output=True)

        if return_code == 0:
            if self.verbose:
                print(f"  ✓ Directory upload complete")
            return True
        else:
            print(f"[ERROR] Directory upload failed: {stderr}")
            return False

    def download_file(self, remote_path: str, local_path: str | Path) -> bool:
        """
        Download file from RunPod via SCP.

        Args:
            remote_path: Remote file path (absolute)
            local_path: Local file path

        Returns:
            True if download succeeded, False otherwise

        Example:
            >>> orch.download_file(
            ...     "/workspace/moola/logs/training.log",
            ...     "/tmp/runpod_training.log"
            ... )
        """
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = (
            f"scp -P {self.port} -i {self.key_path} "
            f"-o StrictHostKeyChecking=no "
            f"root@{self.host}:{remote_path} {local_path}"
        )

        if self.verbose:
            print(f"[DOWNLOAD] {remote_path} → {local_path.name}")

        return_code, stdout, stderr = self._run_command(cmd)

        if return_code == 0:
            if self.verbose:
                print(f"  ✓ Download complete")
            return True
        else:
            print(f"[ERROR] Download failed: {stderr}")
            return False

    def download_directory(
        self,
        remote_dir: str,
        local_dir: str | Path,
        exclude_patterns: list[str] | None = None,
    ) -> bool:
        """
        Download directory recursively from RunPod.

        Args:
            remote_dir: Remote directory path (absolute)
            local_dir: Local directory path
            exclude_patterns: List of patterns to exclude

        Returns:
            True if download succeeded, False otherwise
        """
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)

        # Build exclude arguments
        exclude_args = ""
        if exclude_patterns:
            for pattern in exclude_patterns:
                exclude_args += f" --exclude='{pattern}'"

        cmd = (
            f"rsync -avz --progress "
            f"-e 'ssh -p {self.port} -i {self.key_path} -o StrictHostKeyChecking=no' "
            f"{exclude_args} "
            f"root@{self.host}:{remote_dir}/ {local_dir}/"
        )

        if self.verbose:
            print(f"[DOWNLOAD DIR] {remote_dir} → {local_dir}")

        return_code, stdout, stderr = self._run_command(cmd, stream_output=True)

        if return_code == 0:
            if self.verbose:
                print(f"  ✓ Directory download complete")
            return True
        else:
            print(f"[ERROR] Directory download failed: {stderr}")
            return False

    def execute_command(
        self,
        command: str,
        timeout: int | None = None,
        stream_output: bool = True,
    ) -> int:
        """
        Execute command on RunPod via SSH.

        Args:
            command: Shell command to execute on remote pod
            timeout: Command timeout in seconds (default: self.timeout)
            stream_output: Print output in real-time (default: True)

        Returns:
            Command exit code (0 = success)

        Example:
            >>> orch.execute_command("ls -la /workspace/moola/data/")
            >>> orch.execute_command("nvidia-smi", timeout=10)
        """
        ssh_cmd = (
            f"ssh root@{self.host} -p {self.port} -i {self.key_path} "
            f"-o StrictHostKeyChecking=no "
            f"'{command}'"
        )

        return_code, stdout, stderr = self._run_command(
            ssh_cmd,
            stream_output=stream_output,
            timeout=timeout,
        )

        return return_code

    def verify_environment(self) -> dict[str, bool]:
        """
        Pre-flight checks for RunPod environment.

        Verifies:
        - PyTorch installation
        - CUDA availability
        - Data files present
        - Pre-trained encoder exists (if applicable)
        - Workspace structure

        Returns:
            Dictionary of check names → success status

        Example:
            >>> results = orch.verify_environment()
            >>> if all(results.values()):
            ...     print("Environment OK - ready for training")
        """
        print("[VERIFY] Running pre-flight checks...")

        checks = {
            "SSH Connection": f"echo 'Connection OK'",
            "PyTorch": f"python -c 'import torch; print(f\"PyTorch {{torch.__version__}}\")'",
            "CUDA": f"python -c 'import torch; print(f\"CUDA available: {{torch.cuda.is_available()}}\")'",
            "GPU Info": f"nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            "Workspace": f"ls -d {self.workspace}",
            "Data Files": f"ls {self.workspace}/data/processed/train.parquet",
            "Source Code": f"ls {self.workspace}/src/moola/models/cnn_transformer.py",
            "Artifacts Dir": f"ls -d {self.workspace}/artifacts/",
        }

        results = {}

        for name, cmd in checks.items():
            return_code = self.execute_command(cmd, stream_output=False, timeout=30)
            results[name] = return_code == 0

            status = "✓" if return_code == 0 else "✗"
            print(f"  {status} {name}")

        # Optional: Check for pre-trained encoder
        encoder_check = self.execute_command(
            f"ls {self.workspace}/artifacts/pretrained/encoder_weights.pt",
            stream_output=False,
            timeout=10,
        )
        if encoder_check == 0:
            results["Pre-trained Encoder"] = True
            print(f"  ✓ Pre-trained Encoder")
        else:
            results["Pre-trained Encoder"] = False
            print(f"  ⚠ Pre-trained Encoder (not found - will train from scratch)")

        print()
        if all(results.values()):
            print("[VERIFY] ✓ All checks passed - environment ready")
        else:
            failed = [k for k, v in results.items() if not v]
            print(f"[VERIFY] ✗ Failed checks: {', '.join(failed)}")

        return results

    def deploy_fixes(self, fix_files: list[str | Path]) -> bool:
        """
        Deploy specific fixed files to RunPod.

        This is the core function for incremental deployment - upload only the
        files that have been fixed, without re-uploading the entire codebase.

        Args:
            fix_files: List of local file paths to deploy

        Returns:
            True if all files deployed successfully

        Example:
            >>> orch.deploy_fixes([
            ...     "src/moola/models/cnn_transformer.py",  # Fixed encoder freezing
            ...     "src/moola/config/training_config.py",  # Updated hyperparams
            ...     "src/moola/validation/training_monitor.py",  # New debugging
            ... ])
        """
        print(f"[DEPLOY] Deploying {len(fix_files)} fixed files...")

        success_count = 0

        for local_file in fix_files:
            local_path = Path(local_file)

            # Compute relative path from project root
            try:
                # Assume files are in src/moola/...
                if "src/moola" in str(local_path):
                    relative_path = str(local_path).split("src/moola/")[-1]
                    remote_path = f"{self.workspace}/src/moola/{relative_path}"
                else:
                    # Fallback: use filename only
                    remote_path = f"{self.workspace}/{local_path.name}"
            except Exception as e:
                print(f"[ERROR] Could not determine remote path for {local_file}: {e}")
                continue

            # Upload file
            if self.upload_file(local_path, remote_path):
                success_count += 1
                print(f"  ✓ Deployed: {local_path.name}")
            else:
                print(f"  ✗ Failed: {local_path.name}")

        print(f"[DEPLOY] Complete: {success_count}/{len(fix_files)} files deployed")
        return success_count == len(fix_files)

    def run_training(
        self,
        model: str,
        device: str = "cuda",
        encoder_path: str | None = None,
        extra_args: str = "",
        timeout: int = 3600,
    ) -> int:
        """
        Execute training on RunPod and stream logs.

        Args:
            model: Model name (e.g., "cnn_transformer", "simple_lstm")
            device: Device to train on ("cpu" or "cuda")
            encoder_path: Path to pre-trained encoder (optional)
            extra_args: Additional CLI arguments
            timeout: Training timeout in seconds (default: 3600 = 1 hour)

        Returns:
            Exit code (0 = success)

        Example:
            >>> exit_code = orch.run_training(
            ...     model="cnn_transformer",
            ...     device="cuda",
            ...     encoder_path="/workspace/artifacts/pretrained/encoder_weights.pt",
            ...     timeout=7200  # 2 hours
            ... )
        """
        print(f"[TRAINING] Starting {model} on {device}...")

        # Build training command
        cmd = f"""
        cd {self.workspace} && \\
        source /tmp/moola-venv/bin/activate && \\
        export PYTHONPATH="{self.workspace}/src:$PYTHONPATH" && \\
        python -m moola.cli oof --model {model} --device {device} --seed 1337
        """

        if encoder_path:
            cmd += f" --load-pretrained-encoder {encoder_path}"

        if extra_args:
            cmd += f" {extra_args}"

        # Execute training with streaming output
        return_code = self.execute_command(cmd, timeout=timeout, stream_output=True)

        if return_code == 0:
            print(f"[TRAINING] ✓ Training complete")
        else:
            print(f"[TRAINING] ✗ Training failed with exit code {return_code}")

        return return_code

    def download_results(
        self,
        model: str,
        output_dir: str | Path,
        version: str = "v1",
    ) -> bool:
        """
        Download OOF predictions and artifacts from RunPod.

        Args:
            model: Model name (e.g., "cnn_transformer")
            output_dir: Local directory to save results
            version: Model version (default: "v1")

        Returns:
            True if download succeeded

        Example:
            >>> orch.download_results("cnn_transformer", "/tmp/cnn_results/")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        remote_artifacts = f"{self.workspace}/artifacts/oof/{model}/{version}/"

        print(f"[DOWNLOAD] Fetching results for {model}...")

        success = self.download_directory(remote_artifacts, output_dir)

        if success:
            print(f"[DOWNLOAD] ✓ Results saved to {output_dir}")
        else:
            print(f"[DOWNLOAD] ✗ Failed to download results")

        return success

    def download_logs(
        self,
        output_dir: str | Path,
        log_pattern: str = "*.log",
    ) -> bool:
        """
        Download training logs from RunPod.

        Args:
            output_dir: Local directory to save logs
            log_pattern: Pattern for log files (default: "*.log")

        Returns:
            True if download succeeded
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        remote_logs = f"{self.workspace}/logs/"

        print(f"[DOWNLOAD] Fetching logs...")

        # List log files first
        self.execute_command(f"ls -lh {remote_logs}", stream_output=True, timeout=10)

        success = self.download_directory(remote_logs, output_dir)

        if success:
            print(f"[DOWNLOAD] ✓ Logs saved to {output_dir}")
        else:
            print(f"[DOWNLOAD] ✗ Failed to download logs")

        return success

    def check_training_status(self, model: str) -> dict[str, Any]:
        """
        Check if training is currently running and get status.

        Args:
            model: Model name to check

        Returns:
            Dictionary with training status information
        """
        print(f"[STATUS] Checking training status for {model}...")

        # Check for running Python processes
        ps_check = self.execute_command(
            "ps aux | grep 'python -m moola.cli' | grep -v grep",
            stream_output=False,
            timeout=10,
        )

        status = {
            "training_active": (ps_check == 0),
            "gpu_utilization": None,
            "gpu_memory": None,
        }

        # Get GPU stats if training is active
        if status["training_active"]:
            print("  Training is currently running")

            # GPU utilization
            self.execute_command(
                "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader",
                stream_output=True,
                timeout=10,
            )
        else:
            print("  No active training processes")

        return status

    def cleanup_artifacts(self, keep_latest: int = 3) -> bool:
        """
        Clean up old artifacts to free disk space.

        Args:
            keep_latest: Number of latest versions to keep per model

        Returns:
            True if cleanup succeeded
        """
        print(f"[CLEANUP] Removing old artifacts (keeping latest {keep_latest})...")

        # List artifacts
        self.execute_command(
            f"du -sh {self.workspace}/artifacts/*",
            stream_output=True,
            timeout=30,
        )

        # TODO: Implement intelligent cleanup logic
        print("[CLEANUP] Manual cleanup required - implement version-based pruning")

        return True

    def run_pretraining(
        self,
        unlabeled_data_path: str = "/workspace/data/processed/unlabeled_pretrain.parquet",
        save_path: str = "/workspace/artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt",
        n_epochs: int = 50,
        batch_size: int = 512,
        timeout: int = 3600,
    ) -> int:
        """
        Execute masked LSTM pre-training on RunPod.

        Args:
            unlabeled_data_path: Path to unlabeled sequences (parquet)
            save_path: Path to save pre-trained encoder
            n_epochs: Number of pre-training epochs
            batch_size: Batch size for pre-training
            timeout: Pre-training timeout in seconds (default: 3600 = 1 hour)

        Returns:
            Exit code (0 = success)

        Example:
            >>> exit_code = orch.run_pretraining(
            ...     unlabeled_data_path="/workspace/data/processed/unlabeled_pretrain.parquet",
            ...     n_epochs=50,
            ...     timeout=3600
            ... )
        """
        print(f"[PRE-TRAINING] Starting masked LSTM pre-training...")
        print(f"  Unlabeled data: {unlabeled_data_path}")
        print(f"  Save path: {save_path}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Batch size: {batch_size}")

        # Build pre-training command
        cmd = f"""
        cd {self.workspace} && \\
        source /tmp/moola-venv/bin/activate && \\
        export PYTHONPATH="{self.workspace}/src:$PYTHONPATH" && \\
        python3 -c '
import numpy as np
import pandas as pd
from pathlib import Path
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer

# Load unlabeled data
df = pd.read_parquet("{unlabeled_data_path}")
X_unlabeled = np.stack(df["ohlc_sequence"].values)
print(f"[DATA] Loaded {{len(X_unlabeled)}} unlabeled sequences")

# Initialize pre-trainer
pretrainer = MaskedLSTMPretrainer(
    device="cuda",
    batch_size={batch_size},
    seed=1337
)

# Run pre-training
history = pretrainer.pretrain(
    X_unlabeled=X_unlabeled,
    n_epochs={n_epochs},
    save_path=Path("{save_path}"),
    verbose=True
)

print(f"[PRE-TRAINING] Complete! Best val loss: {{min(history[\\"val_loss\\"]):.4f}}")
'
        """

        # Execute pre-training with streaming output
        return_code = self.execute_command(cmd, timeout=timeout, stream_output=True)

        if return_code == 0:
            print(f"[PRE-TRAINING] ✓ Pre-training complete")
            print(f"  Encoder saved: {save_path}")
        else:
            print(f"[PRE-TRAINING] ✗ Pre-training failed with exit code {return_code}")

        return return_code

    def monitor_pretraining(self) -> dict[str, Any]:
        """
        Monitor pre-training progress and GPU stats.

        Returns:
            Dictionary with pre-training status and metrics
        """
        print(f"[MONITOR] Checking pre-training status...")

        # Check for running Python pre-training process
        ps_check = self.execute_command(
            "ps aux | grep 'MaskedLSTMPretrainer\\|masked_lstm_pretrain' | grep -v grep",
            stream_output=False,
            timeout=10,
        )

        is_pretraining = ps_check == 0

        # Get GPU stats
        gpu_cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
        exit_code, gpu_output = self._run_command(
            f"ssh root@{self.host} -p {self.port} -i {self.key_path} -o StrictHostKeyChecking=no '{gpu_cmd}'",
            stream_output=False,
            timeout=10,
        )[:2]

        gpu_stats = {"utilization": 0, "memory_used": 0, "memory_total": 24000, "temperature": 0}
        if exit_code == 0:
            parts = gpu_output.strip().split(",")
            if len(parts) == 4:
                gpu_stats = {
                    "utilization": int(parts[0].strip()),
                    "memory_used": int(parts[1].strip()),
                    "memory_total": int(parts[2].strip()),
                    "temperature": int(parts[3].strip()),
                }

        status = {
            "is_pretraining": is_pretraining,
            "gpu_stats": gpu_stats,
        }

        # Display status
        print(f"\n[STATUS]")
        if is_pretraining:
            print(f"  ✅ Pre-training in progress")
        else:
            print(f"  ⚠️  No pre-training detected")

        print(f"\n[GPU]")
        print(f"  Utilization: {gpu_stats['utilization']}%")
        print(
            f"  VRAM: {gpu_stats['memory_used']}/{gpu_stats['memory_total']} MB ({gpu_stats['memory_used']/gpu_stats['memory_total']*100:.1f}%)"
        )
        print(f"  Temperature: {gpu_stats['temperature']}°C")

        return status

    def download_pretrained_encoder(
        self,
        remote_path: str = "/workspace/artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt",
        local_path: str | Path = "artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt",
    ) -> bool:
        """
        Download pre-trained encoder from RunPod.

        Args:
            remote_path: Remote encoder path
            local_path: Local save path

        Returns:
            True if download succeeded
        """
        print(f"[DOWNLOAD] Fetching pre-trained encoder...")
        print(f"  Remote: {remote_path}")
        print(f"  Local: {local_path}")

        success = self.download_file(remote_path, local_path)

        if success:
            local_path = Path(local_path)
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"[DOWNLOAD] ✓ Encoder downloaded ({size_mb:.1f} MB)")
        else:
            print(f"[DOWNLOAD] ✗ Failed to download encoder")

        return success
