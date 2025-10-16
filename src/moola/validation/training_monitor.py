"""
Real-time training monitor with automatic error detection.

Monitors training logs and detects issues:
- CUDA OOM errors
- Class collapse
- NaN losses
- Gradient explosions
- Data shape mismatches

Features:
- Real-time log parsing
- Automatic error detection
- Recovery suggestions
- Metric extraction
"""

import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TrainingMetrics:
    """Training metrics extracted from logs."""
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float] = None
    val_accuracy: Optional[float] = None
    gpu_memory_gb: Optional[float] = None


@dataclass
class DetectedError:
    """Detected training error."""
    error_type: str
    severity: str  # "warning", "error", "critical"
    message: str
    suggestion: str
    epoch: Optional[int] = None


class TrainingMonitor:
    """
    Real-time training monitor with error detection.

    Example:
        >>> monitor = TrainingMonitor()
        >>> for line in training_logs:
        ...     monitor.process_line(line)
        ...     if monitor.has_errors():
        ...         print(monitor.get_errors())
    """

    # Error patterns
    ERROR_PATTERNS = {
        "cuda_oom": {
            "pattern": r"torch\.cuda\.OutOfMemoryError|CUDA out of memory",
            "severity": "critical",
            "suggestion": "Reduce batch size or use gradient accumulation",
        },
        "data_shape_mismatch": {
            "pattern": r"RuntimeError.*shape|size mismatch|dimension mismatch",
            "severity": "error",
            "suggestion": "Check data preprocessing and model input dimensions",
        },
        "nan_loss": {
            "pattern": r"loss:\s*nan|Loss:\s*nan",
            "severity": "critical",
            "suggestion": "Reduce learning rate or check for inf/nan in data",
        },
        "gradient_explosion": {
            "pattern": r"gradient.*explod|loss:\s*inf",
            "severity": "error",
            "suggestion": "Reduce learning rate or use gradient clipping",
        },
        "class_collapse": {
            "pattern": r"Class\s+1.*accuracy:\s*0\.0+%",
            "severity": "warning",
            "suggestion": "Check class weights, loss function, or encoder freezing",
        },
        "encoder_not_found": {
            "pattern": r"FileNotFoundError.*encoder_weights",
            "severity": "error",
            "suggestion": "Run SSL pre-training first or train from scratch",
        },
        "import_error": {
            "pattern": r"ImportError|ModuleNotFoundError",
            "severity": "critical",
            "suggestion": "Check virtual environment and installed packages",
        },
    }

    # Metric extraction patterns
    METRIC_PATTERNS = {
        "epoch": r"Epoch\s+\[(\d+)/\d+\]",
        "train_loss": r"Train Loss:\s+([\d.]+)",
        "train_acc": r"Train.*Acc(?:uracy)?:\s+([\d.]+)",
        "val_loss": r"Val Loss:\s+([\d.]+)",
        "val_acc": r"Val.*Acc(?:uracy)?:\s+([\d.]+)",
        "gpu_memory": r"GPU:\s+([\d.]+)GB",
    }

    def __init__(self, verbose: bool = True):
        """
        Initialize training monitor.

        Args:
            verbose: Print detected issues in real-time
        """
        self.verbose = verbose
        self.errors: List[DetectedError] = []
        self.metrics: List[TrainingMetrics] = []
        self.current_epoch: Optional[int] = None

    def process_line(self, line: str) -> None:
        """
        Process a single log line.

        Args:
            line: Log line to process
        """
        # Check for errors
        self._check_errors(line)

        # Extract metrics
        self._extract_metrics(line)

    def _check_errors(self, line: str) -> None:
        """Check line for known error patterns."""
        for error_type, config in self.ERROR_PATTERNS.items():
            pattern = config["pattern"]
            if re.search(pattern, line, re.IGNORECASE):
                error = DetectedError(
                    error_type=error_type,
                    severity=config["severity"],
                    message=line.strip(),
                    suggestion=config["suggestion"],
                    epoch=self.current_epoch,
                )

                self.errors.append(error)

                if self.verbose:
                    severity_emoji = {
                        "warning": "⚠️",
                        "error": "❌",
                        "critical": "🔥",
                    }
                    emoji = severity_emoji.get(error.severity, "⚠️")
                    print(f"\n{emoji} DETECTED: {error_type}")
                    print(f"  Message: {error.message[:100]}...")
                    print(f"  Suggestion: {error.suggestion}")

    def _extract_metrics(self, line: str) -> None:
        """Extract training metrics from log line."""
        # Update current epoch
        epoch_match = re.search(self.METRIC_PATTERNS["epoch"], line)
        if epoch_match:
            self.current_epoch = int(epoch_match.group(1))

        # Check for full metric line (contains train loss and accuracy)
        train_loss_match = re.search(self.METRIC_PATTERNS["train_loss"], line)
        train_acc_match = re.search(self.METRIC_PATTERNS["train_acc"], line)

        if train_loss_match and train_acc_match and self.current_epoch is not None:
            # Extract all available metrics
            metrics = TrainingMetrics(
                epoch=self.current_epoch,
                train_loss=float(train_loss_match.group(1)),
                train_accuracy=float(train_acc_match.group(1)),
            )

            # Extract validation metrics if present
            val_loss_match = re.search(self.METRIC_PATTERNS["val_loss"], line)
            val_acc_match = re.search(self.METRIC_PATTERNS["val_acc"], line)

            if val_loss_match:
                metrics.val_loss = float(val_loss_match.group(1))
            if val_acc_match:
                metrics.val_accuracy = float(val_acc_match.group(1))

            # Extract GPU memory if present
            gpu_mem_match = re.search(self.METRIC_PATTERNS["gpu_memory"], line)
            if gpu_mem_match:
                metrics.gpu_memory_gb = float(gpu_mem_match.group(1))

            self.metrics.append(metrics)

    def has_errors(self) -> bool:
        """Check if any errors detected."""
        return len(self.errors) > 0

    def get_errors(self, severity: Optional[str] = None) -> List[DetectedError]:
        """
        Get detected errors.

        Args:
            severity: Filter by severity ("warning", "error", "critical")

        Returns:
            List of detected errors
        """
        if severity:
            return [e for e in self.errors if e.severity == severity]
        return self.errors

    def get_metrics(self) -> List[TrainingMetrics]:
        """Get extracted training metrics."""
        return self.metrics

    def get_latest_metrics(self) -> Optional[TrainingMetrics]:
        """Get most recent metrics."""
        return self.metrics[-1] if self.metrics else None

    def check_convergence(self, window: int = 5) -> Dict[str, bool]:
        """
        Check if training is converging.

        Args:
            window: Number of recent epochs to check

        Returns:
            Dictionary with convergence indicators
        """
        if len(self.metrics) < window:
            return {"sufficient_data": False}

        recent = self.metrics[-window:]

        # Check if loss is decreasing
        losses = [m.train_loss for m in recent]
        loss_decreasing = losses[-1] < losses[0]

        # Check if accuracy is increasing
        accs = [m.train_accuracy for m in recent]
        acc_increasing = accs[-1] > accs[0]

        # Check for plateau (loss not changing)
        loss_range = max(losses) - min(losses)
        loss_plateau = loss_range < 0.001

        return {
            "sufficient_data": True,
            "loss_decreasing": loss_decreasing,
            "accuracy_increasing": acc_increasing,
            "loss_plateau": loss_plateau,
            "converging": loss_decreasing and acc_increasing and not loss_plateau,
        }

    def generate_report(self) -> str:
        """
        Generate training summary report.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("TRAINING MONITOR REPORT")
        report.append("=" * 60)

        # Error summary
        if self.errors:
            report.append(f"\n🔴 ERRORS DETECTED: {len(self.errors)}")
            by_severity = {}
            for error in self.errors:
                by_severity.setdefault(error.severity, []).append(error)

            for severity in ["critical", "error", "warning"]:
                if severity in by_severity:
                    report.append(f"\n  {severity.upper()}: {len(by_severity[severity])}")
                    for error in by_severity[severity][:3]:  # Show first 3
                        report.append(f"    - {error.error_type}: {error.suggestion}")
        else:
            report.append("\n✅ NO ERRORS DETECTED")

        # Metrics summary
        if self.metrics:
            report.append(f"\n📊 TRAINING PROGRESS")
            report.append(f"  Total epochs: {len(self.metrics)}")

            latest = self.metrics[-1]
            report.append(f"\n  Latest metrics (Epoch {latest.epoch}):")
            report.append(f"    Train Loss: {latest.train_loss:.4f}")
            report.append(f"    Train Accuracy: {latest.train_accuracy:.4f}")

            if latest.val_loss is not None:
                report.append(f"    Val Loss: {latest.val_loss:.4f}")
                report.append(f"    Val Accuracy: {latest.val_accuracy:.4f}")

            if latest.gpu_memory_gb is not None:
                report.append(f"    GPU Memory: {latest.gpu_memory_gb:.2f} GB")

            # Convergence check
            convergence = self.check_convergence()
            if convergence.get("sufficient_data"):
                status = "✓ CONVERGING" if convergence["converging"] else "⚠ NOT CONVERGING"
                report.append(f"\n  Convergence: {status}")

                if convergence["loss_plateau"]:
                    report.append("    ⚠ Loss plateau detected - consider early stopping")

        report.append("\n" + "=" * 60)
        return "\n".join(report)


def monitor_training_with_error_detection(
    orch,
    model: str,
    device: str = "cuda",
    encoder_path: Optional[str] = None,
) -> Tuple[int, List[DetectedError], List[TrainingMetrics]]:
    """
    Monitor training with real-time error detection.

    Args:
        orch: RunPodOrchestrator instance
        model: Model name
        device: Training device
        encoder_path: Path to pre-trained encoder

    Returns:
        Tuple of (exit_code, errors, metrics)

    Example:
        >>> from moola.runpod import RunPodOrchestrator
        >>> orch = RunPodOrchestrator(...)
        >>> exit_code, errors, metrics = monitor_training_with_error_detection(
        ...     orch, "cnn_transformer", device="cuda"
        ... )
        >>> if errors:
        ...     print("Training had errors:", [e.error_type for e in errors])
    """
    monitor = TrainingMonitor(verbose=True)

    # Build training command
    cmd = f"""
    cd {orch.workspace} && \\
    source /tmp/moola-venv/bin/activate && \\
    export PYTHONPATH="{orch.workspace}/src:$PYTHONPATH" && \\
    python -m moola.cli oof --model {model} --device {device} --seed 1337
    """

    if encoder_path:
        cmd += f" --load-pretrained-encoder {encoder_path}"

    print(f"[MONITOR] Starting monitored training for {model}...")

    # Execute command with streaming
    import subprocess

    ssh_cmd = (
        f"ssh root@{orch.host} -p {orch.port} -i {orch.key_path} "
        f"-o StrictHostKeyChecking=no "
        f"'{cmd}'"
    )

    process = subprocess.Popen(
        ssh_cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Monitor output
    for line in iter(process.stdout.readline, ''):
        if line:
            print(line, end='')
            monitor.process_line(line)

    # Wait for completion
    exit_code = process.wait()

    # Generate report
    print("\n" + monitor.generate_report())

    return exit_code, monitor.get_errors(), monitor.get_metrics()
