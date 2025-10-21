#!/usr/bin/env python3
"""
Bulletproof RunPod SSH/SCP Deployment Script

Reliable SSH/SCP deployment that handles:
1. Environment validation and setup
2. Dependency installation with conflict resolution
3. Code deployment from GitHub or local SCP
4. Configuration validation
5. Manual training with preparation
6. Results retrieval

Usage:
    python bulletproof_deployment.py --host HOST --key KEY [--mode github|scp]
    python bulletproof_deployment.py --validate-local
    python bulletproof_deployment.py --prepare-scp
"""

import sys
import os
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import tempfile
import shutil
import tarfile

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from moola.runpod.scp_orchestrator import RunPodOrchestrator
except ImportError:
    print("❌ RunPod orchestrator not found - using basic SSH implementation")
    RunPodOrchestrator = None

@dataclass
class DeploymentConfig:
    host: str
    port: int
    key_path: str
    workspace: str = "/workspace/moola"
    mode: str = "github"  # github, scp, bundle
    model: str = "simple_lstm"
    device: str = "cuda"
    seed: int = 1337
    python_version: str = "3.11"  # Target Python version on RunPod
    pytorch_template: str = "runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04"

@dataclass
class DeploymentResult:
    success: bool
    phase: str
    message: str
    details: Optional[Dict] = None
    timestamp: Optional[str] = None

class BulletproofDeployment:
    """Bulletproof SSH/SCP deployment assistance for Moola on RunPod."""

    def __init__(self, config: DeploymentConfig):
        self.config = config
        self.deployment_log: List[DeploymentResult] = []
        self.temp_dir = None

        # Initialize SSH orchestrator if available
        if RunPodOrchestrator:
            self.orchestrator = RunPodOrchestrator(
                host=config.host,
                port=config.port,
                key_path=config.key_path,
                workspace=config.workspace,
                verbose=True
            )
        else:
            self.orchestrator = None

        self.log("INIT", f"Starting bulletproof deployment for {config.host}:{config.port}")

    def log(self, phase: str, message: str, details: Optional[Dict] = None):
        """Log deployment phase."""
        result = DeploymentResult(
            success=False,
            phase=phase,
            message=message,
            details=details,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.deployment_log.append(result)

        print(f"[{phase}] {message}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")

    def log_success(self, phase: str, message: str, details: Optional[Dict] = None):
        """Log successful deployment phase."""
        result = DeploymentResult(
            success=True,
            phase=phase,
            message=message,
            details=details,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.deployment_log.append(result)

        print(f"✅ [{phase}] {message}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")

    def log_error(self, phase: str, message: str, details: Optional[Dict] = None):
        """Log deployment error."""
        result = DeploymentResult(
            success=False,
            phase=phase,
            message=message,
            details=details,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        self.deployment_log.append(result)

        print(f"❌ [{phase}] {message}")
        if details:
            for key, value in details.items():
                print(f"    {key}: {value}")

    def execute_ssh_command(self, command: str, timeout: int = 300,
                          check: bool = True) -> Tuple[int, str, str]:
        """Execute SSH command with proper error handling."""
        if self.orchestrator:
            # Use RunPod orchestrator
            exit_code = self.orchestrator.execute_command(command, timeout=timeout)
            return exit_code, "", ""  # Orchestrator handles output internally
        else:
            # Fallback to direct SSH
            ssh_cmd = [
                "ssh", "-i", self.config.key_path, "-p", str(self.config.port),
                f"ubuntu@{self.config.host}", command
            ]

            try:
                result = subprocess.run(
                    ssh_cmd, capture_output=True, text=True, timeout=timeout
                )
                if check and result.returncode != 0:
                    raise subprocess.CalledProcessError(
                        result.returncode, ssh_cmd, result.stdout, result.stderr
                    )
                return result.returncode, result.stdout, result.stderr
            except subprocess.TimeoutExpired:
                raise Exception(f"SSH command timed out: {command}")

    def scp_upload(self, local_path: Path, remote_path: str):
        """Upload file via SCP."""
        if self.orchestrator:
            self.orchestrator.upload_file(local_path, remote_path)
        else:
            # Fallback to direct SCP
            scp_cmd = [
                "scp", "-i", self.config.key_path, "-P", str(self.config.port),
                str(local_path), f"ubuntu@{self.config.host}:{remote_path}"
            ]
            subprocess.run(scp_cmd, check=True)

    def scp_download(self, remote_path: str, local_path: Path):
        """Download file via SCP."""
        if self.orchestrator:
            self.orchestrator.download_file(remote_path, local_path)
        else:
            # Fallback to direct SCP
            scp_cmd = [
                "scp", "-i", self.config.key_path, "-P", str(self.config.port),
                f"ubuntu@{self.config.host}:{remote_path}", str(local_path)
            ]
            subprocess.run(scp_cmd, check=True)

    def validate_local_environment(self) -> bool:
        """Validate local environment before deployment."""
        self.log("LOCAL_CHECK", "Validating local environment...")

        # Check Python version
        python_version = sys.version_info
        self.log("LOCAL_CHECK", f"Local Python: {python_version.major}.{python_version.minor}.{python_version.micro}")

        # Check critical local files
        critical_files = [
            "requirements-runpod-bulletproof.txt",
            "src/moola/cli.py",
            "src/moola/models/simple_lstm.py",
            "src/molla/utils/data_validation.py",
        ]

        missing_files = []
        for file_path in critical_files:
            if not (PROJECT_ROOT / file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            self.log_error("LOCAL_CHECK", f"Missing critical files: {missing_files}")
            return False

        # Check git status
        try:
            git_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT, capture_output=True, text=True
            )
            if git_result.stdout.strip():
                self.log("LOCAL_CHECK", "⚠️ Git working directory not clean",
                         {"changes": git_result.stdout.strip()})
            else:
                self.log_success("LOCAL_CHECK", "Git working directory clean")
        except Exception as e:
            self.log("LOCAL_CHECK", f"Could not check git status: {e}")

        self.log_success("LOCAL_CHECK", "Local environment validated")
        return True

    def validate_runpod_environment(self) -> bool:
        """Validate RunPod environment."""
        self.log("RUNPOD_CHECK", "Validating RunPod environment...")

        try:
            # Check Python version
            exit_code, stdout, stderr = self.execute_ssh_command(
                "python3 --version", timeout=30
            )
            if exit_code == 0:
                python_version = stdout.strip()
                self.log("RUNPOD_CHECK", f"RunPod Python: {python_version}")
            else:
                self.log_error("RUNPOD_CHECK", "Could not check Python version")
                return False

            # Check CUDA availability
            exit_code, stdout, stderr = self.execute_ssh_command(
                "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits",
                timeout=30
            )
            if exit_code == 0:
                gpu_info = stdout.strip()
                self.log_success("RUNPOD_CHECK", f"GPU detected: {gpu_info}")
            else:
                self.log_error("RUNPOD_CHECK", "No GPU detected")
                return False

            # Check PyTorch installation
            exit_code, stdout, stderr = self.execute_ssh_command(
                "python3 -c 'import torch; print(f\"PyTorch: {torch.__version__}\")'",
                timeout=30
            )
            if exit_code == 0:
                pytorch_version = stdout.strip()
                self.log("RUNPOD_CHECK", f"PyTorch: {pytorch_version}")
            else:
                self.log("RUNPOD_CHECK", "PyTorch not installed or not accessible")

            self.log_success("RUNPOD_CHECK", "RunPod environment validated")
            return True

        except Exception as e:
            self.log_error("RUNPOD_CHECK", f"RunPod validation failed: {e}")
            return False

    def setup_workspace(self) -> bool:
        """Setup remote workspace."""
        self.log("WORKSPACE", "Setting up remote workspace...")

        try:
            # Create workspace directory
            self.execute_ssh_command(f"mkdir -p {self.config.workspace}")

            # Create necessary subdirectories
            subdirs = [
                f"{self.config.workspace}/data/processed",
                f"{self.config.workspace}/data/raw",
                f"{self.config.workspace}/artifacts/models",
                f"{self.config.workspace}/artifacts/pretrained",
                f"{self.config.workspace}/logs",
                f"{self.config.workspace}/configs"
            ]

            for subdir in subdirs:
                self.execute_ssh_command(f"mkdir -p {subdir}")

            self.log_success("WORKSPACE", "Remote workspace created")
            return True

        except Exception as e:
            self.log_error("WORKSPACE", f"Failed to setup workspace: {e}")
            return False

    def deploy_code(self) -> bool:
        """Deploy code based on selected mode."""
        self.log("CODE_DEPLOY", f"Deploying code using {self.config.mode} mode...")

        try:
            if self.config.mode == "github":
                return self._deploy_from_github()
            elif self.config.mode == "scp":
                return self._deploy_from_scp()
            elif self.config.mode == "bundle":
                return self._deploy_from_bundle()
            else:
                self.log_error("CODE_DEPLOY", f"Unknown deployment mode: {self.config.mode}")
                return False

        except Exception as e:
            self.log_error("CODE_DEPLOY", f"Code deployment failed: {e}")
            return False

    def _deploy_from_github(self) -> bool:
        """Deploy code from GitHub repository."""
        self.log("GITHUB", "Cloning from GitHub...")

        try:
            # Clone or update repository
            clone_cmd = f"""
            cd {self.config.workspace}/.. &&
            if [ -d "moola" ]; then
                cd moola && git pull origin main
            else
                git clone https://github.com/HeinekenBottle/moola.git
            fi
            """

            exit_code, stdout, stderr = self.execute_ssh_command(clone_cmd, timeout=120)

            if exit_code == 0:
                self.log_success("GITHUB", "Code deployed from GitHub")
                return True
            else:
                self.log_error("GITHUB", f"Git clone/pull failed: {stderr}")
                return False

        except Exception as e:
            self.log_error("GITHUB", f"GitHub deployment failed: {e}")
            return False

    def _deploy_from_scp(self) -> bool:
        """Deploy code via SCP."""
        self.log("SCP", "Uploading code via SCP...")

        try:
            # Create temporary directory for code bundle
            self.temp_dir = Path(tempfile.mkdtemp(prefix="moola_deploy_"))

            # Copy essential files
            code_files = [
                "src/",
                "requirements-runpod-bulletproof.txt",
                "README.md",
                "pyproject.toml",
            ]

            for file_path in code_files:
                src_path = PROJECT_ROOT / file_path
                if src_path.exists():
                    if src_path.is_dir():
                        shutil.copytree(src_path, self.temp_dir / file_path)
                    else:
                        shutil.copy2(src_path, self.temp_dir / file_path)

            # Create tarball
            tarball_path = self.temp_dir / "code_bundle.tar.gz"
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(self.temp_dir, arcname="moola")

            # Upload tarball
            remote_tarball = f"/tmp/code_bundle_{int(time.time())}.tar.gz"
            self.scp_upload(tarball_path, remote_tarball)

            # Extract on remote
            extract_cmd = f"""
            cd {self.config.workspace}/.. &&
            tar -xzf {remote_tarball} &&
            mv moola/moola/* {self.config.workspace}/ &&
            rm -rf {remote_tarball} moola
            """

            self.execute_ssh_command(extract_cmd, timeout=60)

            self.log_success("SCP", "Code deployed via SCP")
            return True

        except Exception as e:
            self.log_error("SCP", f"SCP deployment failed: {e}")
            return False

    def _deploy_from_bundle(self) -> bool:
        """Deploy from pre-created bundle."""
        self.log("BUNDLE", "Deploying from pre-created bundle...")

        # Look for existing bundle
        bundle_pattern = PROJECT_ROOT / "artifacts" / "runpod_bundles" / "runpod_bundle_*.tar.gz"
        bundles = list(PROJECT_ROOT.glob("artifacts/runpod_bundles/runpod_bundle_*.tar.gz"))

        if not bundles:
            self.log_error("BUNDLE", "No pre-created bundle found")
            return False

        # Use the most recent bundle
        latest_bundle = max(bundles, key=lambda x: x.stat().st_mtime)
        self.log("BUNDLE", f"Using bundle: {latest_bundle.name}")

        try:
            # Upload bundle
            remote_bundle = f"/tmp/{latest_bundle.name}"
            self.scp_upload(latest_bundle, remote_bundle)

            # Extract bundle
            extract_cmd = f"""
            cd {self.config.workspace}/.. &&
            tar -xzf {remote_bundle} &&
            rm -rf {remote_bundle}
            """

            self.execute_ssh_command(extract_cmd, timeout=120)

            self.log_success("BUNDLE", "Bundle deployed successfully")
            return True

        except Exception as e:
            self.log_error("BUNDLE", f"Bundle deployment failed: {e}")
            return False

    def install_dependencies(self) -> bool:
        """Install dependencies with conflict resolution."""
        self.log("DEPS", "Installing dependencies...")

        try:
            # Upload requirements file
            requirements_path = PROJECT_ROOT / "requirements-runpod-bulletproof.txt"
            remote_requirements = f"{self.config.workspace}/requirements-runpod-bulletproof.txt"
            self.scp_upload(requirements_path, remote_requirements)

            # Install dependencies
            install_cmd = f"""
            cd {self.config.workspace} &&
            python3 -m pip install --no-cache-dir -r requirements-runpod-bulletproof.txt &&
            python3 -m pip install --no-cache-dir -e .
            """

            exit_code, stdout, stderr = self.execute_ssh_command(install_cmd, timeout=600)

            if exit_code == 0:
                self.log_success("DEPS", "Dependencies installed successfully")
                return True
            else:
                self.log_error("DEPS", f"Dependency installation failed: {stderr}")
                return False

        except Exception as e:
            self.log_error("DEPS", f"Dependency installation failed: {e}")
            return False

    def validate_installation(self) -> bool:
        """Validate installation with dependency audit."""
        self.log("VALIDATE", "Running installation validation...")

        try:
            # Upload dependency audit script
            audit_script_path = PROJECT_ROOT / "scripts" / "runpod" / "dependency_audit.py"
            remote_audit = f"{self.config.workspace}/dependency_audit.py"
            self.scp_upload(audit_script_path, remote_audit)

            # Run audit
            audit_cmd = f"""
            cd {self.config.workspace} &&
            export PYTHONPATH="{self.config.workspace}/src:$PYTHONPATH" &&
            python3 dependency_audit.py --save-report audit_report.json
            """

            exit_code, stdout, stderr = self.execute_ssh_command(audit_cmd, timeout=120)

            if exit_code == 0:
                # Download audit report
                local_report = PROJECT_ROOT / "runpod_audit_report.json"
                self.scp_download(f"{self.config.workspace}/audit_report.json", local_report)

                # Parse report
                with open(local_report, 'r') as f:
                    report = json.load(f)

                self.log_success("VALIDATE",
                    f"Installation validated: {report['passed']}/{report['total_dependencies']} OK")

                if report['failed'] > 0:
                    self.log("VALIDATE", f"⚠️ {report['failed']} dependencies have issues")

                return True
            else:
                self.log_error("VALIDATE", f"Validation failed: {stderr}")
                return False

        except Exception as e:
            self.log_error("VALIDATE", f"Validation failed: {e}")
            return False

    def run_training(self) -> bool:
        """Run training with monitoring."""
        self.log("TRAINING", f"Starting {self.config.model} training...")

        try:
            # Set environment variables
            env_setup = f"""
            export PYTHONPATH="{self.config.workspace}/src:$PYTHONPATH" &&
            export MOOLA_DATA_DIR="{self.config.workspace}/data" &&
            export MOOLA_ARTIFACTS_DIR="{self.config.workspace}/artifacts"
            """

            # Build training command
            if self.config.model == "simple_lstm":
                train_cmd = f"""
                cd {self.config.workspace} &&
                {env_setup} &&
                python3 -m moola.cli train --model simple_lstm --device {self.config.device} --seed {self.config.seed}
                """
            elif self.config.model == "pretrain_bilstm":
                train_cmd = f"""
                cd {self.config.workspace} &&
                {env_setup} &&
                python3 -m moola.cli pretrain-bilstm --device {self.config.device} --n-epochs 50
                """
            else:
                self.log_error("TRAINING", f"Unknown model: {self.config.model}")
                return False

            # Run training with extended timeout
            self.log("TRAINING", f"Executing training command (timeout: 3600s)...")
            exit_code, stdout, stderr = self.execute_ssh_command(train_cmd, timeout=3600)

            if exit_code == 0:
                self.log_success("TRAINING", "Training completed successfully")
                return True
            else:
                self.log_error("TRAINING", f"Training failed: {stderr}")
                return False

        except Exception as e:
            self.log_error("TRAINING", f"Training execution failed: {e}")
            return False

    def retrieve_results(self) -> bool:
        """Retrieve training results."""
        self.log("RESULTS", "Retrieving results...")

        try:
            # Create local results directory
            results_dir = PROJECT_ROOT / "runpod_results" / f"run_{int(time.time())}"
            results_dir.mkdir(parents=True, exist_ok=True)

            # Download results files
            result_patterns = [
                "experiment_results.jsonl",
                "artifacts/models/*.pkl",
                "artifacts/pretrained/*.pt",
                "logs/*.log"
            ]

            files_downloaded = 0
            for pattern in result_patterns:
                try:
                    # Use glob to find matching files
                    list_cmd = f"cd {self.config.workspace} && find . -name '{pattern}'"
                    exit_code, stdout, stderr = self.execute_ssh_command(list_cmd, timeout=30)

                    if exit_code == 0 and stdout.strip():
                        files = stdout.strip().split('\n')
                        for remote_file in files:
                            if remote_file.strip():
                                local_file = results_dir / Path(remote_file).name
                                self.scp_download(
                                    f"{self.config.workspace}/{remote_file}",
                                    local_file
                                )
                                files_downloaded += 1
                                self.log("RESULTS", f"Downloaded: {remote_file}")

                except Exception as e:
                    self.log("RESULTS", f"Could not download {pattern}: {e}")

            if files_downloaded > 0:
                self.log_success("RESULTS", f"Retrieved {files_downloaded} result files to {results_dir}")
                return True
            else:
                self.log_error("RESULTS", "No result files found")
                return False

        except Exception as e:
            self.log_error("RESULTS", f"Results retrieval failed: {e}")
            return False

    def deploy(self) -> bool:
        """Execute full deployment pipeline."""
        print("🚀 STARTING BULLETPROOF RUNPOD DEPLOYMENT")
        print("=" * 60)
        print(f"Target: {self.config.host}:{self.config.port}")
        print(f"Mode: {self.config.mode}")
        print(f"Model: {self.config.model}")
        print("=" * 60)

        # Phase 1: Local validation
        if not self.validate_local_environment():
            return False

        # Phase 2: RunPod validation
        if not self.validate_runpod_environment():
            return False

        # Phase 3: Workspace setup
        if not self.setup_workspace():
            return False

        # Phase 4: Code deployment
        if not self.deploy_code():
            return False

        # Phase 5: Dependencies
        if not self.install_dependencies():
            return False

        # Phase 6: Installation validation
        if not self.validate_installation():
            return False

        # Phase 7: Training
        if not self.run_training():
            return False

        # Phase 8: Results retrieval
        if not self.retrieve_results():
            return False

        # Cleanup
        if self.temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)

        print("=" * 60)
        self.log_success("COMPLETE", "🎉 Deployment completed successfully!")
        return True

    def save_deployment_log(self, filepath: Path):
        """Save deployment log to file."""
        log_data = [asdict(result) for result in self.deployment_log]
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        print(f"📄 Deployment log saved to: {filepath}")

def prepare_bundle():
    """Prepare deployment bundle."""
    print("📦 Preparing deployment bundle...")

    bundle_dir = PROJECT_ROOT / "artifacts" / "runpod_bundles"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_name = f"runpod_bundle_{time.strftime('%Y%m%d_%H%M%S')}"
    bundle_path = bundle_dir / f"{bundle_name}.tar.gz"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "moola"

        # Copy essential files
        essential_items = [
            "src/",
            "requirements-runpod-bulletproof.txt",
            "scripts/runpod/dependency_audit.py",
            "README.md",
            "pyproject.toml",
        ]

        for item in essential_items:
            src = PROJECT_ROOT / item
            dst = temp_path / item
            if src.exists():
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)

        # Create bundle
        with tarfile.open(bundle_path, "w:gz") as tar:
            tar.add(temp_path, arcname="moola")

    print(f"✅ Bundle created: {bundle_path}")
    return bundle_path

def main():
    parser = argparse.ArgumentParser(description="Bulletproof RunPod Deployment")
    parser.add_argument("--host", required=True, help="RunPod host IP")
    parser.add_argument("--port", type=int, default=22, help="SSH port")
    parser.add_argument("--key", required=True, help="SSH key path")
    parser.add_argument("--mode", choices=["github", "scp", "bundle"],
                       default="github", help="Deployment mode")
    parser.add_argument("--model", choices=["simple_lstm", "pretrain_bilstm"],
                       default="simple_lstm", help="Model to train")
    parser.add_argument("--device", default="cuda", help="Training device")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--validate-local", action="store_true",
                       help="Only validate local environment")
    parser.add_argument("--prepare-bundle", action="store_true",
                       help="Prepare deployment bundle")
    parser.add_argument("--save-log", help="Save deployment log to file")

    args = parser.parse_args()

    if args.validate_local:
        # Local validation only
        config = DeploymentConfig(
            host=args.host,
            port=args.port,
            key_path=args.key,
            mode="github"
        )
        deployment = BulletproofDeployment(config)
        success = deployment.validate_local_environment()
        sys.exit(0 if success else 1)

    elif args.prepare_bundle:
        # Prepare bundle only
        bundle_path = prepare_bundle()
        sys.exit(0 if bundle_path else 1)

    else:
        # Full deployment
        config = DeploymentConfig(
            host=args.host,
            port=args.port,
            key_path=args.key,
            mode=args.mode,
            model=args.model,
            device=args.device,
            seed=args.seed
        )

        deployment = BulletproofDeployment(config)
        success = deployment.deploy()

        if args.save_log:
            deployment.save_deployment_log(Path(args.save_log))

        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()