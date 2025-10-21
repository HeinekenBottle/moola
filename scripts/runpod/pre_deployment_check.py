#!/usr/bin/env python3
"""
Pre-Deployment Validation Script

Comprehensive checklist and validation that must pass before any RunPod deployment.
Prevents the common issues that cause deployment failures.

Usage:
    python pre_deployment_check.py [--fix] [--strict]
    python pre_deployment_check.py --git-check
    python pre_deployment_check.py --dependency-check
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

@dataclass
class CheckResult:
    name: str
    status: str  # "PASS", "FAIL", "WARN"
    message: str
    details: Optional[Dict] = None
    fix_available: bool = False

@dataclass
class DeploymentReadiness:
    ready: bool
    total_checks: int
    passed: int
    failed: int
    warnings: int
    results: List[CheckResult]
    recommendations: List[str]

class PreDeploymentChecker:
    """Comprehensive pre-deployment validation."""

    def __init__(self, strict: bool = False):
        self.strict = strict
        self.results: List[CheckResult] = []
        self.recommendations: List[str] = []

    def add_result(self, name: str, status: str, message: str,
                   details: Optional[Dict] = None, fix_available: bool = False):
        """Add a check result."""
        result = CheckResult(
            name=name,
            status=status,
            message=message,
            details=details,
            fix_available=fix_available
        )
        self.results.append(result)

        # Print immediate feedback
        icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{icon} {name}: {message}")

    def check_python_version(self) -> bool:
        """Check Python version compatibility."""
        print("\n🐍 Python Version Check")
        print("-" * 40)

        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.micro}"

        if version >= (3, 10):
            self.add_result(
                "Python Version",
                "PASS",
                f"Python {version_str} (compatible)",
                {"version": version_str, "minimum": "3.10+"}
            )
            return True
        else:
            self.add_result(
                "Python Version",
                "FAIL",
                f"Python {version_str} (requires 3.10+)",
                {"version": version_str, "minimum": "3.10+"},
                fix_available=True
            )
            return False

    def check_git_status(self) -> bool:
        """Check git repository status."""
        print("\n📦 Git Repository Check")
        print("-" * 40)

        try:
            # Check if we're in a git repo
            subprocess.run(["git", "rev-parse", "--git-dir"],
                         cwd=PROJECT_ROOT, check=True, capture_output=True)

            # Check for uncommitted changes
            result = subprocess.run(["git", "status", "--porcelain"],
                                  cwd=PROJECT_ROOT, capture_output=True, text=True)

            if result.returncode != 0:
                self.add_result("Git Repository", "FAIL", "Not a git repository")
                return False

            uncommitted = result.stdout.strip()
            if not uncommitted:
                self.add_result("Git Status", "PASS", "Working directory clean")
            else:
                self.add_result(
                    "Git Status",
                    "WARN" if not self.strict else "FAIL",
                    f"Uncommitted changes ({len(uncommitted.splitlines())} files)",
                    {"files": uncommitted.splitlines()},
                    fix_available=True
                )

            # Check current branch
            branch_result = subprocess.run(["git", "branch", "--show-current"],
                                         cwd=PROJECT_ROOT, capture_output=True, text=True)
            current_branch = branch_result.stdout.strip()

            if current_branch == "main":
                self.add_result("Git Branch", "PASS", f"On main branch ({current_branch})")
            else:
                self.add_result(
                    "Git Branch",
                    "WARN",
                    f"On branch '{current_branch}' (recommend main)",
                    {"current_branch": current_branch}
                )

            # Check if push is needed
            ahead_result = subprocess.run(
                ["git", "rev-list", "--count", "HEAD..@{u}"],
                cwd=PROJECT_ROOT, capture_output=True, text=True
            )

            if ahead_result.returncode == 0:
                ahead_count = int(ahead_result.stdout.strip())
                if ahead_count == 0:
                    self.add_result("Git Sync", "PASS", "Up to date with remote")
                else:
                    self.add_result(
                        "Git Sync",
                        "WARN" if not self.strict else "FAIL",
                        f"{ahead_count} commits ahead of remote",
                        {"ahead_count": ahead_count},
                        fix_available=True
                    )
            else:
                self.add_result("Git Sync", "WARN", "Could not check remote sync status")

            return True

        except subprocess.CalledProcessError:
            self.add_result("Git Repository", "FAIL", "Git command failed")
            return False
        except FileNotFoundError:
            self.add_result("Git Repository", "FAIL", "Git not installed")
            return False

    def check_critical_files(self) -> bool:
        """Check for critical project files."""
        print("\n📁 Critical Files Check")
        print("-" * 40)

        critical_files = {
            "requirements-runpod-bulletproof.txt": "Bulletproof requirements file",
            "src/moola/cli.py": "Main CLI interface",
            "src/moola/models/simple_lstm.py": "SimpleLSTM model",
            "src/molla/utils/data_validation.py": "Data validation utilities",
            "src/molla/logging_setup.py": "Logging configuration",
            "src/molla/paths.py": "Path configuration",
            "pyproject.toml": "Project configuration",
            "README.md": "Project documentation",
        }

        all_exist = True
        for file_path, description in critical_files.items():
            full_path = PROJECT_ROOT / file_path
            if full_path.exists():
                self.add_result(
                    f"File: {file_path}",
                    "PASS",
                    f"Found: {description}"
                )
            else:
                self.add_result(
                    f"File: {file_path}",
                    "FAIL",
                    f"Missing: {description}",
                    fix_available=True
                )
                all_exist = False

        return all_exist

    def check_dependencies(self) -> bool:
        """Check local dependencies."""
        print("\n📦 Local Dependencies Check")
        print("-" * 40)

        critical_deps = [
            "numpy",
            "pandas",
            "scipy",
            "scikit-learn",
            "torch",
            "tqdm",
            "loguru",
            "rich",
            "pydantic",
            "click"
        ]

        all_ok = True
        for dep in critical_deps:
            try:
                __import__(dep)
                self.add_result(
                    f"Dependency: {dep}",
                    "PASS",
                    "Importable"
                )
            except ImportError:
                self.add_result(
                    f"Dependency: {dep}",
                    "FAIL",
                    "Not importable",
                    fix_available=True
                )
                all_ok = False

        return all_ok

    def check_moola_modules(self) -> bool:
        """Check Moola-specific modules."""
        print("\n🔧 Moola Modules Check")
        print("-" * 40)

        moola_modules = [
            "moola.cli",
            "molla.models.simple_lstm",
            "molla.pretraining.masked_lstm_pretrain",
            "molla.data.load",
            "molla.features.relative_transform",
            "molla.utils.data_validation",
            "molla.logging_setup",
            "molla.paths",
            "molla.utils.seeds"
        ]

        all_ok = True
        for module in moola_modules:
            try:
                __import__(module)
                self.add_result(
                    f"Module: {module}",
                    "PASS",
                    "Importable"
                )
            except ImportError as e:
                self.add_result(
                    f"Module: {module}",
                    "FAIL",
                    f"Import error: {str(e)[:100]}..."
                )
                all_ok = False
            except Exception as e:
                self.add_result(
                    f"Module: {module}",
                    "WARN",
                    f"Other error: {str(e)[:100]}..."
                )

        return all_ok

    def check_ssh_access(self, host: Optional[str] = None, key_path: Optional[str] = None) -> bool:
        """Check SSH access to RunPod."""
        print("\n🔐 SSH Access Check")
        print("-" * 40)

        if not host or not key_path:
            self.add_result(
                "SSH Access",
                "WARN",
                "Host/key not provided - skipping check"
            )
            return True

        try:
            # Test SSH connection
            result = subprocess.run([
                "ssh", "-i", key_path, "-p", "22",
                f"ubuntu@{host}", "echo 'SSH_OK'"
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and "SSH_OK" in result.stdout:
                self.add_result(
                    "SSH Connection",
                    "PASS",
                    f"Can SSH to {host}"
                )
                return True
            else:
                self.add_result(
                    "SSH Connection",
                    "FAIL",
                    f"SSH failed to {host}",
                    {"error": result.stderr[:200]}
                )
                return False

        except subprocess.TimeoutExpired:
            self.add_result(
                "SSH Connection",
                "FAIL",
                f"SSH timeout to {host}"
            )
            return False
        except Exception as e:
            self.add_result(
                "SSH Connection",
                "WARN",
                f"SSH check failed: {e}"
            )
            return False

    def check_disk_space(self) -> bool:
        """Check available disk space."""
        print("\n💾 Disk Space Check")
        print("-" * 40)

        try:
            import shutil
            total, used, free = shutil.disk_usage(PROJECT_ROOT)
            free_gb = free // (1024**3)

            if free_gb >= 5:
                self.add_result(
                    "Disk Space",
                    "PASS",
                    f"{free_gb}GB free (sufficient)"
                )
                return True
            else:
                self.add_result(
                    "Disk Space",
                    "WARN",
                    f"{free_gb}GB free (recommend 5GB+)"
                )
                return False

        except Exception as e:
            self.add_result(
                "Disk Space",
                "WARN",
                f"Could not check disk space: {e}"
            )
            return False

    def attempt_fixes(self) -> bool:
        """Attempt to fix common issues."""
        print("\n🔧 Attempting Fixes...")
        print("-" * 40)

        fixes_applied = []

        # Create configs directory
        configs_dir = PROJECT_ROOT / "configs"
        if not configs_dir.exists():
            configs_dir.mkdir(exist_ok=True)
            default_config = configs_dir / "default.yaml"
            if not default_config.exists():
                default_config.write_text("# Default Moola Configuration\ndefaults:\n  - _self_\n")
            fixes_applied.append("Created configs/ directory with default config")

        # Create bulletproof requirements if missing
        req_file = PROJECT_ROOT / "requirements-runpod-bulletproof.txt"
        if not req_file.exists():
            # Copy from existing requirements.txt with modifications
            req_content = """# Moola RunPod Bulletproof Requirements
# Template: runpod/pytorch:2.4-py3.11-cuda12.4-ubuntu22.04

numpy>=1.26.4,<2.1
pandas>=2.3,<3.0
scipy>=1.14,<2.0
scikit-learn>=1.7,<2.0
xgboost>=2.0,<3.0
imbalanced-learn==0.14.0
pytorch-lightning>=2.4.0,<3.0
torchmetrics>=1.8,<2.0
pyarrow>=17.0,<18.0
pandera>=0.26.1,<1.0
click>=8.2,<9.0
typer>=0.17,<1.0
hydra-core>=1.3,<2.0
pydantic>=2.11,<3.0
pydantic-settings>=2.9,<3.0
python-dotenv>=1.0
loguru>=0.7,<1.0
rich>=14.0,<15.0
tqdm>=4.66,<5.0
joblib>=1.5,<2.0
PyYAML>=6.0
"""
            req_file.write_text(req_content)
            fixes_applied.append("Created requirements-runpod-bulletproof.txt")

        # Install missing critical dependencies
        failed_deps = [r.name.split(": ")[1] for r in self.results
                      if r.status == "FAIL" and r.name.startswith("Dependency:")]

        if failed_deps:
            print(f"Installing missing dependencies: {', '.join(failed_deps)}")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install"
                ] + failed_deps, check=True, timeout=300)
                fixes_applied.append(f"Installed: {', '.join(failed_deps)}")
            except subprocess.CalledProcessError:
                print("❌ Failed to install dependencies")

        if fixes_applied:
            print("✅ Fixes applied:")
            for fix in fixes_applied:
                print(f"  • {fix}")
            return True
        else:
            print("ℹ️ No fixes needed or possible")
            return False

    def generate_recommendations(self):
        """Generate deployment recommendations."""
        self.recommendations = []

        failed_checks = [r for r in self.results if r.status == "FAIL"]
        warning_checks = [r for r in self.results if r.status == "WARN"]

        if failed_checks:
            self.recommendations.append("🚨 CRITICAL ISSUES - Must fix before deployment:")
            for check in failed_checks:
                if check.fix_available:
                    self.recommendations.append(f"  • Run with --fix to auto-fix: {check.name}")
                else:
                    self.recommendations.append(f"  • Manual fix required: {check.name} - {check.message}")

        if warning_checks:
            self.recommendations.append("⚠️ WARNINGS - Recommended to fix:")
            for check in warning_checks:
                self.recommendations.append(f"  • {check.name}: {check.message}")

        if not failed_checks and not warning_checks:
            self.recommendations.append("🎉 All checks passed! Ready for deployment.")

        # General recommendations
        self.recommendations.extend([
            "",
            "📋 Deployment Checklist:",
            "  • Ensure RunPod pod is running with PyTorch 2.4 template",
            "  • Verify SSH key access to the pod",
            "  • Have at least 10GB free disk space on pod",
            "  • Test with a small training job first",
            "",
            "🚀 Ready to deploy with:",
            f"  python scripts/runpod/bulletproof_deployment.py --host YOUR_POD_IP --key ~/.ssh/runpod_key"
        ])

    def run_all_checks(self, host: Optional[str] = None, key_path: Optional[str] = None) -> DeploymentReadiness:
        """Run all pre-deployment checks."""
        print("🔍 Moola Pre-Deployment Validation")
        print("=" * 60)

        # Run all checks
        checks = [
            self.check_python_version,
            self.check_git_status,
            self.check_critical_files,
            self.check_dependencies,
            self.check_moola_modules,
            lambda: self.check_disk_space(),
            lambda: self.check_ssh_access(host, key_path)
        ]

        for check in checks:
            try:
                check()
            except Exception as e:
                print(f"❌ Check failed with error: {e}")

        # Count results
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        warnings = sum(1 for r in self.results if r.status == "WARN")

        # Generate recommendations
        self.generate_recommendations()

        # Determine readiness
        ready = failed == 0 if not self.strict else failed == 0 and warnings == 0

        return DeploymentReadiness(
            ready=ready,
            total_checks=len(self.results),
            passed=passed,
            failed=failed,
            warnings=warnings,
            results=self.results,
            recommendations=self.recommendations
        )

def main():
    parser = argparse.ArgumentParser(description="Moola Pre-Deployment Validation")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    parser.add_argument("--strict", action="store_true", help="Strict mode - warnings treated as failures")
    parser.add_argument("--host", help="RunPod host IP for SSH check")
    parser.add_argument("--key", help="SSH key path for SSH check")
    parser.add_argument("--git-check", action="store_true", help="Only check git status")
    parser.add_argument("--dependency-check", action="store_true", help="Only check dependencies")
    parser.add_argument("--save-report", type=str, help="Save report to JSON file")

    args = parser.parse_args()

    checker = PreDeploymentChecker(strict=args.strict)

    try:
        if args.git_check:
            checker.check_git_status()
        elif args.dependency_check:
            checker.check_dependencies()
            checker.check_moola_modules()
        else:
            # Run full check
            readiness = checker.run_all_checks(args.host, args.key)

            # Print summary
            print(f"\n📊 SUMMARY")
            print(f"Total: {readiness.total_checks} | ✅ Passed: {readiness.passed} | ❌ Failed: {readiness.failed} | ⚠️ Warnings: {readiness.warnings}")
            print()

            # Print recommendations
            for rec in readiness.recommendations:
                print(rec)

            # Save report if requested
            if args.save_report:
                report_data = asdict(readiness)
                with open(args.save_report, 'w') as f:
                    json.dump(report_data, f, indent=2, default=str)
                print(f"\n📄 Report saved to: {args.save_report}")

            # Attempt fixes if requested and there are failures
            if args.fix and readiness.failed > 0:
                print(f"\n🔧 Attempting fixes...")
                checker.attempt_fixes()
                print(f"\n🔄 Re-running checks after fixes...")
                new_readiness = checker.run_all_checks(args.host, args.key)

                if new_readiness.failed < readiness.failed:
                    print(f"✅ Fixed {readiness.failed - new_readiness.failed} issues!")
                else:
                    print("❌ No issues were fixed")

            # Exit with appropriate code
            if readiness.failed > 0:
                print(f"\n❌ {readiness.failed} critical issues must be fixed before deployment")
                sys.exit(1)
            elif readiness.warnings > 0 and args.strict:
                print(f"\n⚠️ {readiness.warnings} warnings in strict mode")
                sys.exit(1)
            else:
                print(f"\n🎉 Ready for deployment!")
                sys.exit(0)

    except KeyboardInterrupt:
        print("\n⏹️ Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error during validation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()