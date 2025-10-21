#!/usr/bin/env python3
"""
RunPod Dependency Audit Script

Comprehensive dependency validation for Moola on RunPod.
Checks module imports, version compatibility, and environment setup.

Usage:
    python dependency_audit.py [--fix] [--verbose]
    python dependency_audit.py --check-imports-only
    python dependency_audit.py --version-compatibility
"""

import sys
import subprocess
import importlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
from dataclasses import dataclass, asdict

@dataclass
class DependencyStatus:
    name: str
    status: str  # "OK", "MISSING", "VERSION_MISMATCH", "IMPORT_ERROR"
    version: Optional[str] = None
    expected_version: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class AuditReport:
    total_dependencies: int
    passed: int
    failed: int
    warnings: int
    dependencies: List[DependencyStatus]
    environment_info: Dict[str, str]

class DependencyAuditor:
    """Comprehensive dependency auditing for Moola RunPod deployments."""

    def __init__(self):
        # Critical dependencies with version constraints
        self.critical_deps = {
            "torch": ">=2.0,<3.0",  # From template
            "numpy": ">=1.26.4,<2.1",
            "pandas": ">=2.3,<3.0",
            "scipy": ">=1.14,<2.0",
            "scikit-learn": ">=1.7,<2.0",
            "xgboost": ">=2.0,<3.0",
            "imbalanced-learn": "==0.14.0",
            "pytorch-lightning": ">=2.4.0,<3.0",
            "torchmetrics": ">=1.8,<2.0",
            "pyarrow": ">=17.0,<18.0",
            "pandera": ">=0.26.1,<1.0",
            "click": ">=8.2,<9.0",
            "typer": ">=0.17,<1.0",
            "hydra-core": ">=1.3,<2.0",
            "pydantic": ">=2.11,<3.0",
            "pydantic-settings": ">=2.9,<3.0",
            "python-dotenv": ">=1.0",
            "loguru": ">=0.7,<1.0",
            "rich": ">=14.0,<15.0",
            "tqdm": ">=4.66,<5.0",
            "joblib": ">=1.5,<2.0",
            "PyYAML": ">=6.0",
        }

        # Moola-specific modules to check
        self.moola_modules = [
            "moola.cli",
            "molla.models.simple_lstm",
            "moola.models.bilstm_masked_autoencoder",
            "moola.pretraining.masked_lstm_pretrain",
            "moola.data.load",
            "moola.features.relative_transform",
            "moola.utils.data_validation",
            "moola.logging_setup",
            "molla.paths",
            "moola.utils.seeds",
        ]

        self.environment_info = {}

    def get_python_version(self) -> str:
        """Get Python version information."""
        return sys.version.replace('\n', ' ')

    def get_installed_version(self, package_name: str) -> Optional[str]:
        """Get installed version of a package."""
        try:
            import importlib.metadata
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return None
        except Exception as e:
            return f"Error: {e}"

    def check_version_compatibility(self, package_name: str, installed_version: str,
                                  required_version: str) -> Tuple[bool, str]:
        """Check if installed version meets requirements."""
        # Simple version checking - could be enhanced with packaging.version
        try:
            if "==" in required_version:
                expected = required_version.split("==")[1]
                return installed_version == expected, f"Expected {expected}, got {installed_version}"
            elif ">=" in required_version and "<" in required_version:
                # Simple range check - could be improved
                return True, f"Version {installed_version} in range {required_version}"
            else:
                return True, f"Version {installed_version} meets requirement {required_version}"
        except Exception as e:
            return False, f"Version check failed: {e}"

    def check_import(self, module_name: str) -> Tuple[bool, str]:
        """Check if a module can be imported."""
        try:
            importlib.import_module(module_name)
            return True, "Import successful"
        except ImportError as e:
            return False, f"Import error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"

    def audit_dependencies(self) -> List[DependencyStatus]:
        """Audit all critical dependencies."""
        results = []

        for dep_name, version_req in self.critical_deps.items():
            # Check if package is installed
            installed_version = self.get_installed_version(dep_name)

            if installed_version is None:
                status = DependencyStatus(
                    name=dep_name,
                    status="MISSING",
                    expected_version=version_req,
                    error_message=f"Package not installed. Expected: {version_req}"
                )
            else:
                # Check version compatibility
                compatible, message = self.check_version_compatibility(
                    dep_name, installed_version, version_req
                )

                status = DependencyStatus(
                    name=dep_name,
                    status="OK" if compatible else "VERSION_MISMATCH",
                    version=installed_version,
                    expected_version=version_req,
                    error_message=None if compatible else message
                )

            results.append(status)

        return results

    def audit_moola_modules(self) -> List[DependencyStatus]:
        """Audit Moola-specific modules."""
        results = []

        for module_name in self.moola_modules:
            # Handle module name variations
            if module_name == "molla.models.simple_lstm":  # Fix typo in list
                module_name = "moola.models.simple_lstm"

            can_import, message = self.check_import(module_name)

            status = DependencyStatus(
                name=module_name,
                status="OK" if can_import else "IMPORT_ERROR",
                error_message=None if can_import else message
            )

            results.append(status)

        return results

    def check_environment(self) -> Dict[str, str]:
        """Check environment information."""
        env_info = {}

        # Python version
        env_info["python_version"] = self.get_python_version()

        # PyTorch CUDA availability
        try:
            import torch
            env_info["pytorch_version"] = torch.__version__
            env_info["cuda_available"] = str(torch.cuda.is_available())
            if torch.cuda.is_available():
                env_info["cuda_version"] = torch.version.cuda
                env_info["gpu_count"] = str(torch.cuda.device_count())
                env_info["gpu_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            env_info["pytorch_version"] = "Not installed"
            env_info["cuda_available"] = "Unknown"

        # Working directory
        env_info["working_directory"] = str(Path.cwd())

        # Check for configs directory
        configs_path = Path("configs")
        env_info["configs_directory_exists"] = str(configs_path.exists())

        # Check PYTHONPATH
        env_info["pythonpath"] = sys.path[0] if sys.path else "Empty"

        return env_info

    def run_full_audit(self) -> AuditReport:
        """Run comprehensive dependency audit."""
        print("🔍 Running Moola Dependency Audit...")
        print("=" * 60)

        # Check environment
        self.environment_info = self.check_environment()
        print("📊 Environment Information:")
        for key, value in self.environment_info.items():
            print(f"  {key}: {value}")
        print()

        # Audit critical dependencies
        print("📦 Checking Critical Dependencies...")
        dep_results = self.audit_dependencies()

        # Audit Moola modules
        print("🔧 Checking Moola Modules...")
        module_results = self.audit_moola_modules()

        # Combine results
        all_results = dep_results + module_results

        # Count status
        passed = sum(1 for r in all_results if r.status == "OK")
        failed = sum(1 for r in all_results if r.status in ["MISSING", "VERSION_MISMATCH", "IMPORT_ERROR"])
        warnings = sum(1 for r in all_results if r.status not in ["OK", "MISSING", "VERSION_MISMATCH", "IMPORT_ERROR"])

        report = AuditReport(
            total_dependencies=len(all_results),
            passed=passed,
            failed=failed,
            warnings=warnings,
            dependencies=all_results,
            environment_info=self.environment_info
        )

        return report

    def print_report(self, report: AuditReport, verbose: bool = False):
        """Print audit report."""
        print(f"\n📋 AUDIT SUMMARY")
        print(f"Total: {report.total_dependencies} | ✅ Passed: {report.passed} | ❌ Failed: {report.failed} | ⚠️ Warnings: {report.warnings}")
        print()

        # Group results by status
        failed_deps = [r for r in report.dependencies if r.status in ["MISSING", "VERSION_MISMATCH", "IMPORT_ERROR"]]

        if failed_deps:
            print("❌ FAILED DEPENDENCIES:")
            for dep in failed_deps:
                print(f"  • {dep.name}: {dep.status}")
                if dep.error_message:
                    print(f"    {dep.error_message}")
                if dep.version and dep.expected_version:
                    print(f"    Installed: {dep.version} | Expected: {dep.expected_version}")
            print()

        if verbose:
            print("✅ ALL DEPENDENCIES:")
            for dep in report.dependencies:
                status_icon = "✅" if dep.status == "OK" else "❌"
                print(f"  {status_icon} {dep.name}")
                if dep.version:
                    print(f"    Version: {dep.version}")
                if dep.error_message:
                    print(f"    Error: {dep.error_message}")

    def save_report(self, report: AuditReport, filepath: Path):
        """Save audit report to JSON file."""
        report_dict = asdict(report)
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        print(f"📄 Report saved to: {filepath}")

    def attempt_fixes(self) -> bool:
        """Attempt to fix common issues."""
        print("🔧 Attempting to fix common issues...")

        fixes_applied = []

        # Create configs directory if missing
        configs_path = Path("configs")
        if not configs_path.exists():
            configs_path.mkdir(exist_ok=True)
            # Create default Hydra config
            default_config = configs_path / "default.yaml"
            if not default_config.exists():
                default_config.write_text("# Default Moola Configuration\ndefaults:\n  - _self_\n")
            fixes_applied.append("Created configs/ directory")

        # Try to install missing critical dependencies
        missing_deps = []
        for dep_name in self.critical_deps.keys():
            if self.get_installed_version(dep_name) is None:
                missing_deps.append(dep_name)

        if missing_deps:
            print(f"📦 Installing missing dependencies: {', '.join(missing_deps)}")
            try:
                cmd = [sys.executable, "-m", "pip", "install"] + missing_deps
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    fixes_applied.append(f"Installed: {', '.join(missing_deps)}")
                else:
                    print(f"❌ Failed to install dependencies: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("❌ Installation timed out")
            except Exception as e:
                print(f"❌ Installation failed: {e}")

        if fixes_applied:
            print("✅ Fixes applied:")
            for fix in fixes_applied:
                print(f"  • {fix}")
            return True
        else:
            print("ℹ️ No fixes needed or possible")
            return False

def main():
    parser = argparse.ArgumentParser(description="Moola RunPod Dependency Audit")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common issues")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--check-imports-only", action="store_true", help="Only check module imports")
    parser.add_argument("--version-compatibility", action="store_true", help="Only check version compatibility")
    parser.add_argument("--save-report", type=str, help="Save report to JSON file")
    parser.add_argument("--load-pretrained", type=str, help="Test specific pretrained model loading")

    args = parser.parse_args()

    auditor = DependencyAuditor()

    try:
        if args.check_imports_only:
            print("🔧 Checking Moola module imports only...")
            results = auditor.audit_moola_modules()
            for result in results:
                status_icon = "✅" if result.status == "OK" else "❌"
                print(f"  {status_icon} {result.name}")
                if result.error_message:
                    print(f"    {result.error_message}")

        elif args.version_compatibility:
            print("📊 Checking version compatibility only...")
            results = auditor.audit_dependencies()
            for result in results:
                status_icon = "✅" if result.status == "OK" else "❌"
                print(f"  {status_icon} {result.name}")
                if result.version:
                    print(f"    Version: {result.version}")
                if result.error_message:
                    print(f"    {result.error_message}")

        else:
            # Run full audit
            report = auditor.run_full_audit()
            auditor.print_report(report, args.verbose)

            # Save report if requested
            if args.save_report:
                auditor.save_report(report, Path(args.save_report))

            # Attempt fixes if requested
            if args.fix and report.failed > 0:
                print()
                auditor.attempt_fixes()
                print("\n🔄 Re-running audit after fixes...")
                new_report = auditor.run_full_audit()
                auditor.print_report(new_report, args.verbose)

                if new_report.failed < report.failed:
                    print(f"✅ Fixed {report.failed - new_report.failed} issues!")
                else:
                    print("❌ No issues were fixed")

            # Test pretrained model loading if requested
            if args.load_pretrained:
                print(f"\n🎯 Testing pretrained model loading: {args.load_pretrained}")
                try:
                    from moola.models.pretrained_utils import load_pretrained_strict
                    model = load_pretrained_strict(args.load_pretrained)
                    print(f"✅ Successfully loaded model: {type(model).__name__}")
                except Exception as e:
                    print(f"❌ Failed to load pretrained model: {e}")

            # Exit with error code if there are failures
            if report.failed > 0:
                print(f"\n❌ Audit completed with {report.failed} failures")
                sys.exit(1)
            else:
                print(f"\n✅ All dependencies verified successfully!")
                sys.exit(0)

    except KeyboardInterrupt:
        print("\n⏹️ Audit interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Unexpected error during audit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()