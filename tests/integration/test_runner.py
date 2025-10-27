"""Automated test runner for integration tests.

Provides comprehensive test execution with parallel processing,
progress tracking, and detailed reporting.
"""

import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional


class IntegrationTestRunner:
    """Automated runner for integration tests."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        self.config = config or self._default_config()
        self.results = []
        self.start_time = None
        self.end_time = None

    def _default_config(self) -> dict[str, Any]:
        """Default test configuration."""
        return {
            "max_workers": min(multiprocessing.cpu_count(), 4),
            "timeout_seconds": 300,
            "parallel_execution": True,
            "verbose": True,
            "generate_html_report": True,
            "save_results": True,
            "test_patterns": [
                "test_data_pipeline.py",
                "test_model_architecture.py",
                "test_pretraining_integration.py",
                "test_backward_compatibility.py",
            ],
        }

    def run_all_tests(self) -> dict[str, Any]:
        """Run all integration tests."""
        print("🚀 Starting comprehensive integration test suite...")
        self.start_time = time.time()

        # Run test suites
        test_suites = [
            self.run_data_pipeline_tests,
            self.run_model_architecture_tests,
            self.run_pretraining_tests,
            self.run_backward_compatibility_tests,
            self.run_validation_benchmarks,
        ]

        if self.config["parallel_execution"]:
            results = self._run_tests_parallel(test_suites)
        else:
            results = self._run_tests_sequential(test_suites)

        self.end_time = time.time()

        # Generate comprehensive report
        report = self._generate_final_report(results)

        # Save results if configured
        if self.config["save_results"]:
            self._save_results(results, report)

        return report

    def _run_tests_parallel(self, test_suites: list) -> dict[str, Any]:
        """Run tests in parallel using process pool."""
        results = {}

        with ProcessPoolExecutor(max_workers=self.config["max_workers"]) as executor:
            # Submit all test tasks
            future_to_suite = {executor.submit(suite): suite.__name__ for suite in test_suites}

            # Collect results as they complete
            for future in as_completed(future_to_suite):
                suite_name = future_to_suite[future]
                try:
                    result = future.result(timeout=self.config["timeout_seconds"])
                    results[suite_name] = result
                    print(f"✅ {suite_name} completed successfully")
                except Exception as e:
                    error_result = {
                        "suite": suite_name,
                        "success": False,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    }
                    results[suite_name] = error_result
                    print(f"❌ {suite_name} failed: {e}")

        return results

    def _run_tests_sequential(self, test_suites: list) -> dict[str, Any]:
        """Run tests sequentially."""
        results = {}

        for suite in test_suites:
            suite_name = suite.__name__
            try:
                result = suite()
                results[suite_name] = result
                print(f"✅ {suite_name} completed successfully")
            except Exception as e:
                error_result = {
                    "suite": suite_name,
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__,
                }
                results[suite_name] = error_result
                print(f"❌ {suite_name} failed: {e}")

        return results

    def run_data_pipeline_tests(self) -> dict[str, Any]:
        """Run data pipeline integration tests."""
        print("\n🔧 Running data pipeline integration tests...")

        # Import and run pytest
        import subprocess
        import sys

        result = {
            "suite": "data_pipeline",
            "tests": [],
            "success": True,
            "error": None,
        }

        try:
            # Run pytest for data pipeline tests
            test_file = Path(__file__).parent / "test_data_pipeline.py"
            cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]

            process = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config["timeout_seconds"]
            )

            # Parse results
            result["exit_code"] = process.returncode
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr

            if process.returncode == 0:
                result["success"] = True
            else:
                result["success"] = False

        except subprocess.TimeoutExpired:
            result["success"] = False
            result["error"] = "Test timeout"
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result

    def run_model_architecture_tests(self) -> dict[str, Any]:
        """Run enhanced SimpleLSTM architecture tests."""
        print("\n🏗️  Running enhanced SimpleLSTM architecture tests...")

        result = {
            "suite": "model_architecture",
            "tests": [],
            "success": True,
            "error": None,
        }

        try:
            import subprocess
            import sys

            test_file = Path(__file__).parent / "test_model_architecture.py"
            cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]

            process = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config["timeout_seconds"]
            )

            result["exit_code"] = process.returncode
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr

            if process.returncode == 0:
                result["success"] = True
            else:
                result["success"] = False

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result

    def run_pretraining_tests(self) -> dict[str, Any]:
        """Run pre-training integration tests."""
        print("\n🎓 Running pre-training integration tests...")

        result = {
            "suite": "pretraining",
            "tests": [],
            "success": True,
            "error": None,
        }

        try:
            import subprocess
            import sys

            test_file = Path(__file__).parent / "test_pretraining_integration.py"
            cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]

            process = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config["timeout_seconds"]
            )

            result["exit_code"] = process.returncode
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr

            if process.returncode == 0:
                result["success"] = True
            else:
                result["success"] = False

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result

    def run_backward_compatibility_tests(self) -> dict[str, Any]:
        """Run backward compatibility tests."""
        print("\n🔄 Running backward compatibility tests...")

        result = {
            "suite": "backward_compatibility",
            "tests": [],
            "success": True,
            "error": None,
        }

        try:
            import subprocess
            import sys

            test_file = Path(__file__).parent / "test_backward_compatibility.py"
            cmd = [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"]

            process = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self.config["timeout_seconds"]
            )

            result["exit_code"] = process.returncode
            result["stdout"] = process.stdout
            result["stderr"] = process.stderr

            if process.returncode == 0:
                result["success"] = True
            else:
                result["success"] = False

        except Exception as e:
            result["success"] = False
            result["error"] = str(e)

        return result

    def run_validation_benchmarks(self) -> dict[str, Any]:
        """Run validation benchmarks."""
        print("\n📊 Running validation benchmarks...")

        try:
            from .validation_utils import main as validation_main

            validation_results = validation_main()

            result = {
                "suite": "validation_benchmarks",
                "success": validation_results["success"],
                "results": validation_results["results"],
                "regressions": validation_results["regressions"],
            }

        except Exception as e:
            result = {
                "suite": "validation_benchmarks",
                "success": False,
                "error": str(e),
            }

        return result

    def _generate_final_report(self, suite_results: dict[str, Any]) -> dict[str, Any]:
        """Generate final test report."""
        total_time = self.end_time - self.start_time

        # Calculate overall success rate
        total_suites = len(suite_results)
        successful_suites = sum(
            1 for result in suite_results.values() if result.get("success", False)
        )
        success_rate = successful_suites / total_suites if total_suites > 0 else 0

        # Generate summary
        report = {
            "summary": {
                "total_suites": total_suites,
                "successful_suites": successful_suites,
                "success_rate": success_rate,
                "total_time_seconds": total_time,
                "start_time": self.start_time,
                "end_time": self.end_time,
            },
            "suite_results": suite_results,
            "test_timestamp": time.time(),
        }

        # Add detailed analysis
        report["analysis"] = self._analyze_results(suite_results)

        # Print summary
        print("\n📊 Integration Test Summary:")
        print(f"   Total Suites: {total_suites}")
        print(f"   Successful: {successful_suites}")
        print(f"   Success Rate: {success_rate:.2%}")
        print(f"   Total Time: {total_time:.2f}s")

        return report

    def _analyze_results(self, suite_results: dict[str, Any]) -> dict[str, Any]:
        """Analyze test results and provide insights."""
        analysis = {
            "strong_areas": [],
            "improvement_areas": [],
            "critical_failures": [],
            "warnings": [],
        }

        for suite_name, result in suite_results.items():
            if result.get("success", False):
                analysis["strong_areas"].append(suite_name)
            else:
                if result.get("error"):
                    error_msg = result["error"].lower()
                    if "critical" in error_msg or "fail" in error_msg:
                        analysis["critical_failures"].append(
                            {
                                "suite": suite_name,
                                "error": result["error"],
                            }
                        )
                    else:
                        analysis["warnings"].append(
                            {
                                "suite": suite_name,
                                "error": result["error"],
                            }
                        )
                else:
                    analysis["improvement_areas"].append(suite_name)

        return analysis

    def _save_results(self, results: dict[str, Any], report: dict[str, Any]):
        """Save test results to files."""
        # Save JSON results
        json_path = Path("integration_test_results.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)

        # Generate HTML report
        if self.config["generate_html_report"]:
            self._generate_html_report(report)

        # Save validation report
        if "validation_benchmarks" in results:
            validation_results = results["validation_benchmarks"]
            if "results" in validation_results:
                val_report_path = Path("validation_report.json")
                with open(val_report_path, "w") as f:
                    json.dump(validation_results["results"], f, indent=2)

    def _generate_html_report(self, report: dict[str, Any]):
        """Generate HTML test report."""
        html_content = self._create_html_template(report)

        html_path = Path("integration_test_report.html")
        with open(html_path, "w") as f:
            f.write(html_content)

    def _create_html_template(self, report: dict[str, Any]) -> str:
        """Create HTML template for test report."""
        summary = report["summary"]
        analysis = report["analysis"]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        .critical {{ color: darkred; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; }}
        .metric {{ display: inline-block; margin: 10px; }}
    </style>
</head>
<body>
    <h1>Integration Test Report</h1>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">Total Suites: {summary['total_suites']}</div>
        <div class="metric">Success Rate: {summary['success_rate']:.2%}</div>
        <div class="metric">Time: {summary['total_time_seconds']:.2f}s</div>
        <div class="metric success">✅ {summary['successful_suites']} Passed</div>
        <div class="metric failure">❌ {summary['total_suites'] - summary['successful_suites']} Failed</div>
    </div>

    <div class="section">
        <h2>Analysis</h2>
        <h3>Strong Areas</h3>
        <ul>
            {''.join(f'<li class="success">✅ {area}</li>' for area in analysis['strong_areas'])}
        </ul>

        <h3>Improvement Areas</h3>
        <ul>
            {''.join(f'<li class="warning">⚠️ {area}</li>' for area in analysis['improvement_areas'])}
        </ul>

        <h3>Critical Failures</h3>
        <ul>
            {''.join(f'<li class="critical">❌ {failure["suite"]}: {failure["error"]}</li>'
                     for failure in analysis['critical_failures'])}
        </ul>
    </div>

    <div class="section">
        <h2>Detailed Results</h2>
        <pre>{json.dumps(report, indent=2)}</pre>
    </div>
</body>
</html>
        """

        return html


def main():
    """Main test runner entry point."""
    # Test runner configuration
    config = {
        "max_workers": 4,
        "timeout_seconds": 600,
        "parallel_execution": True,
        "verbose": True,
        "generate_html_report": True,
        "save_results": True,
    }

    # Create and run test runner
    runner = IntegrationTestRunner(config)
    results = runner.run_all_tests()

    # Determine exit code
    if results["summary"]["success_rate"] >= 0.8:
        print("\n✅ Integration tests PASSED (80% success rate)")
        exit_code = 0
    else:
        print("\n❌ Integration tests FAILED (below 80% success rate)")
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
