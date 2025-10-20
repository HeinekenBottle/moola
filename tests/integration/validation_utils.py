"""Validation utilities for enhanced architecture testing.

Provides comprehensive validation procedures, benchmarking utilities,
and performance regression detection.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger


class PerformanceBenchmark:
    """Performance benchmarking utilities for architecture validation."""

    def __init__(self):
        self.benchmark_results = []

    def benchmark_model_training(self, model_name: str, X: np.ndarray, y: np.ndarray,
                               n_epochs: int = 5) -> Dict[str, Any]:
        """Benchmark model training performance."""
        from moola.models import get_model

        # Record start time
        start_time = time.time()

        # Create and train model
        model = get_model(model_name, device="cpu", n_epochs=n_epochs)
        model.fit(X, y)

        # Record end time
        end_time = time.time()
        training_time = end_time - start_time

        # Measure memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated(0) / 1024**3  # GB
        else:
            memory_used = 0

        # Count parameters
        param_count = sum(p.numel() for p in model.model.parameters())

        # Generate predictions for latency benchmark
        prediction_start = time.time()
        predictions = model.predict(X[:10])  # First 10 samples
        prediction_time = time.time() - prediction_start

        # Benchmark probability calculation
        proba_start = time.time()
        probabilities = model.predict_proba(X[:10])
        proba_time = time.time() - proba_start

        result = {
            "model_name": model_name,
            "training_time": training_time,
            "memory_used_gb": memory_used,
            "parameter_count": param_count,
            "prediction_time": prediction_time,
            "proba_time": proba_time,
            "samples_per_second": len(X) / training_time,
            "n_epochs": n_epochs,
        }

        self.benchmark_results.append(result)
        return result

    def benchmark_feature_engineering(self, X_ohlc: np.ndarray, iterations: int = 10) -> Dict[str, Any]:
        """Benchmark feature engineering performance."""
        from moola.features import AdvancedFeatureEngineer, FeatureConfig

        engineer = AdvancedFeatureEngineer(FeatureConfig())

        # Benchmark transformation time
        start_time = time.time()
        for _ in range(iterations):
            _ = engineer.transform(X_ohlc)
        total_time = time.time() - start_time

        # Calculate feature generation speed
        samples_per_second = (len(X_ohlc) * iterations) / total_time

        result = {
            "samples_processed": len(X_ohlc),
            "iterations": iterations,
            "total_time": total_time,
            "samples_per_second": samples_per_second,
            "avg_time_per_sample": total_time / (len(X_ohlc) * iterations),
        }

        return result

    def benchmark_data_augmentation(self, X: np.ndarray, iterations: int = 50) -> Dict[str, Any]:
        """Benchmark data augmentation performance."""
        from moola.utils.augmentation import mixup_cutmix
        from moola.utils.temporal_augmentation import TemporalAugmentation

        # Benchmark mixup/cutmix
        start_time = time.time()
        for _ in range(iterations):
            X_aug, _, _, _ = mixup_cutmix(X, np.random.randint(0, 2, len(X)), mixup_alpha=0.4)
        mixup_time = time.time() - start_time

        # Benchmark temporal augmentation
        temporal_aug = TemporalAugmentation()
        start_time = time.time()
        for _ in range(iterations):
            X_temp = temporal_aug.apply_augmentation(X)
        temporal_time = time.time() - start_time

        result = {
            "samples_processed": len(X),
            "iterations": iterations,
            "mixup_time": mixup_time,
            "temporal_time": temporal_time,
            "mixup_samples_per_second": (len(X) * iterations) / mixup_time,
            "temporal_samples_per_second": (len(X) * iterations) / temporal_time,
        }

        return result


class ArchitectureValidator:
    """Comprehensive architecture validation utilities."""

    def __init__(self):
        self.validation_results = {}

    def validate_parameter_count(self, model: torch.nn.Module,
                                expected_range: Tuple[int, int],
                                model_name: str) -> Dict[str, Any]:
        """Validate model parameter count meets expectations."""
        param_count = sum(p.numel() for p in model.parameters())

        result = {
            "model_name": model_name,
            "actual_params": param_count,
            "expected_range": expected_range,
            "within_range": expected_range[0] <= param_count <= expected_range[1],
            "param_efficiency": param_count / np.prod([p.shape for p in model.parameters()]),
        }

        self.validation_results[f"param_count_{model_name}"] = result
        return result

    def validate_gradient_flow(self, model: torch.nn.Module,
                             X: torch.Tensor, y: torch.Tensor,
                             threshold: float = 1e-8) -> Dict[str, Any]:
        """Validate gradient flow through model architecture."""
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Forward pass
        model.train()
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)

        # Backward pass
        loss.backward()

        # Analyze gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradients[name] = grad_norm

        # Check for vanishing/exploding gradients
        min_grad = min(gradients.values()) if gradients else 0
        max_grad = max(gradients.values()) if gradients else 0

        result = {
            "min_gradient_norm": min_grad,
            "max_gradient_norm": max_grad,
            "vanishing_gradients": min_grad < threshold,
            "exploding_gradients": max_grad > 1.0,
            "healthy_gradient_flow": not (min_grad < threshold or max_grad > 1.0),
            "gradients": gradients,
        }

        self.validation_results["gradient_flow"] = result
        return result

    def validate_model_consistency(self, model1: torch.nn.Module,
                                 model2: torch.nn.Module,
                                 X_test: torch.Tensor) -> Dict[str, Any]:
        """Validate model consistency between two instances."""
        model1.eval()
        model2.eval()

        with torch.no_grad():
            outputs1 = model1(X_test)
            outputs2 = model2(X_test)

        # Compare outputs
        output_diff = torch.abs(outputs1 - outputs2).mean().item()

        result = {
            "output_difference": output_diff,
            "consistency_threshold": 1e-5,
            "models_consistent": output_diff < 1e-5,
        }

        self.validation_results["model_consistency"] = result
        return result

    def validate_data_compatibility(self, model: torch.nn.Module,
                                   test_inputs: List[np.ndarray]) -> Dict[str, Any]:
        """Validate model compatibility with different data configurations."""
        results = []

        for i, X in enumerate(test_inputs):
            try:
                X_tensor = torch.FloatTensor(X)
                with torch.no_grad():
                    outputs = model(X_tensor)

                result = {
                    "test_case": i,
                    "input_shape": X.shape,
                    "output_shape": outputs.shape,
                    "success": True,
                    "no_nan": not torch.isnan(outputs).any(),
                    "no_inf": not torch.isinf(outputs).any(),
                }

            except Exception as e:
                result = {
                    "test_case": i,
                    "input_shape": X.shape,
                    "success": False,
                    "error": str(e),
                }

            results.append(result)

        all_success = all(r["success"] for r in results)
        any_nan = any(not r["no_nan"] for r in results if r["success"])
        any_inf = any(not r["no_inf"] for r in results if r["success"])

        validation_result = {
            "all_cases_successful": all_success,
            "contains_nan": any_nan,
            "contains_inf": any_inf,
            "details": results,
        }

        self.validation_results["data_compatibility"] = validation_result
        return validation_result


class PerformanceRegressionDetector:
    """Detect performance regressions in enhanced architecture."""

    def __init__(self, baseline_file: Optional[Path] = None):
        self.baseline_file = baseline_file
        self.baseline_data = self._load_baseline()

    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline performance metrics."""
        if self.baseline_file and self.baseline_file.exists():
            with open(self.baseline_file, "r") as f:
                return json.load(f)
        return None

    def detect_regressions(self, current_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect performance regressions compared to baseline."""
        if not self.baseline_data:
            return {"regressions": [], "warnings": []}

        regressions = []
        warnings = []

        # Compare training time
        if "training_time" in current_results and "training_time" in self.baseline_data:
            current_time = current_results["training_time"]
            baseline_time = self.baseline_data["training_time"]
            time_change = (current_time - baseline_time) / baseline_time

            if time_change > 0.2:  # 20% increase
                regressions.append({
                    "metric": "training_time",
                    "baseline": baseline_time,
                    "current": current_time,
                    "change_percent": time_change * 100,
                    "severity": "high" if time_change > 0.5 else "medium",
                })

        # Compare memory usage
        if "memory_used_gb" in current_results and "memory_used_gb" in self.baseline_data:
            current_memory = current_results["memory_used_gb"]
            baseline_memory = self.baseline_data["memory_used_gb"]
            memory_change = (current_memory - baseline_memory) / baseline_memory

            if memory_change > 0.3:  # 30% increase
                warnings.append({
                    "metric": "memory_usage",
                    "baseline": baseline_memory,
                    "current": current_memory,
                    "change_percent": memory_change * 100,
                    "severity": "medium" if memory_change > 0.5 else "low",
                })

        # Compare accuracy
        if "accuracy" in current_results and "accuracy" in self.baseline_data:
            current_acc = current_results["accuracy"]
            baseline_acc = self.baseline_data["accuracy"]
            acc_change = current_acc - baseline_acc

            if acc_change < -0.05:  # 5% decrease
                regressions.append({
                    "metric": "accuracy",
                    "baseline": baseline_acc,
                    "current": current_acc,
                    "change_percent": acc_change * 100,
                    "severity": "high",
                })

        return {
            "regressions": regressions,
            "warnings": warnings,
            "has_regressions": len(regressions) > 0,
            "has_warnings": len(warnings) > 0,
        }

    def save_current_baseline(self, results: Dict[str, Any], path: Path):
        """Save current results as new baseline."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)


class SmallDatasetValidator:
    """Validate model performance with small datasets (production constraints)."""

    def validate_small_dataset_training(self, model_name: str, n_samples: int = 98) -> Dict[str, Any]:
        """Validate training with small dataset size."""
        from moola.models import get_model

        # Generate small dataset
        np.random.seed(42)
        X = np.random.randn(n_samples, 105, 4)
        y = np.random.choice(["class_A", "class_B"], n_samples)

        # Train model
        model = get_model(model_name, device="cpu", n_epochs=10)
        model.fit(X, y)

        # Evaluate performance
        predictions = model.predict(X)
        accuracy = (predictions == y).mean()

        # Check for overfitting
        train_predictions = model.predict(X[:n_samples//2])
        val_predictions = model.predict(X[n_samples//2:])
        train_acc = (train_predictions == y[:n_samples//2]).mean()
        val_acc = (val_predictions == y[n_samples//2:])

        overfitting_ratio = train_acc / val_acc if val_acc > 0 else float('inf')

        result = {
            "n_samples": n_samples,
            "training_accuracy": accuracy,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "overfitting_ratio": overfitting_ratio,
            "overfitting_detected": overfitting_ratio > 1.5,
            "small_dataset_compatible": n_samples <= 100 and accuracy > 0.6,
        }

        return result

    def validate_class_balance_robustness(self, model_name: str) -> Dict[str, Any]:
        """Validate robustness to class imbalance."""
        imbalance_ratios = [0.1, 0.2, 0.3]  # Minority class ratios
        results = []

        for ratio in imbalance_ratios:
            n_samples = 100
            n_minority = int(n_samples * ratio)
            n_majority = n_samples - n_minority

            # Create imbalanced dataset
            X = np.random.randn(n_samples, 105, 4)
            y = ["class_B"] * n_minority + ["class_A"] * n_majority

            # Train and evaluate
            from moola.models import get_model
            model = get_model(model_name, device="cpu", n_epochs=5)
            model.fit(X, y)

            predictions = model.predict(X)
            accuracy = (predictions == y).mean()

            # Check class prediction distribution
            class_predictions = pd.Series(predictions).value_counts()
            pred_ratio = class_predictions.min() / class_predictions.max()

            results.append({
                "imbalance_ratio": ratio,
                "accuracy": accuracy,
                "prediction_balance": pred_ratio,
                "robust": accuracy > 0.5 and pred_ratio > 0.3,
            })

        all_robust = all(r["robust"] for r in results)
        avg_accuracy = np.mean([r["accuracy"] for r in results])

        return {
            "all_imbalanced_scenarios_robust": all_robust,
            "average_accuracy": avg_accuracy,
            "details": results,
        }


class ValidationReporter:
    """Generate comprehensive validation reports."""

    @staticmethod
    def generate_comprehensive_report(validation_results: Dict[str, Any],
                                   benchmark_results: Dict[str, Any],
                                   regression_results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report."""
        report = []

        report.append("# Enhanced Architecture Validation Report")
        report.append("=" * 50)

        # Summary
        report.append("\n## Summary")
        report.append(f"Total Validation Tests: {len(validation_results)}")
        report.append(f"Benchmark Results: {len(benchmark_results)}")

        # Performance Analysis
        report.append("\n## Performance Analysis")
        if "training_performance" in benchmark_results:
            perf = benchmark_results["training_performance"]
            report.append(f"Training Time: {perf['training_time']:.2f}s")
            report.append(f"Memory Usage: {perf['memory_used_gb']:.2f}GB")
            report.append(f"Parameters: {perf['parameter_count']:,}")

        # Validation Results
        report.append("\n## Validation Results")
        for test_name, result in validation_results.items():
            if isinstance(result, dict):
                status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
                report.append(f"{test_name}: {status}")
                if "details" in result:
                    report.append(f"  Details: {result['details']}")

        # Regression Detection
        report.append("\n## Regression Detection")
        if regression_results["has_regressions"]:
            report.append("⚠️  REGRESSIONS DETECTED:")
            for regression in regression_results["regressions"]:
                severity = regression["severity"].upper()
                report.append(f"  {severity}: {regression['metric']} - {regression['change_percent']:.1f}% change")
        else:
            report.append("✅ No performance regressions detected")

        # Recommendations
        report.append("\n## Recommendations")
        if regression_results["has_regressions"]:
            report.append("- Address detected regressions before deployment")
        if regression_results["has_warnings"]:
            report.append("- Monitor warning metrics in production")

        return "\n".join(report)

    @staticmethod
    def save_validation_report(report: str, path: Path):
        """Save validation report to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(report)


def main():
    """Main validation workflow."""
    # Initialize validation components
    benchmark = PerformanceBenchmark()
    validator = ArchitectureValidator()
    detector = PerformanceRegressionDetector()
    small_dataset_validator = SmallDatasetValidator()

    # Generate test data
    np.random.seed(42)
    X = np.random.randn(100, 105, 4)
    y = np.random.choice(["class_A", "class_B"], 100)

    # Run benchmarks
    training_results = benchmark.benchmark_model_training("simple_lstm", X, y)
    feature_results = benchmark.benchmark_feature_engineering(X[:10])
    augmentation_results = benchmark.benchmark_data_augmentation(X[:10])

    # Run validations
    from moola.models import get_model
    model = get_model("simple_lstm", device="cpu")

    param_validation = validator.validate_parameter_count(
        model.model, (40000, 60000), "SimpleLSTM"
    )

    # Create test tensors for gradient flow
    X_tensor = torch.FloatTensor(X[:10])
    y_tensor = torch.LongTensor([0] * 5 + [1] * 5)
    gradient_validation = validator.validate_gradient_flow(model.model, X_tensor, y_tensor)

    # Test small dataset validation
    small_dataset_validation = small_dataset_validator.validate_small_dataset_training("simple_lstm")

    # Combine results
    all_results = {
        "benchmarking": {
            "training": training_results,
            "feature_engineering": feature_results,
            "data_augmentation": augmentation_results,
        },
        "validation": {
            "parameter_count": param_validation,
            "gradient_flow": gradient_validation,
            "small_dataset": small_dataset_validation,
        }
    }

    # Check for regressions
    regression_results = detector.detect_regressions(training_results)

    # Generate report
    report = ValidationReporter.generate_comprehensive_report(
        all_results["validation"],
        all_results["benchmarking"],
        regression_results
    )

    # Save results
    results_path = Path("validation_report.json")
    ValidationReporter.save_validation_report(report, Path("validation_report.md"))

    print("✅ Validation completed successfully")
    print(f"📄 Report saved to: validation_report.md")

    return {
        "results": all_results,
        "regressions": regression_results,
        "success": not regression_results["has_regressions"],
    }


if __name__ == "__main__":
    main()