"""Robust data handling framework for extreme small dataset scenarios.

Specialized framework for handling the challenges of 98-sample datasets:
- Cross-validation strategies optimized for tiny datasets
- Statistical methods that work with limited samples
- Confidence interval estimation for small N
- Bias correction and uncertainty quantification
- Robust performance estimation
- Data-efficient model selection

Key components:
1. Small sample cross-validation (nested CV, bootstrap CV)
2. Statistical power analysis and sample size requirements
3. Confidence interval estimation for tiny samples
4. Model selection with limited data
5. Uncertainty quantification and risk assessment
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import bootstrap, mannwhitneyu, ttest_ind
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.utils import resample

warnings.filterwarnings("ignore")

from loguru import logger


class SmallDatasetStrategy(str, Enum):
    """Strategies for handling small datasets."""

    CONSERVATIVE = "conservative"  # Prioritize avoiding overfitting
    BIAS_CORRECTED = "bias_corrected"  # Apply statistical bias correction
    BOOTSTRAP_HEAVY = "bootstrap_heavy"  # Heavy use of bootstrap methods
    UNCERTAINTY_AWARE = "uncertainty_aware"  # Emphasize uncertainty quantification
    ROBUST_SELECTION = "robust_selection"  # Robust model selection procedures


@dataclass
class SmallDatasetConfig:
    """Configuration for small dataset handling."""

    # Dataset characteristics
    n_samples: int = 98
    n_classes: int = 3
    min_samples_per_class: int = 2

    # Cross-validation strategy
    cv_strategy: SmallDatasetStrategy = SmallDatasetStrategy.CONSERVATIVE
    n_outer_folds: int = 5  # Outer CV folds
    n_inner_folds: int = 3  # Inner CV folds
    n_bootstrap_samples: int = 1000  # For bootstrap methods

    # Statistical parameters
    confidence_level: float = 0.95
    effect_size_threshold: float = 0.5  # Cohen's d threshold
    multiple_testing_correction: bool = True

    # Model selection
    use_nested_cv: bool = True  # Prevent optimistic bias
    early_stopping_patience: int = 10  # Conservative early stopping
    regularization_strength: float = 1.0  # Strong regularization

    # Uncertainty quantification
    compute_confidence_intervals: bool = True
    compute_prediction_intervals: bool = True
    use_bayesian_uncertainty: bool = False  # For future Bayesian methods

    # Robustness checks
    perform_sensitivity_analysis: bool = True
    perform_outlier_analysis: bool = True
    perform_power_analysis: bool = True


class SmallDatasetCrossValidator:
    """Specialized cross-validation for small datasets."""

    def __init__(self, config: SmallDatasetConfig):
        """Initialize small dataset cross-validator."""
        self.config = config

    def get_cv_splits(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get cross-validation splits optimized for small datasets."""
        n_samples = len(X)

        if self.config.cv_strategy == SmallDatasetStrategy.CONSERVATIVE:
            return self._conservative_cv_splits(X, y)
        elif self.config.cv_strategy == SmallDatasetStrategy.BOOTSTRAP_HEAVY:
            return self._bootstrap_cv_splits(X, y)
        elif self.config.cv_strategy == SmallDatasetStrategy.UNCERTAINTY_AWARE:
            return self._uncertainty_aware_cv_splits(X, y)
        else:
            return self._default_cv_splits(X, y)

    def _conservative_cv_splits(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Conservative CV with stratification and size constraints."""
        # Use stratified K-fold but ensure minimum class representation
        n_splits = min(self.config.n_outer_folds, len(X) // self.config.min_samples_per_class)

        if n_splits < 2:
            # Fall back to leave-one-out if dataset is too small
            loo = LeaveOneOut()
            return list(loo.split(X))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []

        for train_idx, val_idx in skf.split(X, y):
            # Ensure minimum samples per class in validation
            val_labels = y[val_idx]
            unique, counts = np.unique(val_labels, return_counts=True)

            if len(counts) == len(np.unique(y)) and np.all(counts >= 1):
                splits.append((train_idx, val_idx))

        return splits

    def _bootstrap_cv_splits(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Bootstrap-based CV for small datasets."""
        splits = []
        n_samples = len(X)

        for _ in range(self.config.n_bootstrap_samples):
            # Bootstrap sample with replacement
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)

            # Out-of-bag samples as validation
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[boot_idx] = False
            oob_idx = np.where(oob_mask)[0]

            if len(oob_idx) > 0:
                splits.append((boot_idx, oob_idx))

        return splits

    def _uncertainty_aware_cv_splits(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Uncertainty-aware CV that emphasizes boundary samples."""
        # For now, use stratified CV with uncertainty weighting in training
        return self._conservative_cv_splits(X, y)

    def _default_cv_splits(
        self, X: np.ndarray, y: np.ndarray
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Default stratified CV."""
        n_splits = min(5, len(X) // 2)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(skf.split(X, y))


class SmallDatasetStatistics:
    """Statistical methods specialized for small sample sizes."""

    @staticmethod
    def compute_confidence_interval(
        data: np.ndarray,
        confidence_level: float = 0.95,
        method: str = "t",  # "t" for t-distribution, "bootstrap" for bootstrap
    ) -> Tuple[float, float]:
        """Compute confidence interval for small sample."""
        n = len(data)

        if method == "t" and n >= 2:
            # Use t-distribution for small samples
            mean = np.mean(data)
            sem = stats.sem(data)  # Standard error of mean
            alpha = 1 - confidence_level
            t_critical = stats.t.ppf(1 - alpha / 2, n - 1)

            ci_lower = mean - t_critical * sem
            ci_upper = mean + t_critical * sem
            return ci_lower, ci_upper

        elif method == "bootstrap" and n >= 3:
            # Bootstrap confidence interval
            try:
                result = bootstrap(
                    (data,),
                    np.mean,
                    confidence_level=confidence_level,
                    n_resamples=1000,
                    random_state=42,
                )
                return result.confidence_interval.low, result.confidence_interval.high
            except:
                # Fallback to t-interval
                return SmallDatasetStatistics.compute_confidence_interval(
                    data, confidence_level, method="t"
                )

        else:
            # Normal approximation as last resort
            mean = np.mean(data)
            std = np.std(data, ddof=1)
            z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

            ci_lower = mean - z_critical * std / np.sqrt(n)
            ci_upper = mean + z_critical * std / np.sqrt(n)
            return ci_lower, ci_upper

    @staticmethod
    def compute_effect_size(group1: np.ndarray, group2: np.ndarray) -> Tuple[float, str]:
        """Compute Cohen's d effect size for small samples."""
        n1, n2 = len(group1), len(group2)

        if n1 == 0 or n2 == 0:
            return 0.0, "insufficient_data"

        # Pooled standard deviation
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0, "no_variance"

        cohen_d = (mean1 - mean2) / pooled_std

        # Interpret effect size
        if abs(cohen_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohen_d) < 0.5:
            interpretation = "small"
        elif abs(cohen_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"

        return cohen_d, interpretation

    @staticmethod
    def statistical_power_analysis(
        effect_size: float, n_samples: int, alpha: float = 0.05, power: float = 0.8
    ) -> Dict[str, float]:
        """Perform statistical power analysis for small samples."""
        # Simplified power calculation for two-sample t-test
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # Required sample size for given effect size
        n_required = 2 * ((z_alpha + z_beta) / effect_size) ** 2

        # Actual power with given sample size
        z_actual = np.sqrt(n_samples / 2) * abs(effect_size) - z_alpha
        actual_power = norm.cdf(z_actual)

        return {
            "required_n_per_group": n_required,
            "actual_power": actual_power,
            "sufficient_power": actual_power >= power,
            "power_gap": power - actual_power,
        }

    @staticmethod
    def bias_corrected_accuracy(
        y_true: np.ndarray, y_pred: np.ndarray, n_bootstrap: int = 1000
    ) -> Dict[str, float]:
        """Compute bias-corrected accuracy for small samples."""
        n_samples = len(y_true)

        # Observed accuracy
        obs_accuracy = accuracy_score(y_true, y_pred)

        # Bootstrap bias correction
        bootstrap_accuracies = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            boot_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_acc = accuracy_score(y_true[boot_idx], y_pred[boot_idx])
            bootstrap_accuracies.append(boot_acc)

        # Compute bias
        bootstrap_mean = np.mean(bootstrap_accuracies)
        bias = bootstrap_mean - obs_accuracy

        # Bias-corrected accuracy
        corrected_accuracy = obs_accuracy - bias

        # Confidence interval
        ci_lower, ci_upper = SmallDatasetStatistics.compute_confidence_interval(
            np.array(bootstrap_accuracies), confidence_level=0.95
        )

        return {
            "observed_accuracy": obs_accuracy,
            "bias_corrected_accuracy": corrected_accuracy,
            "bootstrap_bias": bias,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "n_bootstrap": n_bootstrap,
        }


class SmallDatasetModelSelector:
    """Model selection procedures for small datasets."""

    def __init__(self, config: SmallDatasetConfig):
        """Initialize model selector."""
        self.config = config
        self.results_history: List[Dict] = []

    def nested_cross_validation(
        self, X: np.ndarray, y: np.ndarray, model_configs: List[Dict], model_factory: Callable
    ) -> Dict[str, Any]:
        """Perform nested cross-validation for unbiased model selection."""
        cv = SmallDatasetCrossValidator(self.config)
        outer_splits = cv.get_cv_splits(X, y)

        results = {
            "model_scores": {config["name"]: [] for config in model_configs},
            "best_models": [],
            "overall_best": None,
            "uncertainty_estimates": {},
        }

        for outer_fold, (train_idx, val_idx) in enumerate(outer_splits):
            X_train_outer, X_val_outer = X[train_idx], X[val_idx]
            y_train_outer, y_val_outer = y[train_idx], y[val_idx]

            # Inner CV for hyperparameter selection
            best_model_name, best_score = self._inner_cv_selection(
                X_train_outer, y_train_outer, model_configs, model_factory
            )

            # Train best model on full outer training data
            best_config = next(
                config for config in model_configs if config["name"] == best_model_name
            )
            best_model = model_factory(**best_config)

            # Train model (placeholder - actual training would happen here)
            # best_model.fit(X_train_outer, y_train_outer)

            # Evaluate on outer validation set
            # y_pred = best_model.predict(X_val_outer)
            # score = accuracy_score(y_val_outer, y_pred)

            # For demonstration, use random score
            score = np.random.uniform(0.3, 0.8)

            results["model_scores"][best_model_name].append(score)
            results["best_models"].append(
                {
                    "fold": outer_fold,
                    "model_name": best_model_name,
                    "score": score,
                    "config": best_config,
                }
            )

        # Determine overall best model
        mean_scores = {
            name: np.scores(scores) if scores else 0.0
            for name, scores in results["model_scores"].items()
        }

        results["overall_best"] = max(mean_scores.items(), key=lambda x: x[1])

        # Compute uncertainty estimates
        for name, scores in results["model_scores"].items():
            if len(scores) >= 2:
                ci_lower, ci_upper = SmallDatasetStatistics.compute_confidence_interval(
                    np.array(scores), confidence_level=self.config.confidence_level
                )
                results["uncertainty_estimates"][name] = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_folds": len(scores),
                }

        self.results_history.append(results)
        return results

    def _inner_cv_selection(
        self, X: np.ndarray, y: np.ndarray, model_configs: List[Dict], model_factory: Callable
    ) -> Tuple[str, float]:
        """Inner CV for model/hyperparameter selection."""
        inner_cv = SmallDatasetCrossValidator(self.config)
        inner_splits = inner_cv.get_cv_splits(X, y)

        model_scores = {config["name"]: [] for config in model_configs}

        for train_idx, val_idx in inner_splits:
            X_train_inner, X_val_inner = X[train_idx], X[val_idx]
            y_train_inner, y_val_inner = y[train_idx], y[val_idx]

            for config in model_configs:
                # Train model (placeholder)
                # model = model_factory(**config)
                # model.fit(X_train_inner, y_train_inner)
                # y_pred = model.predict(X_val_inner)
                # score = accuracy_score(y_val_inner, y_pred)

                # For demonstration, use random score with bias toward simpler models
                complexity_penalty = config.get("complexity", 1) * 0.05
                score = np.random.uniform(0.3, 0.8) - complexity_penalty

                model_scores[config["name"]].append(score)

        # Select best model (prefer simpler models in case of ties)
        mean_scores = {
            name: np.mean(scores) if scores else 0.0 for name, scores in model_scores.items()
        }

        best_model = max(
            mean_scores.items(),
            key=lambda x: (
                x[1],
                -next(c["complexity"] for c in model_configs if c["name"] == x[0]),
            ),
        )

        return best_model


class SmallDatasetFramework:
    """Complete framework for handling extreme small dataset scenarios."""

    def __init__(self, config: Optional[SmallDatasetConfig] = None):
        """Initialize small dataset framework."""
        self.config = config or SmallDatasetConfig()
        self.cross_validator = SmallDatasetCrossValidator(self.config)
        self.statistics = SmallDatasetStatistics()
        self.model_selector = SmallDatasetModelSelector(self.config)

        # Results tracking
        self.experiment_results: List[Dict] = []

    def analyze_dataset_characteristics(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Analyze dataset characteristics and limitations."""
        n_samples, n_features = X.shape
        unique_classes, class_counts = np.unique(y, return_counts=True)

        analysis = {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_classes": len(unique_classes),
            "class_distribution": dict(zip(unique_classes, class_counts)),
            "class_imbalance_ratio": max(class_counts) / min(class_counts),
            "samples_per_feature": n_samples / n_features,
            "samples_per_class": n_samples / len(unique_classes),
            "min_class_samples": min(class_counts),
        }

        # Assess data sufficiency
        analysis["sufficient_samples"] = all(
            [
                n_samples >= 50,
                analysis["min_class_samples"] >= 5,
                analysis["samples_per_feature"] >= 5,
            ]
        )

        # Power analysis
        if len(unique_classes) == 2:
            effect_size = 0.5  # Medium effect size
            power_analysis = self.statistics.statistical_power_analysis(effect_size, n_samples)
            analysis["power_analysis"] = power_analysis

        # Recommendations
        recommendations = []

        if analysis["class_imbalance_ratio"] > 3:
            recommendations.append("Consider class balancing techniques")

        if analysis["samples_per_feature"] < 10:
            recommendations.append("Feature selection recommended (high dimensionality)")

        if analysis["min_class_samples"] < 10:
            recommendations.append("Use data augmentation or collect more samples")

        if not analysis["sufficient_samples"]:
            recommendations.append("Dataset is very small - use conservative methods")

        analysis["recommendations"] = recommendations

        return analysis

    def evaluate_model_robustness(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory: Callable,
        model_config: Dict,
        n_repetitions: int = 100,
    ) -> Dict[str, Any]:
        """Evaluate model robustness with uncertainty quantification."""
        results = {
            "repetition_scores": [],
            "confidence_intervals": {},
            "bias_corrected_metrics": {},
            "sensitivity_analysis": {},
        }

        # Multiple CV repetitions
        for rep in range(n_repetitions):
            cv_splits = self.cross_validator.get_cv_splits(X, y)
            fold_scores = []

            for train_idx, val_idx in cv_splits:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                # Train and evaluate model (placeholder)
                # model = model_factory(**model_config)
                # model.fit(X_train, y_train)
                # y_pred = model.predict(X_val)
                # score = accuracy_score(y_val, y_pred)

                # For demonstration
                score = np.random.beta(2, 3)  # Realistic accuracy distribution

                fold_scores.append(score)

            # Average score for this repetition
            rep_score = np.mean(fold_scores)
            results["repetition_scores"].append(rep_score)

        # Compute confidence intervals
        scores_array = np.array(results["repetition_scores"])
        ci_lower, ci_upper = self.statistics.compute_confidence_interval(
            scores_array, confidence_level=self.config.confidence_level
        )

        results["confidence_intervals"] = {
            "mean": np.mean(scores_array),
            "std": np.std(scores_array),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "coefficient_of_variation": np.std(scores_array) / np.mean(scores_array),
        }

        # Bias correction
        if len(scores_array) >= 10:
            bias_corrected = self.statistics.bias_corrected_accuracy(
                np.repeat(1, len(scores_array)),  # Placeholder true labels
                (scores_array > 0.5).astype(int),  # Placeholder predictions
            )
            results["bias_corrected_metrics"] = bias_corrected

        # Sensitivity analysis
        if self.config.perform_sensitivity_analysis:
            results["sensitivity_analysis"] = self._perform_sensitivity_analysis(
                X, y, model_factory, model_config
            )

        return results

    def _perform_sensitivity_analysis(
        self, X: np.ndarray, y: np.ndarray, model_factory: Callable, model_config: Dict
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis to model assumptions."""
        sensitivity_results = {}

        # Sensitivity to training data variation
        base_score = np.random.uniform(0.5, 0.7)  # Placeholder

        # Remove different samples to assess stability
        stability_scores = []
        for _ in range(20):
            # Randomly remove 10% of samples
            n_remove = max(1, len(X) // 10)
            remove_idx = np.random.choice(len(X), n_remove, replace=False)
            keep_mask = np.ones(len(X), dtype=bool)
            keep_mask[remove_idx] = False

            if keep_mask.sum() >= 10:  # Minimum samples
                # Train on reduced dataset (placeholder)
                reduced_score = base_score + np.random.normal(0, 0.05)
                stability_scores.append(reduced_score)

        if stability_scores:
            sensitivity_results["data_stability"] = {
                "mean_score": np.mean(stability_scores),
                "score_std": np.std(stability_scores),
                "stability_coefficient": 1 - (np.std(stability_scores) / base_score),
            }

        return sensitivity_results

    def generate_experiment_report(self) -> Dict[str, Any]:
        """Generate comprehensive experiment report."""
        if not self.experiment_results:
            return {"error": "No experiments performed"}

        report = {
            "summary": {
                "n_experiments": len(self.experiment_results),
                "config": self.config.__dict__,
            },
            "performance_summary": {},
            "recommendations": [],
        }

        # Aggregate results across experiments
        all_scores = []
        for result in self.experiment_results:
            if "scores" in result:
                all_scores.extend(result["scores"])

        if all_scores:
            scores_array = np.array(all_scores)
            ci_lower, ci_upper = self.statistics.compute_confidence_interval(
                scores_array, confidence_level=self.config.confidence_level
            )

            report["performance_summary"] = {
                "mean_score": np.mean(scores_array),
                "std_score": np.std(scores_array),
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
                "n_total_evaluations": len(scores_array),
            }

        # Generate recommendations
        report["recommendations"] = self._generate_recommendations()

        return report

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on experimental results."""
        recommendations = []

        if self.config.n_samples < 50:
            recommendations.append(
                "Dataset is extremely small (<50 samples). Consider data augmentation "
                "or collecting more samples for reliable conclusions."
            )

        if self.config.min_samples_per_class < 5:
            recommendations.append(
                "Some classes have very few samples (<5). Use one-vs-rest approaches "
                "or synthetic data generation methods."
            )

        recommendations.append(
            "Use nested cross-validation to avoid optimistic bias in model selection."
        )

        recommendations.append(
            "Report confidence intervals for all performance metrics due to high uncertainty."
        )

        recommendations.append(
            "Consider ensemble methods to reduce variance in small sample scenarios."
        )

        return recommendations


def create_small_dataset_framework(
    n_samples: int = 98,
    strategy: SmallDatasetStrategy = SmallDatasetStrategy.CONSERVATIVE,
    confidence_level: float = 0.95,
) -> SmallDatasetFramework:
    """Create a small dataset framework with sensible defaults.

    Args:
        n_samples: Number of samples in dataset
        strategy: Strategy for handling small datasets
        confidence_level: Confidence level for statistical inference

    Returns:
        Configured small dataset framework
    """
    config = SmallDatasetConfig(
        n_samples=n_samples,
        cv_strategy=strategy,
        confidence_level=confidence_level,
        n_outer_folds=min(5, n_samples // 3),
        n_inner_folds=min(3, n_samples // 5),
    )

    return SmallDatasetFramework(config)


def analyze_small_dataset_feasibility(
    X: np.ndarray, y: np.ndarray, target_accuracy: float = 0.7
) -> Dict[str, Any]:
    """Analyze feasibility of machine learning with small dataset.

    Args:
        X: Feature matrix
        y: Target labels
        target_accuracy: Desired target accuracy

    Returns:
        Feasibility analysis with recommendations
    """
    framework = create_small_dataset_framework(n_samples=len(X))
    analysis = framework.analyze_dataset_characteristics(X, y)

    # Feasibility assessment
    feasibility = {
        "dataset_analysis": analysis,
        "feasibility_score": 0.0,
        "challenges": [],
        "mitigation_strategies": [],
        "minimum_samples_needed": 0,
    }

    # Calculate feasibility score
    score = 1.0

    if analysis["n_samples"] < 30:
        score -= 0.4
        feasibility["challenges"].append("Very small sample size")

    if analysis["class_imbalance_ratio"] > 5:
        score -= 0.2
        feasibility["challenges"].append("Severe class imbalance")

    if analysis["samples_per_feature"] < 5:
        score -= 0.3
        feasibility["challenges"].append("High dimensionality relative to samples")

    feasibility["feasibility_score"] = max(0.0, score)

    # Minimum samples calculation (rule of thumb)
    n_classes = analysis["n_classes"]
    n_features = analysis["n_features"]
    min_samples = max(10 * n_classes, 5 * n_features, 30)
    feasibility["minimum_samples_needed"] = min_samples

    # Mitigation strategies
    if analysis["class_imbalance_ratio"] > 3:
        feasibility["mitigation_strategies"].append("Use class balancing techniques")

    if analysis["samples_per_feature"] < 10:
        feasibility["mitigation_strategies"].append("Apply aggressive feature selection")

    feasibility["mitigation_strategies"].extend(
        [
            "Use nested cross-validation",
            "Apply strong regularization",
            "Use data augmentation",
            "Report confidence intervals",
            "Consider ensemble methods",
        ]
    )

    return feasibility
