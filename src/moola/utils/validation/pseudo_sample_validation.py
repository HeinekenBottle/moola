"""Comprehensive validation and quality assessment framework for pseudo-samples.

This module provides tools to validate the quality of generated pseudo-samples
against real financial data, ensuring statistical realism and market consistency.

Key validation metrics:
1. Statistical distribution similarity
2. Temporal dependency preservation
3. Market microstructure consistency
4. Financial realism checks
5. Pattern similarity assessment
6. Risk factor validation
"""

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import euclidean
from scipy.stats import wasserstein_distance
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ValidationReport:
    """Comprehensive validation report for pseudo-samples."""

    overall_quality_score: float
    statistical_similarity: Dict[str, float]
    temporal_consistency: Dict[str, float]
    market_realism: Dict[str, float]
    pattern_similarity: Dict[str, float]
    risk_metrics: Dict[str, float]
    ohlc_integrity: float
    recommendations: List[str]
    passed_thresholds: Dict[str, bool]


class FinancialDataValidator:
    """Comprehensive validator for financial pseudo-samples."""

    def __init__(self, significance_level: float = 0.05, strict_mode: bool = False):
        """Initialize financial data validator.

        Args:
            significance_level: Statistical significance level for tests
            strict_mode: Whether to apply stricter validation criteria
        """
        self.significance_level = significance_level
        self.strict_mode = strict_mode

        # Validation thresholds (can be adjusted based on requirements)
        self.thresholds = {
            "ohlc_integrity": 0.95 if strict_mode else 0.90,
            "return_distribution": 0.80 if strict_mode else 0.70,
            "volatility_clustering": 0.75 if strict_mode else 0.65,
            "autocorrelation": 0.70 if strict_mode else 0.60,
            "kurtosis_similarity": 0.70 if strict_mode else 0.60,
            "pattern_dtw": 0.65 if strict_mode else 0.55,
            "risk_metrics": 0.75 if strict_mode else 0.65,
            "overall_quality": 0.75 if strict_mode else 0.65,
        }

    def _calculate_returns(self, data: np.ndarray) -> np.ndarray:
        """Calculate log returns for OHLC data.

        Args:
            data: OHLC data [N, T, 4]

        Returns:
            Log returns array [N, T-1]
        """
        returns = []
        for sample in data:
            close_prices = sample[:, 3]
            log_returns = np.diff(np.log(close_prices + 1e-8))
            returns.append(log_returns)
        return np.array(returns)

    def _validate_ohlc_relationships(self, data: np.ndarray) -> Tuple[float, Dict]:
        """Validate OHLC relationships in generated data.

        Args:
            data: OHLC data to validate [N, T, 4]

        Returns:
            Tuple of (integrity_score, violation_details)
        """
        violations = {
            "open_high": 0,
            "open_low": 0,
            "close_high": 0,
            "close_low": 0,
            "high_low_order": 0,
            "negative_prices": 0,
        }

        total_checks = len(data) * data.shape[1]

        for sample in data:
            for t in range(sample.shape[0]):
                o, h, l, c = sample[t]

                # Check for negative prices
                if any(x <= 0 for x in [o, h, l, c]):
                    violations["negative_prices"] += 1
                    continue

                # Check OHLC relationships
                if o > h:
                    violations["open_high"] += 1
                if o < l:
                    violations["open_low"] += 1
                if c > h:
                    violations["close_high"] += 1
                if c < l:
                    violations["close_low"] += 1
                if h < l:
                    violations["high_low_order"] += 1

        # Calculate integrity score
        total_violations = sum(violations.values())
        integrity_score = 1.0 - (total_violations / total_checks)

        return integrity_score, violations

    def _validate_statistical_properties(
        self, original_data: np.ndarray, generated_data: np.ndarray
    ) -> Dict[str, float]:
        """Validate statistical properties of generated data.

        Args:
            original_data: Original OHLC data [N, T, 4]
            generated_data: Generated OHLC data [M, T, 4]

        Returns:
            Dictionary of statistical similarity metrics
        """
        metrics = {}

        # Calculate returns for both datasets
        orig_returns = self._calculate_returns(original_data).flatten()
        gen_returns = self._calculate_returns(generated_data).flatten()

        # Remove infinite values
        orig_returns = orig_returns[np.isfinite(orig_returns)]
        gen_returns = gen_returns[np.isfinite(gen_returns)]

        if len(orig_returns) == 0 or len(gen_returns) == 0:
            return {"error": "No valid returns data"}

        # Distribution similarity tests
        try:
            # Kolmogorov-Smirnov test
            ks_stat = stats.ks_2samp(orig_returns, gen_returns).statistic
            metrics["ks_similarity"] = 1.0 - ks_stat

            # Wasserstein distance
            wasserstein_dist = wasserstein_distance(orig_returns, gen_returns)
            # Normalize by data range
            data_range = np.percentile(np.abs(orig_returns), 99)
            metrics["wasserstein_similarity"] = 1.0 - (wasserstein_dist / (data_range + 1e-8))
        except Exception as e:
            metrics["distribution_error"] = str(e)

        # Statistical moments comparison
        try:
            orig_moments = {
                "mean": np.mean(orig_returns),
                "std": np.std(orig_returns),
                "skew": stats.skew(orig_returns),
                "kurtosis": stats.kurtosis(orig_returns),
            }

            gen_moments = {
                "mean": np.mean(gen_returns),
                "std": np.std(gen_returns),
                "skew": stats.skew(gen_returns),
                "kurtosis": stats.kurtosis(gen_returns),
            }

            # Calculate similarity scores for each moment
            for moment in orig_moments:
                orig_val = orig_moments[moment]
                gen_val = gen_moments[moment]
                similarity = 1.0 - abs(gen_val - orig_val) / (abs(orig_val) + 1e-8)
                similarity = max(0.0, min(1.0, similarity))
                metrics[f"{moment}_similarity"] = similarity

        except Exception as e:
            metrics["moments_error"] = str(e)

        # Price level distributions
        try:
            orig_prices = original_data[:, :, 3].flatten()
            gen_prices = generated_data[:, :, 3].flatten()

            # Remove outliers for comparison
            orig_prices_clean = orig_prices[
                (orig_prices > np.percentile(orig_prices, 1))
                & (orig_prices < np.percentile(orig_prices, 99))
            ]
            gen_prices_clean = gen_prices[
                (gen_prices > np.percentile(gen_prices, 1))
                & (gen_prices < np.percentile(gen_prices, 99))
            ]

            if len(orig_prices_clean) > 0 and len(gen_prices_clean) > 0:
                ks_stat_prices = stats.ks_2samp(orig_prices_clean, gen_prices_clean).statistic
                metrics["price_distribution_similarity"] = 1.0 - ks_stat_prices
        except Exception as e:
            metrics["price_distribution_error"] = str(e)

        return metrics

    def _validate_temporal_dependencies(
        self, original_data: np.ndarray, generated_data: np.ndarray
    ) -> Dict[str, float]:
        """Validate temporal dependencies and autocorrelation structure.

        Args:
            original_data: Original OHLC data [N, T, 4]
            generated_data: Generated OHLC data [M, T, 4]

        Returns:
            Dictionary of temporal consistency metrics
        """
        metrics = {}

        try:
            # Calculate returns
            orig_returns = self._calculate_returns(original_data)
            gen_returns = self._calculate_returns(generated_data)

            # Autocorrelation comparison for different lags
            lags = [1, 2, 5, 10, 20]
            autocorr_similarities = []

            for lag in lags:
                if orig_returns.shape[1] > lag:
                    # Calculate autocorrelations for original data
                    orig_autocorrs = []
                    for sample in orig_returns:
                        if len(sample) > lag:
                            autocorr = np.corrcoef(sample[:-lag], sample[lag:])[0, 1]
                            if not np.isnan(autocorr):
                                orig_autocorrs.append(autocorr)

                    # Calculate autocorrelations for generated data
                    gen_autocorrs = []
                    for sample in gen_returns:
                        if len(sample) > lag:
                            autocorr = np.corrcoef(sample[:-lag], sample[lag:])[0, 1]
                            if not np.isnan(autocorr):
                                gen_autocorrs.append(autocorr)

                    if orig_autocorrs and gen_autocorrs:
                        orig_mean = np.mean(orig_autocorrs)
                        gen_mean = np.mean(gen_autocorrs)
                        similarity = 1.0 - abs(orig_mean - gen_mean)
                        autocorr_similarities.append(similarity)

                        metrics[f"autocorr_lag_{lag}_similarity"] = similarity

            if autocorr_similarities:
                metrics["autocorr_overall_similarity"] = np.mean(autocorr_similarities)

            # Volatility clustering (absolute return autocorrelation)
            orig_abs_returns = np.abs(orig_returns)
            gen_abs_returns = np.abs(gen_returns)

            if orig_abs_returns.shape[1] > 1:
                orig_vol_clust = []
                gen_vol_clust = []

                for sample in orig_abs_returns:
                    if len(sample) > 1:
                        autocorr = np.corrcoef(sample[:-1], sample[1:])[0, 1]
                        if not np.isnan(autocorr):
                            orig_vol_clust.append(autocorr)

                for sample in gen_abs_returns:
                    if len(sample) > 1:
                        autocorr = np.corrcoef(sample[:-1], sample[1:])[0, 1]
                        if not np.isnan(autocorr):
                            gen_vol_clust.append(autocorr)

                if orig_vol_clust and gen_vol_clust:
                    orig_mean = np.mean(orig_vol_clust)
                    gen_mean = np.mean(gen_vol_clust)
                    similarity = 1.0 - abs(orig_mean - gen_mean)
                    metrics["volatility_clustering_similarity"] = similarity

        except Exception as e:
            metrics["temporal_error"] = str(e)

        return metrics

    def _validate_market_realism(
        self, original_data: np.ndarray, generated_data: np.ndarray
    ) -> Dict[str, float]:
        """Validate market microstructure and realism aspects.

        Args:
            original_data: Original OHLC data [N, T, 4]
            generated_data: Generated OHLC data [M, T, 4]

        Returns:
            Dictionary of market realism metrics
        """
        metrics = {}

        try:
            # Spread analysis (High-Low spread)
            orig_spreads = []
            gen_spreads = []

            for sample in original_data:
                spreads = (sample[:, 1] - sample[:, 2]) / sample[:, 2]  # (High-Low)/Low
                orig_spreads.extend(spreads[spreads < 1.0])  # Filter out unreasonable spreads

            for sample in generated_data:
                spreads = (sample[:, 1] - sample[:, 2]) / sample[:, 2]
                gen_spreads.extend(spreads[spreads < 1.0])

            if orig_spreads and gen_spreads:
                orig_mean_spread = np.mean(orig_spreads)
                gen_mean_spread = np.mean(gen_spreads)
                similarity = 1.0 - abs(gen_mean_spread - orig_mean_spread) / (
                    orig_mean_spread + 1e-8
                )
                metrics["spread_similarity"] = similarity

            # Gap analysis (Open-Previous close gaps)
            orig_gaps = []
            gen_gaps = []

            for sample in original_data:
                gaps = (sample[1:, 0] - sample[:-1, 3]) / sample[
                    :-1, 3
                ]  # (Open-Prev Close)/Prev Close
                orig_gaps.extend(gaps)

            for sample in generated_data:
                gaps = (sample[1:, 0] - sample[:-1, 3]) / sample[:-1, 3]
                gen_gaps.extend(gaps)

            if orig_gaps and gen_gaps:
                # Filter extreme gaps
                orig_gaps_clean = [g for g in orig_gaps if abs(g) < 0.2]
                gen_gaps_clean = [g for g in gen_gaps if abs(g) < 0.2]

                if orig_gaps_clean and gen_gaps_clean:
                    orig_gap_std = np.std(orig_gaps_clean)
                    gen_gap_std = np.std(gen_gaps_clean)
                    similarity = 1.0 - abs(gen_gap_std - orig_gap_std) / (orig_gap_std + 1e-8)
                    metrics["gap_similarity"] = similarity

            # Range utilization (typical price movement within daily range)
            orig_range_util = []
            gen_range_util = []

            for sample in original_data:
                daily_range = sample[:, 1] - sample[:, 2]  # High - Low
                price_movement = np.abs(sample[:, 3] - sample[:, 0])  # |Close - Open|
                utilization = price_movement / (daily_range + 1e-8)
                orig_range_util.extend(utilization[utilization <= 1.0])

            for sample in generated_data:
                daily_range = sample[:, 1] - sample[:, 2]
                price_movement = np.abs(sample[:, 3] - sample[:, 0])
                utilization = price_movement / (daily_range + 1e-8)
                gen_range_util.extend(utilization[utilization <= 1.0])

            if orig_range_util and gen_range_util:
                orig_mean_util = np.mean(orig_range_util)
                gen_mean_util = np.mean(gen_range_util)
                similarity = 1.0 - abs(gen_mean_util - orig_mean_util)
                metrics["range_utilization_similarity"] = similarity

        except Exception as e:
            metrics["market_realism_error"] = str(e)

        return metrics

    def _validate_pattern_similarity(
        self, original_data: np.ndarray, generated_data: np.ndarray
    ) -> Dict[str, float]:
        """Validate pattern similarity using Dynamic Time Warping.

        Args:
            original_data: Original OHLC data [N, T, 4]
            generated_data: Generated OHLC data [M, T, 4]

        Returns:
            Dictionary of pattern similarity metrics
        """
        metrics = {}

        def dtw_distance(s1, s2):
            """Calculate Dynamic Time Warping distance."""
            n, m = len(s1), len(s2)
            dtw = np.full((n + 1, m + 1), np.inf)
            dtw[0, 0] = 0

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(s1[i - 1] - s2[j - 1])
                    dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

            return dtw[n, m]

        try:
            # Sample subset for efficiency
            n_samples = min(20, len(original_data), len(generated_data))
            orig_sample = np.random.choice(len(original_data), n_samples, replace=False)
            gen_sample = np.random.choice(len(generated_data), n_samples, replace=False)

            dtw_distances = []
            pattern_correlations = []

            for i in range(n_samples):
                orig_close = original_data[orig_sample[i], :, 3]
                gen_close = generated_data[gen_sample[i], :, 3]

                # Normalize sequences
                orig_norm = (orig_close - np.mean(orig_close)) / (np.std(orig_close) + 1e-8)
                gen_norm = (gen_close - np.mean(gen_close)) / (np.std(gen_close) + 1e-8)

                # DTW distance
                dist = dtw_distance(orig_norm, gen_norm)
                dtw_distances.append(dist)

                # Correlation
                if len(orig_norm) == len(gen_norm):
                    corr = np.corrcoef(orig_norm, gen_norm)[0, 1]
                    if not np.isnan(corr):
                        pattern_correlations.append(abs(corr))

            if dtw_distances:
                # Convert DTW distances to similarity scores
                max_dist = max(dtw_distances) if dtw_distances else 1.0
                dtw_similarities = [1.0 - (d / max_dist) for d in dtw_distances]
                metrics["dtw_pattern_similarity"] = np.mean(dtw_similarities)

            if pattern_correlations:
                metrics["correlation_pattern_similarity"] = np.mean(pattern_correlations)

        except Exception as e:
            metrics["pattern_error"] = str(e)

        return metrics

    def _validate_risk_metrics(
        self, original_data: np.ndarray, generated_data: np.ndarray
    ) -> Dict[str, float]:
        """Validate risk-related metrics and extreme events.

        Args:
            original_data: Original OHLC data [N, T, 4]
            generated_data: Generated OHLC data [M, T, 4]

        Returns:
            Dictionary of risk metric similarities
        """
        metrics = {}

        try:
            # Calculate returns
            orig_returns = self._calculate_returns(original_data).flatten()
            gen_returns = self._calculate_returns(generated_data).flatten()

            # Remove infinite values
            orig_returns = orig_returns[np.isfinite(orig_returns)]
            gen_returns = gen_returns[np.isfinite(gen_returns)]

            if len(orig_returns) == 0 or len(gen_returns) == 0:
                return {"error": "No valid returns data"}

            # Value at Risk (VaR) comparison
            var_levels = [0.95, 0.99, 0.999]
            var_similarities = []

            for var_level in var_levels:
                orig_var = np.percentile(orig_returns, (1 - var_level) * 100)
                gen_var = np.percentile(gen_returns, (1 - var_level) * 100)

                # Compare VaR values
                similarity = 1.0 - abs(gen_var - orig_var) / (abs(orig_var) + 1e-8)
                similarity = max(0.0, min(1.0, similarity))
                var_similarities.append(similarity)

                metrics[f"var_{int(var_level*100)}_similarity"] = similarity

            if var_similarities:
                metrics["var_overall_similarity"] = np.mean(var_similarities)

            # Maximum drawdown comparison
            def calculate_max_drawdown(prices):
                """Calculate maximum drawdown."""
                peak = prices[0]
                max_dd = 0
                for price in prices[1:]:
                    if price > peak:
                        peak = price
                    drawdown = (peak - price) / peak
                    max_dd = max(max_dd, drawdown)
                return max_dd

            orig_max_dds = []
            gen_max_dds = []

            for sample in original_data:
                prices = sample[:, 3]
                max_dd = calculate_max_drawdown(prices)
                orig_max_dds.append(max_dd)

            for sample in generated_data:
                prices = sample[:, 3]
                max_dd = calculate_max_drawdown(prices)
                gen_max_dds.append(max_dd)

            if orig_max_dds and gen_max_dds:
                orig_mean_dd = np.mean(orig_max_dds)
                gen_mean_dd = np.mean(gen_max_dds)
                similarity = 1.0 - abs(gen_mean_dd - orig_mean_dd)
                metrics["max_drawdown_similarity"] = similarity

            # Tail risk comparison (extreme return frequency)
            extreme_threshold = np.percentile(np.abs(orig_returns), 99)
            orig_extreme_freq = np.mean(np.abs(orig_returns) > extreme_threshold)
            gen_extreme_freq = np.mean(np.abs(gen_returns) > extreme_threshold)

            if orig_extreme_freq > 0:
                similarity = 1.0 - abs(gen_extreme_freq - orig_extreme_freq) / orig_extreme_freq
                metrics["tail_risk_similarity"] = similarity

        except Exception as e:
            metrics["risk_metrics_error"] = str(e)

        return metrics

    def validate_pseudo_samples(
        self, original_data: np.ndarray, generated_data: np.ndarray
    ) -> ValidationReport:
        """Perform comprehensive validation of generated pseudo-samples.

        Args:
            original_data: Original OHLC data [N, T, 4]
            generated_data: Generated OHLC data [M, T, 4]

        Returns:
            Comprehensive validation report
        """
        # Initialize all metric categories
        statistical_similarity = self._validate_statistical_properties(
            original_data, generated_data
        )
        temporal_consistency = self._validate_temporal_dependencies(original_data, generated_data)
        market_realism = self._validate_market_realism(original_data, generated_data)
        pattern_similarity = self._validate_pattern_similarity(original_data, generated_data)
        risk_metrics = self._validate_risk_metrics(original_data, generated_data)

        # OHLC integrity validation
        ohlc_integrity, ohlc_violations = self._validate_ohlc_relationships(generated_data)

        # Calculate overall quality score
        all_metrics = {
            **statistical_similarity,
            **temporal_consistency,
            **market_realism,
            **pattern_similarity,
            **risk_metrics,
            "ohlc_integrity": ohlc_integrity,
        }

        # Filter out error metrics and calculate mean
        valid_metrics = {
            k: v for k, v in all_metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)
        }
        overall_quality = np.mean(list(valid_metrics.values())) if valid_metrics else 0.0

        # Generate recommendations
        recommendations = self._generate_recommendations(all_metrics, self.thresholds)

        # Check which thresholds are passed
        passed_thresholds = self._check_thresholds(all_metrics, self.thresholds)

        return ValidationReport(
            overall_quality_score=overall_quality,
            statistical_similarity=statistical_similarity,
            temporal_consistency=temporal_consistency,
            market_realism=market_realism,
            pattern_similarity=pattern_similarity,
            risk_metrics=risk_metrics,
            ohlc_integrity=ohlc_integrity,
            recommendations=recommendations,
            passed_thresholds=passed_thresholds,
        )

    def _generate_recommendations(
        self, metrics: Dict[str, float], thresholds: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on validation results.

        Args:
            metrics: Validation metrics
            thresholds: Quality thresholds

        Returns:
            List of recommendations
        """
        recommendations = []

        # OHLC integrity
        if metrics.get("ohlc_integrity", 1.0) < thresholds["ohlc_integrity"]:
            recommendations.append(
                "OHLC integrity violations detected. Review price relationship constraints."
            )

        # Statistical properties
        if metrics.get("ks_similarity", 1.0) < thresholds["return_distribution"]:
            recommendations.append(
                "Return distribution mismatch. Consider adjusting volatility parameters."
            )

        if metrics.get("kurtosis_similarity", 1.0) < thresholds["kurtosis_similarity"]:
            recommendations.append(
                "Tail behavior differs from original data. Adjust extreme event modeling."
            )

        # Temporal consistency
        if (
            metrics.get("volatility_clustering_similarity", 1.0)
            < thresholds["volatility_clustering"]
        ):
            recommendations.append(
                "Volatility clustering not preserved. Enhance temporal dependency modeling."
            )

        if metrics.get("autocorr_overall_similarity", 1.0) < thresholds["autocorrelation"]:
            recommendations.append(
                "Autocorrelation structure differs. Review time series dynamics."
            )

        # Pattern similarity
        if metrics.get("dtw_pattern_similarity", 1.0) < thresholds["pattern_dtw"]:
            recommendations.append(
                "Pattern shapes differ significantly. Improve pattern preservation techniques."
            )

        # Market realism
        if metrics.get("spread_similarity", 1.0) < 0.7:
            recommendations.append(
                "Spread characteristics unrealistic. Review intraday volatility modeling."
            )

        # Risk metrics
        if metrics.get("var_overall_similarity", 1.0) < thresholds["risk_metrics"]:
            recommendations.append(
                "Risk metrics differ. Review extreme event modeling and tail risk."
            )

        if not recommendations:
            recommendations.append("Quality metrics look good. Generated samples are acceptable.")

        return recommendations

    def _check_thresholds(
        self, metrics: Dict[str, float], thresholds: Dict[str, float]
    ) -> Dict[str, bool]:
        """Check which quality thresholds are passed.

        Args:
            metrics: Validation metrics
            thresholds: Quality thresholds

        Returns:
            Dictionary of threshold pass/fail status
        """
        passed = {}

        # Map metrics to thresholds
        metric_threshold_map = {
            "ohlc_integrity": "ohlc_integrity",
            "ks_similarity": "return_distribution",
            "wasserstein_similarity": "return_distribution",
            "volatility_clustering_similarity": "volatility_clustering",
            "autocorr_overall_similarity": "autocorrelation",
            "kurtosis_similarity": "kurtosis_similarity",
            "dtw_pattern_similarity": "pattern_dtw",
            "var_overall_similarity": "risk_metrics",
            "max_drawdown_similarity": "risk_metrics",
        }

        for metric, threshold_name in metric_threshold_map.items():
            if metric in metrics and threshold_name in thresholds:
                passed[metric] = metrics[metric] >= thresholds[threshold_name]

        # Overall quality check
        valid_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float)) and not np.isnan(v)
        }
        if valid_metrics:
            overall_quality = np.mean(list(valid_metrics.values()))
            passed["overall_quality"] = overall_quality >= thresholds["overall_quality"]

        return passed

    def generate_validation_report(self, report: ValidationReport) -> str:
        """Generate human-readable validation report.

        Args:
            report: Validation report object

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 60)
        lines.append("FINANCIAL PSEUDO-SAMPLE VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Overall Quality Score: {report.overall_quality_score:.3f}")
        lines.append(f"OHLC Integrity: {report.ohlc_integrity:.3f}")
        lines.append("")

        # Statistical similarity
        lines.append("STATISTICAL SIMILARITY:")
        for metric, value in report.statistical_similarity.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {metric}: {value:.3f}")
        lines.append("")

        # Temporal consistency
        lines.append("TEMPORAL CONSISTENCY:")
        for metric, value in report.temporal_consistency.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {metric}: {value:.3f}")
        lines.append("")

        # Market realism
        lines.append("MARKET REALISM:")
        for metric, value in report.market_realism.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {metric}: {value:.3f}")
        lines.append("")

        # Pattern similarity
        lines.append("PATTERN SIMILARITY:")
        for metric, value in report.pattern_similarity.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {metric}: {value:.3f}")
        lines.append("")

        # Risk metrics
        lines.append("RISK METRICS:")
        for metric, value in report.risk_metrics.items():
            if isinstance(value, (int, float)):
                lines.append(f"  {metric}: {value:.3f}")
        lines.append("")

        # Recommendations
        lines.append("RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

        # Threshold status
        lines.append("THRESHOLD STATUS:")
        passed_count = sum(report.passed_thresholds.values())
        total_count = len(report.passed_thresholds)
        lines.append(f"  Passed: {passed_count}/{total_count} thresholds")
        for metric, passed in report.passed_thresholds.items():
            status = "✓" if passed else "✗"
            lines.append(f"  {status} {metric}")

        return "\n".join(lines)


class QualityMetricsVisualizer:
    """Visualizer for quality metrics and validation results."""

    def __init__(self):
        """Initialize visualizer."""
        self.colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E83"]

    def plot_validation_results(self, report: ValidationReport, save_path: Optional[str] = None):
        """Create comprehensive validation visualization.

        Args:
            report: Validation report
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Pseudo-Sample Validation Results", fontsize=16, fontweight="bold")

        # Overall quality score
        axes[0, 0].bar(
            ["Overall Quality"],
            [report.overall_quality_score],
            color=self.colors[0] if report.overall_quality_score >= 0.7 else self.colors[3],
        )
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].set_title("Overall Quality Score")
        axes[0, 0].set_ylabel("Score")

        # OHLC integrity
        axes[0, 1].bar(
            ["OHLC Integrity"],
            [report.ohlc_integrity],
            color=self.colors[0] if report.ohlc_integrity >= 0.9 else self.colors[3],
        )
        axes[0, 1].set_ylim([0, 1])
        axes[0, 1].set_title("OHLC Integrity")
        axes[0, 1].set_ylabel("Score")

        # Statistical similarity
        stat_metrics = {
            k: v for k, v in report.statistical_similarity.items() if isinstance(v, (int, float))
        }
        if stat_metrics:
            axes[0, 2].barh(
                list(stat_metrics.keys()), list(stat_metrics.values()), color=self.colors[1]
            )
            axes[0, 2].set_xlim([0, 1])
            axes[0, 2].set_title("Statistical Similarity")
            axes[0, 2].set_xlabel("Score")

        # Temporal consistency
        temp_metrics = {
            k: v for k, v in report.temporal_consistency.items() if isinstance(v, (int, float))
        }
        if temp_metrics:
            axes[1, 0].barh(
                list(temp_metrics.keys()), list(temp_metrics.values()), color=self.colors[2]
            )
            axes[1, 0].set_xlim([0, 1])
            axes[1, 0].set_title("Temporal Consistency")
            axes[1, 0].set_xlabel("Score")

        # Market realism
        market_metrics = {
            k: v for k, v in report.market_realism.items() if isinstance(v, (int, float))
        }
        if market_metrics:
            axes[1, 1].barh(
                list(market_metrics.keys()), list(market_metrics.values()), color=self.colors[3]
            )
            axes[1, 1].set_xlim([0, 1])
            axes[1, 1].set_title("Market Realism")
            axes[1, 1].set_xlabel("Score")

        # Risk metrics
        risk_metrics = {k: v for k, v in report.risk_metrics.items() if isinstance(v, (int, float))}
        if risk_metrics:
            axes[1, 2].barh(
                list(risk_metrics.keys()), list(risk_metrics.values()), color=self.colors[4]
            )
            axes[1, 2].set_xlim([0, 1])
            axes[1, 2].set_title("Risk Metrics")
            axes[1, 2].set_xlabel("Score")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def plot_distribution_comparison(
        self, original_data: np.ndarray, generated_data: np.ndarray, save_path: Optional[str] = None
    ):
        """Plot distribution comparisons between original and generated data.

        Args:
            original_data: Original OHLC data [N, T, 4]
            generated_data: Generated OHLC data [M, T, 4]
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            "Distribution Comparison: Original vs Generated", fontsize=16, fontweight="bold"
        )

        # Calculate returns
        orig_returns = []
        gen_returns = []
        for sample in original_data:
            returns = np.diff(np.log(sample[:, 3] + 1e-8))
            orig_returns.extend(returns)
        for sample in generated_data:
            returns = np.diff(np.log(sample[:, 3] + 1e-8))
            gen_returns.extend(returns)

        orig_returns = np.array(orig_returns)
        gen_returns = np.array(gen_returns)

        # Filter extreme values for visualization
        orig_returns = orig_returns[
            (orig_returns > np.percentile(orig_returns, 1))
            & (orig_returns < np.percentile(orig_returns, 99))
        ]
        gen_returns = gen_returns[
            (gen_returns > np.percentile(gen_returns, 1))
            & (gen_returns < np.percentile(gen_returns, 99))
        ]

        # Returns distribution
        axes[0, 0].hist(
            orig_returns, bins=50, alpha=0.7, label="Original", density=True, color=self.colors[0]
        )
        axes[0, 0].hist(
            gen_returns, bins=50, alpha=0.7, label="Generated", density=True, color=self.colors[1]
        )
        axes[0, 0].set_title("Returns Distribution")
        axes[0, 0].set_xlabel("Returns")
        axes[0, 0].set_ylabel("Density")
        axes[0, 0].legend()

        # Q-Q plot
        from scipy import stats as scipy_stats

        scipy_stats.probplot(gen_returns, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title("Q-Q Plot (Generated Returns)")

        # Autocorrelation comparison
        def calculate_autocorr(returns, max_lag=20):
            autocorr = []
            for lag in range(1, min(max_lag, len(returns) // 4)):
                if len(returns) > lag:
                    corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    if not np.isnan(corr):
                        autocorr.append(corr)
            return autocorr

        orig_autocorr = calculate_autocorr(orig_returns)
        gen_autocorr = calculate_autocorr(gen_returns)

        lags = range(1, len(orig_autocorr) + 1)
        axes[0, 2].plot(lags, orig_autocorr, "o-", label="Original", color=self.colors[0])
        axes[1, 0].plot(
            lags[: len(gen_autocorr)], gen_autocorr, "s-", label="Generated", color=self.colors[1]
        )
        axes[0, 2].set_title("Autocorrelation Comparison")
        axes[0, 2].set_xlabel("Lag")
        axes[0, 2].set_ylabel("Autocorrelation")
        axes[0, 2].legend()

        # Volatility clustering
        orig_abs_returns = np.abs(orig_returns)
        gen_abs_returns = np.abs(gen_returns)

        axes[1, 0].hist(
            orig_abs_returns,
            bins=30,
            alpha=0.7,
            label="Original",
            density=True,
            color=self.colors[0],
        )
        axes[1, 0].hist(
            gen_abs_returns,
            bins=30,
            alpha=0.7,
            label="Generated",
            density=True,
            color=self.colors[1],
        )
        axes[1, 0].set_title("Absolute Returns Distribution")
        axes[1, 0].set_xlabel("Absolute Returns")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].legend()

        # Price levels
        orig_prices = original_data[:, :, 3].flatten()
        gen_prices = generated_data[:, :, 3].flatten()

        axes[1, 1].hist(
            orig_prices, bins=50, alpha=0.7, label="Original", density=True, color=self.colors[0]
        )
        axes[1, 1].hist(
            gen_prices, bins=50, alpha=0.7, label="Generated", density=True, color=self.colors[1]
        )
        axes[1, 1].set_title("Price Level Distribution")
        axes[1, 1].set_xlabel("Price")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].legend()

        # Sample time series
        sample_idx = min(0, len(original_data) - 1, len(generated_data) - 1)
        timesteps = range(len(original_data[sample_idx]))

        axes[1, 2].plot(
            timesteps, original_data[sample_idx, :, 3], label="Original", color=self.colors[0]
        )
        axes[1, 2].plot(
            timesteps, generated_data[sample_idx, :, 3], label="Generated", color=self.colors[1]
        )
        axes[1, 2].set_title("Sample Time Series Comparison")
        axes[1, 2].set_xlabel("Time")
        axes[1, 2].set_ylabel("Price")
        axes[1, 2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    print("Financial Pseudo-Sample Validation Framework")
    print("This module provides comprehensive validation tools for generated financial data.")
    print("Available components:")
    print("1. FinancialDataValidator - Comprehensive validation")
    print("2. QualityMetricsVisualizer - Visualization tools")
    print("3. ValidationReport - Structured validation results")
