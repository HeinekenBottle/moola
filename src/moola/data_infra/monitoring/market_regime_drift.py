"""Market regime-aware drift detection for financial time series.

Enhanced drift detection specifically designed for financial market data:
- Regime-specific drift detection (trending, ranging, volatile)
- Temporal pattern drift analysis
- Multi-scale drift detection (short-term vs long-term)
- Market microstructure drift monitoring
- Early warning system for regime changes

Key improvements over basic drift detection:
1. Concept drift detection for changing market patterns
2. Distribution drift detection for statistical properties
3. Temporal drift detection for time series dependencies
4. Adaptive thresholds based on market conditions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import StandardScaler
from loguru import logger


class MarketRegime(str, Enum):
    """Market regime classification."""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    UNCERTAIN = "uncertain"


@dataclass
class DriftSignal:
    """Individual drift detection signal."""

    signal_name: str
    drift_score: float
    threshold: float
    drift_detected: bool
    confidence: float
    regime_specific: bool
    description: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RegimeDriftReport:
    """Comprehensive drift detection report."""

    baseline_regime: MarketRegime
    current_regime: MarketRegime
    regime_change_detected: bool

    # Drift signals
    distribution_drift: List[DriftSignal]
    concept_drift: List[DriftSignal]
    temporal_drift: List[DriftSignal]
    microstructure_drift: List[DriftSignal]

    # Overall assessment
    overall_drift_score: float
    drift_severity: str  # "none", "low", "medium", "high", "critical"
    confidence: float

    # Recommendations
    recommendations: List[str]

    timestamp: datetime = field(default_factory=datetime.utcnow)


class MarketRegimeDriftDetector:
    """Advanced drift detection specialized for financial market regimes."""

    def __init__(
        self,
        window_size: int = 105,
        regime_detection_period: int = 30,
        sensitivity: str = "medium",  # "low", "medium", "high"
        adaptive_thresholds: bool = True
    ):
        """Initialize market regime drift detector.

        Args:
            window_size: Size of analysis window (matches Moola's 105 bars)
            regime_detection_period: Period for regime classification
            sensitivity: Detection sensitivity (affects thresholds)
            adaptive_thresholds: Enable adaptive thresholding based on market conditions
        """
        self.window_size = window_size
        self.regime_detection_period = regime_detection_period
        self.sensitivity = sensitivity
        self.adaptive_thresholds = adaptive_thresholds

        # Sensitivity-based thresholds
        self.thresholds = self._get_sensitivity_thresholds()

        # Regime-specific drift parameters
        self.regime_params = self._initialize_regime_parameters()

        # Historical drift signals for adaptive learning
        self.drift_history: List[RegimeDriftReport] = []

    def _get_sensitivity_thresholds(self) -> Dict[str, float]:
        """Get thresholds based on sensitivity setting."""
        base_thresholds = {
            "low": {
                "ks_threshold": 0.01,
                "psi_threshold": 0.05,
                "wasserstein_threshold": 0.02,
                "concept_threshold": 0.15,
                "temporal_threshold": 0.1,
                "microstructure_threshold": 0.05
            },
            "medium": {
                "ks_threshold": 0.05,
                "psi_threshold": 0.1,
                "wasserstein_threshold": 0.05,
                "concept_threshold": 0.1,
                "temporal_threshold": 0.15,
                "microstructure_threshold": 0.1
            },
            "high": {
                "ks_threshold": 0.1,
                "psi_threshold": 0.15,
                "wasserstein_threshold": 0.1,
                "concept_threshold": 0.05,
                "temporal_threshold": 0.2,
                "microstructure_threshold": 0.15
            }
        }

        return base_thresholds[self.sensitivity]

    def _initialize_regime_parameters(self) -> Dict[MarketRegime, Dict]:
        """Initialize regime-specific drift detection parameters."""
        return {
            MarketRegime.TRENDING_UP: {
                "expected_return_sign": 1,
                "volatility_tolerance": 0.3,
                "trend_consistency_threshold": 0.7,
                "reversal_penalty": 2.0
            },
            MarketRegime.TRENDING_DOWN: {
                "expected_return_sign": -1,
                "volatility_tolerance": 0.3,
                "trend_consistency_threshold": 0.7,
                "reversal_penalty": 2.0
            },
            MarketRegime.RANGING: {
                "expected_return_sign": 0,
                "volatility_tolerance": 0.5,
                "trend_consistency_threshold": 0.3,
                "reversal_penalty": 1.0
            },
            MarketRegime.VOLATILE: {
                "expected_return_sign": 0,
                "volatility_tolerance": 1.0,
                "trend_consistency_threshold": 0.2,
                "reversal_penalty": 0.5
            },
            MarketRegime.UNCERTAIN: {
                "expected_return_sign": 0,
                "volatility_tolerance": 0.7,
                "trend_consistency_threshold": 0.4,
                "reversal_penalty": 1.0
            }
        }

    def detect_market_regime_drift(
        self,
        baseline_data: Union[np.ndarray, pd.DataFrame],
        current_data: Union[np.ndarray, pd.DataFrame],
        baseline_labels: Optional[np.ndarray] = None,
        current_labels: Optional[np.ndarray] = None
    ) -> RegimeDriftReport:
        """Comprehensive market regime drift detection.

        Args:
            baseline_data: Reference/baseline OHLC data [N, 105, 4] or DataFrame
            current_data: Current/production OHLC data [N, 105, 4] or DataFrame
            baseline_labels: Optional baseline pattern labels
            current_labels: Optional current pattern labels

        Returns:
            Comprehensive drift detection report
        """
        # Convert to consistent format
        baseline_df = self._to_dataframe(baseline_data)
        current_df = self._to_dataframe(current_data)

        if baseline_df is None or current_df is None:
            raise ValueError("Invalid data format provided")

        # Detect market regimes
        baseline_regime = self._classify_market_regime(baseline_df)
        current_regime = self._classify_market_regime(current_df)

        logger.info(f"Regime detection: baseline={baseline_regime}, current={current_regime}")

        # Initialize report
        report = RegimeDriftReport(
            baseline_regime=baseline_regime,
            current_regime=current_regime,
            regime_change_detected=baseline_regime != current_regime,
            distribution_drift=[],
            concept_drift=[],
            temporal_drift=[],
            microstructure_drift=[],
            overall_drift_score=0.0,
            drift_severity="none",
            confidence=0.0,
            recommendations=[]
        )

        # 1. Distribution drift detection
        report.distribution_drift = self._detect_distribution_drift(
            baseline_df, current_df
        )

        # 2. Concept drift detection (pattern relationships)
        if baseline_labels is not None and current_labels is not None:
            report.concept_drift = self._detect_concept_drift(
                baseline_df, current_df, baseline_labels, current_labels
            )

        # 3. Temporal drift detection
        report.temporal_drift = self._detect_temporal_drift(
            baseline_df, current_df
        )

        # 4. Microstructure drift detection
        report.microstructure_drift = self._detect_microstructure_drift(
            baseline_df, current_df
        )

        # Compute overall drift assessment
        report.overall_drift_score, report.drift_severity, report.confidence = \
            self._compute_overall_drift_assessment(report)

        # Generate recommendations
        report.recommendations = self._generate_recommendations(report)

        # Store in history
        self.drift_history.append(report)

        return report

    def _to_dataframe(self, data: Union[np.ndarray, pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Convert data to DataFrame format."""
        if isinstance(data, pd.DataFrame):
            return data

        elif isinstance(data, np.ndarray):
            if data.ndim == 3 and data.shape[1:] == (105, 4):
                # Reshape [N, 105, 4] to [N*105, 4]
                n_samples = data.shape[0]
                reshaped = data.reshape(-1, 4)
                df = pd.DataFrame(reshaped, columns=['open', 'high', 'low', 'close'])

                # Add sample_id and timestep for analysis
                sample_ids = np.repeat(np.arange(n_samples), 105)
                timesteps = np.tile(np.arange(105), n_samples)
                df['sample_id'] = sample_ids
                df['timestep'] = timesteps

                return df

            elif data.ndim == 2 and data.shape[1] == 4:
                df = pd.DataFrame(data, columns=['open', 'high', 'low', 'close'])
                return df

        return None

    def _classify_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """Classify market regime based on price behavior."""
        if len(df) < self.regime_detection_period:
            return MarketRegime.UNCERTAIN

        # Use last N bars for regime classification
        recent_df = df.tail(self.regime_detection_period)

        # Calculate returns
        returns = recent_df['close'].pct_change().dropna()

        # Calculate trend strength
        x = np.arange(len(recent_df))
        slope, _, r_value, _, _ = stats.linregress(x, recent_df['close'])
        trend_strength = abs(slope) * (r_value ** 2)

        # Calculate volatility
        volatility = returns.std() * np.sqrt(252)  # Annualized

        # Calculate range ratio (how much price moves relative to level)
        price_range = (recent_df['high'].max() - recent_df['low'].min()) / recent_df['close'].mean()

        # Regime classification logic
        if trend_strength > 0.5 and volatility < 0.3:
            if slope > 0:
                return MarketRegime.TRENDING_UP
            else:
                return MarketRegime.TRENDING_DOWN

        elif volatility > 0.5:
            return MarketRegime.VOLATILE

        elif price_range < 0.05 and volatility < 0.2:
            return MarketRegime.RANGING

        else:
            return MarketRegime.UNCERTAIN

    def _detect_distribution_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> List[DriftSignal]:
        """Detect distribution drift using multiple statistical methods."""
        signals = []

        # Extract OHLC features
        baseline_features = baseline_df[['open', 'high', 'low', 'close']].values
        current_features = current_df[['open', 'high', 'low', 'close']].values

        # 1. Kolmogorov-Smirnov test for each OHLC component
        for i, feature in enumerate(['open', 'high', 'low', 'close']):
            baseline_vals = baseline_features[:, i]
            current_vals = current_features[:, i]

            ks_stat, ks_pvalue = stats.ks_2samp(baseline_vals, current_vals)
            drift_detected = ks_pvalue < self.thresholds["ks_threshold"]

            signals.append(DriftSignal(
                signal_name=f"ks_test_{feature}",
                drift_score=float(ks_stat),
                threshold=self.thresholds["ks_threshold"],
                drift_detected=drift_detected,
                confidence=float(1 - ks_pvalue),
                regime_specific=False,
                description=f"Kolmogorov-Smirnov test for {feature} distribution"
            ))

        # 2. Population Stability Index (PSI)
        for i, feature in enumerate(['open', 'high', 'low', 'close']):
            psi_score = self._compute_psi(
                baseline_features[:, i],
                current_features[:, i]
            )
            drift_detected = psi_score > self.thresholds["psi_threshold"]

            signals.append(DriftSignal(
                signal_name=f"psi_{feature}",
                drift_score=psi_score,
                threshold=self.thresholds["psi_threshold"],
                drift_detected=drift_detected,
                confidence=min(psi_score / self.thresholds["psi_threshold"], 1.0),
                regime_specific=False,
                description=f"Population Stability Index for {feature}"
            ))

        # 3. Wasserstein distance
        wasserstein_dist = stats.wasserstein_distance(
            baseline_features.flatten(),
            current_features.flatten()
        )

        # Normalize by baseline range
        baseline_range = baseline_features.max() - baseline_features.min()
        normalized_distance = wasserstein_dist / baseline_range if baseline_range > 0 else 0

        drift_detected = normalized_distance > self.thresholds["wasserstein_threshold"]

        signals.append(DriftSignal(
            signal_name="wasserstein_all_features",
            drift_score=normalized_distance,
            threshold=self.thresholds["wasserstein_threshold"],
            drift_detected=drift_detected,
            confidence=min(normalized_distance / self.thresholds["wasserstein_threshold"], 1.0),
            regime_specific=False,
            description="Wasserstein distance across all OHLC features"
        ))

        return signals

    def _compute_psi(self, baseline: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
        """Compute Population Stability Index."""
        # Create bins based on baseline
        bins_edges = np.percentile(baseline, np.linspace(0, 100, bins + 1))
        bins_edges = np.unique(bins_edges)

        if len(bins_edges) <= 1:
            return 0.0

        # Compute distributions
        baseline_counts, _ = np.histogram(baseline, bins=bins_edges)
        current_counts, _ = np.histogram(current, bins=bins_edges)

        # Normalize to percentages
        baseline_pct = baseline_counts / len(baseline)
        current_pct = current_counts / len(current)

        # Avoid division by zero
        baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
        current_pct = np.where(current_pct == 0, 0.0001, current_pct)

        # Compute PSI
        psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))

        return float(psi)

    def _detect_concept_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        baseline_labels: np.ndarray,
        current_labels: np.ndarray
    ) -> List[DriftSignal]:
        """Detect concept drift in pattern-label relationships."""
        signals = []

        # 1. Label distribution shift
        baseline_label_dist = pd.Series(baseline_labels).value_counts(normalize=True)
        current_label_dist = pd.Series(current_labels).value_counts(normalize=True)

        # Compute Jensen-Shannon divergence between label distributions
        all_labels = sorted(set(baseline_labels) | set(current_labels))
        baseline_probs = [baseline_label_dist.get(label, 0) for label in all_labels]
        current_probs = [current_label_dist.get(label, 0) for label in all_labels]

        js_divergence = jensenshannon(baseline_probs, current_probs) ** 2
        drift_detected = js_divergence > self.thresholds["concept_threshold"]

        signals.append(DriftSignal(
            signal_name="label_distribution_shift",
            drift_score=js_divergence,
            threshold=self.thresholds["concept_threshold"],
            drift_detected=drift_detected,
            confidence=min(js_divergence / self.thresholds["concept_threshold"], 1.0),
            regime_specific=True,
            description="Shift in pattern label distribution"
        ))

        # 2. Pattern-feature relationship change
        # Compute average OHLC patterns per label
        baseline_patterns = self._compute_pattern_features(baseline_df, baseline_labels)
        current_patterns = self._compute_pattern_features(current_df, current_labels)

        for label in baseline_patterns.keys():
            if label in current_patterns:
                baseline_avg = baseline_patterns[label]
                current_avg = current_patterns[label]

                # Compute pattern distance
                pattern_distance = np.linalg.norm(baseline_avg - current_avg)
                drift_detected = pattern_distance > self.thresholds["concept_threshold"]

                signals.append(DriftSignal(
                    signal_name=f"pattern_drift_{label}",
                    drift_score=pattern_distance,
                    threshold=self.thresholds["concept_threshold"],
                    drift_detected=drift_detected,
                    confidence=min(pattern_distance / self.thresholds["concept_threshold"], 1.0),
                    regime_specific=True,
                    description=f"Pattern feature drift for label {label}"
                ))

        return signals

    def _compute_pattern_features(self, df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """Compute average pattern features per label."""
        pattern_features = {}

        for label in np.unique(labels):
            mask = labels == label
            label_data = df[mask]

            # Compute pattern features
            features = [
                label_data['close'].mean(),
                label_data['close'].std(),
                (label_data['high'] - label_data['low']).mean() / label_data['close'].mean(),
                (label_data['close'] - label_data['open']).mean() / label_data['open'].mean(),
                label_data['close'].autocorr(lag=1) if len(label_data) > 1 else 0
            ]

            pattern_features[label] = np.array(features)

        return pattern_features

    def _detect_temporal_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> List[DriftSignal]:
        """Detect temporal drift in time series dependencies."""
        signals = []

        # 1. Autocorrelation drift
        baseline_autocorr = self._compute_autocorrelation_features(baseline_df)
        current_autocorr = self._compute_autocorrelation_features(current_df)

        autocorr_distance = np.linalg.norm(baseline_autocorr - current_autocorr)
        drift_detected = autocorr_distance > self.thresholds["temporal_threshold"]

        signals.append(DriftSignal(
            signal_name="autocorrelation_drift",
            drift_score=autocorr_distance,
            threshold=self.thresholds["temporal_threshold"],
            drift_detected=drift_detected,
            confidence=min(autocorr_distance / self.thresholds["temporal_threshold"], 1.0),
            regime_specific=True,
            description="Drift in temporal autocorrelation structure"
        ))

        # 2. Volatility clustering drift
        baseline_vol_cluster = self._compute_volatility_clustering(baseline_df)
        current_vol_cluster = self._compute_volatility_clustering(current_df)

        vol_cluster_drift = abs(baseline_vol_cluster - current_vol_cluster)
        drift_detected = vol_cluster_drift > self.thresholds["temporal_threshold"]

        signals.append(DriftSignal(
            signal_name="volatility_clustering_drift",
            drift_score=vol_cluster_drift,
            threshold=self.thresholds["temporal_threshold"],
            drift_detected=drift_detected,
            confidence=min(vol_cluster_drift / self.thresholds["temporal_threshold"], 1.0),
            regime_specific=True,
            description="Drift in volatility clustering behavior"
        ))

        return signals

    def _compute_autocorrelation_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute autocorrelation features for temporal analysis."""
        returns = df['close'].pct_change().dropna()

        # Compute autocorrelations at different lags
        lags = [1, 5, 10, 20]  # Different time scales
        autocorr_features = []

        for lag in lags:
            if len(returns) > lag:
                autocorr = returns.autocorr(lag=lag)
                autocorr_features.append(autocorr if not np.isnan(autocorr) else 0)
            else:
                autocorr_features.append(0)

        return np.array(autocorr_features)

    def _compute_volatility_clustering(self, df: pd.DataFrame) -> float:
        """Compute volatility clustering metric."""
        returns = df['close'].pct_change().dropna()

        if len(returns) < 10:
            return 0.0

        # Compute absolute returns and their autocorrelation
        abs_returns = abs(returns)
        vol_clustering = abs_returns.autocorr(lag=5)  # 5-period lag

        return vol_clustering if not np.isnan(vol_clustering) else 0.0

    def _detect_microstructure_drift(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> List[DriftSignal]:
        """Detect microstructure drift in market properties."""
        signals = []

        # 1. Spread drift
        baseline_spread = self._compute_average_spread(baseline_df)
        current_spread = self._compute_average_spread(current_df)

        spread_change = abs(current_spread - baseline_spread) / baseline_spread
        drift_detected = spread_change > self.thresholds["microstructure_threshold"]

        signals.append(DriftSignal(
            signal_name="spread_drift",
            drift_score=spread_change,
            threshold=self.thresholds["microstructure_threshold"],
            drift_detected=drift_detected,
            confidence=min(spread_change / self.thresholds["microstructure_threshold"], 1.0),
            regime_specific=True,
            description="Drift in bid-ask spread characteristics"
        ))

        # 2. Price impact drift
        baseline_impact = self._compute_price_impact(baseline_df)
        current_impact = self._compute_price_impact(current_df)

        impact_change = abs(current_impact - baseline_impact)
        drift_detected = impact_change > self.thresholds["microstructure_threshold"]

        signals.append(DriftSignal(
            signal_name="price_impact_drift",
            drift_score=impact_change,
            threshold=self.thresholds["microstructure_threshold"],
            drift_detected=drift_detected,
            confidence=min(impact_change / self.thresholds["microstructure_threshold"], 1.0),
            regime_specific=True,
            description="Drift in price impact characteristics"
        ))

        return signals

    def _compute_average_spread(self, df: pd.DataFrame) -> float:
        """Compute average spread (proxy using high-low range)."""
        spread = (df['high'] - df['low']) / df['close']
        return spread.mean()

    def _compute_price_impact(self, df: pd.DataFrame) -> float:
        """Compute price impact proxy (volume-weighted price movement)."""
        # Use range-based volume proxy
        volume_proxy = df['high'] - df['low']
        price_movement = abs(df['close'] - df['open']).mean()

        if volume_proxy.sum() > 0:
            impact = (price_movement * volume_proxy).sum() / volume_proxy.sum()
        else:
            impact = price_movement

        return impact

    def _compute_overall_drift_assessment(
        self,
        report: RegimeDriftReport
    ) -> Tuple[float, str, float]:
        """Compute overall drift score and severity."""
        all_signals = (
            report.distribution_drift +
            report.concept_drift +
            report.temporal_drift +
            report.microstructure_drift
        )

        if not all_signals:
            return 0.0, "none", 0.0

        # Compute weighted average drift score
        total_score = 0.0
        total_weight = 0.0
        detected_count = 0

        for signal in all_signals:
            weight = 2.0 if signal.regime_specific else 1.0  # Higher weight for regime-specific
            total_score += signal.drift_score * weight
            total_weight += weight

            if signal.drift_detected:
                detected_count += 1

        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        confidence = detected_count / len(all_signals)

        # Determine severity
        if confidence < 0.2:
            severity = "none"
        elif overall_score < 0.1:
            severity = "low"
        elif overall_score < 0.2:
            severity = "medium"
        elif overall_score < 0.3:
            severity = "high"
        else:
            severity = "critical"

        return overall_score, severity, confidence

    def _generate_recommendations(self, report: RegimeDriftReport) -> List[str]:
        """Generate actionable recommendations based on drift detection."""
        recommendations = []

        if report.regime_change_detected:
            recommendations.append(
                f"Regime changed from {report.baseline_regime} to {report.current_regime}. "
                "Consider retraining model with regime-specific data."
            )

        if report.drift_severity in ["high", "critical"]:
            recommendations.append(
                "Significant drift detected. Immediate model retraining recommended."
            )

        # Specific recommendations based on drift types
        for signal in report.distribution_drift:
            if signal.drift_detected:
                recommendations.append(
                    f"Distribution drift detected in {signal.signal_name}. "
                    "Consider updating normalization parameters."
                )

        for signal in report.concept_drift:
            if signal.drift_detected:
                recommendations.append(
                    f"Concept drift detected in {signal.signal_name}. "
                    "Pattern-label relationships have changed - model retraining needed."
                )

        for signal in report.temporal_drift:
            if signal.drift_detected:
                recommendations.append(
                    f"Temporal drift detected in {signal.signal_name}. "
                    "Consider updating time series preprocessing."
                )

        if not recommendations:
            recommendations.append("No significant drift detected. Continue monitoring.")

        return recommendations


def detect_market_regime_drift(
    baseline_data: Union[np.ndarray, pd.DataFrame],
    current_data: Union[np.ndarray, pd.DataFrame],
    baseline_labels: Optional[np.ndarray] = None,
    current_labels: Optional[np.ndarray] = None,
    sensitivity: str = "medium"
) -> RegimeDriftReport:
    """Convenience function for market regime drift detection.

    Args:
        baseline_data: Reference/baseline OHLC data
        current_data: Current/production OHLC data
        baseline_labels: Optional baseline pattern labels
        current_labels: Optional current pattern labels
        sensitivity: Detection sensitivity ("low", "medium", "high")

    Returns:
        Comprehensive drift detection report
    """
    detector = MarketRegimeDriftDetector(sensitivity=sensitivity)
    return detector.detect_market_regime_drift(
        baseline_data, current_data, baseline_labels, current_labels
    )