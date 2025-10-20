"""Optimized feature engineering for small dataset (98-sample) financial pattern recognition.

Specialized feature extraction that addresses the extreme small dataset challenge:
- Signal preservation: Minimizes signal dilution in multi-scale extraction
- Noise reduction: Robust features that work with limited samples
- Pattern specificity: Features optimized for consolidation/retracement/expansion patterns
- Statistical efficiency: Features with high signal-to-noise ratio
- Cross-validation stability: Features that perform consistently across folds

Key principles for small dataset feature engineering:
1. Focus on pattern-specific characteristics rather than generic technical indicators
2. Use relative features that normalize across different price levels
3. Emphasize geometric features that capture pattern morphology
4. Include context features that relate patterns to surrounding market structure
5. Avoid overfitting through feature selection and regularization
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from scipy import stats
from scipy.signal import find_peaks
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from loguru import logger


class FeatureCategory(str, Enum):
    """Categories of optimized features for small datasets."""
    PATTERN_MORPHOLOGY = "pattern_morphology"  # Shape and structure of patterns
    RELATIVE_DYNAMICS = "relative_dynamics"  # Pattern vs context relationships
    MARKET_MICROSTRUCTURE = "market_microstructure"  # Price action features
    GEOMETRIC_INVARIANTS = "geometric_invariants"  # Scale-invariant features
    TEMPORAL_SIGNATURES = "temporal_signatures"  # Time-based pattern characteristics


@dataclass
class SmallDatasetFeatureConfig:
    """Configuration optimized for small dataset feature engineering."""

    # Feature selection
    max_features_per_category: int = 5  # Limit features to avoid overfitting
    use_pattern_specific_features: bool = True
    use_context_aware_features: bool = True
    use_normalized_features: bool = True

    # Pattern analysis parameters
    pattern_window_start: int = 30
    pattern_window_end: int = 75
    context_window_size: int = 20

    # Feature engineering parameters
    robust_scaling: bool = True  # Use robust scaling for outlier resistance
    feature_selection_threshold: float = 0.1  # Minimum feature importance
    cross_validation_folds: int = 5  # For stability assessment

    # Noise reduction
    apply_smoothing: bool = True
    smoothing_window: int = 3
    outlier_removal: bool = True
    outlier_threshold: float = 3.0  # Z-score threshold

    # Geometric analysis
    use_fractal_features: bool = True
    use_curvature_features: bool = True
    use_symmetry_features: bool = True

    # Statistical features
    use_distribution_features: bool = True
    use_moment_features: bool = True
    use_quantile_features: bool = True


class SmallDatasetFeatureEngineer:
    """Specialized feature engineer for small dataset financial pattern recognition."""

    def __init__(self, config: Optional[SmallDatasetFeatureConfig] = None):
        """Initialize small dataset feature engineer.

        Args:
            config: Feature engineering configuration (uses defaults if None)
        """
        self.config = config or SmallDatasetFeatureConfig()
        self.feature_names: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self.scaler = RobustScaler() if self.config.robust_scaling else None

    def extract_features(
        self,
        X: np.ndarray,
        expansion_start: Optional[np.ndarray] = None,
        expansion_end: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Extract optimized features for small dataset pattern recognition.

        Args:
            X: OHLC data [N, 105, 4] or [N, 420]
            expansion_start: Pattern start indices [N]
            expansion_end: Pattern end indices [N]
            y: Optional labels for supervised feature selection

        Returns:
            Engineered features [N, F] with F optimized for small datasets
        """
        # Reshape if needed
        if X.ndim == 2 and X.shape[1] == 420:
            X = X.reshape(-1, 105, 4)

        N, T, F = X.shape
        logger.info(f"Extracting small dataset optimized features from {X.shape}")

        # Validate pattern indices
        if expansion_start is None or expansion_end is None:
            # Use default pattern window
            expansion_start = np.full(N, self.config.pattern_window_start)
            expansion_end = np.full(N, self.config.pattern_window_end - 1)

        all_features = []

        # 1. Pattern morphology features
        pattern_features = self._extract_pattern_morphology_features(
            X, expansion_start, expansion_end
        )
        all_features.append(pattern_features)

        # 2. Relative dynamics features
        relative_features = self._extract_relative_dynamics_features(
            X, expansion_start, expansion_end
        )
        all_features.append(relative_features)

        # 3. Market microstructure features
        microstructure_features = self._extract_market_microstructure_features(
            X, expansion_start, expansion_end
        )
        all_features.append(microstructure_features)

        # 4. Geometric invariant features
        geometric_features = self._extract_geometric_invariant_features(
            X, expansion_start, expansion_end
        )
        all_features.append(geometric_features)

        # 5. Temporal signature features
        temporal_features = self._extract_temporal_signature_features(
            X, expansion_start, expansion_end
        )
        all_features.append(temporal_features)

        # Concatenate all features
        feature_matrix = np.hstack(all_features)

        # Apply preprocessing
        feature_matrix = self._preprocess_features(feature_matrix)

        # Feature selection if labels are available
        if y is not None:
            feature_matrix = self._select_features(feature_matrix, y)

        logger.success(
            f"Feature extraction complete: {X.shape} -> {feature_matrix.shape} "
            f"(reduction ratio: {1 - feature_matrix.shape[1] / (T * F):.2%})"
        )

        return feature_matrix

    def _extract_pattern_morphology_features(
        self,
        X: np.ndarray,
        expansion_start: np.ndarray,
        expansion_end: np.ndarray
    ) -> np.ndarray:
        """Extract features describing the shape and structure of patterns."""
        N = X.shape[0]
        features = []

        for i in range(N):
            start = int(expansion_start[i])
            end = int(expansion_end[i])

            # Extract pattern region
            pattern = X[i, start:end+1, :]
            o, h, l, c = pattern[:, 0], pattern[:, 1], pattern[:, 2], pattern[:, 3]

            sample_features = []

            # 1. Pattern complexity (fractal dimension approximation)
            if len(c) > 2:
                # Simple fractal dimension using box-counting approximation
                price_range = h.max() - l.min()
                if price_range > 0:
                    # Normalize prices
                    norm_prices = (c - l.min()) / price_range

                    # Count "boxes" at different scales
                    scales = [2, 4, 8]
                    box_counts = []
                    for scale in scales:
                        n_boxes = int(np.ceil(len(norm_prices) / scale))
                        unique_boxes = len(set(
                            int(p * n_boxes) for p in norm_prices
                        ))
                        box_counts.append(unique_boxes)

                    # Estimate fractal dimension
                    if len(box_counts) >= 2 and len(scales) >= 2:
                        log_scales = np.log(scales[:len(box_counts)])
                        log_counts = np.log(box_counts)
                        if len(log_scales) > 1:
                            fractal_dim = -np.polyfit(log_scales, log_counts, 1)[0]
                            sample_features.append(fractal_dim)
                        else:
                            sample_features.append(1.0)  # Linear
                    else:
                        sample_features.append(1.0)
                else:
                    sample_features.append(1.0)
            else:
                sample_features.append(1.0)

            # 2. Pattern symmetry
            if len(c) > 1:
                # Mirror symmetry around pattern center
                center_idx = len(c) // 2
                left_half = c[:center_idx]
                right_half = c[center_idx:][::-1]  # Reverse right half

                # Pad shorter half
                min_len = min(len(left_half), len(right_half))
                if min_len > 0:
                    left_norm = (left_half[:min_len] - left_half[:min_len].mean()) / (left_half[:min_len].std() + 1e-8)
                    right_norm = (right_half[:min_len] - right_half[:min_len].mean()) / (right_half[:min_len].std() + 1e-8)
                    symmetry = 1 - np.mean(np.abs(left_norm - right_norm)) / 2
                    sample_features.append(np.clip(symmetry, 0, 1))
                else:
                    sample_features.append(0.5)
            else:
                sample_features.append(0.5)

            # 3. Pattern curvature
            if len(c) > 2:
                # Second derivative approximation
                first_deriv = np.diff(c)
                second_deriv = np.diff(first_deriv)
                curvature = np.mean(np.abs(second_deriv))
                sample_features.append(curvature)
            else:
                sample_features.append(0.0)

            # 4. Pattern directionality
            if len(c) > 1:
                # Net movement vs total movement
                net_change = c[-1] - c[0]
                total_movement = np.sum(np.abs(np.diff(c)))
                directionality = net_change / (total_movement + 1e-8)
                sample_features.append(directionality)
            else:
                sample_features.append(0.0)

            # 5. Pattern compactness
            pattern_range = h.max() - l.min()
            if pattern_range > 0:
                # How tightly prices stay within the range
                mean_deviation = np.mean(np.abs(c - c.mean()))
                compactness = 1 - (mean_deviation / pattern_range)
                sample_features.append(np.clip(compactness, 0, 1))
            else:
                sample_features.append(1.0)

            features.append(sample_features)

        # Limit features per category
        features_array = np.array(features)
        n_features = min(features_array.shape[1], self.config.max_features_per_category)
        if features_array.shape[1] > n_features:
            # Select features with highest variance (most informative)
            feature_variances = np.var(features_array, axis=0)
            top_indices = np.argsort(feature_variances)[-n_features:]
            features_array = features_array[:, top_indices]
            self.feature_names.extend([f"pattern_morphology_{i}" for i in top_indices])
        else:
            self.feature_names.extend([f"pattern_morphology_{i}" for i in range(features_array.shape[1])])

        return features_array

    def _extract_relative_dynamics_features(
        self,
        X: np.ndarray,
        expansion_start: np.ndarray,
        expansion_end: np.ndarray
    ) -> np.ndarray:
        """Extract features describing pattern dynamics relative to context."""
        N = X.shape[0]
        features = []

        for i in range(N):
            start = int(expansion_start[i])
            end = int(expansion_end[i])

            # Define context windows
            left_context_start = max(0, start - self.config.context_window_size)
            left_context = X[i, left_context_start:start, :]
            right_context_end = min(105, end + self.config.context_window_size + 1)
            right_context = X[i, end+1:right_context_end, :]

            pattern = X[i, start:end+1, :]

            sample_features = []

            # 1. Pattern vs context volatility ratio
            if len(pattern) > 1 and len(left_context) > 1:
                pattern_vol = np.std(pattern[:, 3]) / np.mean(pattern[:, 3])
                context_vol = np.std(left_context[:, 3]) / np.mean(left_context[:, 3])
                vol_ratio = pattern_vol / (context_vol + 1e-8)
                sample_features.append(np.log1p(vol_ratio))
            else:
                sample_features.append(0.0)

            # 2. Momentum continuity
            if len(left_context) > 0 and len(pattern) > 0:
                left_momentum = (left_context[-1, 3] - left_context[0, 3]) / (left_context[0, 3] + 1e-8)
                pattern_momentum = (pattern[-1, 3] - pattern[0, 3]) / (pattern[0, 3] + 1e-8)
                momentum_change = pattern_momentum - left_momentum
                sample_features.append(momentum_change)
            else:
                sample_features.append(0.0)

            # 3. Price level positioning
            if len(left_context) > 0 and len(pattern) > 0:
                context_range = np.max(left_context[:, 1]) - np.min(left_context[:, 2])
                if context_range > 0:
                    pattern_level = np.mean(pattern[:, 3])
                    context_level = np.mean(left_context[:, 3])
                    position = (pattern_level - context_level) / context_range
                    sample_features.append(position)
                else:
                    sample_features.append(0.0)
            else:
                sample_features.append(0.0)

            # 4. Range expansion ratio
            if len(left_context) > 0 and len(pattern) > 0:
                context_range = np.max(left_context[:, 1]) - np.min(left_context[:, 2])
                pattern_range = np.max(pattern[:, 1]) - np.min(pattern[:, 2])
                if context_range > 0:
                    expansion_ratio = pattern_range / context_range
                    sample_features.append(np.log1p(expansion_ratio))
                else:
                    sample_features.append(0.0)
            else:
                sample_features.append(0.0)

            # 5. Volume proxy change
            if len(left_context) > 0 and len(pattern) > 0:
                # Use range as volume proxy
                context_vol_proxy = np.mean(left_context[:, 1] - left_context[:, 2])
                pattern_vol_proxy = np.mean(pattern[:, 1] - pattern[:, 2])
                vol_change = (pattern_vol_proxy - context_vol_proxy) / (context_vol_proxy + 1e-8)
                sample_features.append(vol_change)
            else:
                sample_features.append(0.0)

            features.append(sample_features)

        # Limit features per category
        features_array = np.array(features)
        n_features = min(features_array.shape[1], self.config.max_features_per_category)
        if features_array.shape[1] > n_features:
            feature_variances = np.var(features_array, axis=0)
            top_indices = np.argsort(feature_variances)[-n_features:]
            features_array = features_array[:, top_indices]
            self.feature_names.extend([f"relative_dynamics_{i}" for i in top_indices])
        else:
            self.feature_names.extend([f"relative_dynamics_{i}" for i in range(features_array.shape[1])])

        return features_array

    def _extract_market_microstructure_features(
        self,
        X: np.ndarray,
        expansion_start: np.ndarray,
        expansion_end: np.ndarray
    ) -> np.ndarray:
        """Extract market microstructure features from OHLC data."""
        N = X.shape[0]
        features = []

        for i in range(N):
            start = int(expansion_start[i])
            end = int(expansion_end[i])
            pattern = X[i, start:end+1, :]
            o, h, l, c = pattern[:, 0], pattern[:, 1], pattern[:, 2], pattern[:, 3]

            sample_features = []

            # 1. Average body ratio
            if len(pattern) > 0:
                body_sizes = np.abs(c - o)
                ranges = h - l + 1e-8
                body_ratios = body_sizes / ranges
                sample_features.append(np.mean(body_ratios))
            else:
                sample_features.append(0.5)

            # 2. Upper wick dominance
            if len(pattern) > 0:
                upper_wicks = (h - np.maximum(o, c)) / (ranges + 1e-8)
                sample_features.append(np.mean(upper_wicks))
            else:
                sample_features.append(0.25)

            # 3. Lower wick dominance
            if len(pattern) > 0:
                lower_wicks = (np.minimum(o, c) - l) / (ranges + 1e-8)
                sample_features.append(np.mean(lower_wicks))
            else:
                sample_features.append(0.25)

            # 4. Price efficiency coefficient
            if len(pattern) > 1:
                # How much of the range is captured by open/close
                price_efficiency = np.abs(c - o) / (h - l + 1e-8)
                sample_features.append(np.mean(price_efficiency))
            else:
                sample_features.append(0.5)

            # 5. Gap frequency
            if len(pattern) > 1:
                # Count gaps between consecutive bars
                gaps_up = (l[1:] > h[:-1]).sum()
                gaps_down = (h[1:] < l[:-1]).sum()
                total_gaps = gaps_up + gaps_down
                gap_frequency = total_gaps / (len(pattern) - 1)
                sample_features.append(gap_frequency)
            else:
                sample_features.append(0.0)

            features.append(sample_features)

        # Limit features per category
        features_array = np.array(features)
        n_features = min(features_array.shape[1], self.config.max_features_per_category)
        if features_array.shape[1] > n_features:
            feature_variances = np.var(features_array, axis=0)
            top_indices = np.argsort(feature_variances)[-n_features:]
            features_array = features_array[:, top_indices]
            self.feature_names.extend([f"microstructure_{i}" for i in top_indices])
        else:
            self.feature_names.extend([f"microstructure_{i}" for i in range(features_array.shape[1])])

        return features_array

    def _extract_geometric_invariant_features(
        self,
        X: np.ndarray,
        expansion_start: np.ndarray,
        expansion_end: np.ndarray
    ) -> np.ndarray:
        """Extract scale-invariant geometric features."""
        N = X.shape[0]
        features = []

        for i in range(N):
            start = int(expansion_start[i])
            end = int(expansion_end[i])
            pattern = X[i, start:end+1, :]
            o, h, l, c = pattern[:, 0], pattern[:, 1], pattern[:, 2], pattern[:, 3]

            sample_features = []

            # 1. Normalized price path length
            if len(c) > 1:
                # Total path length normalized by direct distance
                path_length = np.sum(np.abs(np.diff(c)))
                direct_distance = abs(c[-1] - c[0])
                if direct_distance > 0:
                    path_ratio = path_length / direct_distance
                    sample_features.append(np.log1p(path_ratio))
                else:
                    sample_features.append(0.0)
            else:
                sample_features.append(0.0)

            # 2. Hurst exponent approximation
            if len(c) > 4:
                # Simple Hurst exponent estimation
                log_ranges = []
                log_scales = []

                for scale in [2, 4, max(1, len(c)//2)]:
                    if scale < len(c):
                        # Rescaled range analysis
                        scaled_data = c[:len(c)//scale * scale].reshape(-1, scale)
                        ranges = scaled_data.max(axis=1) - scaled_data.min(axis=1)
                        mean_range = np.mean(ranges)

                        log_ranges.append(np.log(mean_range))
                        log_scales.append(np.log(scale))

                if len(log_ranges) >= 2:
                    hurst = np.polyfit(log_scales, log_ranges, 1)[0]
                    sample_features.append(hurst)
                else:
                    sample_features.append(0.5)  # Random walk
            else:
                sample_features.append(0.5)

            # 3. Shape regularity
            if len(c) > 2:
                # Compare local slopes
                slopes = np.diff(c)
                slope_changes = np.diff(slopes)
                regularity = 1 - (np.std(slope_changes) / (np.std(slopes) + 1e-8))
                sample_features.append(np.clip(regularity, 0, 1))
            else:
                sample_features.append(0.5)

            # 4. Turning point density
            if len(c) > 2:
                # Count local extrema
                peaks, _ = find_peaks(c)
                troughs, _ = find_peaks(-c)
                turning_points = len(peaks) + len(troughs)
                density = turning_points / len(c)
                sample_features.append(density)
            else:
                sample_features.append(0.0)

            # 5. Aspect ratio
            if len(c) > 0:
                time_span = len(c)
                price_span = h.max() - l.min()
                if price_span > 0:
                    aspect_ratio = time_span / price_span
                    sample_features.append(np.log1p(aspect_ratio))
                else:
                    sample_features.append(0.0)
            else:
                sample_features.append(0.0)

            features.append(sample_features)

        # Limit features per category
        features_array = np.array(features)
        n_features = min(features_array.shape[1], self.config.max_features_per_category)
        if features_array.shape[1] > n_features:
            feature_variances = np.var(features_array, axis=0)
            top_indices = np.argsort(feature_variances)[-n_features:]
            features_array = features_array[:, top_indices]
            self.feature_names.extend([f"geometric_invariant_{i}" for i in top_indices])
        else:
            self.feature_names.extend([f"geometric_invariant_{i}" for i in range(features_array.shape[1])])

        return features_array

    def _extract_temporal_signature_features(
        self,
        X: np.ndarray,
        expansion_start: np.ndarray,
        expansion_end: np.ndarray
    ) -> np.ndarray:
        """Extract temporal signature features."""
        N = X.shape[0]
        features = []

        for i in range(N):
            start = int(expansion_start[i])
            end = int(expansion_end[i])
            pattern = X[i, start:end+1, :]
            c = pattern[:, 3]

            sample_features = []

            # 1. Autocorrelation at different lags
            if len(c) > 10:
                autocorr_1 = np.corrcoef(c[:-1], c[1:])[0, 1] if not np.isnan(np.corrcoef(c[:-1], c[1:])[0, 1]) else 0
                autocorr_5 = np.corrcoef(c[:-5], c[5:])[0, 1] if len(c) > 5 and not np.isnan(np.corrcoef(c[:-5], c[5:])[0, 1]) else 0
                sample_features.extend([autocorr_1, autocorr_5])
            else:
                sample_features.extend([0.0, 0.0])

            # 2. Partial autocorrelation (simplified)
            if len(c) > 5:
                # Simple approximation using lag correlations
                pacf_1 = np.corrcoef(c[:-1], c[1:])[0, 1]
                if len(c) > 2:
                    pacf_2 = np.corrcoef(c[:-2], c[2:])[0, 1]
                    pacf_2 = pacf_2 - pacf_1**2  # Remove first-order effect
                else:
                    pacf_2 = 0.0

                sample_features.extend([pacf_1 if not np.isnan(pacf_1) else 0,
                                       pacf_2 if not np.isnan(pacf_2) else 0])
            else:
                sample_features.extend([0.0, 0.0])

            # 3. Periodicity strength
            if len(c) > 6:
                # Simple periodicity detection using FFT
                fft_vals = np.fft.fft(c - c.mean())
                power_spectrum = np.abs(fft_vals) ** 2

                # Dominant frequency strength
                dominant_power = np.max(power_spectrum[1:len(power_spectrum)//2])
                total_power = np.sum(power_spectrum[1:len(power_spectrum)//2])

                if total_power > 0:
                    periodicity = dominant_power / total_power
                else:
                    periodicity = 0.0

                sample_features.append(periodicity)
            else:
                sample_features.append(0.0)

            features.append(sample_features)

        # Limit features per category
        features_array = np.array(features)
        n_features = min(features_array.shape[1], self.config.max_features_per_category)
        if features_array.shape[1] > n_features:
            feature_variances = np.var(features_array, axis=0)
            top_indices = np.argsort(feature_variances)[-n_features:]
            features_array = features_array[:, top_indices]
            self.feature_names.extend([f"temporal_signature_{i}" for i in top_indices])
        else:
            self.feature_names.extend([f"temporal_signature_{i}" for i in range(features_array.shape[1])])

        return features_array

    def _preprocess_features(self, features: np.ndarray) -> np.ndarray:
        """Preprocess extracted features."""
        # Handle NaN and infinite values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)

        # Remove outliers if enabled
        if self.config.outlier_removal:
            z_scores = np.abs(stats.zscore(features, axis=0))
            outlier_mask = z_scores > self.config.outlier_threshold
            features[outlier_mask] = 0.0

        # Apply smoothing if enabled
        if self.config.apply_smoothing and self.config.smoothing_window > 1:
            # Apply smoothing across samples (assuming similar samples are nearby)
            from scipy.ndimage import uniform_filter1d
            features = uniform_filter1d(features, size=self.config.smoothing_window, axis=0)

        # Apply robust scaling if enabled
        if self.scaler is not None:
            features = self.scaler.fit_transform(features)

        return features

    def _select_features(self, features: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Select most informative features using mutual information."""
        try:
            # Compute mutual information scores
            mi_scores = mutual_info_classif(features, y, random_state=42)

            # Select features above threshold
            selected_mask = mi_scores >= self.config.feature_selection_threshold
            selected_features = features[:, selected_mask]

            # Update feature names
            selected_names = [name for name, mask in zip(self.feature_names, selected_mask) if mask]
            self.feature_names = selected_names

            # Store feature importance
            self.feature_importance = dict(zip(selected_names, mi_scores[selected_mask]))

            logger.info(f"Feature selection: {features.shape[1]} -> {selected_features.shape[1]} features")

            return selected_features

        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")
            return features

    def get_feature_names(self) -> List[str]:
        """Get the names of extracted features."""
        return self.feature_names.copy()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self.feature_importance.copy()


def create_small_dataset_feature_engineer(
    max_features_per_category: int = 5,
    robust_scaling: bool = True,
    feature_selection: bool = True
) -> SmallDatasetFeatureEngineer:
    """Create a feature engineer optimized for small datasets.

    Args:
        max_features_per_category: Maximum features per category to avoid overfitting
        robust_scaling: Use robust scaling for outlier resistance
        feature_selection: Enable supervised feature selection

    Returns:
        Configured small dataset feature engineer
    """
    config = SmallDatasetFeatureConfig(
        max_features_per_category=max_features_per_category,
        robust_scaling=robust_scaling,
        feature_selection_threshold=0.1 if feature_selection else 0.0
    )

    return SmallDatasetFeatureEngineer(config)


def extract_optimized_features(
    X: np.ndarray,
    expansion_start: Optional[np.ndarray] = None,
    expansion_end: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    max_total_features: int = 25
) -> Tuple[np.ndarray, List[str]]:
    """Convenience function for optimized feature extraction.

    Args:
        X: OHLC data [N, 105, 4] or [N, 420]
        expansion_start: Pattern start indices [N]
        expansion_end: Pattern end indices [N]
        y: Optional labels for supervised feature selection
        max_total_features: Maximum total features to avoid overfitting

    Returns:
        Tuple of (features, feature_names)
    """
    # Create engineer with conservative settings for small datasets
    engineer = create_small_dataset_feature_engineer(
        max_features_per_category=max_total_features // 5,  # Distribute across 5 categories
        robust_scaling=True,
        feature_selection=True
    )

    # Extract features
    features = engineer.extract_features(X, expansion_start, expansion_end, y)
    feature_names = engineer.get_feature_names()

    # Limit total features if needed
    if features.shape[1] > max_total_features:
        # Select features with highest importance
        importance = engineer.get_feature_importance()
        if importance:
            sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_features = sorted_features[:max_total_features]
            top_indices = [feature_names.index(name) for name, _ in top_features]
            features = features[:, top_indices]
            feature_names = [name for name, _ in top_features]
        else:
            # Fallback: select first N features
            features = features[:, :max_total_features]
            feature_names = feature_names[:max_total_features]

    return features, feature_names