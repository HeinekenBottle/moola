"""Advanced financial time series pseudo-sample generation strategies.

This module implements comprehensive augmentation and synthesis methods specifically
designed for OHLC financial data while maintaining market realism and statistical
properties. It focuses on generating high-quality pseudo-samples for small datasets
with binary classification tasks (consolidation vs retracement).

Key strategies implemented:
1. Temporal augmentation with market microstructure preservation
2. Pattern-based synthesis with realistic price action
3. Statistical simulation with regime-specific dynamics
4. Self-supervised pseudo-labeling using pre-trained encoders
5. Market condition simulation across volatility regimes

References:
- Financial Time Series Generation: Jaganathan et al., "Deep Generative Modeling for Financial Time Series"
- Market Microstructure: O'Hara, "Market Microstructure Theory"
- Augmentation for Time Series: Fawaz et al., "Data augmentation using time series generative adversarial networks"
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class MarketRegime:
    """Market regime parameters for realistic simulation."""
    volatility_level: float  # Volatility multiplier (0.5-3.0)
    trend_strength: float    # Trend bias (-1 to 1)
    noise_ratio: float      # Signal-to-noise ratio (0.1-1.0)
    gap_probability: float  # Probability of gaps/jumps (0.01-0.2)
    mean_reversion_speed: float  # Speed of mean reversion (0.01-0.5)


class BasePseudoGenerator(ABC):
    """Abstract base class for pseudo-sample generators."""

    def __init__(self, seed: int = 1337):
        """Initialize generator with random seed.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

    @abstractmethod
    def generate(self, data: np.ndarray, labels: np.ndarray,
                n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pseudo-samples from original data.

        Args:
            data: Original OHLC data [N, T, 4]
            labels: Original labels [N]
            n_samples: Number of pseudo-samples to generate

        Returns:
            Tuple of (generated_data, generated_labels)
        """
        pass

    @abstractmethod
    def validate_quality(self, original: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
        """Validate quality of generated samples.

        Args:
            original: Original data samples
            generated: Generated data samples

        Returns:
            Dictionary of quality metrics
        """
        pass


class TemporalAugmentationGenerator(BasePseudoGenerator):
    """Temporal augmentation preserving market microstructure and OHLC relationships."""

    def __init__(self, seed: int = 1337,
                 time_warp_std: float = 0.1,
                 magnitude_warp_std: float = 0.1,
                 permutation_segments: int = 4):
        """Initialize temporal augmentation generator.

        Args:
            seed: Random seed
            time_warp_std: Standard deviation for time warping
            magnitude_warp_std: Standard deviation for magnitude warping
            permutation_segments: Number of segments for permutation
        """
        super().__init__(seed)
        self.time_warp_std = time_warp_std
        self.magnitude_warp_std = magnitude_warp_std
        self.permutation_segments = permutation_segments

    def _validate_ohlc(self, ohlc: np.ndarray) -> np.ndarray:
        """Ensure OHLC relationships are maintained (O <= H >= L <= C).

        Args:
            ohlc: OHLC array [T, 4]

        Returns:
            Validated OHLC with corrected relationships
        """
        corrected = ohlc.copy()

        for t in range(corrected.shape[0]):
            o, h, l, c = corrected[t]

            # Ensure O <= H and L <= H
            corrected[t, 1] = max(h, o, c)  # High is max of all
            # Ensure O >= L and C >= L
            corrected[t, 2] = min(l, o, c)  # Low is min of all
            # Ensure high >= low
            if corrected[t, 1] < corrected[t, 2]:
                corrected[t, 1] = corrected[t, 2] + 1e-6

        return corrected

    def _time_warp(self, sequence: np.ndarray) -> np.ndarray:
        """Apply time warping to sequence while preserving OHLC relationships.

        Args:
            sequence: Input sequence [T, 4]

        Returns:
            Time-warped sequence
        """
        seq_len = sequence.shape[0]

        # Generate smooth warping path
        warp_points = np.random.normal(1.0, self.time_warp_std, size=5)
        warp_points = np.clip(warp_points, 0.5, 1.5)

        # Create smooth interpolation
        original_indices = np.linspace(0, 1, len(warp_points))
        warp_path = interp1d(original_indices, warp_points, kind='cubic')

        # Generate warped indices
        new_indices = warp_path(np.linspace(0, 1, seq_len))
        new_indices = np.clip(new_indices * (seq_len - 1), 0, seq_len - 1)

        # Interpolate each OHLC component
        warped = np.zeros_like(sequence)
        for i in range(4):
            warped[:, i] = np.interp(new_indices, np.arange(seq_len), sequence[:, i])

        return self._validate_ohlc(warped)

    def _magnitude_warp(self, sequence: np.ndarray) -> np.ndarray:
        """Apply magnitude warping with realistic market dynamics.

        Args:
            sequence: Input sequence [T, 4]

        Returns:
            Magnitude-warped sequence
        """
        seq_len = sequence.shape[0]

        # Generate smooth warping curve
        warp_curve = np.random.normal(1.0, self.magnitude_warp_std, size=seq_len)
        # Smooth the curve to avoid unrealistic jumps
        from scipy.ndimage import gaussian_filter1d
        warp_curve = gaussian_filter1d(warp_curve, sigma=2)
        warp_curve = np.clip(warp_curve, 0.7, 1.3)  # Reasonable magnitude bounds

        # Apply warping centered around sequence mean
        seq_mean = np.mean(sequence)
        warped = (sequence - seq_mean) * warp_curve[:, np.newaxis] + seq_mean

        return self._validate_ohlc(warped)

    def _window_warping(self, sequence: np.ndarray) -> np.ndarray:
        """Apply window-based warping for localized distortions.

        Args:
            sequence: Input sequence [T, 4]

        Returns:
            Window-warped sequence
        """
        seq_len = sequence.shape[0]

        # Select random window to warp
        window_size = np.random.randint(seq_len // 4, seq_len // 2)
        window_start = np.random.randint(0, seq_len - window_size)

        warped = sequence.copy()
        window = warped[window_start:window_start + window_size]

        # Apply compression or expansion
        warp_factor = np.random.choice([0.8, 1.2])
        if warp_factor < 1:  # Compression
            new_length = int(window_size * warp_factor)
            indices = np.linspace(0, window_size - 1, new_length)
            for i in range(4):
                warped[window_start:window_start + new_length, i] = np.interp(
                    indices, np.arange(window_size), window[:, i]
                )
        else:  # Expansion
            new_length = int(window_size * warp_factor)
            if window_start + new_length <= seq_len:
                indices = np.linspace(0, window_size - 1, new_length)
                for i in range(4):
                    warped[window_start:window_start + new_length, i] = np.interp(
                        indices, np.arange(window_size), window[:, i]
                    )

        return self._validate_ohlc(warped)

    def generate(self, data: np.ndarray, labels: np.ndarray,
                n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate temporally augmented pseudo-samples.

        Args:
            data: Original OHLC data [N, T, 4]
            labels: Original labels [N]
            n_samples: Number of samples to generate

        Returns:
            Tuple of (generated_data, generated_labels)
        """
        generated_data = []
        generated_labels = []

        for _ in range(n_samples):
            # Randomly select base sample
            idx = np.random.randint(0, len(data))
            sample = data[idx].copy()
            label = labels[idx]

            # Apply random combination of augmentations
            augmentation_type = np.random.choice(['time_warp', 'magnitude_warp',
                                               'window_warping', 'combined'])

            if augmentation_type == 'time_warp':
                sample = self._time_warp(sample)
            elif augmentation_type == 'magnitude_warp':
                sample = self._magnitude_warp(sample)
            elif augmentation_type == 'window_warping':
                sample = self._window_warping(sample)
            else:  # combined
                sample = self._time_warp(sample)
                sample = self._magnitude_warp(sample)

            generated_data.append(sample)
            generated_labels.append(label)

        return np.array(generated_data), np.array(generated_labels)

    def validate_quality(self, original: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
        """Validate temporal augmentation quality.

        Args:
            original: Original samples [N, T, 4]
            generated: Generated samples [N, T, 4]

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Statistical similarity
        for i, component in enumerate(['open', 'high', 'low', 'close']):
            orig_mean = np.mean(original[:, :, i])
            gen_mean = np.mean(generated[:, :, i])
            orig_std = np.std(original[:, :, i])
            gen_std = np.std(generated[:, :, i])

            metrics[f'{component}_mean_ratio'] = gen_mean / (orig_mean + 1e-8)
            metrics[f'{component}_std_ratio'] = gen_std / (orig_std + 1e-8)

        # Temporal correlation preservation
        orig_autocorr = [np.correlate(original[j, :, 3], original[j, :, 3], mode='full')[len(original[j, :, 3])//2]
                        for j in range(min(10, len(original)))]
        gen_autocorr = [np.correlate(generated[j, :, 3], generated[j, :, 3], mode='full')[len(generated[j, :, 3])//2]
                       for j in range(min(10, len(generated)))]

        metrics['autocorr_preservation'] = np.mean(gen_autocorr) / (np.mean(orig_autocorr) + 1e-8)

        # OHLC relationship preservation
        ohlc_violations = 0
        total_checks = len(generated) * generated.shape[1]
        for sample in generated:
            for t in range(sample.shape[0]):
                o, h, l, c = sample[t]
                if not (o <= h and l <= h and o >= l and c >= l and h >= l):
                    ohlc_violations += 1

        metrics['ohlc_preservation_rate'] = 1.0 - (ohlc_violations / total_checks)

        return metrics


class PatternBasedSynthesisGenerator(BasePseudoGenerator):
    """Pattern-based synthesis creating realistic variations of existing patterns."""

    def __init__(self, seed: int = 1337,
                 pattern_variation_strength: float = 0.15,
                 noise_level: float = 0.02,
                 preserve_trend: bool = True):
        """Initialize pattern-based synthesis generator.

        Args:
            seed: Random seed
            pattern_variation_strength: Strength of pattern variations
            noise_level: Level of realistic market noise
            preserve_trend: Whether to preserve overall trend
        """
        super().__init__(seed)
        self.pattern_variation_strength = pattern_variation_strength
        self.noise_level = noise_level
        self.preserve_trend = preserve_trend

    def _extract_pattern_components(self, sequence: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract trend, seasonal, and noise components from sequence.

        Args:
            sequence: Input sequence [T, 4]

        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        components = {}

        for i, name in enumerate(['open', 'high', 'low', 'close']):
            series = sequence[:, i]

            # Extract trend using moving average
            trend = pd.Series(series).rolling(window=20, min_periods=1, center=True).mean().values

            # Extract detrended series
            detrended = series - trend

            # Store components
            components[f'{name}_trend'] = trend
            components[f'{name}_detrended'] = detrended

        return components

    def _modify_pattern(self, sequence: np.ndarray, variation_strength: float) -> np.ndarray:
        """Apply realistic pattern modifications while preserving key characteristics.

        Args:
            sequence: Input sequence [T, 4]
            variation_strength: Strength of pattern modification

        Returns:
            Modified sequence
        """
        modified = sequence.copy()

        # Extract pattern components
        components = self._extract_pattern_components(sequence)

        # Modify each component realistically
        for i, name in enumerate(['open', 'high', 'low', 'close']):
            trend = components[f'{name}_trend']
            detrended = components[f'{name}_detrended']

            if self.preserve_trend:
                # Keep trend, modify pattern
                # Apply smooth variations to detrended component
                from scipy.ndimage import gaussian_filter1d
                variation = np.random.normal(0, variation_strength * np.std(detrended), len(detrended))
                smooth_variation = gaussian_filter1d(variation, sigma=3)

                new_detrended = detrended + smooth_variation
                modified[:, i] = trend + new_detrended
            else:
                # Modify both trend and pattern
                trend_variation = np.random.normal(0, variation_strength * np.std(trend), len(trend))
                smooth_trend_var = gaussian_filter1d(trend_variation, sigma=5)

                pattern_variation = np.random.normal(0, variation_strength * np.std(detrended), len(detrended))
                smooth_pattern_var = gaussian_filter1d(pattern_variation, sigma=2)

                modified[:, i] = trend + smooth_trend_var + detrended + smooth_pattern_var

        # Add realistic market noise
        noise = np.random.normal(0, self.noise_level * np.std(modified), modified.shape)
        modified = modified + noise

        # Ensure OHLC relationships
        return self._ensure_ohlc_relationships(modified)

    def _pattern_morphing(self, seq1: np.ndarray, seq2: np.ndarray,
                         morph_factor: float) -> np.ndarray:
        """Morph between two patterns smoothly.

        Args:
            seq1: First pattern [T, 4]
            seq2: Second pattern [T, 4]
            morph_factor: Morphing factor (0-1, 0=seq1, 1=seq2)

        Returns:
            Morphed pattern
        """
        # Normalize both sequences to same scale
        seq1_norm = (seq1 - np.mean(seq1)) / (np.std(seq1) + 1e-8)
        seq2_norm = (seq2 - np.mean(seq2)) / (np.std(seq2) + 1e-8)

        # Smooth morphing with time-varying factor
        seq_len = seq1.shape[0]
        time_varying_factor = np.linspace(morph_factor * 0.8, morph_factor * 1.2, seq_len)
        time_varying_factor = np.clip(time_varying_factor, 0, 1)[:, np.newaxis]

        morphed = (1 - time_varying_factor) * seq1_norm + time_varying_factor * seq2_norm

        # Rescale back to original range
        morphed = morphed * np.std(seq1) + np.mean(seq1)

        return self._ensure_ohlc_relationships(morphed)

    def _ensure_ohlc_relationships(self, ohlc: np.ndarray) -> np.ndarray:
        """Ensure proper OHLC relationships in generated data.

        Args:
            ohlc: OHLC array [T, 4]

        Returns:
            OHLC with proper relationships
        """
        corrected = ohlc.copy()

        for t in range(corrected.shape[0]):
            o, h, l, c = corrected[t]

            # Calculate reasonable corrections
            price_level = (o + h + l + c) / 4

            # Ensure reasonable relationships
            corrected[t, 0] = o  # Keep open as-is
            corrected[t, 1] = max(h, o, c, price_level * 0.999)  # High must be >= others
            corrected[t, 2] = min(l, o, c, price_level * 1.001)  # Low must be <= others
            corrected[t, 3] = c  # Keep close as-is

            # Final sanity check
            if corrected[t, 1] < corrected[t, 2]:
                # Swap if necessary
                corrected[t, 1], corrected[t, 2] = corrected[t, 2], corrected[t, 1]
                corrected[t, 1] += 1e-6  # Small buffer

        return corrected

    def generate(self, data: np.ndarray, labels: np.ndarray,
                n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pattern-based pseudo-samples.

        Args:
            data: Original OHLC data [N, T, 4]
            labels: Original labels [N]
            n_samples: Number of samples to generate

        Returns:
            Tuple of (generated_data, generated_labels)
        """
        generated_data = []
        generated_labels = []

        for _ in range(n_samples):
            generation_method = np.random.choice(['single_modify', 'pattern_morph', 'hybrid'])

            if generation_method == 'single_modify':
                # Modify single pattern
                idx = np.random.randint(0, len(data))
                sample = data[idx].copy()
                label = labels[idx]

                # Apply pattern modification
                strength = np.random.uniform(0.05, self.pattern_variation_strength)
                modified = self._modify_pattern(sample, strength)
                generated_data.append(modified)
                generated_labels.append(label)

            elif generation_method == 'pattern_morph':
                # Morph between two patterns of same class
                class_samples = np.where(labels == labels[np.random.randint(0, len(labels))])[0]
                if len(class_samples) >= 2:
                    idx1, idx2 = np.random.choice(class_samples, 2, replace=False)
                    sample1 = data[idx1].copy()
                    sample2 = data[idx2].copy()
                    label = labels[idx1]

                    morph_factor = np.random.uniform(0.2, 0.8)
                    morphed = self._pattern_morphing(sample1, sample2, morph_factor)
                    generated_data.append(morphed)
                    generated_labels.append(label)
                else:
                    # Fallback to single modification
                    idx = np.random.randint(0, len(data))
                    sample = data[idx].copy()
                    modified = self._modify_pattern(sample, self.pattern_variation_strength)
                    generated_data.append(modified)
                    generated_labels.append(labels[idx])

            else:  # hybrid
                # Combine modification and morphing
                idx = np.random.randint(0, len(data))
                sample = data[idx].copy()
                label = labels[idx]

                # First modify
                modified = self._modify_pattern(sample, self.pattern_variation_strength * 0.5)

                # Then find another sample and partially morph
                class_samples = np.where(labels == label)[0]
                if len(class_samples) >= 2:
                    idx2 = np.random.choice(class_samples)
                    sample2 = data[idx2].copy()
                    morph_factor = np.random.uniform(0.1, 0.3)
                    modified = self._pattern_morphing(modified, sample2, morph_factor)

                generated_data.append(modified)
                generated_labels.append(label)

        return np.array(generated_data), np.array(generated_labels)

    def validate_quality(self, original: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
        """Validate pattern-based synthesis quality.

        Args:
            original: Original samples [N, T, 4]
            generated: Generated samples [N, T, 4]

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Distribution similarity for each component
        for i, component in enumerate(['open', 'high', 'low', 'close']):
            orig_data = original[:, :, i].flatten()
            gen_data = generated[:, :, i].flatten()

            # Kolmogorov-Smirnov test for distribution similarity
            ks_statistic = stats.ks_2samp(orig_data, gen_data).statistic
            metrics[f'{component}_ks_similarity'] = 1.0 - ks_statistic

            # Moments comparison
            orig_mean, orig_std = np.mean(orig_data), np.std(orig_data)
            gen_mean, gen_std = np.mean(gen_data), np.std(gen_data)

            metrics[f'{component}_mean_error'] = abs(gen_mean - orig_mean) / (abs(orig_mean) + 1e-8)
            metrics[f'{component}_std_error'] = abs(gen_std - orig_std) / (orig_std + 1e-8)

        # Pattern similarity using Dynamic Time Warping distance
        def dtw_distance(s1, s2):
            """Simple DTW implementation for pattern similarity."""
            n, m = len(s1), len(s2)
            dtw = np.full((n + 1, m + 1), np.inf)
            dtw[0, 0] = 0

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(s1[i-1] - s2[j-1])
                    dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

            return dtw[n, m]

        # Calculate average DTW distance for close prices
        dtw_distances = []
        n_samples = min(10, len(original), len(generated))
        for i in range(n_samples):
            dist = dtw_distance(original[i, :, 3], generated[i, :, 3])
            dtw_distances.append(dist)

        metrics['avg_dtw_similarity'] = 1.0 / (1.0 + np.mean(dtw_distances))

        # OHLC relationship preservation
        ohlc_violations = 0
        total_checks = len(generated) * generated.shape[1]
        for sample in generated:
            for t in range(sample.shape[0]):
                o, h, l, c = sample[t]
                if not (o <= h and l <= h and o >= l and c >= l):
                    ohlc_violations += 1

        metrics['ohlc_preservation_rate'] = 1.0 - (ohlc_violations / total_checks)

        return metrics


class StatisticalSimulationGenerator(BasePseudoGenerator):
    """Statistical simulation generating samples matching real data distributions."""

    def __init__(self, seed: int = 1337,
                 use_gaussian_process: bool = True,
                 n_regimes: int = 3,
                 regime_detection_window: int = 20):
        """Initialize statistical simulation generator.

        Args:
            seed: Random seed
            use_gaussian_process: Whether to use Gaussian Process for simulation
            n_regimes: Number of market regimes to detect
            regime_detection_window: Window for regime detection
        """
        super().__init__(seed)
        self.use_gaussian_process = use_gaussian_process
        self.n_regimes = n_regimes
        self.regime_detection_window = regime_detection_window
        self.regimes = {}
        self.regime_parameters = {}

    def _detect_market_regimes(self, data: np.ndarray) -> Dict[int, List[int]]:
        """Detect market regimes using volatility clustering.

        Args:
            data: OHLC data [N, T, 4]

        Returns:
            Dictionary mapping regime_id to list of sample indices
        """
        # Calculate volatility for each sample
        volatilities = []
        for i in range(len(data)):
            returns = np.diff(data[i, :, 3]) / (data[i, :-1, 3] + 1e-8)
            volatility = np.std(returns)
            volatilities.append(volatility)

        volatilities = np.array(volatilities)

        # Use K-means clustering on log volatility
        from sklearn.cluster import KMeans

        log_vol = np.log(volatilities + 1e-8)
        log_vol = log_vol.reshape(-1, 1)

        kmeans = KMeans(n_clusters=self.n_regimes, random_state=self.seed)
        regime_labels = kmeans.fit_predict(log_vol)

        # Group samples by regime
        regimes = {}
        for i, label in enumerate(regime_labels):
            if label not in regimes:
                regimes[label] = []
            regimes[label].append(i)

        return regimes

    def _estimate_regime_parameters(self, data: np.ndarray, labels: np.ndarray,
                                  regime_samples: List[int]) -> MarketRegime:
        """Estimate parameters for a specific market regime.

        Args:
            data: OHLC data [N, T, 4]
            labels: Class labels [N]
            regime_samples: List of sample indices for this regime

        Returns:
            MarketRegime parameters
        """
        regime_data = data[regime_samples]
        regime_labels = labels[regime_samples]

        # Calculate volatility level
        all_returns = []
        for sample in regime_data:
            returns = np.diff(sample[:, 3]) / (sample[:-1, 3] + 1e-8)
            all_returns.extend(returns)

        volatility_level = np.std(all_returns)
        volatility_level = np.clip(volatility_level / np.std(all_returns), 0.5, 3.0)

        # Calculate trend strength (bias in returns)
        mean_return = np.mean(all_returns)
        trend_strength = np.clip(mean_return * 100, -1.0, 1.0)  # Scale to [-1, 1]

        # Calculate noise ratio
        signal_levels = []
        for sample in regime_data:
            # Use low-frequency component as signal
            signal = pd.Series(sample[:, 3]).rolling(window=20, min_periods=1).mean()
            signal_levels.append(np.std(signal))

        noise_ratio = np.mean(signal_levels) / (np.std(all_returns) + 1e-8)
        noise_ratio = np.clip(noise_ratio, 0.1, 1.0)

        # Calculate gap probability (large price jumps)
        large_jumps = [abs(r) for r in all_returns if abs(r) > 2 * np.std(all_returns)]
        gap_probability = len(large_jumps) / len(all_returns)
        gap_probability = np.clip(gap_probability, 0.01, 0.2)

        # Calculate mean reversion speed
        autocorr_1 = [np.corrcoef(sample[:-1, 3], sample[1:, 3])[0, 1]
                     for sample in regime_data if len(sample) > 1]
        mean_reversion_speed = 1.0 - np.mean(np.abs(autocorr_1))
        mean_reversion_speed = np.clip(mean_reversion_speed, 0.01, 0.5)

        return MarketRegime(
            volatility_level=volatility_level,
            trend_strength=trend_strength,
            noise_ratio=noise_ratio,
            gap_probability=gap_probability,
            mean_reversion_speed=mean_reversion_speed
        )

    def _simulate_geometric_brownian_motion(self, n_steps: int,
                                          regime: MarketRegime,
                                          initial_price: float = 100.0) -> np.ndarray:
        """Simulate price path using Geometric Brownian Motion with regime parameters.

        Args:
            n_steps: Number of time steps
            regime: Market regime parameters
            initial_price: Starting price

        Returns:
            Simulated price series
        """
        dt = 1.0  # Daily steps

        # Calculate drift and volatility from regime parameters
        drift = regime.trend_strength * 0.001  # Scale down for daily returns
        volatility = regime.volatility_level * 0.02  # Scale to reasonable daily volatility

        # Generate random shocks
        shocks = np.random.normal(0, 1, n_steps)

        # Add occasional jumps based on gap probability
        for i in range(1, n_steps):
            if np.random.random() < regime.gap_probability:
                jump_size = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.05)
                shocks[i] += jump_size / volatility

        # Generate price path
        prices = [initial_price]
        for i in range(1, n_steps):
            price_prev = prices[-1]

            # GBM formula: S_t = S_{t-1} * exp((drift - 0.5*vol^2)*dt + vol*sqrt(dt)*shock)
            price_new = price_prev * np.exp((drift - 0.5 * volatility**2) * dt +
                                          volatility * np.sqrt(dt) * shocks[i])
            prices.append(price_new)

        return np.array(prices)

    def _create_ohlc_from_prices(self, prices: np.ndarray) -> np.ndarray:
        """Create realistic OHLC data from price series.

        Args:
            prices: Close price series

        Returns:
            OHLC array [T, 4]
        """
        n_steps = len(prices)
        ohlc = np.zeros((n_steps, 4))

        for i in range(n_steps):
            close = prices[i]

            # Generate realistic intraday variation
            daily_volatility = 0.001 * close  # 0.1% typical intraday variation

            # Open price (near previous close with gap)
            if i == 0:
                open_price = close
            else:
                gap = np.random.normal(0, daily_volatility * 0.5)
                open_price = max(1e-6, prices[i-1] + gap)

            # High and low prices
            high_low_range = np.random.uniform(0.0005, 0.002) * close

            if close >= open_price:
                # Up day
                high = max(open_price, close) + np.random.uniform(0, high_low_range)
                low = min(open_price, close) - np.random.uniform(0, high_low_range * 0.5)
            else:
                # Down day
                high = max(open_price, close) + np.random.uniform(0, high_low_range * 0.5)
                low = min(open_price, close) - np.random.uniform(0, high_low_range)

            # Ensure positive prices
            high = max(high, 1e-6)
            low = max(low, 1e-6)
            open_price = max(open_price, 1e-6)
            close = max(close, 1e-6)

            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            ohlc[i] = [open_price, high, low, close]

        return ohlc

    def _simulate_with_gaussian_process(self, template_sample: np.ndarray,
                                       regime: MarketRegime) -> np.ndarray:
        """Simulate using Gaussian Process for more realistic dynamics.

        Args:
            template_sample: Template OHLC sample [T, 4]
            regime: Market regime parameters

        Returns:
            Simulated OHLC sample
        """
        seq_len = template_sample.shape[0]

        # Use close prices as target
        close_prices = template_sample[:, 3]

        # Design kernel for financial time series
        kernel = (RBF(length_scale=10.0) * regime.volatility_level +
                 Matern(length_scale=5.0, nu=1.5) * 0.1 +
                 WhiteKernel(noise_level=regime.noise_ratio * 0.01))

        # Fit GP to log returns
        log_returns = np.diff(np.log(close_prices))
        gp = GaussianProcessRegressor(kernel=kernel, random_state=self.seed)

        # Generate time indices
        X = np.arange(len(log_returns)).reshape(-1, 1)

        # Fit GP
        gp.fit(X, log_returns)

        # Sample new returns
        X_new = np.arange(seq_len - 1).reshape(-1, 1)
        sampled_returns, _ = gp.sample_y(X_new, n_samples=1, random_state=self.seed)
        sampled_returns = sampled_returns.flatten()

        # Add trend component
        trend_component = regime.trend_strength * 0.0001 * np.arange(seq_len - 1)
        sampled_returns += trend_component

        # Convert back to prices
        simulated_log_prices = np.concatenate([[np.log(close_prices[0])],
                                              np.log(close_prices[0]) + np.cumsum(sampled_returns)])
        simulated_prices = np.exp(simulated_log_prices)

        # Create OHLC from prices
        ohlc = self._create_ohlc_from_prices(simulated_prices)

        return ohlc

    def fit(self, data: np.ndarray, labels: np.ndarray):
        """Fit the generator by detecting regimes and estimating parameters.

        Args:
            data: OHLC data [N, T, 4]
            labels: Class labels [N]
        """
        # Detect market regimes
        self.regimes = self._detect_market_regimes(data)

        # Estimate parameters for each regime
        for regime_id, sample_indices in self.regimes.items():
            self.regime_parameters[regime_id] = self._estimate_regime_parameters(
                data, labels, sample_indices
            )

    def generate(self, data: np.ndarray, labels: np.ndarray,
                n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate statistically simulated pseudo-samples.

        Args:
            data: Original OHLC data [N, T, 4]
            labels: Original labels [N]
            n_samples: Number of samples to generate

        Returns:
            Tuple of (generated_data, generated_labels)
        """
        # Fit regimes if not already done
        if not self.regimes:
            self.fit(data, labels)

        generated_data = []
        generated_labels = []

        for _ in range(n_samples):
            # Select regime and class
            regime_id = np.random.choice(list(self.regimes.keys()))
            class_label = np.random.choice(['consolidation', 'retracement'])

            # Get template samples from this class
            class_samples = np.where(labels == class_label)[0]
            if len(class_samples) == 0:
                continue

            template_idx = np.random.choice(class_samples)
            template_sample = data[template_idx].copy()
            regime = self.regime_parameters[regime_id]

            # Generate sample
            if self.use_gaussian_process:
                simulated = self._simulate_with_gaussian_process(template_sample, regime)
            else:
                # Use GBM approach
                initial_price = template_sample[0, 3]  # Start with template's first close
                prices = self._simulate_geometric_brownian_motion(
                    len(template_sample), regime, initial_price
                )
                simulated = self._create_ohlc_from_prices(prices)

            generated_data.append(simulated)
            generated_labels.append(class_label)

        return np.array(generated_data), np.array(generated_labels)

    def validate_quality(self, original: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
        """Validate statistical simulation quality.

        Args:
            original: Original samples [N, T, 4]
            generated: Generated samples [N, T, 4]

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Compare return distributions
        def calculate_returns(samples):
            all_returns = []
            for sample in samples:
                returns = np.diff(sample[:, 3]) / (sample[:-1, 3] + 1e-8)
                all_returns.extend(returns)
            return np.array(all_returns)

        orig_returns = calculate_returns(original)
        gen_returns = calculate_returns(generated)

        # Distribution similarity tests
        ks_stat = stats.ks_2samp(orig_returns, gen_returns).statistic
        metrics['return_distribution_similarity'] = 1.0 - ks_stat

        # Volatility clustering comparison
        orig_abs_returns = np.abs(orig_returns)
        gen_abs_returns = np.abs(gen_returns)

        orig_autocorr = np.correlate(orig_abs_returns, orig_abs_returns, mode='full')
        orig_autocorr = orig_autocorr[len(orig_autocorr)//2] / orig_autocorr[len(orig_autocorr)//2]

        gen_autocorr = np.correlate(gen_abs_returns, gen_abs_returns, mode='full')
        gen_autocorr = gen_autocorr[len(gen_autocorr)//2] / gen_autocorr[len(gen_autocorr)//2]

        metrics['volatility_clustering_similarity'] = 1.0 - abs(orig_autocorr - gen_autocorr)

        # Kurtosis comparison (important for financial returns)
        orig_kurtosis = stats.kurtosis(orig_returns)
        gen_kurtosis = stats.kurtosis(gen_returns)
        metrics['kurtosis_similarity'] = 1.0 - abs(orig_kurtosis - gen_kurtosis) / (abs(orig_kurtosis) + 1.0)

        # Price level distribution
        orig_prices = original[:, :, 3].flatten()
        gen_prices = generated[:, :, 3].flatten()

        orig_price_mean, orig_price_std = np.mean(orig_prices), np.std(orig_prices)
        gen_price_mean, gen_price_std = np.mean(gen_prices), np.std(gen_prices)

        metrics['price_level_similarity'] = 1.0 - abs(gen_price_mean - orig_price_mean) / (orig_price_mean + 1e-8)
        metrics['price_volatility_similarity'] = 1.0 - abs(gen_price_std - orig_price_std) / (orig_price_std + 1e-8)

        # OHLC relationship preservation
        ohlc_violations = 0
        total_checks = len(generated) * generated.shape[1]
        for sample in generated:
            for t in range(sample.shape[0]):
                o, h, l, c = sample[t]
                if not (o <= h and l <= h and o >= l and c >= l):
                    ohlc_violations += 1

        metrics['ohlc_preservation_rate'] = 1.0 - (ohlc_violations / total_checks)

        return metrics


class SelfSupervisedPseudoLabelingGenerator(BasePseudoGenerator):
    """Self-supervised pseudo-labeling using pre-trained encoder for confident predictions."""

    def __init__(self, seed: int = 1337,
                 confidence_threshold: float = 0.95,
                 encoder_model: Optional[nn.Module] = None,
                 feature_extractor: Optional[callable] = None):
        """Initialize self-supervised pseudo-labeling generator.

        Args:
            seed: Random seed
            confidence_threshold: Minimum confidence for pseudo-labeling
            encoder_model: Pre-trained encoder model
            feature_extractor: Function to extract features from raw data
        """
        super().__init__(seed)
        self.confidence_threshold = confidence_threshold
        self.encoder_model = encoder_model
        self.feature_extractor = feature_extractor
        self.scaler = StandardScaler()

    def _extract_features(self, ohlc_data: np.ndarray) -> np.ndarray:
        """Extract features from OHLC data for encoder input.

        Args:
            ohlc_data: OHLC data [N, T, 4]

        Returns:
            Feature array [N, T, F] where F is feature dimension
        """
        features = []

        for sample in ohlc_data:
            sample_features = []

            # Raw OHLC values (normalized)
            normalized = (sample - sample.mean(axis=0)) / (sample.std(axis=0) + 1e-8)
            sample_features.append(normalized)

            # Price changes
            price_changes = np.diff(sample[:, 3], prepend=sample[0, 3])
            price_changes = price_changes.reshape(-1, 1)
            sample_features.append(price_changes)

            # Returns
            returns = np.diff(sample[:, 3]) / (sample[:-1, 3] + 1e-8)
            returns = np.concatenate([[0], returns])
            returns = returns.reshape(-1, 1)
            sample_features.append(returns)

            # High-Low spread
            hl_spread = (sample[:, 1] - sample[:, 2]) / (sample[:, 2] + 1e-8)
            hl_spread = hl_spread.reshape(-1, 1)
            sample_features.append(hl_spread)

            # Open-Close spread
            oc_spread = (sample[:, 3] - sample[:, 0]) / (sample[:, 0] + 1e-8)
            oc_spread = oc_spread.reshape(-1, 1)
            sample_features.append(oc_spread)

            # Combine all features
            combined_features = np.concatenate(sample_features, axis=1)
            features.append(combined_features)

        return np.array(features)

    def _generate_unlabeled_candidates(self, data: np.ndarray, n_candidates: int) -> np.ndarray:
        """Generate candidate unlabeled samples using various strategies.

        Args:
            data: Original labeled data [N, T, 4]
            n_candidates: Number of candidates to generate

        Returns:
            Candidate samples [M, T, 4]
        """
        candidates = []

        # Strategy 1: Temporal augmentations of existing samples
        temp_aug = TemporalAugmentationGenerator(seed=self.seed)
        aug_samples, _ = temp_aug.generate(data, np.zeros(len(data)), n_candidates // 3)
        candidates.extend(aug_samples)

        # Strategy 2: Pattern-based modifications
        pattern_gen = PatternBasedSynthesisGenerator(seed=self.seed)
        pattern_samples, _ = pattern_gen.generate(data, np.zeros(len(data)), n_candidates // 3)
        candidates.extend(pattern_samples)

        # Strategy 3: Statistical simulations
        stat_gen = StatisticalSimulationGenerator(seed=self.seed)
        stat_samples, _ = stat_gen.generate(data, np.zeros(len(data)), n_candidates // 3)
        candidates.extend(stat_samples)

        # Fill remaining if needed
        while len(candidates) < n_candidates:
            idx = np.random.randint(0, len(data))
            sample = data[idx].copy()
            # Apply random augmentation
            temp_aug = TemporalAugmentationGenerator(seed=self.seed + len(candidates))
            aug_sample, _ = temp_aug.generate(sample.reshape(1, -1, 4), np.array([0]), 1)
            candidates.extend(aug_sample)

        return np.array(candidates[:n_candidates])

    def _predict_with_confidence(self, samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict labels and confidence scores using encoder model.

        Args:
            samples: Input samples [N, T, 4]

        Returns:
            Tuple of (predicted_labels, confidence_scores)
        """
        if self.encoder_model is None:
            raise ValueError("Encoder model must be provided for pseudo-labeling")

        # Extract features
        features = self._extract_features(samples)

        # Convert to tensor if needed
        if isinstance(features, np.ndarray):
            features_tensor = torch.FloatTensor(features)
        else:
            features_tensor = features

        # Get model predictions
        self.encoder_model.eval()
        with torch.no_grad():
            if hasattr(self.encoder_model, 'predict_proba'):
                probabilities = self.encoder_model.predict_proba(features_tensor)
            else:
                logits = self.encoder_model(features_tensor)
                probabilities = torch.softmax(logits, dim=-1)

            # Get predicted labels and max probability (confidence)
            confidence_scores, predicted_labels = torch.max(probabilities, dim=-1)

            if torch.is_tensor(predicted_labels):
                predicted_labels = predicted_labels.cpu().numpy()
            if torch.is_tensor(confidence_scores):
                confidence_scores = confidence_scores.cpu().numpy()

        return predicted_labels, confidence_scores

    def generate(self, data: np.ndarray, labels: np.ndarray,
                n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate pseudo-labeled samples using self-supervised learning.

        Args:
            data: Original OHLC data [N, T, 4]
            labels: Original labels [N]
            n_samples: Number of pseudo-labeled samples to generate

        Returns:
            Tuple of (pseudo_labeled_data, pseudo_labels)
        """
        if self.encoder_model is None:
            raise ValueError("Pre-trained encoder model is required for pseudo-labeling")

        # Generate more candidates than needed (to filter by confidence)
        n_candidates = int(n_samples * 3)  # Generate 3x candidates
        candidates = self._generate_unlabeled_candidates(data, n_candidates)

        # Predict labels and confidence for candidates
        pred_labels, confidence_scores = self._predict_with_confidence(candidates)

        # Filter high-confidence predictions
        confident_mask = confidence_scores >= self.confidence_threshold
        confident_candidates = candidates[confident_mask]
        confident_labels = pred_labels[confident_mask]

        # If we have enough confident samples, return requested amount
        if len(confident_candidates) >= n_samples:
            indices = np.random.choice(len(confident_candidates), n_samples, replace=False)
            return confident_candidates[indices], confident_labels[indices]
        else:
            # Return all confident samples (may be fewer than requested)
            print(f"Warning: Only generated {len(confident_candidates)} confident samples "
                  f"out of {n_samples} requested")
            return confident_candidates, confident_labels

    def validate_quality(self, original: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
        """Validate self-supervised pseudo-labeling quality.

        Args:
            original: Original samples [N, T, 4]
            generated: Generated samples [N, T, 4]

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Feature distribution similarity
        orig_features = self._extract_features(original)
        gen_features = self._extract_features(generated)

        for i in range(orig_features.shape[2]):  # For each feature dimension
            orig_feat = orig_features[:, :, i].flatten()
            gen_feat = gen_features[:, :, i].flatten()

            ks_stat = stats.ks_2samp(orig_feat, gen_feat).statistic
            metrics[f'feature_{i}_similarity'] = 1.0 - ks_stat

        # Return distribution comparison
        def calculate_returns(samples):
            returns = []
            for sample in samples:
                sample_returns = np.diff(sample[:, 3]) / (sample[:-1, 3] + 1e-8)
                returns.extend(sample_returns)
            return np.array(returns)

        orig_returns = calculate_returns(original)
        gen_returns = calculate_returns(generated)

        ks_stat = stats.ks_2samp(orig_returns, gen_returns).statistic
        metrics['return_distribution_similarity'] = 1.0 - ks_stat

        # Pattern similarity using DTW
        def dtw_distance(s1, s2):
            n, m = len(s1), len(s2)
            dtw = np.full((n + 1, m + 1), np.inf)
            dtw[0, 0] = 0

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    cost = abs(s1[i-1] - s2[j-1])
                    dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])

            return dtw[n, m]

        dtw_distances = []
        n_samples = min(5, len(original), len(generated))
        for i in range(n_samples):
            dist = dtw_distance(original[i, :, 3], generated[i, :, 3])
            dtw_distances.append(dist)

        metrics['avg_dtw_similarity'] = 1.0 / (1.0 + np.mean(dtw_distances))

        # Confidence score distribution
        if self.encoder_model is not None:
            _, confidence_scores = self._predict_with_confidence(generated)
            metrics['avg_confidence_score'] = np.mean(confidence_scores)
            metrics['high_confidence_ratio'] = np.mean(confidence_scores >= self.confidence_threshold)

        return metrics


class MarketConditionSimulationGenerator(BasePseudoGenerator):
    """Market condition simulation across different volatility and trend regimes."""

    def __init__(self, seed: int = 1337,
                 regime_config: Optional[Dict[str, MarketRegime]] = None):
        """Initialize market condition simulation generator.

        Args:
            seed: Random seed
            regime_config: Dictionary of regime configurations
        """
        super().__init__(seed)
        self.regime_config = regime_config or self._get_default_regimes()
        self.regime_names = list(self.regime_config.keys())

    def _get_default_regimes(self) -> Dict[str, MarketRegime]:
        """Get default market regime configurations.

        Returns:
            Dictionary of default regimes
        """
        return {
            'low_volatility': MarketRegime(
                volatility_level=0.5,
                trend_strength=0.1,
                noise_ratio=0.8,
                gap_probability=0.01,
                mean_reversion_speed=0.3
            ),
            'normal_market': MarketRegime(
                volatility_level=1.0,
                trend_strength=0.0,
                noise_ratio=0.5,
                gap_probability=0.05,
                mean_reversion_speed=0.1
            ),
            'high_volatility': MarketRegime(
                volatility_level=2.0,
                trend_strength=-0.2,
                noise_ratio=0.3,
                gap_probability=0.15,
                mean_reversion_speed=0.05
            ),
            'strong_trend_up': MarketRegime(
                volatility_level=1.2,
                trend_strength=0.8,
                noise_ratio=0.4,
                gap_probability=0.08,
                mean_reversion_speed=0.02
            ),
            'strong_trend_down': MarketRegime(
                volatility_level=1.5,
                trend_strength=-0.8,
                noise_ratio=0.4,
                gap_probability=0.12,
                mean_reversion_speed=0.02
            )
        }

    def _simulate_regime_specific_path(self, regime: MarketRegime,
                                     seq_len: int,
                                     initial_price: float = 100.0,
                                     pattern_type: str = 'random') -> np.ndarray:
        """Simulate price path specific to market regime.

        Args:
            regime: Market regime parameters
            seq_len: Length of sequence
            initial_price: Starting price
            pattern_type: Type of pattern to simulate

        Returns:
            Simulated OHLC data [T, 4]
        """
        if pattern_type == 'trend_following':
            return self._simulate_trend_following(regime, seq_len, initial_price)
        elif pattern_type == 'mean_reverting':
            return self._simulate_mean_reverting(regime, seq_len, initial_price)
        elif pattern_type == 'breakout':
            return self._simulate_breakout(regime, seq_len, initial_price)
        else:  # random walk with regime characteristics
            return self._simulate_regime_random_walk(regime, seq_len, initial_price)

    def _simulate_trend_following(self, regime: MarketRegime, seq_len: int,
                                initial_price: float) -> np.ndarray:
        """Simulate trend-following price action.

        Args:
            regime: Market regime parameters
            seq_len: Sequence length
            initial_price: Starting price

        Returns:
            OHLC data [T, 4]
        """
        prices = [initial_price]

        # Trend component
        trend_drift = regime.trend_strength * 0.001

        momentum = 0
        for i in range(1, seq_len):
            # Strong momentum component
            if len(prices) > 1:
                momentum = (prices[-1] - prices[-2]) / prices[-2]
                momentum_component = momentum * 0.3
            else:
                momentum_component = 0

            # Random component
            random_component = np.random.normal(0, regime.volatility_level * 0.01)

            # Mean reversion damping
            mean_reversion = -regime.mean_reversion_speed * momentum

            # Update price
            price_change = trend_drift + momentum_component + random_component + mean_reversion
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(1e-6, new_price))

        return self._create_ohlc_from_prices(np.array(prices), regime)

    def _simulate_mean_reverting(self, regime: MarketRegime, seq_len: int,
                               initial_price: float) -> np.ndarray:
        """Simulate mean-reverting price action.

        Args:
            regime: Market regime parameters
            seq_len: Sequence length
            initial_price: Starting price

        Returns:
            OHLC data [T, 4]
        """
        prices = [initial_price]
        mean_price = initial_price  # Reversion target

        for i in range(1, seq_len):
            # Distance from mean
            deviation = (prices[-1] - mean_price) / mean_price

            # Mean reversion force
            reversion_force = -regime.mean_reversion_speed * deviation

            # Random component
            random_component = np.random.normal(0, regime.volatility_level * 0.01)

            # Small trend component
            trend_component = regime.trend_strength * 0.0001

            # Update price
            price_change = reversion_force + random_component + trend_component
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(1e-6, new_price))

            # Slowly adapt mean price
            mean_price = 0.99 * mean_price + 0.01 * new_price

        return self._create_ohlc_from_prices(np.array(prices), regime)

    def _simulate_breakout(self, regime: MarketRegime, seq_len: int,
                         initial_price: float) -> np.ndarray:
        """Simulate breakout price action.

        Args:
            regime: Market regime parameters
            seq_len: Sequence length
            initial_price: Starting price

        Returns:
            OHLC data [T, 4]
        """
        prices = [initial_price]

        # Define consolidation range (initially tight)
        range_width = initial_price * 0.02 * regime.volatility_level
        range_center = initial_price
        breakout_time = np.random.randint(seq_len // 3, 2 * seq_len // 3)
        breakout_direction = np.random.choice([-1, 1])

        for i in range(1, seq_len):
            if i < breakout_time:
                # Consolidation phase
                # Bounce within range
                if prices[-1] > range_center + range_width:
                    price_change = -regime.volatility_level * 0.005
                elif prices[-1] < range_center - range_width:
                    price_change = regime.volatility_level * 0.005
                else:
                    price_change = np.random.normal(0, regime.volatility_level * 0.003)

            else:
                # Breakout phase
                if i == breakout_time:
                    # Initial breakout jump
                    price_change = breakout_direction * regime.gap_probability * 0.05
                else:
                    # Continue trend with momentum
                    trend_strength = abs(regime.trend_strength) * 0.001 * breakout_direction
                    momentum = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0
                    momentum_component = momentum * 0.4
                    noise = np.random.normal(0, regime.volatility_level * 0.008)
                    price_change = trend_strength + momentum_component + noise

            new_price = prices[-1] * (1 + price_change)
            prices.append(max(1e-6, new_price))

        return self._create_ohlc_from_prices(np.array(prices), regime)

    def _simulate_regime_random_walk(self, regime: MarketRegime, seq_len: int,
                                   initial_price: float) -> np.ndarray:
        """Simulate random walk with regime characteristics.

        Args:
            regime: Market regime parameters
            seq_len: Sequence length
            initial_price: Starting price

        Returns:
            OHLC data [T, 4]
        """
        prices = [initial_price]

        for i in range(1, seq_len):
            # Base random walk
            drift = regime.trend_strength * 0.001
            volatility = regime.volatility_level * 0.01

            # Random shock
            shock = np.random.normal(drift, volatility)

            # Add occasional jumps
            if np.random.random() < regime.gap_probability:
                jump = np.random.choice([-1, 1]) * np.random.uniform(0.02, 0.05)
                shock += jump

            # Update price
            new_price = prices[-1] * (1 + shock)
            prices.append(max(1e-6, new_price))

        return self._create_ohlc_from_prices(np.array(prices), regime)

    def _create_ohlc_from_prices(self, prices: np.ndarray,
                               regime: MarketRegime) -> np.ndarray:
        """Create realistic OHLC from price series with regime characteristics.

        Args:
            prices: Price series
            regime: Market regime for realistic parameters

        Returns:
            OHLC data [T, 4]
        """
        n_steps = len(prices)
        ohlc = np.zeros((n_steps, 4))

        # Base intraday volatility based on regime
        base_volatility = regime.volatility_level * 0.001

        for i in range(n_steps):
            close = prices[i]

            # Calculate intraday volatility with regime-specific scaling
            intraday_vol = base_volatility * close * (1 + 0.5 * np.random.random())

            # Open price with gap probability
            if i == 0:
                open_price = close
            else:
                if np.random.random() < regime.gap_probability:
                    gap = np.random.normal(0, intraday_vol * 2)
                else:
                    gap = np.random.normal(0, intraday_vol * 0.3)
                open_price = max(1e-6, prices[i-1] + gap)

            # Determine if up or down day
            is_up_day = close >= open_price

            # High and low calculation
            if is_up_day:
                # Up day: high is above close, low is below open
                high_range = np.random.uniform(0.0005, 0.002) * close * regime.volatility_level
                low_range = np.random.uniform(0.0002, 0.001) * close * regime.volatility_level

                high = max(close, open_price) + np.random.uniform(0, high_range)
                low = min(close, open_price) - np.random.uniform(0, low_range)
            else:
                # Down day: high is above open, low is below close
                high_range = np.random.uniform(0.0002, 0.001) * close * regime.volatility_level
                low_range = np.random.uniform(0.0005, 0.002) * close * regime.volatility_level

                high = max(close, open_price) + np.random.uniform(0, high_range)
                low = min(close, open_price) - np.random.uniform(0, low_range)

            # Ensure positive prices and OHLC relationships
            high = max(high, open_price, close, 1e-6)
            low = min(low, open_price, close, 1e-6)
            open_price = max(open_price, 1e-6)
            close = max(close, 1e-6)

            ohlc[i] = [open_price, high, low, close]

        return ohlc

    def generate(self, data: np.ndarray, labels: np.ndarray,
                n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate samples across different market conditions.

        Args:
            data: Original OHLC data [N, T, 4] (used for sequence length reference)
            labels: Original labels [N] (used for class balance)
            n_samples: Number of samples to generate

        Returns:
            Tuple of (generated_data, generated_labels)
        """
        generated_data = []
        generated_labels = []

        seq_len = data.shape[1] if len(data) > 0 else 105

        # Determine class distribution from original labels
        unique_labels, counts = np.unique(labels, return_counts=True)
        class_probabilities = counts / len(labels)

        for _ in range(n_samples):
            # Select regime
            regime_name = np.random.choice(self.regime_names)
            regime = self.regime_config[regime_name]

            # Select class based on original distribution
            class_label = np.random.choice(unique_labels, p=class_probabilities)

            # Select pattern type based on regime characteristics
            if regime.trend_strength > 0.5:
                pattern_types = ['trend_following', 'breakout', 'random']
                pattern_probs = [0.5, 0.3, 0.2]
            elif regime.trend_strength < -0.5:
                pattern_types = ['trend_following', 'breakout', 'random']
                pattern_probs = [0.5, 0.3, 0.2]
            elif regime.mean_reversion_speed > 0.2:
                pattern_types = ['mean_reverting', 'random']
                pattern_probs = [0.7, 0.3]
            else:
                pattern_types = ['random', 'breakout']
                pattern_probs = [0.6, 0.4]

            pattern_type = np.random.choice(pattern_types, p=pattern_probs)

            # Get initial price from original data distribution
            if len(data) > 0:
                initial_price = np.random.choice(data[:, 0, 3])  # First close price
            else:
                initial_price = 100.0

            # Generate sample
            simulated = self._simulate_regime_specific_path(
                regime, seq_len, initial_price, pattern_type
            )

            generated_data.append(simulated)
            generated_labels.append(class_label)

        return np.array(generated_data), np.array(generated_labels)

    def validate_quality(self, original: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
        """Validate market condition simulation quality.

        Args:
            original: Original samples [N, T, 4]
            generated: Generated samples [N, T, 4]

        Returns:
            Dictionary of quality metrics
        """
        metrics = {}

        # Return distribution comparison
        def calculate_returns(samples):
            returns = []
            for sample in samples:
                sample_returns = np.diff(sample[:, 3]) / (sample[:-1, 3] + 1e-8)
                returns.extend(sample_returns)
            return np.array(returns)

        orig_returns = calculate_returns(original)
        gen_returns = calculate_returns(generated)

        # Statistical moments comparison
        orig_mean, orig_std, orig_skew, orig_kurt = (
            np.mean(orig_returns), np.std(orig_returns),
            stats.skew(orig_returns), stats.kurtosis(orig_returns)
        )
        gen_mean, gen_std, gen_skew, gen_kurt = (
            np.mean(gen_returns), np.std(gen_returns),
            stats.skew(gen_returns), stats.kurtosis(gen_returns)
        )

        metrics['mean_similarity'] = 1.0 - abs(gen_mean - orig_mean) / (abs(orig_mean) + 1e-8)
        metrics['volatility_similarity'] = 1.0 - abs(gen_std - orig_std) / (orig_std + 1e-8)
        metrics['skewness_similarity'] = 1.0 - abs(gen_skew - orig_skew) / (abs(orig_skew) + 1.0)
        metrics['kurtosis_similarity'] = 1.0 - abs(gen_kurt - orig_kurt) / (abs(orig_kurt) + 1.0)

        # Volatility clustering
        orig_abs_returns = np.abs(orig_returns)
        gen_abs_returns = np.abs(gen_returns)

        orig_autocorr = np.corrcoef(orig_abs_returns[:-1], orig_abs_returns[1:])[0, 1]
        gen_autocorr = np.corrcoef(gen_abs_returns[:-1], gen_abs_returns[1:])[0, 1]

        metrics['volatility_clustering_similarity'] = 1.0 - abs(orig_autocorr - gen_autocorr)

        # Distribution similarity
        ks_stat = stats.ks_2samp(orig_returns, gen_returns).statistic
        metrics['return_distribution_similarity'] = 1.0 - ks_stat

        # OHLC relationship preservation
        ohlc_violations = 0
        total_checks = len(generated) * generated.shape[1]
        for sample in generated:
            for t in range(sample.shape[0]):
                o, h, l, c = sample[t]
                if not (o <= h and l <= h and o >= l and c >= l):
                    ohlc_violations += 1

        metrics['ohlc_preservation_rate'] = 1.0 - (ohlc_violations / total_checks)

        return metrics


class PseudoSampleGenerationPipeline:
    """Main pipeline integrating all pseudo-sample generation strategies."""

    def __init__(self, seed: int = 1337,
                 strategy_weights: Optional[Dict[str, float]] = None,
                 validation_threshold: float = 0.7):
        """Initialize pseudo-sample generation pipeline.

        Args:
            seed: Random seed for reproducibility
            strategy_weights: Weights for each generation strategy
            validation_threshold: Minimum quality score for sample acceptance
        """
        self.seed = seed
        self.validation_threshold = validation_threshold

        # Default strategy weights
        self.strategy_weights = strategy_weights or {
            'temporal_augmentation': 0.25,
            'pattern_synthesis': 0.25,
            'statistical_simulation': 0.2,
            'market_condition': 0.3
        }

        # Initialize generators
        self.generators = {
            'temporal_augmentation': TemporalAugmentationGenerator(seed=seed),
            'pattern_synthesis': PatternBasedSynthesisGenerator(seed=seed),
            'statistical_simulation': StatisticalSimulationGenerator(seed=seed),
            'market_condition': MarketConditionSimulationGenerator(seed=seed)
        }

        # Quality metrics storage
        self.quality_history = []

    def set_encoder_model(self, encoder_model: nn.Module):
        """Set encoder model for self-supervised pseudo-labeling.

        Args:
            encoder_model: Pre-trained encoder model
        """
        self.generators['self_supervised'] = SelfSupervisedPseudoLabelingGenerator(
            seed=self.seed, encoder_model=encoder_model
        )
        self.strategy_weights['self_supervised'] = 0.0  # Start with 0 weight

    def enable_self_supervised(self, confidence_threshold: float = 0.95):
        """Enable self-supervised pseudo-labeling.

        Args:
            confidence_threshold: Minimum confidence for pseudo-labels
        """
        if 'self_supervised' in self.generators:
            self.generators['self_supervised'].confidence_threshold = confidence_threshold
            self.strategy_weights['self_supervised'] = 0.1  # Add to strategy mix
        else:
            raise ValueError("Encoder model must be set first using set_encoder_model()")

    def generate_samples(self, data: np.ndarray, labels: np.ndarray,
                        n_samples: int, quality_check: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Generate pseudo-samples using integrated strategies.

        Args:
            data: Original OHLC data [N, T, 4]
            labels: Original labels [N]
            n_samples: Number of samples to generate
            quality_check: Whether to perform quality validation

        Returns:
            Tuple of (generated_data, generated_labels, generation_metadata)
        """
        if n_samples <= 0:
            return np.array([]), np.array([]), {}

        # Calculate samples per strategy based on weights
        total_weight = sum(self.strategy_weights.values())
        samples_per_strategy = {}

        remaining_samples = n_samples
        for strategy, weight in self.strategy_weights.items():
            if weight > 0 and strategy in self.generators:
                strategy_samples = int(n_samples * weight / total_weight)
                samples_per_strategy[strategy] = strategy_samples
                remaining_samples -= strategy_samples

        # Distribute remaining samples
        if remaining_samples > 0 and samples_per_strategy:
            strategies_with_weight = [s for s, w in self.strategy_weights.items() if w > 0 and s in self.generators]
            if strategies_with_weight:
                strategy = np.random.choice(strategies_with_weight)
                samples_per_strategy[strategy] = samples_per_strategy.get(strategy, 0) + remaining_samples

        # Generate samples from each strategy
        all_generated_data = []
        all_generated_labels = []
        generation_metadata = {
            'strategies_used': list(samples_per_strategy.keys()),
            'samples_per_strategy': samples_per_strategy,
            'quality_scores': {},
            'accepted_samples': 0,
            'rejected_samples': 0
        }

        for strategy, n_strategy_samples in samples_per_strategy.items():
            if n_strategy_samples <= 0:
                continue

            try:
                generator = self.generators[strategy]

                # Generate samples
                gen_data, gen_labels = generator.generate(data, labels, n_strategy_samples)

                if quality_check:
                    # Validate quality
                    quality_metrics = generator.validate_quality(data, gen_data)

                    # Calculate overall quality score
                    quality_score = np.mean(list(quality_metrics.values()))
                    generation_metadata['quality_scores'][strategy] = quality_score

                    # Accept or reject based on quality threshold
                    if quality_score >= self.validation_threshold:
                        all_generated_data.extend(gen_data)
                        all_generated_labels.extend(gen_labels)
                        generation_metadata['accepted_samples'] += len(gen_data)
                    else:
                        generation_metadata['rejected_samples'] += len(gen_data)
                        print(f"Rejected {len(gen_data)} samples from {strategy} (quality: {quality_score:.3f})")
                else:
                    # Accept all samples without quality check
                    all_generated_data.extend(gen_data)
                    all_generated_labels.extend(gen_labels)
                    generation_metadata['accepted_samples'] += len(gen_data)

            except Exception as e:
                print(f"Error generating samples with {strategy}: {e}")
                generation_metadata['rejected_samples'] += n_strategy_samples

        # Convert to arrays
        generated_data = np.array(all_generated_data) if all_generated_data else np.array([]).reshape(0, data.shape[1], 4)
        generated_labels = np.array(all_generated_labels) if all_generated_labels else np.array([])

        # Store quality history
        self.quality_history.append(generation_metadata)

        return generated_data, generated_labels, generation_metadata

    def get_generation_report(self) -> Dict:
        """Get comprehensive report of generation performance.

        Returns:
            Dictionary with generation statistics and quality metrics
        """
        if not self.quality_history:
            return {"message": "No generation history available"}

        report = {
            'total_generations': len(self.quality_history),
            'total_samples_generated': sum(meta['accepted_samples'] for meta in self.quality_history),
            'total_samples_rejected': sum(meta['rejected_samples'] for meta in self.quality_history),
            'average_quality_scores': {},
            'strategy_usage': {},
            'acceptance_rate': 0.0
        }

        # Calculate strategy usage and quality
        strategy_counts = {}
        strategy_quality = {}
        for meta in self.quality_history:
            for strategy in meta['strategies_used']:
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
                if strategy in meta['quality_scores']:
                    if strategy not in strategy_quality:
                        strategy_quality[strategy] = []
                    strategy_quality[strategy].append(meta['quality_scores'][strategy])

        report['strategy_usage'] = strategy_counts

        # Calculate average quality scores
        for strategy, scores in strategy_quality.items():
            report['average_quality_scores'][strategy] = np.mean(scores)

        # Calculate acceptance rate
        total_generated = report['total_samples_generated'] + report['total_samples_rejected']
        if total_generated > 0:
            report['acceptance_rate'] = report['total_samples_generated'] / total_generated

        return report


# Continue with the remaining generators and main integration class...
if __name__ == "__main__":
    # Example usage and testing
    print("Financial Time Series Pseudo-Sample Generation Framework")
    print("This module provides advanced strategies for generating realistic financial data.")
    print("Available generators:")
    print("1. TemporalAugmentationGenerator")
    print("2. PatternBasedSynthesisGenerator")
    print("3. StatisticalSimulationGenerator")
    print("4. SelfSupervisedPseudoLabelingGenerator (to be implemented)")
    print("5. MarketConditionSimulationGenerator (to be implemented)")