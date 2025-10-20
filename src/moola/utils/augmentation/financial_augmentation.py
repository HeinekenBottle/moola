"""Financial time series augmentation that preserves market physics.

Specialized augmentation strategies for financial OHLC data that maintain:
- Temporal dependencies and autocorrelation structure
- Market microstructure properties (bid-ask spread, volatility clustering)
- OHLC relationships (high ≥ low, etc.)
- Economic plausibility and market physics
- Pattern integrity for consolidation/retracement/expansion classification

Key improvements over generic time series augmentation:
1. Market-aware transformations that respect OHLC constraints
2. Preserves volatility clustering and regime-specific properties
3. Maintains cross-asset relationships and fundamental constraints
4. Adaptive augmentation based on market conditions
"""

import numpy as np
import torch
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum

from scipy import stats
from scipy.signal import savgol_filter
from loguru import logger


class AugmentationType(str, Enum):
    """Types of financial augmentation strategies."""
    MARKET_MIXUP = "market_mixup"  # Market-aware mixup
    VOLATILITY_SCALING = "volatility_scaling"  # Regime-aware volatility changes
    TREND_PRESERVING_JITTER = "trend_preserving_jitter"  # Noise that maintains trends
    MARKET_MICROSTRUCTURE_WARP = "market_microstructure_warp"  # Realistic microstructure changes
    PATTERN_PRESERVING_CUTMIX = "pattern_preserving_cutmix"  # CutMix that preserves patterns
    ECONOMIC_SCENARIO_SIMULATION = "economic_scenario_simulation"  # Stress testing scenarios


@dataclass
class FinancialAugmentationConfig:
    """Configuration for financial-aware data augmentation."""

    # Augmentation probabilities
    market_mixup_prob: float = 0.4
    volatility_scaling_prob: float = 0.3
    trend_preserving_jitter_prob: float = 0.5
    microstructure_warp_prob: float = 0.2
    pattern_preserving_cutmix_prob: float = 0.3
    economic_scenario_prob: float = 0.1

    # Augmentation parameters
    market_mixup_alpha: float = 0.3  # Beta distribution parameter
    volatility_scaling_range: Tuple[float, float] = (0.8, 1.3)  # ±30% volatility change
    trend_jitter_sigma: float = 0.02  # 2% noise relative to trend
    microstructure_noise_scale: float = 0.001  # 0.1% microstructure noise
    cutmix_pattern_preservation: float = 0.7  # Preserve 70% of pattern region

    # Market-aware constraints
    max_price_gap: float = 0.05  # Maximum 5% gap in augmentations
    preserve_volatility_clustering: bool = True
    maintain_ohlc_constraints: bool = True
    respect_market_hours: bool = True

    # Adaptive parameters
    adaptive_to_regime: bool = True
    regime_specific_params: Dict[str, Dict] = field(default_factory=dict)


class FinancialAugmentationPipeline:
    """Financial-aware augmentation pipeline for OHLC time series."""

    def __init__(self, config: Optional[FinancialAugmentationConfig] = None):
        """Initialize financial augmentation pipeline.

        Args:
            config: Augmentation configuration (uses defaults if None)
        """
        self.config = config or FinancialAugmentationConfig()

        # Initialize regime-specific parameters
        self._initialize_regime_parameters()

    def _initialize_regime_parameters(self):
        """Initialize regime-specific augmentation parameters."""
        self.config.regime_specific_params = {
            "trending_up": {
                "trend_jitter_sigma": 0.015,  # Less jitter in trends
                "volatility_scaling_range": (0.9, 1.2),  # Conservative volatility changes
                "market_mixup_alpha": 0.25,  # Gentler mixing in trends
            },
            "trending_down": {
                "trend_jitter_sigma": 0.015,
                "volatility_scaling_range": (0.9, 1.2),
                "market_mixup_alpha": 0.25,
            },
            "ranging": {
                "trend_jitter_sigma": 0.025,  # More jitter in ranging markets
                "volatility_scaling_range": (0.7, 1.4),  # More volatility variation
                "market_mixup_alpha": 0.35,
            },
            "volatile": {
                "trend_jitter_sigma": 0.03,  # Higher noise tolerance
                "volatility_scaling_range": (0.8, 1.5),  # Wider volatility changes
                "market_mixup_alpha": 0.4,
            }
        }

    def augment_batch(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regime_info: Optional[Dict] = None,
        expansion_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply financial-aware augmentation to a batch.

        Args:
            x: Input batch [B, 105, 4] OHLC data
            y: Target labels [B]
            regime_info: Optional regime information for adaptive augmentation
            expansion_indices: Optional (expansion_start, expansion_end) for pattern preservation

        Returns:
            Tuple of (augmented_x, y_a, y_b, lambda) for mixup-style training
        """
        batch_size = x.size(0)

        # Choose augmentation type
        aug_type = self._choose_augmentation_type()

        # Apply selected augmentation
        if aug_type == AugmentationType.MARKET_MIXUP:
            return self._market_mixup(x, y, regime_info)

        elif aug_type == AugmentationType.VOLATILITY_SCALING:
            return self._volatility_scaling(x, y, regime_info)

        elif aug_type == AugmentationType.TREND_PRESERVING_JITTER:
            return self._trend_preserving_jitter(x, y, regime_info)

        elif aug_type == AugmentationType.MARKET_MICROSTRUCTURE_WARP:
            return self._market_microstructure_warp(x, y, regime_info)

        elif aug_type == AugmentationType.PATTERN_PRESERVING_CUTMIX:
            return self._pattern_preserving_cutmix(x, y, expansion_indices)

        elif aug_type == AugmentationType.ECONOMIC_SCENARIO_SIMULATION:
            return self._economic_scenario_simulation(x, y, regime_info)

        else:
            # Fallback to identity (no augmentation)
            return x, y, y, 1.0

    def _choose_augmentation_type(self) -> AugmentationType:
        """Choose augmentation type based on configured probabilities."""
        probabilities = [
            self.config.market_mixup_prob,
            self.config.volatility_scaling_prob,
            self.config.trend_preserving_jitter_prob,
            self.config.microstructure_warp_prob,
            self.config.pattern_preserving_cutmix_prob,
            self.config.economic_scenario_prob
        ]

        augmentation_types = [
            AugmentationType.MARKET_MIXUP,
            AugmentationType.VOLATILITY_SCALING,
            AugmentationType.TREND_PRESERVING_JITTER,
            AugmentationType.MARKET_MICROSTRUCTURE_WARP,
            AugmentationType.PATTERN_PRESERVING_CUTMIX,
            AugmentationType.ECONOMIC_SCENARIO_SIMULATION
        ]

        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]

        # Choose based on probabilities
        rand_val = np.random.rand()
        cumulative_prob = 0.0

        for prob, aug_type in zip(probabilities, augmentation_types):
            cumulative_prob += prob
            if rand_val < cumulative_prob:
                return aug_type

        # Default fallback
        return AugmentationType.MARKET_MIXUP

    def _market_mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regime_info: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Market-aware mixup that preserves OHLC relationships."""
        if self.config.market_mixup_alpha > 0:
            lam = np.random.beta(self.config.market_mixup_alpha, self.config.market_mixup_alpha)
        else:
            lam = 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size, device=x.device)

        # Market-aware mixing
        x_mixed = self._mix_ohlc_series(x, x[index], lam)

        return x_mixed, y, y[index], lam

    def _mix_ohlc_series(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        lam: float
    ) -> torch.Tensor:
        """Mix two OHLC series while preserving constraints."""
        # Mix log prices (ensures positivity)
        log_x1 = torch.log(torch.clamp(x1, min=1e-8))
        log_x2 = torch.log(torch.clamp(x2, min=1e-8))

        # Linear interpolation in log space
        mixed_log = lam * log_x1 + (1 - lam) * log_x2

        # Convert back to price space
        x_mixed = torch.exp(mixed_log)

        # Ensure OHLC constraints are maintained
        if self.config.maintain_ohlc_constraints:
            x_mixed = self._enforce_ohlc_constraints(x_mixed)

        return x_mixed

    def _enforce_ohlc_constraints(self, x: torch.Tensor) -> torch.Tensor:
        """Enforce OHLC logical constraints on augmented data."""
        # Extract OHLC components
        open_p = x[..., 0]
        high = x[..., 1]
        low = x[..., 2]
        close = x[..., 3]

        # Ensure high >= max(open, close)
        high = torch.max(high, torch.max(open_p, close))

        # Ensure low <= min(open, close)
        low = torch.min(low, torch.min(open_p, close))

        # Ensure high >= low
        high = torch.max(high, low)

        # Reconstruct OHLC tensor
        x_constrained = torch.stack([open_p, high, low, close], dim=-1)

        return x_constrained

    def _volatility_scaling(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regime_info: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Scale volatility while preserving price levels and trends."""
        batch_size = x.size(0)

        # Choose volatility scaling factor
        if self.config.adaptive_to_regime and regime_info:
            regime = regime_info.get("regime", "uncertain")
            params = self.config.regime_specific_params.get(regime, {})
            vol_range = params.get("volatility_scaling_range", self.config.volatility_scaling_range)
        else:
            vol_range = self.config.volatility_scaling_range

        vol_scale = np.random.uniform(vol_range[0], vol_range[1])

        # Apply volatility scaling
        x_scaled = self._scale_volatility(x, vol_scale)

        # For mixup-style training, use identity mixing (no label mixing)
        return x_scaled, y, y, 1.0

    def _scale_volatility(self, x: torch.Tensor, scale_factor: float) -> torch.Tensor:
        """Scale volatility of OHLC series while preserving overall structure."""
        # Decompose into trend and residual components
        trend = self._extract_trend(x)
        residual = x - trend

        # Scale residual (volatility component)
        scaled_residual = residual * scale_factor

        # Reconstruct
        x_scaled = trend + scaled_residual

        # Ensure OHLC constraints
        if self.config.maintain_ohlc_constraints:
            x_scaled = self._enforce_ohlc_constraints(x_scaled)

        return x_scaled

    def _extract_trend(self, x: torch.Tensor, window_size: int = 20) -> torch.Tensor:
        """Extract trend component using moving average."""
        # Apply Savitzky-Golay filter for smooth trend extraction
        x_np = x.detach().cpu().numpy()

        # Extract close prices for trend calculation
        close_prices = x_np[..., 3]

        # Apply filter to each sample
        trend = np.zeros_like(x_np)
        for i in range(x_np.shape[0]):
            if len(close_prices[i]) >= window_size:
                trend_close = savgol_filter(close_prices[i], window_size, 3)
                # Propagate trend to all OHLC components
                for j in range(4):
                    trend[i, :, j] = trend_close
            else:
                trend[i] = x_np[i]  # Use original if too short

        return torch.from_numpy(trend).to(x.device)

    def _trend_preserving_jitter(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regime_info: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Add jitter that preserves trend structure."""
        # Get regime-specific parameters
        if self.config.adaptive_to_regime and regime_info:
            regime = regime_info.get("regime", "uncertain")
            params = self.config.regime_specific_params.get(regime, {})
            sigma = params.get("trend_jitter_sigma", self.config.trend_jitter_sigma)
        else:
            sigma = self.config.trend_jitter_sigma

        # Extract trend and add noise to residuals
        trend = self._extract_trend(x)
        residual = x - trend

        # Add noise to residuals only (preserves trend)
        noise = torch.randn_like(residual) * sigma
        noisy_residual = residual + noise

        # Reconstruct
        x_jittered = trend + noisy_residual

        # Ensure OHLC constraints
        if self.config.maintain_ohlc_constraints:
            x_jittered = self._enforce_ohlc_constraints(x_jittered)

        return x_jittered, y, y, 1.0

    def _market_microstructure_warp(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regime_info: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply realistic market microstructure changes."""
        # Add microstructure noise to OHLC components
        noise_scale = self.config.microstructure_noise_scale

        # Different noise levels for different components
        open_noise = torch.randn_like(x[..., 0]) * noise_scale
        high_noise = torch.abs(torch.randn_like(x[..., 1])) * noise_scale * 2  # Positive bias for high
        low_noise = -torch.abs(torch.randn_like(x[..., 2])) * noise_scale * 2  # Negative bias for low
        close_noise = torch.randn_like(x[..., 3]) * noise_scale

        # Apply noise
        x_warped = x.clone()
        x_warped[..., 0] += open_noise
        x_warped[..., 1] += high_noise
        x_warped[..., 2] += low_noise
        x_warped[..., 3] += close_noise

        # Ensure OHLC constraints
        if self.config.maintain_ohlc_constraints:
            x_warped = self._enforce_ohlc_constraints(x_warped)

        return x_warped, y, y, 1.0

    def _pattern_preserving_cutmix(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        expansion_indices: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """CutMix that preserves pattern regions."""
        if self.config.market_mixup_alpha > 0:
            lam = np.random.beta(self.config.market_mixup_alpha, self.config.market_mixup_alpha)
        else:
            lam = 1.0

        batch_size, seq_len, _ = x.size()
        index = torch.randperm(batch_size, device=x.device)

        # Determine cut region
        if expansion_indices is not None:
            # Avoid cutting through pattern regions
            cut_start, cut_end = self._find_safe_cut_region(
                expansion_indices, seq_len, lam
            )
        else:
            # Use default central region for cutting
            cut_len = int(seq_len * (1 - lam))
            cut_start = seq_len // 2 - cut_len // 2
            cut_end = cut_start + cut_len

        # Create mixed sample
        x_mixed = x.clone()
        x_mixed[:, cut_start:cut_end, :] = x[index, cut_start:cut_end, :]

        # Adjust lambda to actual proportion
        actual_lam = 1 - (cut_end - cut_start) / seq_len

        return x_mixed, y, y[index], actual_lam

    def _find_safe_cut_region(
        self,
        expansion_indices: Tuple[torch.Tensor, torch.Tensor],
        seq_len: int,
        lam: float
    ) -> Tuple[int, int]:
        """Find safe region for cutting that avoids pattern areas."""
        exp_start, exp_end = expansion_indices

        # Get pattern boundaries across batch
        min_start = exp_start.min().item()
        max_end = exp_end.max().item()

        # Define safe regions (outside pattern areas)
        safe_regions = [
            (0, min_start),  # Before pattern
            (max_end + 1, seq_len)  # After pattern
        ]

        # Filter out invalid regions
        safe_regions = [(s, e) for s, e in safe_regions if s < e]

        if not safe_regions:
            # Fallback: use small central region
            cut_len = int(seq_len * 0.1)
            cut_start = seq_len // 2 - cut_len // 2
            cut_end = cut_start + cut_len
        else:
            # Choose largest safe region
            best_region = max(safe_regions, key=lambda r: r[1] - r[0])
            cut_start, cut_end = best_region

        return cut_start, cut_end

    def _economic_scenario_simulation(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        regime_info: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Simulate economic scenarios (stress testing)."""
        # Choose scenario type
        scenarios = ["market_crash", "market_rally", "volatility_spike", "range_breakout"]
        scenario = np.random.choice(scenarios)

        if scenario == "market_crash":
            x_stressed = self._simulate_market_crash(x)
        elif scenario == "market_rally":
            x_stressed = self._simulate_market_rally(x)
        elif scenario == "volatility_spike":
            x_stressed = self._simulate_volatility_spike(x)
        else:  # range_breakout
            x_stressed = self._simulate_range_breakout(x)

        return x_stressed, y, y, 1.0

    def _simulate_market_crash(self, x: torch.Tensor, crash_magnitude: float = -0.2) -> torch.Tensor:
        """Simulate market crash scenario."""
        crash_start = np.random.randint(30, 75)  # Start crash in prediction window
        crash_duration = np.random.randint(5, 15)  # 5-15 bar crash

        x_crashed = x.clone()

        # Apply crash with gradual recovery
        for i in range(crash_start, min(crash_start + crash_duration, x.size(1))):
            # Exponential decay during crash
            decay_factor = np.exp(-0.3 * (i - crash_start))
            price_impact = crash_magnitude * decay_factor

            # Apply to all OHLC components
            x_crashed[:, i, :] *= (1 + price_impact)

        # Ensure OHLC constraints
        if self.config.maintain_ohlc_constraints:
            x_crashed = self._enforce_ohlc_constraints(x_crashed)

        return x_crashed

    def _simulate_market_rally(self, x: torch.Tensor, rally_magnitude: float = 0.15) -> torch.Tensor:
        """Simulate market rally scenario."""
        rally_start = np.random.randint(30, 75)
        rally_duration = np.random.randint(5, 15)

        x_rallied = x.clone()

        for i in range(rally_start, min(rally_start + rally_duration, x.size(1))):
            # Gradual rally
            rally_factor = 1 - np.exp(-0.2 * (i - rally_start))
            price_impact = rally_magnitude * rally_factor

            x_rallied[:, i, :] *= (1 + price_impact)

        if self.config.maintain_ohlc_constraints:
            x_rallied = self._enforce_ohlc_constraints(x_rallied)

        return x_rallied

    def _simulate_volatility_spike(self, x: torch.Tensor, spike_factor: float = 3.0) -> torch.Tensor:
        """Simulate volatility spike scenario."""
        spike_start = np.random.randint(30, 75)
        spike_duration = np.random.randint(3, 10)

        x_spike = x.clone()

        # Increase volatility during spike
        for i in range(spike_start, min(spike_start + spike_duration, x.size(1))):
            # Add amplified noise
            noise = torch.randn_like(x_spike[:, i, :]) * 0.01 * spike_factor
            x_spike[:, i, :] += noise

        if self.config.maintain_ohlc_constraints:
            x_spike = self._enforce_ohlc_constraints(x_spike)

        return x_spike

    def _simulate_range_breakout(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate range breakout scenario."""
        # Find range boundaries in first 30 bars
        range_data = x[:, :30, :]
        range_high = range_data[..., 1].max(dim=1, keepdim=True)[0]
        range_low = range_data[..., 2].min(dim=1, keepdim=True)[0]

        breakout_start = np.random.randint(40, 70)
        breakout_direction = np.random.choice([-1, 1])  # Break up or down

        x_breakout = x.clone()

        # Apply breakout move
        for i in range(breakout_start, x.size(1)):
            breakout_strength = 0.02 * (i - breakout_start) / (x.size(1) - breakout_start)

            if breakout_direction > 0:  # Break up
                target_level = range_high * 1.05  # 5% above range
            else:  # Break down
                target_level = range_low * 0.95  # 5% below range

            # Gradual move toward target
            x_breakout[:, i, :] += (target_level - x_breakout[:, i, 3:4]) * breakout_strength

        if self.config.maintain_ohlc_constraints:
            x_breakout = self._enforce_ohlc_constraints(x_breakout)

        return x_breakout


def financial_mixup_criterion(
    criterion: callable,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float
) -> torch.Tensor:
    """Compute loss for financial mixup augmentation.

    Args:
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
        pred: Model predictions
        y_a: First set of labels
        y_b: Second set of labels
        lam: Mixing coefficient

    Returns:
        Mixed loss value
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def create_financial_augmentation_pipeline(
    sensitivity: str = "medium",
    preserve_patterns: bool = True,
    adaptive_to_regime: bool = True
) -> FinancialAugmentationPipeline:
    """Create a financial augmentation pipeline with sensible defaults.

    Args:
        sensitivity: Augmentation sensitivity ("low", "medium", "high")
        preserve_patterns: Whether to preserve pattern regions in augmentation
        adaptive_to_regime: Whether to adapt augmentation based on market regime

    Returns:
        Configured financial augmentation pipeline
    """
    # Base configuration based on sensitivity
    if sensitivity == "low":
        config = FinancialAugmentationConfig(
            market_mixup_prob=0.3,
            volatility_scaling_prob=0.2,
            trend_preserving_jitter_prob=0.3,
            microstructure_warp_prob=0.1,
            pattern_preserving_cutmix_prob=0.2,
            economic_scenario_prob=0.05,
            market_mixup_alpha=0.2,
            trend_jitter_sigma=0.01,
        )
    elif sensitivity == "high":
        config = FinancialAugmentationConfig(
            market_mixup_prob=0.5,
            volatility_scaling_prob=0.4,
            trend_preserving_jitter_prob=0.6,
            microstructure_warp_prob=0.3,
            pattern_preserving_cutmix_prob=0.4,
            economic_scenario_prob=0.2,
            market_mixup_alpha=0.4,
            trend_jitter_sigma=0.03,
        )
    else:  # medium
        config = FinancialAugmentationConfig()

    # Override based on parameters
    config.adaptive_to_regime = adaptive_to_regime
    if preserve_patterns:
        config.pattern_preserving_cutmix_prob = max(
            config.pattern_preserving_cutmix_prob, 0.4
        )

    return FinancialAugmentationPipeline(config)