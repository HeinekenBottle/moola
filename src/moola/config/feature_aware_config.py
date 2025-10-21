"""Configuration for Feature-Aware Pre-training System.

Centralized configuration for feature-aware bidirectional masked LSTM autoencoder
pre-training and enhanced SimpleLSTM fine-tuning.

Usage:
    >>> from moola.config.feature_aware_config import (
    ...     get_feature_aware_pretraining_config,
    ...     get_enhanced_simple_lstm_config
    ... )
    >>> pretrain_config = get_feature_aware_pretraining_config("concat")
    >>> lstm_config = get_enhanced_simple_lstm_config()
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class FeatureAwarePretrainingConfig:
    """Configuration for feature-aware masked LSTM pre-training."""

    # Model architecture
    ohlc_dim: int = 4
    feature_dim: int = 25
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    feature_fusion: Literal["concat", "add", "gate"] = "concat"

    # Masking strategy
    mask_ratio: float = 0.15
    mask_strategy: Literal["random", "block", "patch"] = "patch"
    patch_size: int = 7

    # Loss weights
    loss_weights: dict = field(
        default_factory=lambda: {
            "ohlc_weight": 0.4,
            "feature_weight": 0.4,
            "regularization_weight": 0.2,
        }
    )

    # Training hyperparameters
    learning_rate: float = 1e-3
    batch_size: int = 256
    n_epochs: int = 50
    early_stopping_patience: int = 10
    val_split: float = 0.1

    # Performance settings
    device: str = "cuda"
    seed: int = 1337
    use_amp: bool = True
    num_workers: int = 8

    # Checkpointing
    save_encoder: bool = True
    save_best_only: bool = True
    checkpoint_frequency: int = 10


@dataclass
class EnhancedSimpleLSTMConfig:
    """Configuration for enhanced SimpleLSTM with feature-aware support."""

    # Model architecture
    hidden_size: int = 128
    num_layers: int = 1
    num_heads: int = 2
    dropout: float = 0.1
    feature_fusion: Literal["concat", "add", "gate"] = "concat"

    # Training hyperparameters
    learning_rate: float = 5e-4
    batch_size: int = 512
    n_epochs: int = 60
    early_stopping_patience: int = 20
    val_split: float = 0.15

    # Augmentation
    mixup_alpha: float = 0.4
    cutmix_prob: float = 0.5
    use_temporal_aug: bool = True
    jitter_prob: float = 0.5
    scaling_prob: float = 0.3
    time_warp_prob: float = 0.0

    # Transfer learning
    unfreeze_encoder_after: int = 10  # Two-phase training
    freeze_encoder_initially: bool = True

    # Performance settings
    device: str = "cuda"
    seed: int = 1337
    use_amp: bool = True
    num_workers: int = 16


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering in pre-training."""

    # Feature selection
    use_returns: bool = True
    use_zscore: bool = True
    use_moving_averages: bool = True
    use_rsi: bool = True
    use_macd: bool = True
    use_volatility: bool = True
    use_bollinger: bool = True
    use_atr: bool = True
    use_candle_patterns: bool = True
    use_swing_points: bool = True
    use_gaps: bool = True
    use_volume_proxy: bool = True

    # Feature hyperparameters
    ma_windows: list = field(default_factory=lambda: [5, 10, 20])
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    volatility_windows: list = field(default_factory=lambda: [5, 10, 20])
    bollinger_window: int = 20
    bollinger_num_std: float = 2.0
    atr_period: int = 14
    swing_window: int = 5

    # Processing
    robust_scaling: bool = True
    apply_smoothing: bool = False  # Generally not needed for pre-training
    outlier_removal: bool = True
    outlier_threshold: float = 3.0


# Preset configurations for different scenarios


def get_feature_aware_pretraining_config(
    feature_fusion: str = "concat", preset: str = "default"
) -> FeatureAwarePretrainingConfig:
    """Get feature-aware pre-training configuration.

    Args:
        feature_fusion: Fusion strategy ('concat', 'add', 'gate')
        preset: Preset configuration ('default', 'fast', 'high_quality')

    Returns:
        FeatureAwarePretrainingConfig instance
    """
    base_config = FeatureAwarePretrainingConfig(feature_fusion=feature_fusion)

    if preset == "fast":
        # Faster pre-training for experimentation
        base_config.batch_size = 512
        base_config.n_epochs = 30
        base_config.early_stopping_patience = 5
        base_config.patch_size = 15  # Larger patches for faster training

    elif preset == "high_quality":
        # High-quality pre-training for production
        base_config.hidden_dim = 256
        base_config.num_layers = 3
        base_config.batch_size = 128
        base_config.n_epochs = 100
        base_config.early_stopping_patience = 20
        base_config.learning_rate = 5e-4
        base_config.loss_weights = {
            "ohlc_weight": 0.3,
            "feature_weight": 0.5,
            "regularization_weight": 0.2,
        }

    return base_config


def get_enhanced_simple_lstm_config(preset: str = "default") -> EnhancedSimpleLSTMConfig:
    """Get enhanced SimpleLSTM configuration.

    Args:
        preset: Preset configuration ('default', 'transfer_learning', 'small_dataset')

    Returns:
        EnhancedSimpleLSTMConfig instance
    """
    base_config = EnhancedSimpleLSTMConfig()

    if preset == "transfer_learning":
        # Optimized for transfer learning from pre-trained encoder
        base_config.unfreeze_encoder_after = 10
        base_config.freeze_encoder_initially = True
        base_config.learning_rate = 3e-4  # Lower LR for fine-tuning
        base_config.n_epochs = 40
        base_config.early_stopping_patience = 15

    elif preset == "small_dataset":
        # Optimized for small dataset (98 samples)
        base_config.num_heads = 1  # Reduce complexity
        base_config.dropout = 0.05  # Less regularization for small data
        base_config.mixup_alpha = 0.2  # Less aggressive augmentation
        base_config.use_temporal_aug = True
        base_config.time_warp_prob = 0.0  # Disable time warping
        base_config.n_epochs = 100  # More epochs for small data
        base_config.early_stopping_patience = 30

    return base_config


def get_feature_engineering_config(preset: str = "default") -> FeatureEngineeringConfig:
    """Get feature engineering configuration.

    Args:
        preset: Preset configuration ('default', 'minimal', 'comprehensive')

    Returns:
        FeatureEngineeringConfig instance
    """
    base_config = FeatureEngineeringConfig()

    if preset == "minimal":
        # Minimal feature set for faster training
        base_config.use_macd = False
        base_config.use_bollinger = False
        base_config.use_atr = False
        base_config.use_swing_points = False
        base_config.use_gaps = False
        base_config.use_volume_proxy = False
        base_config.ma_windows = [10, 20]

    elif preset == "comprehensive":
        # Comprehensive feature set for maximum performance
        base_config.ma_windows = [5, 10, 20, 50]
        base_config.volatility_windows = [5, 10, 20, 50]
        base_config.bollinger_num_std = 1.5
        base_config.swing_window = 3

    return base_config


# GPU-optimized configurations


def get_gpu_optimized_config(gpu_memory_gb: int = 24) -> dict:
    """Get GPU-optimized configuration based on available memory.

    Args:
        gpu_memory_gb: Available GPU memory in GB

    Returns:
        Dictionary with optimized settings
    """
    if gpu_memory_gb >= 24:  # RTX 4090, A100
        return {
            "batch_size": 512,
            "num_workers": 16,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
        }
    elif gpu_memory_gb >= 16:  # RTX 3080, V100
        return {
            "batch_size": 256,
            "num_workers": 12,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": True,
        }
    elif gpu_memory_gb >= 8:  # RTX 3070, GTX 1080 Ti
        return {
            "batch_size": 128,
            "num_workers": 8,
            "pin_memory": True,
            "prefetch_factor": 2,
            "persistent_workers": False,
        }
    else:  # Low memory GPUs
        return {
            "batch_size": 64,
            "num_workers": 4,
            "pin_memory": False,
            "prefetch_factor": 1,
            "persistent_workers": False,
        }


# Environment-specific configurations


def get_environment_config() -> dict:
    """Get environment-specific configuration.

    Returns:
        Dictionary with environment settings
    """
    import psutil
    import torch

    # GPU detection
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
        )
    else:
        gpu_count = 0
        gpu_name = "None"
        gpu_memory = 0

    # CPU detection
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total / (1024**3)

    return {
        "gpu_available": gpu_count > 0,
        "gpu_count": gpu_count,
        "gpu_name": gpu_name,
        "gpu_memory_gb": gpu_memory,
        "cpu_count": cpu_count,
        "memory_gb": memory_gb,
        "recommended_device": "cuda" if gpu_count > 0 else "cpu",
        "recommended_batch_size": (
            get_gpu_optimized_config(gpu_memory)["batch_size"] if gpu_count > 0 else 32
        ),
    }


# Validation functions


def validate_feature_aware_config(config: FeatureAwarePretrainingConfig) -> bool:
    """Validate feature-aware pre-training configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    # Validate dimensions
    if config.ohlc_dim <= 0:
        raise ValueError("ohlc_dim must be positive")
    if config.feature_dim < 0:
        raise ValueError("feature_dim must be non-negative")
    if config.hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")

    # Validate fusion strategy
    if config.feature_fusion not in ["concat", "add", "gate"]:
        raise ValueError(f"Invalid feature_fusion: {config.feature_fusion}")

    # Validate masking
    if not 0 < config.mask_ratio < 1:
        raise ValueError("mask_ratio must be between 0 and 1")
    if config.mask_strategy not in ["random", "block", "patch"]:
        raise ValueError(f"Invalid mask_strategy: {config.mask_strategy}")
    if config.patch_size <= 0:
        raise ValueError("patch_size must be positive")

    # Validate loss weights
    total_weight = sum(config.loss_weights.values())
    if not abs(total_weight - 1.0) < 1e-6:
        raise ValueError(f"Loss weights must sum to 1.0, got {total_weight}")

    # Validate training parameters
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.n_epochs <= 0:
        raise ValueError("n_epochs must be positive")

    return True


def validate_enhanced_lstm_config(config: EnhancedSimpleLSTMConfig) -> bool:
    """Validate enhanced SimpleLSTM configuration.

    Args:
        config: Configuration to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    # Validate architecture
    if config.hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    if config.num_layers <= 0:
        raise ValueError("num_layers must be positive")
    if config.num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if not 0 <= config.dropout < 1:
        raise ValueError("dropout must be between 0 and 1")

    # Validate fusion strategy
    if config.feature_fusion not in ["concat", "add", "gate"]:
        raise ValueError(f"Invalid feature_fusion: {config.feature_fusion}")

    # Validate training parameters
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if config.n_epochs <= 0:
        raise ValueError("n_epochs must be positive")

    # Validate augmentation
    if not 0 <= config.mixup_alpha <= 1:
        raise ValueError("mixup_alpha must be between 0 and 1")
    if not 0 <= config.cutmix_prob <= 1:
        raise ValueError("cutmix_prob must be between 0 and 1")

    return True
