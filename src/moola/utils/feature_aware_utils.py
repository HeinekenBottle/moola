"""Utility functions for Feature-Aware Pre-training System.

Helper functions for data preparation, model initialization, training orchestration,
and result analysis for feature-aware pre-training and enhanced SimpleLSTM fine-tuning.

Usage:
    >>> from moola.utils.feature_aware_utils import (
    ...     prepare_feature_aware_data,
    ...     create_feature_aware_pretrainer,
    ...     run_feature_aware_pretraining,
    ...     evaluate_transfer_learning
    ... )
    >>>
    >>> # Prepare data
    >>> X_ohlc, X_features, y = prepare_feature_aware_data(
    ...     raw_ohlc, feature_config, split_ratio=0.8
    ... )
    >>>
    >>> # Run pre-training
    >>> encoder_path = run_feature_aware_pretraining(
    ...     X_ohlc_unlabeled, X_features_unlabeled, config
    ... )
    >>>
    >>> # Evaluate transfer learning
    >>> results = evaluate_transfer_learning(
    ...     X_ohlc_train, X_features_train, y_train,
    ...     X_ohlc_val, X_features_val, y_val,
    ...     encoder_path
    ... )
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from ..config.feature_aware_config import (
    EnhancedSimpleLSTMConfig,
    FeatureAwarePretrainingConfig,
    FeatureEngineeringConfig,
    get_environment_config,
    get_gpu_optimized_config,
    validate_enhanced_lstm_config,
    validate_feature_aware_config,
)
from ..features.feature_engineering import AdvancedFeatureEngineer, FeatureConfig
from ..models.enhanced_simple_lstm import EnhancedSimpleLSTMModel
from ..pretraining.feature_aware_masked_lstm_pretrain import (
    FeatureAwareMaskedLSTMPretrainer,
    visualize_feature_aware_reconstruction,
)


def prepare_feature_aware_data(
    X_ohlc: np.ndarray,
    feature_config: Optional[FeatureEngineeringConfig] = None,
    split_ratio: float = 0.8,
    seed: int = 1337,
    unlabeled_only: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]],  # Labeled: (ohlc, features, labels)
    Tuple[np.ndarray, np.ndarray],  # Unlabeled: (ohlc, features)
]:
    """Prepare feature-aware data for pre-training and/or fine-tuning.

    Args:
        X_ohlc: Raw OHLC data [N, T, 4] or [N, T*4]
        feature_config: Feature engineering configuration
        split_ratio: Train/validation split ratio (for labeled data)
        seed: Random seed for reproducibility
        unlabeled_only: If True, only return processed data without labels

    Returns:
        For labeled data: (X_ohlc, X_features, y) or (X_ohlc_train, X_features_train, y_train)
        For unlabeled data: (X_ohlc_processed, X_features_processed)
    """
    logger.info(f"Preparing feature-aware data from {X_ohlc.shape}")

    # Reshape if needed
    if X_ohlc.ndim == 2 and X_ohlc.shape[1] == 420:  # 105 * 4
        X_ohlc = X_ohlc.reshape(-1, 105, 4)
    elif X_ohlc.ndim != 3 or X_ohlc.shape[-1] != 4:
        raise ValueError(f"Expected OHLC shape [N, 105, 4], got {X_ohlc.shape}")

    # Feature engineering
    if feature_config is None:
        feature_config = FeatureEngineeringConfig()

    # Convert FeatureConfig to FeatureConfig for AdvancedFeatureEngineer
    advanced_config = FeatureConfig(
        use_returns=feature_config.use_returns,
        use_zscore=feature_config.use_zscore,
        use_moving_averages=feature_config.use_moving_averages,
        use_rsi=feature_config.use_rsi,
        use_macd=feature_config.use_macd,
        use_volatility=feature_config.use_volatility,
        use_bollinger=feature_config.use_bollinger,
        use_atr=feature_config.use_atr,
        use_candle_patterns=feature_config.use_candle_patterns,
        use_swing_points=feature_config.use_swing_points,
        use_gaps=feature_config.use_gaps,
        use_volume_proxy=feature_config.use_volume_proxy,
        ma_windows=feature_config.ma_windows,
        rsi_period=feature_config.rsi_period,
        macd_fast=feature_config.macd_fast,
        macd_slow=feature_config.macd_slow,
        macd_signal=feature_config.macd_signal,
        volatility_windows=feature_config.volatility_windows,
        bollinger_window=feature_config.bollinger_window,
        bollinger_num_std=feature_config.bollinger_num_std,
        atr_period=feature_config.atr_period,
        swing_window=feature_config.swing_window,
    )

    engineer = AdvancedFeatureEngineer(advanced_config)
    X_features = engineer.transform(X_ohlc)

    logger.success(f"Feature engineering complete: {X_ohlc.shape} → {X_features.shape}")

    if unlabeled_only:
        return X_ohlc, X_features
    else:
        # For labeled data, we would expect y to be provided separately
        # This function assumes the caller will handle labels
        return X_ohlc, X_features


def create_feature_aware_pretrainer(
    config: FeatureAwarePretrainingConfig, ohlc_dim: int = 4, feature_dim: Optional[int] = None
) -> FeatureAwareMaskedLSTMPretrainer:
    """Create feature-aware pre-trainer with configuration.

    Args:
        config: Pre-training configuration
        ohlc_dim: OHLC feature dimension
        feature_dim: Engineered feature dimension (if None, use config value)

    Returns:
        Configured FeatureAwareMaskedLSTMPretrainer
    """
    # Validate configuration
    validate_feature_aware_config(config)

    # Get environment settings
    env_config = get_environment_config()
    gpu_config = get_gpu_optimized_config(env_config["gpu_memory_gb"])

    # Override batch size if needed for GPU memory constraints
    if config.batch_size > gpu_config["batch_size"]:
        logger.warning(
            f"Reducing batch size from {config.batch_size} to {gpu_config['batch_size']} "
            f"for GPU memory constraints"
        )
        config.batch_size = gpu_config["batch_size"]

    # Update config with environment settings
    config.device = env_config["recommended_device"]
    config.num_workers = gpu_config["num_workers"]

    logger.info(f"Creating feature-aware pre-trainer:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Feature fusion: {config.feature_fusion}")
    logger.info(f"  Mask strategy: {config.mask_strategy}")
    logger.info(f"  OHLC dim: {ohlc_dim}")
    logger.info(f"  Feature dim: {feature_dim or config.feature_dim}")

    # Create pre-trainer
    pretrainer = FeatureAwareMaskedLSTMPretrainer(
        ohlc_dim=ohlc_dim,
        feature_dim=feature_dim or config.feature_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        feature_fusion=config.feature_fusion,
        mask_ratio=config.mask_ratio,
        mask_strategy=config.mask_strategy,
        patch_size=config.patch_size,
        loss_weights=config.loss_weights,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        device=config.device,
        seed=config.seed,
    )

    return pretrainer


def run_feature_aware_pretraining(
    X_ohlc: np.ndarray,
    X_features: np.ndarray,
    config: Optional[FeatureAwarePretrainingConfig] = None,
    save_dir: Optional[Path] = None,
    verbose: bool = True,
) -> Path:
    """Run complete feature-aware pre-training pipeline.

    Args:
        X_ohlc: OHLC data [N, T, 4]
        X_features: Engineered features [N, T, F]
        config: Pre-training configuration (uses default if None)
        save_dir: Directory to save results (uses default if None)
        verbose: Print training progress

    Returns:
        Path to saved encoder checkpoint
    """
    if config is None:
        config = FeatureAwarePretrainingConfig()

    if save_dir is None:
        save_dir = Path("artifacts/pretrained/feature_aware")

    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting feature-aware pre-training")
    logger.info(f"  OHLC data shape: {X_ohlc.shape}")
    logger.info(f"  Features shape: {X_features.shape}")
    logger.info(f"  Save directory: {save_dir}")

    # Create pre-trainer
    pretrainer = create_feature_aware_pretrainer(
        config, ohlc_dim=X_ohlc.shape[-1], feature_dim=X_features.shape[-1]
    )

    # Set save path
    encoder_path = save_dir / f"feature_aware_encoder_{config.feature_fusion}.pt"

    # Start pre-training
    start_time = time.time()

    history = pretrainer.pretrain(
        X_ohlc=X_ohlc,
        X_features=X_features,
        n_epochs=config.n_epochs,
        val_split=config.val_split,
        patience=config.early_stopping_patience,
        save_path=encoder_path if config.save_encoder else None,
        verbose=verbose,
    )

    training_time = time.time() - start_time

    # Save training history
    history_path = save_dir / f"training_history_{config.feature_fusion}.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    # Save configuration
    config_path = save_dir / f"config_{config.feature_fusion}.json"
    with open(config_path, "w") as f:
        # Convert dataclass to dict for JSON serialization
        config_dict = {
            "ohlc_dim": config.ohlc_dim,
            "feature_dim": config.feature_dim,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "feature_fusion": config.feature_fusion,
            "mask_ratio": config.mask_ratio,
            "mask_strategy": config.mask_strategy,
            "patch_size": config.patch_size,
            "loss_weights": config.loss_weights,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "n_epochs": config.n_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "val_split": config.val_split,
            "device": config.device,
            "seed": config.seed,
            "use_amp": config.use_amp,
            "num_workers": config.num_workers,
        }
        json.dump(config_dict, f, indent=2)

    logger.success(f"Feature-aware pre-training complete in {training_time:.1f} seconds")
    logger.info(f"  Encoder saved: {encoder_path}")
    logger.info(f"  Final training loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"  Final validation loss: {history['val_loss'][-1]:.4f}")
    logger.info(f"  Best validation loss: {min(history['val_loss']):.4f}")

    return encoder_path


def evaluate_transfer_learning(
    X_ohlc_train: np.ndarray,
    X_features_train: np.ndarray,
    y_train: np.ndarray,
    X_ohlc_val: np.ndarray,
    X_features_val: np.ndarray,
    y_val: np.ndarray,
    encoder_path: Path,
    lstm_config: Optional[EnhancedSimpleLSTMConfig] = None,
    modes: List[str] = ["ohlc_only", "feature_aware", "pretrained_ohlc", "pretrained_features"],
) -> Dict[str, Dict]:
    """Evaluate transfer learning performance across different modes.

    Args:
        X_ohlc_train: Training OHLC data [N_train, T, 4]
        X_features_train: Training features [N_train, T, F]
        y_train: Training labels [N_train]
        X_ohlc_val: Validation OHLC data [N_val, T, 4]
        X_features_val: Validation features [N_val, T, F]
        y_val: Validation labels [N_val]
        encoder_path: Path to pre-trained encoder
        lstm_config: Enhanced SimpleLSTM configuration
        modes: List of modes to evaluate

    Returns:
        Dictionary with results for each mode
    """
    if lstm_config is None:
        lstm_config = EnhancedSimpleLSTMConfig()

    validate_enhanced_lstm_config(lstm_config)

    results = {}

    logger.info("Evaluating transfer learning performance")

    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating mode: {mode}")
        logger.info(f"{'='*60}")

        if mode == "ohlc_only":
            # OHLC-only baseline
            X_train = X_ohlc_train
            X_val = X_ohlc_val
            encoder_path_mode = None

        elif mode == "feature_aware":
            # Feature-aware from scratch
            X_train = np.concatenate([X_ohlc_train, X_features_train], axis=-1)
            X_val = np.concatenate([X_ohlc_val, X_features_val], axis=-1)
            encoder_path_mode = None

        elif mode == "pretrained_ohlc":
            # Transfer learning with OHLC only
            X_train = X_ohlc_train
            X_val = X_ohlc_val
            encoder_path_mode = encoder_path

        elif mode == "pretrained_features":
            # Transfer learning with features
            X_train = np.concatenate([X_ohlc_train, X_features_train], axis=-1)
            X_val = np.concatenate([X_ohlc_val, X_features_val], axis=-1)
            encoder_path_mode = encoder_path

        else:
            logger.warning(f"Unknown mode: {mode}, skipping")
            continue

        # Create and train model
        model = EnhancedSimpleLSTMModel(
            hidden_size=lstm_config.hidden_size,
            num_layers=lstm_config.num_layers,
            num_heads=lstm_config.num_heads,
            dropout=lstm_config.dropout,
            feature_fusion=lstm_config.feature_fusion,
            n_epochs=lstm_config.n_epochs,
            batch_size=lstm_config.batch_size,
            learning_rate=lstm_config.learning_rate,
            device=lstm_config.device,
            seed=lstm_config.seed,
            mixup_alpha=lstm_config.mixup_alpha,
            cutmix_prob=lstm_config.cutmix_prob,
            use_temporal_aug=lstm_config.use_temporal_aug,
            jitter_prob=lstm_config.jitter_prob,
            scaling_prob=lstm_config.scaling_prob,
            time_warp_prob=lstm_config.time_warp_prob,
        )

        # Train model
        start_time = time.time()
        model.fit(
            X_train,
            y_train,
            pretrained_encoder_path=encoder_path_mode,
            freeze_encoder=lstm_config.freeze_encoder_initially,
            unfreeze_encoder_after=lstm_config.unfreeze_encoder_after,
        )
        training_time = time.time() - start_time

        # Evaluate
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)

        results[mode] = {
            "accuracy": accuracy,
            "classification_report": report,
            "training_time": training_time,
            "input_shape": X_train.shape,
            "model_type": mode,
            "used_pretrained": encoder_path_mode is not None,
        }

        logger.info(f"Mode: {mode}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Training time: {training_time:.1f}s")
        logger.info(f"  Input shape: {X_train.shape}")
        logger.info(f"  Pre-trained: {encoder_path_mode is not None}")

    # Summary comparison
    logger.info(f"\n{'='*60}")
    logger.info("TRANSFER LEARNING COMPARISON")
    logger.info(f"{'='*60}")

    for mode, result in results.items():
        logger.info(
            f"{mode:20s}: {result['accuracy']:.4f} "
            f"({result['training_time']:.1f}s) "
            f"[{'PT' if result['used_pretrained'] else 'Scratch'}]"
        )

    return results


def analyze_encoder_importance(
    encoder_path: Path,
    X_ohlc_sample: np.ndarray,
    X_features_sample: np.ndarray,
    device: str = "cuda",
) -> Dict[str, np.ndarray]:
    """Analyze feature importance in pre-trained encoder.

    Args:
        encoder_path: Path to pre-trained encoder
        X_ohlc_sample: Sample OHLC data [N_samples, T, 4]
        X_features_sample: Sample features [N_samples, T, F]
        device: Device for computation

    Returns:
        Dictionary with importance analysis results
    """
    logger.info("Analyzing encoder feature importance")

    # Load encoder
    checkpoint = torch.load(encoder_path, map_location=device)
    hyperparams = checkpoint["hyperparams"]

    # Create model for analysis
    from ..models.feature_aware_bilstm_masked_autoencoder import (
        FeatureAwareBiLSTMMaskedAutoencoder,
    )

    model = FeatureAwareBiLSTMMaskedAutoencoder(
        ohlc_dim=hyperparams["ohlc_dim"],
        feature_dim=hyperparams["feature_dim"],
        hidden_dim=hyperparams["hidden_dim"],
        num_layers=hyperparams["num_layers"],
        feature_fusion=hyperparams["feature_fusion"],
    ).to(device)

    model.load_state_dict(checkpoint["encoder_state_dict"], strict=False)
    model.eval()

    # Convert to tensors
    ohlc_tensor = torch.FloatTensor(X_ohlc_sample).to(device)
    features_tensor = torch.FloatTensor(X_features_sample).to(device)

    # Feature-wise ablation study
    ohlc_importance = []
    feature_importance = []

    # OHLC importance (mask each feature and measure reconstruction)
    with torch.no_grad():
        # Baseline reconstruction
        ohlc_masked = torch.zeros_like(ohlc_tensor)
        features_masked = torch.zeros_like(features_tensor)

        # Apply original mask tokens
        ohlc_masked[:] = model.ohlc_mask_token
        features_masked[:] = model.feature_mask_token

        baseline_ohlc, baseline_features = model(ohlc_masked, features_masked)

        # Ablate OHLC features one by one
        for i in range(hyperparams["ohlc_dim"]):
            ohlc_ablated = ohlc_tensor.clone()
            ohlc_ablated[..., i] = model.ohlc_mask_token[..., i]

            recon_ohlc, recon_features = model(ohlc_ablated, features_tensor)

            # Measure change in reconstruction quality
            ohlc_change = F.mse_loss(recon_ohlc, ohlc_tensor).item()
            ohlc_importance.append(ohlc_change)

        # Ablate engineered features (sample a few if too many)
        feature_indices = range(min(10, hyperparams["feature_dim"]))  # Sample up to 10 features

        for i in feature_indices:
            features_ablated = features_tensor.clone()
            features_ablated[..., i] = model.feature_mask_token[..., i]

            recon_ohlc, recon_features = model(ohlc_tensor, features_ablated)

            # Measure change in reconstruction quality
            feature_change = F.mse_loss(recon_features, features_tensor).item()
            feature_importance.append(feature_change)

    logger.info("Feature importance analysis complete")
    logger.info(f"  OHLC features analyzed: {len(ohlc_importance)}")
    logger.info(f"  Engineered features analyzed: {len(feature_importance)}")

    return {
        "ohlc_importance": np.array(ohlc_importance),
        "feature_importance": np.array(feature_importance),
        "feature_indices": list(feature_indices),
        "ohlc_feature_names": ["open", "high", "low", "close"],
    }


def create_experiment_report(results: Dict[str, Dict], save_path: Optional[Path] = None) -> str:
    """Create comprehensive experiment report.

    Args:
        results: Results from evaluate_transfer_learning
        save_path: Path to save report (optional)

    Returns:
        Report as string
    """
    report_lines = [
        "FEATURE-AWARE PRE-TRAINING EXPERIMENT REPORT",
        "=" * 60,
        "",
        "SUMMARY",
        "-" * 30,
    ]

    # Summary table
    report_lines.extend(
        [
            f"{'Mode':<20} {'Accuracy':<10} {'Time (s)':<10} {'Type':<10}",
            "-" * 60,
        ]
    )

    for mode, result in results.items():
        model_type = "Pre-trained" if result["used_pretrained"] else "From scratch"
        report_lines.append(
            f"{mode:<20} {result['accuracy']:<10.4f} {result['training_time']:<10.1f} {model_type:<10}"
        )

    # Detailed analysis
    report_lines.extend(["", "DETAILED RESULTS", "-" * 30, ""])

    for mode, result in results.items():
        report_lines.extend(
            [
                f"Mode: {mode}",
                f"  Accuracy: {result['accuracy']:.4f}",
                f"  Training time: {result['training_time']:.1f} seconds",
                f"  Input shape: {result['input_shape']}",
                f"  Pre-trained: {result['used_pretrained']}",
                f"  Classification report:",
            ]
        )

        # Add classification report details
        for class_name, metrics in result["classification_report"].items():
            if isinstance(metrics, dict):
                report_lines.append(f"    {class_name}:")
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"      {metric_name}: {value:.4f}")
                    else:
                        report_lines.append(f"      {metric_name}: {value}")

        report_lines.append("")

    # Best performing model
    best_mode = max(results.keys(), key=lambda k: results[k]["accuracy"])
    report_lines.extend(
        [
            "BEST PERFORMING MODEL",
            "-" * 30,
            f"Mode: {best_mode}",
            f"Accuracy: {results[best_mode]['accuracy']:.4f}",
            f"Training time: {results[best_mode]['training_time']:.1f}s",
            "",
        ]
    )

    # Recommendations
    report_lines.extend(
        [
            "RECOMMENDATIONS",
            "-" * 30,
        ]
    )

    if results["pretrained_features"]["accuracy"] > results["ohlc_only"]["accuracy"]:
        improvement = results["pretrained_features"]["accuracy"] - results["ohlc_only"]["accuracy"]
        report_lines.append(
            f"✅ Feature-aware pre-training improves accuracy by {improvement:.2%} "
            f"({improvement*100:.1f} percentage points)"
        )
    else:
        report_lines.append("⚠️  Feature-aware pre-training does not improve accuracy")

    if results["pretrained_features"]["accuracy"] > results["feature_aware"]["accuracy"]:
        report_lines.append("✅ Transfer learning provides benefits over from-scratch training")

    if results["pretrained_ohlc"]["accuracy"] > results["ohlc_only"]["accuracy"]:
        report_lines.append("✅ Even OHLC-only transfer learning is beneficial")

    report = "\n".join(report_lines)

    # Save report if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        logger.info(f"Experiment report saved: {save_path}")

    return report
