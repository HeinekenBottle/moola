"""Feature-Aware Pre-training CLI Commands.

CLI commands for feature-aware bidirectional masked LSTM autoencoder pre-training
and enhanced SimpleLSTM fine-tuning with transfer learning.

Usage:
    # Feature-aware pre-training
    python -m moola.cli_feature_aware pretrain-features \\
        --ohlc data/unlabeled_ohlc.npy \\
        --features data/unlabeled_features.npy \\
        --output artifacts/pretrained/feature_aware_encoder.pt \\
        --fusion concat

    # Transfer learning evaluation
    python -m moola.cli_feature_aware evaluate-transfer \\
        --train-ohlc data/train_ohlc.npy \\
        --train-features data/train_features.npy \\
        --train-labels data/train_labels.npy \\
        --val-ohlc data/val_ohlc.npy \\
        --val-features data/val_features.npy \\
        --val-labels data/val_labels.npy \\
        --encoder artifacts/pretrained/feature_aware_encoder.pt
"""

import click
import numpy as np
from pathlib import Path
from rich import print as rprint

from .config.feature_aware_config import (
    FeatureAwarePretrainingConfig,
    EnhancedSimpleLSTMConfig,
    FeatureEngineeringConfig,
    get_feature_aware_pretraining_config,
    get_enhanced_simple_lstm_config,
    get_feature_engineering_config,
)
from .features.feature_engineering import AdvancedFeatureEngineer, FeatureConfig
from .utils.feature_aware_utils import (
    prepare_feature_aware_data,
    run_feature_aware_pretraining,
    evaluate_transfer_learning,
    analyze_encoder_importance,
    create_experiment_report,
)


@click.group(help="Feature-Aware Pre-training CLI")
def app():
    pass


@app.command()
@click.option("--ohlc", "ohlc_path", type=click.Path(exists=True), required=True,
              help="Path to OHLC data (.npy file) [N, 105, 4] or [N, 420]")
@click.option("--features", "features_path", type=click.Path(exists=True),
              help="Path to engineered features (.npy file) [N, 105, F]. If not provided, will compute from OHLC.")
@click.option("--feature-config", "feature_config_preset", default="default",
              type=click.Choice(["default", "minimal", "comprehensive"]),
              help="Feature engineering preset (used if features not provided)")
@click.option("--output", "output_path", type=click.Path(),
              default="artifacts/pretrained/feature_aware_encoder.pt",
              help="Output path for trained encoder")
@click.option("--fusion", "feature_fusion", default="concat",
              type=click.Choice(["concat", "add", "gate"]),
              help="Feature fusion strategy")
@click.option("--preset", "training_preset", default="default",
              type=click.Choice(["default", "fast", "high_quality"]),
              help="Training preset configuration")
@click.option("--epochs", default=50, type=int, help="Number of training epochs")
@click.option("--batch-size", default=256, type=int, help="Training batch size")
@click.option("--learning-rate", default=1e-3, type=float, help="Learning rate")
@click.option("--mask-strategy", default="patch", type=click.Choice(["random", "block", "patch"]),
              help="Masking strategy")
@click.option("--mask-ratio", default=0.15, type=float, help="Masking ratio (0.15 = 15%)")
@click.option("--device", default="cuda", type=click.Choice(["cpu", "cuda"]),
              help="Training device")
@click.option("--save-dir", type=click.Path(),
              default="artifacts/pretrained/feature_aware",
              help="Directory to save training artifacts")
@click.option("--verbose/--quiet", default=True, help="Print training progress")
def pretrain_features(
    ohlc_path, features_path, feature_config_preset, output_path,
    feature_fusion, training_preset, epochs, batch_size, learning_rate,
    mask_strategy, mask_ratio, device, save_dir, verbose
):
    """Pre-train feature-aware bidirectional masked LSTM autoencoder.

    Examples:
        # With pre-computed features
        moola cli-feature-aware pretrain-features \\
            --ohlc data/unlabeled_ohlc.npy \\
            --features data/unlabeled_features.npy \\
            --fusion concat

        # Compute features automatically
        moola cli-feature-aware pretrain-features \\
            --ohlc data/unlabeled_ohlc.npy \\
            --feature-config comprehensive \\
            --fusion gate \\
            --preset high_quality
    """
    rprint(f"\n{'='*70}")
    rprint(f"FEATURE-AWARE PRE-TRAINING")
    rprint(f"{'='*70}")
    rprint(f"OHLC data: {ohlc_path}")
    rprint(f"Features: {features_path or 'Computed from OHLC'}")
    rprint(f"Fusion strategy: {feature_fusion}")
    rprint(f"Training preset: {training_preset}")
    rprint(f"Device: {device}")
    rprint(f"{'='*70}\n")

    # Load OHLC data
    rprint("Loading OHLC data...")
    X_ohlc = np.load(ohlc_path)
    rprint(f"  Loaded OHLC shape: {X_ohlc.shape}")

    # Load or compute features
    if features_path:
        rprint("Loading pre-computed features...")
        X_features = np.load(features_path)
        rprint(f"  Loaded features shape: {X_features.shape}")
    else:
        rprint("Computing features from OHLC data...")
        feature_config = get_feature_engineering_config(feature_config_preset)

        # Convert to AdvancedFeatureEngineer config
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
        rprint(f"  Computed features shape: {X_features.shape}")

    # Get training configuration
    config = get_feature_aware_pretraining_config(feature_fusion, training_preset)

    # Override with command line arguments
    config.n_epochs = epochs
    config.batch_size = batch_size
    config.learning_rate = learning_rate
    config.mask_strategy = mask_strategy
    config.mask_ratio = mask_ratio
    config.device = device

    # Update output path
    output_path = Path(output_path)
    save_dir = Path(save_dir)

    # Run pre-training
    encoder_path = run_feature_aware_pretraining(
        X_ohlc=X_ohlc,
        X_features=X_features,
        config=config,
        save_dir=save_dir,
        verbose=verbose
    )

    rprint(f"\n✅ Feature-aware pre-training complete!")
    rprint(f"   Encoder saved: {encoder_path}")
    rprint(f"   Training artifacts: {save_dir}")


@app.command()
@click.option("--train-ohlc", "train_ohlc_path", type=click.Path(exists=True), required=True,
              help="Path to training OHLC data")
@click.option("--train-features", "train_features_path", type=click.Path(exists=True),
              help="Path to training features data")
@click.option("--train-labels", "train_labels_path", type=click.Path(exists=True), required=True,
              help="Path to training labels")
@click.option("--val-ohlc", "val_ohlc_path", type=click.Path(exists=True), required=True,
              help="Path to validation OHLC data")
@click.option("--val-features", "val_features_path", type=click.Path(exists=True),
              help="Path to validation features data")
@click.option("--val-labels", "val_labels_path", type=click.Path(exists=True), required=True,
              help="Path to validation labels")
@click.option("--encoder", "encoder_path", type=click.Path(exists=True), required=True,
              help="Path to pre-trained encoder")
@click.option("--modes", "eval_modes", multiple=True, default=["ohlc_only", "feature_aware", "pretrained_ohlc", "pretrained_features"],
              help="Evaluation modes")
@click.option("--fusion", "feature_fusion", default="concat",
              type=click.Choice(["concat", "add", "gate"]),
              help="Feature fusion strategy")
@click.option("--epochs", default=60, type=int, help="Fine-tuning epochs")
@click.option("--batch-size", default=512, type=int, help="Fine-tuning batch size")
@click.option("--learning-rate", default=5e-4, type=float, help="Fine-tuning learning rate")
@click.option("--device", default="cuda", type=click.Choice(["cpu", "cuda"]),
              help="Training device")
@click.option("--output", "output_path", type=click.Path(),
              default="experiments/feature_aware_transfer_results.json",
              help="Output path for evaluation results")
@click.option("--report", "report_path", type=click.Path(),
              default="experiments/feature_aware_transfer_report.txt",
              help="Output path for experiment report")
def evaluate_transfer(
    train_ohlc_path, train_features_path, train_labels_path,
    val_ohlc_path, val_features_path, val_labels_path,
    encoder_path, eval_modes, feature_fusion, epochs, batch_size,
    learning_rate, device, output_path, report_path
):
    """Evaluate transfer learning performance.

    Compares different training strategies:
    - OHLC-only baseline
    - Feature-aware from scratch
    - Pre-trained encoder with OHLC only
    - Pre-trained encoder with features

    Example:
        moola cli-feature-aware evaluate-transfer \\
            --train-ohlc data/train_ohlc.npy \\
            --train-features data/train_features.npy \\
            --train-labels data/train_labels.npy \\
            --val-ohlc data/val_ohlc.npy \\
            --val-features data/val_features.npy \\
            --val-labels data/val_labels.npy \\
            --encoder artifacts/pretrained/feature_aware_encoder.pt \\
            --modes pretrained_features
    """
    rprint(f"\n{'='*70}")
    rprint(f"TRANSFER LEARNING EVALUATION")
    rprint(f"{'='*70}")
    rprint(f"Encoder: {encoder_path}")
    rprint(f"Fusion strategy: {feature_fusion}")
    rprint(f"Evaluation modes: {list(eval_modes)}")
    rprint(f"{'='*70}\n")

    # Load data
    rprint("Loading data...")
    X_ohlc_train = np.load(train_ohlc_path)
    y_train = np.load(train_labels_path)
    X_ohlc_val = np.load(val_ohlc_path)
    y_val = np.load(val_labels_path)

    rprint(f"  Train OHLC: {X_ohlc_train.shape}")
    rprint(f"  Train labels: {y_train.shape}")
    rprint(f"  Val OHLC: {X_ohlc_val.shape}")
    rprint(f"  Val labels: {y_val.shape}")

    # Load features
    if train_features_path and val_features_path:
        X_features_train = np.load(train_features_path)
        X_features_val = np.load(val_features_path)
        rprint(f"  Train features: {X_features_train.shape}")
        rprint(f"  Val features: {X_features_val.shape}")
    else:
        rprint("⚠️  No features provided, will compute from OHLC data")
        # Compute features for feature-aware modes
        feature_config = get_feature_engineering_config("default")
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
        )
        engineer = AdvancedFeatureEngineer(advanced_config)
        X_features_train = engineer.transform(X_ohlc_train)
        X_features_val = engineer.transform(X_ohlc_val)
        rprint(f"  Computed train features: {X_features_train.shape}")
        rprint(f"  Computed val features: {X_features_val.shape}")

    # Configure fine-tuning
    lstm_config = get_enhanced_simple_lstm_config("transfer_learning")
    lstm_config.feature_fusion = feature_fusion
    lstm_config.n_epochs = epochs
    lstm_config.batch_size = batch_size
    lstm_config.learning_rate = learning_rate
    lstm_config.device = device

    # Run evaluation
    results = evaluate_transfer_learning(
        X_ohlc_train=X_ohlc_train,
        X_features_train=X_features_train,
        y_train=y_train,
        X_ohlc_val=X_ohlc_val,
        X_features_val=X_features_val,
        y_val=y_val,
        encoder_path=Path(encoder_path),
        lstm_config=lstm_config,
        modes=list(eval_modes)
    )

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for mode, result in results.items():
        json_results[mode] = {k: v for k, v in result.items() if k != 'classification_report'}
        json_results[mode]['classification_report'] = result['classification_report']

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Generate and save report
    report = create_experiment_report(results, Path(report_path))

    # Summary
    rprint(f"\n{'='*70}")
    rprint(f"EVALUATION COMPLETE")
    rprint(f"{'='*70}")
    rprint(f"Results saved: {output_path}")
    rprint(f"Report saved: {report_path}")

    # Best model
    best_mode = max(results.keys(), key=lambda k: results[k]['accuracy'])
    rprint(f"\n🏆 Best performing model: {best_mode}")
    rprint(f"   Accuracy: {results[best_mode]['accuracy']:.4f}")
    rprint(f"   Training time: {results[best_mode]['training_time']:.1f}s")


@app.command()
@click.option("--encoder", "encoder_path", type=click.Path(exists=True), required=True,
              help="Path to pre-trained encoder")
@click.option("--ohlc", "ohlc_path", type=click.Path(exists=True), required=True,
              help="Path to OHLC sample data")
@click.option("--features", "features_path", type=click.Path(exists=True), required=True,
              help="Path to features sample data")
@click.option("--samples", default=100, type=int, help="Number of samples to analyze")
@click.option("--device", default="cuda", type=click.Choice(["cpu", "cuda"]),
              help="Device for computation")
@click.option("--output", "output_path", type=click.Path(),
              default="experiments/encoder_importance_analysis.json",
              help="Output path for importance analysis")
def analyze_importance(encoder_path, ohlc_path, features_path, samples, device, output_path):
    """Analyze feature importance in pre-trained encoder.

    Performs ablation study to understand which features are most important
    for the pre-trained encoder's reconstruction capability.

    Example:
        moola cli-feature-aware analyze-importance \\
            --encoder artifacts/pretrained/feature_aware_encoder.pt \\
            --ohlc data/sample_ohlc.npy \\
            --features data/sample_features.npy \\
            --samples 50
    """
    rprint(f"\n{'='*70}")
    rprint(f"ENCODER FEATURE IMPORTANCE ANALYSIS")
    rprint(f"{'='*70}")
    rprint(f"Encoder: {encoder_path}")
    rprint(f"Samples: {samples}")
    rprint(f"Device: {device}")
    rprint(f"{'='*70}\n")

    # Load sample data
    rprint("Loading sample data...")
    X_ohlc = np.load(ohlc_path)[:samples]
    X_features = np.load(features_path)[:samples]

    rprint(f"  OHLC samples: {X_ohlc.shape}")
    rprint(f"  Feature samples: {X_features.shape}")

    # Run importance analysis
    importance_results = analyze_encoder_importance(
        encoder_path=Path(encoder_path),
        X_ohlc_sample=X_ohlc,
        X_features_sample=X_features,
        device=device
    )

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    # Convert numpy arrays to lists
    json_results = {
        'ohlc_importance': importance_results['ohlc_importance'].tolist(),
        'feature_importance': importance_results['feature_importance'].tolist(),
        'feature_indices': importance_results['feature_indices'],
        'ohlc_feature_names': importance_results['ohlc_feature_names'],
    }

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    # Display results
    rprint(f"\n📊 Feature Importance Results:")

    rprint("\nOHLC Feature Importance:")
    for i, (name, importance) in enumerate(zip(importance_results['ohlc_feature_names'], importance_results['ohlc_importance'])):
        rprint(f"  {name:8s}: {importance:.6f}")

    rprint("\nTop Engineered Features:")
    feature_importance = importance_results['feature_importance']
    feature_indices = importance_results['feature_indices']
    sorted_indices = np.argsort(feature_importance)[::-1][:10]

    for rank, idx in enumerate(sorted_indices):
        feature_idx = feature_indices[idx]
        importance = feature_importance[idx]
        rprint(f"  {rank+1:2d}. Feature {feature_idx:2d}: {importance:.6f}")

    rprint(f"\n✅ Analysis complete! Results saved: {output_path}")


if __name__ == "__main__":
    app()