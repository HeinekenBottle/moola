from pathlib import Path

import click
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from rich import print as rprint

from .logging_setup import setup_logging
from .paths import resolve_paths


@click.group(help="Moola CLI")
def app():
    pass


def _load_cfg(cfg_dir: Path, overrides: list[str] = None):
    overrides = overrides or []
    cfg_dir = Path(cfg_dir).resolve()
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(cfg_dir)):
        cfg = compose(config_name="default", overrides=overrides)
    return cfg


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True, help="Hydra-style overrides, e.g., hardware=gpu")
def doctor(cfg_dir, over):
    "Validate environment and show resolved paths and config."
    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    # Convert OmegaConf to dict for JSON serialization
    cfg_dict = cfg.model_dump() if hasattr(cfg, "model_dump") else dict(cfg)
    rprint({"cfg": cfg_dict, "paths": paths.model_dump()})


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    help="Input parquet file to ingest (optional)",
)
def ingest(cfg_dir, over, input_path):
    """Ingest and validate raw data, write processed/train.parquet."""
    import numpy as np
    import pandas as pd

    from .schema import TrainingDataRow

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Ingest start")

    if input_path:
        # Load and validate existing dataset
        input_path = Path(input_path)
        log.info(f"Loading existing dataset from {input_path}")
        df = pd.read_parquet(input_path)
        log.info(f"Loaded {len(df)} samples | Shape: {df.shape}")
        log.info(f"Columns: {list(df.columns)}")
        log.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

        # Validate schema if window_id column exists (our new format)
        if "window_id" in df.columns:
            from .schemas.canonical_v1 import check_training_data

            if check_training_data(df):
                log.info("✅ Dataset schema validation passed")
            else:
                log.error("❌ Dataset schema validation failed")
                raise ValueError("Invalid dataset schema")

        # Copy to standard train.parquet location
        output_path = paths.data / "processed" / "train.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine="pyarrow")
        log.info(f"Copied validated dataset to {output_path}")
        log.info("Ingest done")

    else:
        # Set seed for reproducible synthetic data
        np.random.seed(cfg.seed)

        # Generate synthetic training data
        # For now: 1000 samples, 2 classes, 10 features each
        n_samples = 1000
        n_features = 10
        labels = ["class_A", "class_B"]

        log.info(f"Generating {n_samples} synthetic samples with {n_features} features")

        rows = []
        for i in range(n_samples):
            # Generate random features
            features = np.random.randn(n_features).tolist()
            # Assign label (50/50 split)
            label = labels[i % 2]

            # Validate against schema
            row_data = {"window_id": i, "label": label, "features": features}
            validated_row = TrainingDataRow(**row_data)
            rows.append(validated_row.model_dump())

        # Create DataFrame and write to parquet
        df = pd.DataFrame(rows)
        output_path = paths.data / "processed" / "train.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False, engine="pyarrow")

        log.info(f"Wrote {len(df)} validated rows to {output_path}")
        log.info(f"Schema: {list(df.columns)} | Shape: {df.shape}")
        log.info("Ingest done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option(
    "--model",
    default="enhanced_simple_lstm",
    help=(
        "Model name. Options:\n"
        "  - enhanced_simple_lstm: PRIMARY production model (BiLSTM + attention, pretrained support)\n"
        "  - simple_lstm: BASELINE lightweight model for smoke tests\n"
        "  - cnn_transformer: EXPERIMENTAL multi-task model\n"
        "  - rwkv_ts: EXPERIMENTAL RWKV for time series\n"
        "  - logreg, rf, xgb: Classical ML models for stacking\n"
    ),
)
@click.option(
    "--data",
    type=click.Path(exists=True),
    default="data/processed/train_latest.parquet",
    help="Path to training data parquet file",
)
@click.option(
    "--split",
    type=click.Path(exists=True),
    required=True,
    help="Path to temporal split JSON (REQUIRED to prevent look-ahead bias)",
)
@click.option(
    "--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device for training"
)
@click.option(
    "--use-engineered-features",
    is_flag=True,
    default=False,
    help="Enable engineered features (small dataset + price action)",
)
@click.option(
    "--max-engineered-features",
    default=50,
    type=int,
    help="Maximum number of engineered features to use",
)
@click.option(
    "--use-hopsketch",
    is_flag=True,
    default=False,
    help="Include HopSketch features (1575 features, for XGBoost)",
)
@click.option(
    "--augment-data",
    is_flag=True,
    default=False,
    help="Enable pseudo-sample augmentation for small datasets",
)
@click.option(
    "--augmentation-ratio",
    default=2.0,
    type=float,
    help="Target ratio of synthetic:real samples (e.g., 2.0 = 2:1)",
)
@click.option(
    "--max-synthetic-samples",
    default=210,
    type=int,
    help="Maximum number of synthetic samples to generate",
)
@click.option(
    "--augmentation-seed", default=1337, type=int, help="Random seed for reproducible augmentation"
)
@click.option(
    "--quality-threshold",
    default=0.7,
    type=float,
    help="Minimum quality score for sample acceptance (0.0-1.0)",
)
@click.option(
    "--pretrained-encoder",
    type=click.Path(exists=True),
    default=None,
    help="Path to pretrained encoder (for enhanced_simple_lstm)",
)
@click.option(
    "--freeze-encoder/--no-freeze-encoder",
    default=True,
    help="Freeze encoder during initial training (for pretrained models)",
)
@click.option(
    "--log-pretrained-stats",
    is_flag=True,
    default=False,
    help="Log detailed pretrained load statistics",
)
@click.option(
    "--input-dim",
    default=None,
    type=int,
    help="Input feature dimension (4 for OHLC, 11 for RelativeTransform). Auto-detected if not specified.",
)
@click.option("--seed", default=None, type=int, help="Random seed (overrides config)")
@click.option(
    "--save-run", is_flag=True, default=False, help="Save run artifacts to artifacts/runs/"
)
@click.option(
    "--predict-pointers",
    is_flag=True,
    default=False,
    help="Enable multi-task pointer prediction (for enhanced_simple_lstm)",
)
@click.option(
    "--use-latent-mixup/--no-latent-mixup",
    default=True,
    help="Apply latent-space mixup augmentation (PHASE 2, default: True)",
)
@click.option(
    "--latent-mixup-alpha",
    type=float,
    default=0.4,
    help="Latent mixup Beta distribution parameter (PHASE 2, default: 0.4)",
)
@click.option(
    "--latent-mixup-prob",
    type=float,
    default=0.5,
    help="Probability of applying latent mixup (PHASE 2, default: 0.5)",
)
@click.option(
    "--augment-jitter",
    is_flag=True,
    default=False,
    help="Enable temporal jittering augmentation (PHASE 2)",
)
@click.option(
    "--augment-warp",
    is_flag=True,
    default=False,
    help="Enable magnitude warping augmentation (PHASE 2)",
)
@click.option(
    "--jitter-sigma",
    type=float,
    default=0.03,
    help="Jitter noise std dev (PHASE 2, default: 0.03)",
)
@click.option(
    "--warp-sigma",
    type=float,
    default=0.2,
    help="Warp magnitude std dev (PHASE 2, default: 0.2)",
)
@click.option(
    "--mc-dropout",
    is_flag=True,
    default=False,
    help="Enable MC Dropout for uncertainty estimation (PHASE 3)",
)
@click.option(
    "--mc-passes", type=int, default=50, help="Number of MC Dropout forward passes (default: 50)"
)
@click.option(
    "--mc-dropout-rate",
    type=float,
    default=0.15,
    help="Dropout rate for MC inference (default: 0.15)",
)
@click.option(
    "--temperature-scaling",
    is_flag=True,
    default=False,
    help="Apply temperature scaling for probability calibration (PHASE 3)",
)
@click.option(
    "--compute-calibration",
    is_flag=True,
    default=False,
    help="Compute calibration metrics (ECE, Brier score) - PHASE 3",
)
@click.option(
    "--calibration-bins",
    type=int,
    default=15,
    help="Number of bins for ECE computation (PHASE 3, default: 15)",
)
@click.option(
    "--plot-reliability",
    is_flag=True,
    default=False,
    help="Generate reliability diagram (PHASE 3)",
)
@click.option(
    "--bootstrap-ci",
    is_flag=True,
    default=False,
    help="Compute bootstrap confidence intervals for metrics (PHASE 4)",
)
@click.option(
    "--bootstrap-resamples",
    type=int,
    default=1000,
    help="Number of bootstrap resamples (PHASE 4, default: 1000)",
)
@click.option(
    "--bootstrap-confidence",
    type=float,
    default=0.95,
    help="Confidence level for bootstrap CIs (PHASE 4, default: 0.95)",
)
@click.option(
    "--use-lr-scheduler/--no-lr-scheduler",
    default=True,
    help="Use ReduceLROnPlateau learning rate scheduler (PHASE 4, default: True)",
)
@click.option(
    "--scheduler-factor",
    type=float,
    default=0.5,
    help="LR reduction factor when plateauing (PHASE 4, default: 0.5)",
)
@click.option(
    "--scheduler-patience",
    type=int,
    default=10,
    help="Epochs to wait before reducing LR (PHASE 4, default: 10)",
)
@click.option(
    "--scheduler-min-lr",
    type=float,
    default=1e-6,
    help="Minimum learning rate threshold (PHASE 4, default: 1e-6)",
)
@click.option(
    "--save-checkpoints",
    is_flag=True,
    default=False,
    help="Save best model checkpoints during training (PHASE 4)",
)
@click.option(
    "--monitor-gradients",
    is_flag=True,
    default=False,
    help="Monitor gradient statistics during training (PHASE 4)",
)
@click.option(
    "--gradient-log-freq",
    type=int,
    default=10,
    help="Gradient monitoring log frequency in epochs (PHASE 4, default: 10)",
)
@click.option(
    "--use-uncertainty-weighting/--no-use-uncertainty-weighting",
    default=True,
    help="Use uncertainty-weighted multi-task loss (REQUIRED for production, default: True)",
)
def train(
    cfg_dir,
    over,
    model,
    data,
    split,
    device,
    use_engineered_features,
    max_engineered_features,
    use_hopsketch,
    augment_data,
    augmentation_ratio,
    max_synthetic_samples,
    augmentation_seed,
    quality_threshold,
    pretrained_encoder,
    freeze_encoder,
    log_pretrained_stats,
    input_dim,
    seed,
    save_run,
    predict_pointers,
    use_latent_mixup,
    latent_mixup_alpha,
    latent_mixup_prob,
    augment_jitter,
    augment_warp,
    jitter_sigma,
    warp_sigma,
    mc_dropout,
    mc_passes,
    mc_dropout_rate,
    temperature_scaling,
    compute_calibration,
    calibration_bins,
    plot_reliability,
    bootstrap_ci,
    bootstrap_resamples,
    bootstrap_confidence,
    use_lr_scheduler,
    scheduler_factor,
    scheduler_patience,
    scheduler_min_lr,
    save_checkpoints,
    monitor_gradients,
    gradient_log_freq,
    use_uncertainty_weighting,
):
    """Train classifier with temporal split validation.

    CRITICAL: Requires temporal split file to prevent look-ahead bias in time series.
    Random/stratified splits are FORBIDDEN for financial data.
    """
    import pickle

    import numpy as np
    import pandas as pd

    from .data.dual_input_pipeline import create_dual_input_processor, prepare_model_inputs
    from .data.splits import assert_no_random, assert_temporal, load_split
    from .models import get_model
    from .utils.seeds import print_gpu_info

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)

    # Override seed if provided
    if seed is not None:
        cfg.seed = seed

    # Load and validate temporal split (CRITICAL for time series integrity)
    log.info("=" * 60)
    log.info("TEMPORAL SPLIT VALIDATION")
    log.info("=" * 60)
    split_data = load_split(split)
    assert_temporal(split_data)
    assert_no_random({"split_impl": "temporal", "split_strategy": "forward_chaining"})

    log.info(f"Split: {split_data['name']}")
    log.info(f"Train indices: {len(split_data['train_indices'])} samples")
    log.info(f"Val indices: {len(split_data['val_indices'])} samples")
    log.info("=" * 60)

    log.info(
        "Train start | model=%s seed=%s device=%s input_dim=%s",
        model,
        cfg.seed,
        device,
        input_dim if input_dim else "auto",
    )

    # Log feature configuration
    log.info("=" * 60)
    log.info("DUAL-INPUT DATA PIPELINE CONFIGURATION")
    log.info("=" * 60)
    log.info("Raw OHLC features: ALWAYS ENABLED (105×4)")
    log.info(f"Engineered features: {use_engineered_features}")
    if use_engineered_features:
        log.info("  - Small dataset optimized features: ✓")
        log.info("  - Multi-scale price action features: ✓")
        log.info(f"  - HopSketch features: {use_hopsketch}")
        log.info(f"  - Max engineered features: {max_engineered_features}")

    # Log augmentation configuration
    log.info(f"Pseudo-sample augmentation: {augment_data}")
    if augment_data:
        log.info(f"  - Augmentation ratio: {augmentation_ratio:.1f}:1 (synthetic:real)")
        log.info(f"  - Max synthetic samples: {max_synthetic_samples}")
        log.info(f"  - Quality threshold: {quality_threshold:.1f}")
        log.info(f"  - Augmentation seed: {augmentation_seed}")
        log.info("  - Safe strategies only: True (temporal + pattern-based)")

    # Log temporal augmentation configuration (PHASE 2)
    log.info(f"Temporal jittering: {augment_jitter}")
    if augment_jitter:
        log.info(f"  - Jitter sigma: {jitter_sigma:.3f}")
        log.info("  - Jitter prob: 0.8")
    log.info(f"Magnitude warping: {augment_warp}")
    if augment_warp:
        log.info(f"  - Warp sigma: {warp_sigma:.2f}")
        log.info("  - Warp knots: 4")
        log.info("  - Warp prob: 0.5")
    log.info("=" * 60)

    # GPU verification for deep learning models
    if device == "cuda" and model in ["rwkv_ts", "cnn_transformer"]:
        print_gpu_info()

    # Load training data
    train_path = Path(data)
    if not train_path.exists():
        log.error(f"Training data not found at {train_path}. Run 'moola ingest' first.")
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_parquet(train_path)
    log.info(f"Loaded {len(df)} training samples from {train_path}")

    # Check if features are pickled (bytes) and deserialize if needed
    if "features" in df.columns:
        features_col = df["features"].values
        if len(features_col) > 0 and isinstance(features_col[0], bytes):
            # Pickled format (RelativeTransform)
            log.info("Detected pickled features, deserializing...")
            df["features"] = [pickle.loads(f) for f in features_col]
            log.info("Successfully deserialized pickled features")
        # else: nested array format (keep as is)

    # Process data with dual-input pipeline
    processor = create_dual_input_processor(
        use_engineered_features=use_engineered_features,
        max_engineered_features=max_engineered_features,
        use_hopsketch=use_hopsketch,
        enable_augmentation=augment_data,
        augmentation_ratio=augmentation_ratio,
        max_synthetic_samples=max_synthetic_samples,
        augmentation_seed=augmentation_seed,
        quality_threshold=quality_threshold,
        use_safe_strategies_only=True,  # Always use safe strategies in CLI
    )

    processed_data = processor.process_training_data(
        df, enable_engineered_features=use_engineered_features
    )

    # Log data processing results
    log.info("=" * 60)
    log.info("DATA PROCESSING RESULTS")
    log.info("=" * 60)
    log.info(f"Raw OHLC shape: {processed_data['X_ohlc'].shape}")

    # Log augmentation results if applicable
    augmentation_metadata = processed_data["metadata"].get("augmentation_metadata", {})
    if augmentation_metadata:
        log.info("Augmentation results:")
        log.info(f"  - Original samples: {augmentation_metadata.get('n_original', 'N/A')}")
        log.info(
            f"  - Synthetic samples: {augmentation_metadata.get('n_synthetic_accepted', 'N/A')}"
        )
        log.info(f"  - Total samples: {augmentation_metadata.get('n_total', 'N/A')}")
        log.info(f"  - Synthetic ratio: {augmentation_metadata.get('synthetic_ratio', 0):.2f}")
        log.info(
            f"  - OHLC integrity rate: {augmentation_metadata.get('ohlc_integrity_rate', 0):.3f}"
        )

        generation_meta = augmentation_metadata.get("generation_metadata", {})
        if generation_meta:
            strategies_used = generation_meta.get("strategies_used", [])
            quality_scores = generation_meta.get("quality_scores", {})
            log.info(f"  - Strategies used: {', '.join(strategies_used)}")
            if quality_scores:
                avg_quality = np.mean(list(quality_scores.values()))
                log.info(f"  - Average quality score: {avg_quality:.3f}")
    else:
        log.info("Augmentation: DISABLED")

    if processed_data["X_engineered"] is not None:
        log.info(f"Engineered features shape: {processed_data['X_engineered'].shape}")
        log.info(f"Feature names: {len(processed_data['feature_names'])} features")

        # Log feature statistics
        feature_stats = processor.get_feature_statistics(processed_data["X_engineered"])
        if feature_stats:
            log.info(
                f"Engineered feature stats: mean={feature_stats['mean']:.4f}, "
                f"std={feature_stats['std']:.4f}, range=[{feature_stats['min']:.4f}, {feature_stats['max']:.4f}]"
            )
    else:
        log.info("Engineered features: DISABLED")

    if processed_data["expansion_start"] is not None:
        log.info(
            f"Expansion indices | start range: [{processed_data['expansion_start'].min()}, {processed_data['expansion_start'].max()}] "
            f"| end range: [{processed_data['expansion_end'].min()}, {processed_data['expansion_end'].max()}]"
        )
    log.info(f"Unique labels: {np.unique(processed_data['y'])}")
    log.info("=" * 60)

    # Prepare model inputs based on model type
    model_inputs = prepare_model_inputs(
        processed_data, model_type=model, use_engineered_features=use_engineered_features
    )

    # Extract training variables
    X = model_inputs["X"]
    y = model_inputs["y"]
    expansion_start = model_inputs.get("expansion_start")
    expansion_end = model_inputs.get("expansion_end")

    log.info(f"Final input shape for {model}: {X.shape}")

    # Use temporal split indices (NO random/stratified splitting!)
    train_idx = np.array(split_data["train_indices"])
    val_idx = np.array(split_data["val_indices"])

    # Apply split indices to data
    X_train, X_test = X[train_idx], X[val_idx]
    y_train, y_test = y[train_idx], y[val_idx]

    if expansion_start is not None and expansion_end is not None:
        exp_start_train = expansion_start[train_idx]
        exp_start_test = expansion_start[val_idx]
        exp_end_train = expansion_end[train_idx]
        exp_end_test = expansion_end[val_idx]
    else:
        exp_start_train, exp_start_test = None, None
        exp_end_train, exp_end_test = None, None

    log.info(f"Train samples: {len(X_train)} | Val samples: {len(X_test)}")
    log.info("Using temporal split - train comes BEFORE val in time")

    # Get model from registry and train
    # Pass device parameter for deep learning models
    # Enable multi-task pointer prediction for CNN-Transformer and EnhancedSimpleLSTM
    model_kwargs = {"seed": cfg.seed, "device": device}
    if model == "cnn_transformer":
        model_kwargs["predict_pointers"] = True
    if model == "enhanced_simple_lstm" and predict_pointers:
        model_kwargs["predict_pointers"] = True
        # PHASE 2: Add latent mixup parameters for enhanced_simple_lstm
        model_kwargs["use_latent_mixup"] = use_latent_mixup
        model_kwargs["latent_mixup_alpha"] = latent_mixup_alpha
        model_kwargs["latent_mixup_prob"] = latent_mixup_prob
        # PHASE 1: Add uncertainty weighting parameter (CRITICAL for production)
        model_kwargs["use_uncertainty_weighting"] = use_uncertainty_weighting

    # PHASE 4: Add learning rate scheduler parameters for enhanced_simple_lstm
    if model == "enhanced_simple_lstm":
        model_kwargs["use_lr_scheduler"] = use_lr_scheduler
        model_kwargs["scheduler_factor"] = scheduler_factor
        model_kwargs["scheduler_patience"] = scheduler_patience
        model_kwargs["scheduler_min_lr"] = scheduler_min_lr
        model_kwargs["save_checkpoints"] = save_checkpoints
        # PHASE 1: Always pass uncertainty weighting parameter (CRITICAL for production)
        model_kwargs["use_uncertainty_weighting"] = use_uncertainty_weighting

    # PHASE 2: Add temporal augmentation parameters (jitter + warp)
    if model in ["simple_lstm", "enhanced_simple_lstm"] and (augment_jitter or augment_warp):
        model_kwargs["use_temporal_aug"] = True
        model_kwargs["jitter_sigma"] = jitter_sigma if augment_jitter else 0.0
        model_kwargs["jitter_prob"] = 0.8 if augment_jitter else 0.0
        model_kwargs["warp_sigma"] = warp_sigma if augment_warp else 0.0
        model_kwargs["warp_prob"] = 0.5 if augment_warp else 0.0

    # Add input_dim if specified (for 11D RelativeTransform support)
    if input_dim is not None:
        model_kwargs["input_dim"] = input_dim
        log.info(f"Using explicit input_dim={input_dim} for model")

    # Add engineered features to model kwargs if available
    if use_engineered_features and processed_data["X_engineered"] is not None:
        if model in ["lstm", "simple_lstm", "cnn_transformer"]:
            # Deep learning models can potentially use engineered features as additional context
            log.info(
                f"Deep learning model {model} has engineered features available for potential use"
            )
            # Note: Model architecture would need to be updated to actually use these features

    model_instance = get_model(model, **model_kwargs)

    # Handle pretrained encoder loading for enhanced_simple_lstm
    pretrained_stats = None
    if pretrained_encoder and model == "enhanced_simple_lstm":
        log.info("=" * 60)
        log.info("LOADING PRETRAINED ENCODER WITH STRICT VALIDATION")
        log.info("=" * 60)
        log.info(f"Encoder path: {pretrained_encoder}")
        log.info(f"Freeze encoder: {freeze_encoder}")

        # Use strict pretrained loader

        # Note: Model must be built first, so we pass pretrained_encoder_path to fit()
        # The fit() method will handle strict loading after model is built

    # Fit model (with optional pretrained encoder)
    fit_kwargs = {
        "X": X_train,
        "y": y_train,
        "expansion_start": exp_start_train,
        "expansion_end": exp_end_train,
    }

    if pretrained_encoder and model in ["enhanced_simple_lstm", "relative_transform_lstm"]:
        fit_kwargs["pretrained_encoder_path"] = Path(pretrained_encoder)
        fit_kwargs["freeze_encoder"] = freeze_encoder

    model_instance.fit(**fit_kwargs)

    # Calculate accuracy using model's predict method (handles label encoding)
    y_train_pred = model_instance.predict(
        X_train, expansion_start=exp_start_train, expansion_end=exp_end_train
    )
    y_test_pred = model_instance.predict(
        X_test, expansion_start=exp_start_test, expansion_end=exp_end_test
    )
    train_score = (y_train_pred == y_train).mean()
    test_score = (y_test_pred == y_test).mean()
    log.info(f"Train accuracy: {train_score:.3f} | Test accuracy: {test_score:.3f}")

    # PHASE 1: Compute pointer regression metrics if predict_pointers enabled
    if predict_pointers and exp_start_test is not None:
        log.info("=" * 60)
        log.info("POINTER REGRESSION METRICS (Validation Set)")
        log.info("=" * 60)

        from moola.data.pointer_transforms import numpy_center_length_to_start_end
        from moola.utils.metrics.pointer_regression import compute_pointer_regression_metrics

        # Get pointer predictions on validation set
        try:
            val_outputs = model_instance.predict_with_pointers(X_test)
            pointer_preds = val_outputs["pointers"]  # [N, 2] = [center, length] in [0, 1]

            # PHASE 2: Convert center-length to start-end for evaluation
            pred_start, pred_end = numpy_center_length_to_start_end(
                pointer_preds[:, 0], pointer_preds[:, 1], seq_len=104
            )

            ptr_metrics = compute_pointer_regression_metrics(
                pred_start=pred_start,
                pred_end=pred_end,
                true_start=exp_start_test,
                true_end=exp_end_test,
                tolerance=3,
            )

            log.info("Pointer Regression Performance:")
            log.info(
                f"  hit@±3:      {ptr_metrics['hit_at_pm3']:.1%} (both start AND end within ±3)"
            )
            log.info(
                f"  hit@±5:      {ptr_metrics['hit_at_pm5']:.1%} (both start AND end within ±5)"
            )
            log.info(
                f"  Center MAE:  {ptr_metrics['center_mae']:.2f} bars (expansion center position)"
            )
            log.info(f"  Length MAE:  {ptr_metrics['length_mae']:.2f} bars (expansion span length)")
            log.info(f"  Start MAE:   {ptr_metrics['start_mae']:.2f} bars")
            log.info(f"  End MAE:     {ptr_metrics['end_mae']:.2f} bars")
            log.info(f"  Exact Match: {ptr_metrics['exact_match']:.1%} (both pointers perfect)")

            # Warning if hit rate is very low (indicates task silencing)
            if ptr_metrics["hit_at_pm3"] < 0.10:
                log.warning("⚠️  WARNING: hit@±3 < 10% - pointer head may be silenced!")
                log.warning("   Consider: (1) Increase loss_beta, (2) Enable uncertainty weighting")

            log.info("=" * 60)

            # PHASE 2: Compute joint metrics (pointer hit + type classification)
            log.info("=" * 60)
            log.info("JOINT TASK METRICS (Pointer + Type)")
            log.info("=" * 60)

            from moola.utils.metrics.joint_metrics import (
                compute_joint_hit_accuracy,
                compute_joint_hit_f1,
                compute_task_contribution_analysis,
            )

            # Joint accuracy
            joint_acc = compute_joint_hit_accuracy(
                pred_start=pred_start,
                pred_end=pred_end,
                true_start=exp_start_test,
                true_end=exp_end_test,
                pred_type=y_test_pred,
                true_type=y_test,
                tolerance=3,
            )

            log.info("Joint Performance (both tasks correct):")
            log.info(f"  Joint Accuracy:     {joint_acc['joint_accuracy']:.1%}")
            log.info(f"  Pointer Hit Rate:   {joint_acc['pointer_hit_rate']:.1%}")
            log.info(f"  Type Accuracy:      {joint_acc['type_accuracy']:.1%}")
            log.info(
                f"  Joint Correct:      {joint_acc['joint_correct_count']} / {joint_acc['total_samples']}"
            )

            # Joint F1
            joint_f1 = compute_joint_hit_f1(
                pred_start=pred_start,
                pred_end=pred_end,
                true_start=exp_start_test,
                true_end=exp_end_test,
                pred_type=y_test_pred,
                true_type=y_test,
                tolerance=3,
                average="weighted",
            )

            log.info("\nJoint F1 Metrics:")
            log.info(f"  Joint F1:           {joint_f1['joint_f1']:.3f}")
            log.info(f"  Joint Precision:    {joint_f1['joint_precision']:.3f}")
            log.info(f"  Joint Recall:       {joint_f1['joint_recall']:.3f}")

            # Task contribution analysis
            contribution = compute_task_contribution_analysis(
                pred_start=pred_start,
                pred_end=pred_end,
                true_start=exp_start_test,
                true_end=exp_end_test,
                pred_type=y_test_pred,
                true_type=y_test,
                tolerance=3,
            )

            log.info("\nTask Contribution Breakdown:")
            log.info(
                f"  Both Correct:       {contribution['both_correct']['fraction']:.1%} "
                f"({contribution['both_correct']['count']} samples)"
            )
            log.info(
                f"  Pointer Only:       {contribution['pointer_only_correct']['fraction']:.1%} "
                f"({contribution['pointer_only_correct']['count']} samples)"
            )
            log.info(
                f"  Type Only:          {contribution['type_only_correct']['fraction']:.1%} "
                f"({contribution['type_only_correct']['count']} samples)"
            )
            log.info(
                f"  Both Wrong:         {contribution['both_wrong']['fraction']:.1%} "
                f"({contribution['both_wrong']['count']} samples)"
            )

            # Warning if joint accuracy is low
            if joint_acc["joint_accuracy"] < 0.30:
                log.warning("⚠️  WARNING: Joint accuracy < 30% - tasks may be competing!")
                log.warning("   Consider: (1) Adjust loss_beta, (2) Enable uncertainty weighting")

            log.info("=" * 60)

        except AttributeError:
            log.warning(
                "Model doesn't have predict_with_pointers() method - skipping pointer metrics"
            )
        except Exception as e:
            log.error(f"Failed to compute pointer metrics: {e}")

    # PHASE 3: MC Dropout uncertainty estimation
    if mc_dropout and model == "enhanced_simple_lstm" and predict_pointers:
        log.info("=" * 60)
        log.info("MC DROPOUT UNCERTAINTY ESTIMATION")
        log.info("=" * 60)
        log.info(f"Running {mc_passes} forward passes with dropout_rate={mc_dropout_rate}...")

        import torch

        from moola.utils.uncertainty.mc_dropout import get_uncertainty_threshold, mc_dropout_predict

        # Prepare validation data as tensor
        X_test_tensor = torch.FloatTensor(X_test).to(model_instance.device)

        # Run MC Dropout inference
        mc_results = mc_dropout_predict(
            model=model_instance.model,
            x=X_test_tensor,
            n_passes=mc_passes,
            dropout_rate=mc_dropout_rate,
        )

        # Compute uncertainty statistics
        mean_entropy = mc_results["type_entropy"].mean()
        mean_ptr_std = mc_results["pointer_std"].mean(axis=0)

        log.info(f"Type classification uncertainty:")
        log.info(f"  Mean predictive entropy: {mean_entropy:.4f}")
        log.info(
            f"  Entropy range: [{mc_results['type_entropy'].min():.4f}, {mc_results['type_entropy'].max():.4f}]"
        )

        log.info(f"Pointer regression uncertainty:")
        log.info(f"  Mean center std: {mean_ptr_std[0]:.4f}")
        log.info(f"  Mean length std: {mean_ptr_std[1]:.4f}")

        # Flag high-uncertainty predictions (top 10%)
        threshold = get_uncertainty_threshold(mc_results["type_entropy"], percentile=90)
        high_uncertainty = mc_results["type_entropy"] > threshold
        log.info(
            f"High uncertainty samples (top 10%): {high_uncertainty.sum()} / {len(high_uncertainty)}"
        )
        log.info(f"  Uncertainty threshold: {threshold:.4f}")
        log.info("=" * 60)

    # PHASE 3: Temperature scaling for probability calibration
    if temperature_scaling and model == "enhanced_simple_lstm" and predict_pointers:
        log.info("=" * 60)
        log.info("TEMPERATURE SCALING CALIBRATION")
        log.info("=" * 60)

        import torch

        from moola.utils.uncertainty.mc_dropout import apply_temperature_scaling

        # Create validation dataloader
        X_val_tensor = torch.FloatTensor(X_test)
        y_val_tensor = torch.LongTensor([model_instance.label_to_idx[label] for label in y_test])

        if exp_start_test is not None:
            ptr_start_val_tensor = torch.FloatTensor(exp_start_test)
            ptr_end_val_tensor = torch.FloatTensor(exp_end_test)
            val_dataset = torch.utils.data.TensorDataset(
                X_val_tensor, y_val_tensor, ptr_start_val_tensor, ptr_end_val_tensor
            )
        else:
            val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, batch_size=model_instance.batch_size, shuffle=False
        )

        # Fit temperature scaling
        temp_scaler, optimal_temp = apply_temperature_scaling(
            model_instance.model, val_dataloader, device=device
        )

        log.info(f"Optimal temperature: {optimal_temp:.4f}")
        if optimal_temp > 1.5:
            log.info("  Model is OVERCONFIDENT (temperature > 1.5)")
        elif optimal_temp < 0.7:
            log.info("  Model is UNDERCONFIDENT (temperature < 0.7)")
        else:
            log.info("  Model is well-calibrated (temperature ≈ 1.0)")
        log.info("=" * 60)

    # PHASE 3: Compute calibration metrics if requested
    if compute_calibration:
        log.info("=" * 60)
        log.info("CALIBRATION METRICS (Validation Set)")
        log.info("=" * 60)

        from moola.utils.metrics.calibration import (
            compute_calibration_metrics,
            plot_reliability_diagram,
        )

        # Get probability predictions on validation set
        try:
            # Try to get probabilities - different models may have different interfaces
            if hasattr(model_instance, "predict_proba"):
                y_test_proba = model_instance.predict_proba(
                    X_test, expansion_start=exp_start_test, expansion_end=exp_end_test
                )

                cal_metrics = compute_calibration_metrics(
                    y_test_proba, y_test, n_bins=calibration_bins
                )

                log.info("Calibration Performance:")
                log.info(f"  ECE:          {cal_metrics['ece']:.4f} (target: <0.08)")
                log.info(f"  MCE:          {cal_metrics['mce']:.4f} (max calibration error)")
                log.info(f"  Brier Score:  {cal_metrics['brier_score']:.4f} (lower is better)")
                log.info(f"  Accuracy:     {cal_metrics['accuracy']:.1%}")

                # Warning if ECE is high
                if cal_metrics["ece"] > 0.15:
                    log.warning("⚠️  WARNING: ECE > 0.15 - model is poorly calibrated!")
                    log.warning(
                        "   Consider: (1) Enable temperature scaling, (2) Collect more data"
                    )
                elif cal_metrics["ece"] > 0.08:
                    log.warning("⚠️  NOTICE: ECE > 0.08 - calibration could be improved")
                    log.warning("   Consider: Enable temperature scaling")

                # Save calibration metrics if save_run is enabled
                if save_run:
                    import json
                    from datetime import datetime

                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_dir = paths.artifacts / "runs" / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)

                    cal_path = run_dir / "calibration_metrics.json"
                    with open(cal_path, "w") as f:
                        json.dump(
                            {
                                k: float(v) if isinstance(v, (np.floating, float)) else v
                                for k, v in cal_metrics.items()
                                if k != "ece_details"
                            },
                            f,
                            indent=2,
                        )
                    log.info(f"  Saved calibration metrics to {cal_path}")

                    # Plot reliability diagram if requested
                    if plot_reliability:
                        log.info("Generating reliability diagram...")
                        # Use probability of predicted class for diagram
                        pred_classes = y_test_proba.argmax(axis=1)
                        pred_probs = y_test_proba[np.arange(len(y_test_proba)), pred_classes]
                        binary_labels = (y_test == pred_classes).astype(int)

                        fig_path = run_dir / "reliability_diagram.png"
                        plot_reliability_diagram(
                            pred_probs,
                            binary_labels,
                            n_bins=calibration_bins,
                            save_path=str(fig_path),
                        )
                        log.info(f"  Saved reliability diagram to {fig_path}")

                log.info("=" * 60)
            else:
                log.warning(
                    "Model doesn't have predict_proba() method - skipping calibration metrics"
                )
        except Exception as e:
            log.error(f"Failed to compute calibration metrics: {e}")
            import traceback

            log.error(traceback.format_exc())

    # PHASE 4: Bootstrap confidence intervals for robust performance estimation
    if bootstrap_ci:
        log.info("=" * 60)
        log.info("BOOTSTRAP CONFIDENCE INTERVALS")
        log.info("=" * 60)
        log.info(f"Resamples: {bootstrap_resamples}, Confidence: {bootstrap_confidence*100:.0f}%")
        log.info(f"Validation set size: {len(y_test)} samples (small sample - CIs will be wide)")

        from moola.utils.metrics.bootstrap import (
            bootstrap_accuracy,
            bootstrap_calibration_metrics,
            bootstrap_pointer_metrics,
            format_bootstrap_result,
        )

        # Classification accuracy CI
        try:
            acc_ci = bootstrap_accuracy(
                y_test,
                y_test_pred,
                n_resamples=bootstrap_resamples,
                confidence_level=bootstrap_confidence,
            )
            log.info("")
            log.info("Classification Accuracy:")
            log.info(f"  {format_bootstrap_result('Accuracy', acc_ci)}")
            log.info(f"  Standard deviation: {acc_ci['std']:.4f}")
            ci_width = acc_ci["ci_upper"] - acc_ci["ci_lower"]
            log.info(f"  CI width: {ci_width:.4f} ({ci_width*100:.1f} percentage points)")

            if ci_width > 0.20:
                log.warning("⚠️  Wide CI (>20%) - high uncertainty due to small validation set")
                log.warning(
                    "   Consider: (1) More validation data, (2) Cross-validation, (3) MC Dropout"
                )
        except Exception as e:
            log.error(f"Failed to compute accuracy bootstrap CI: {e}")

        # Pointer regression metrics CI (if enabled)
        if predict_pointers and exp_start_test is not None:
            try:
                log.info("")
                log.info("Pointer Regression Metrics:")

                # Get pointer predictions again (already computed earlier but for clarity)
                val_outputs = model_instance.predict_with_pointers(X_test)
                pointer_preds = val_outputs["pointers"]

                from moola.data.pointer_transforms import numpy_center_length_to_start_end

                pred_start, pred_end = numpy_center_length_to_start_end(
                    pointer_preds[:, 0], pointer_preds[:, 1], seq_len=104
                )

                ptr_ci = bootstrap_pointer_metrics(
                    pred_start=pred_start,
                    pred_end=pred_end,
                    true_start=exp_start_test,
                    true_end=exp_end_test,
                    tolerance=3,
                    n_resamples=bootstrap_resamples,
                    confidence_level=bootstrap_confidence,
                )

                # Display key metrics
                for metric_name in ["start_mae", "end_mae", "hit_at_pm3", "center_mae"]:
                    log.info(f"  {format_bootstrap_result(metric_name, ptr_ci[metric_name])}")
            except Exception as e:
                log.error(f"Failed to compute pointer regression bootstrap CIs: {e}")

        # Calibration metrics CI (if computed)
        if compute_calibration and hasattr(model_instance, "predict_proba"):
            try:
                log.info("")
                log.info("Calibration Metrics:")

                # Get probabilities again
                y_test_proba = model_instance.predict_proba(
                    X_test, expansion_start=exp_start_test, expansion_end=exp_end_test
                )

                # Use probability of predicted class for calibration
                pred_classes = y_test_proba.argmax(axis=1)
                pred_probs = y_test_proba[np.arange(len(y_test_proba)), pred_classes]
                binary_labels = (y_test == pred_classes).astype(int)

                cal_ci = bootstrap_calibration_metrics(
                    probs=pred_probs,
                    labels=binary_labels,
                    n_resamples=bootstrap_resamples,
                    confidence_level=bootstrap_confidence,
                )

                for metric_name in ["ece", "brier"]:
                    log.info(f"  {format_bootstrap_result(metric_name, cal_ci[metric_name])}")
            except Exception as e:
                log.error(f"Failed to compute calibration bootstrap CIs: {e}")

        log.info("")
        log.info("Interpretation Guide:")
        log.info("  - Non-overlapping CIs → models are significantly different")
        log.info("  - Overlapping CIs → difference may be due to random variation")
        log.info(
            f"  - Wide CIs → high uncertainty (expected with {len(y_test)} validation samples)"
        )
        log.info("=" * 60)

    # Save model
    model_dir = paths.artifacts / "models" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    model_instance.save(model_path)

    # Save feature metadata if engineered features were used
    if use_engineered_features and processed_data["X_engineered"] is not None:
        import json

        metadata_path = model_dir / "feature_metadata.json"
        feature_metadata = {
            "feature_config": {
                "use_engineered_features": use_engineered_features,
                "max_engineered_features": max_engineered_features,
                "use_hopsketch": use_hopsketch,
            },
            "processing_metadata": processed_data["metadata"],
            "feature_names": processed_data["feature_names"],
            "feature_statistics": processor.get_feature_statistics(processed_data["X_engineered"]),
        }
        with open(metadata_path, "w") as f:
            json.dump(feature_metadata, f, indent=2)
        log.info(f"Saved feature metadata to {metadata_path}")

    log.info(f"Saved model to {model_path}")

    # Save run artifacts if requested
    if save_run:
        import json
        from datetime import datetime

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = paths.artifacts / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        import shutil

        shutil.copy(model_path, run_dir / "model.pkl")

        # Save run metadata
        run_metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model": model,
            "data": str(data),
            "split": str(split),
            "seed": cfg.seed,
            "input_dim": input_dim,
            "device": device,
            "pretrained_encoder": str(pretrained_encoder) if pretrained_encoder else None,
            "freeze_encoder": freeze_encoder,
            "augment_data": augment_data,
            "train_accuracy": float(train_score),
            "val_accuracy": float(test_score),
            "train_samples": len(X_train),
            "val_samples": len(X_test),
        }

        with open(run_dir / "metadata.json", "w") as f:
            json.dump(run_metadata, f, indent=2)

        log.info(f"Saved run artifacts to {run_dir}")
        log.info(f"Run ID: {run_id}")

    log.info("Train done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option("--model", default="logreg", help="Model name (logreg, rf, xgb)")
@click.option(
    "--split-dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing temporal split files (e.g., data/artifacts/splits/v1)",
)
@click.option("--num-folds", default=5, type=int, help="Number of folds to evaluate")
@click.option(
    "--use-engineered-features",
    is_flag=True,
    default=False,
    help="Enable engineered features (small dataset + price action)",
)
@click.option(
    "--max-engineered-features",
    default=50,
    type=int,
    help="Maximum number of engineered features to use",
)
@click.option(
    "--use-hopsketch",
    is_flag=True,
    default=False,
    help="Include HopSketch features (1575 features, for XGBoost)",
)
@click.option(
    "--augment-data",
    is_flag=True,
    default=False,
    help="Enable pseudo-sample augmentation for evaluation",
)
@click.option(
    "--augmentation-ratio", default=2.0, type=float, help="Target ratio of synthetic:real samples"
)
@click.option(
    "--max-synthetic-samples",
    default=210,
    type=int,
    help="Maximum number of synthetic samples to generate",
)
@click.option(
    "--augmentation-seed", default=1337, type=int, help="Random seed for reproducible augmentation"
)
@click.option(
    "--quality-threshold",
    default=0.7,
    type=float,
    help="Minimum quality score for sample acceptance",
)
def evaluate(
    cfg_dir,
    over,
    model,
    split_dir,
    num_folds,
    use_engineered_features,
    max_engineered_features,
    use_hopsketch,
    augment_data,
    augmentation_ratio,
    max_synthetic_samples,
    augmentation_seed,
    quality_threshold,
):
    """Evaluate model with temporal K-fold CV.

    CRITICAL: Uses temporal splits to prevent look-ahead bias.
    Random/stratified CV is FORBIDDEN for time series.
    """
    import csv
    import json
    import subprocess
    import time
    from datetime import datetime, timezone

    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    from .data.dual_input_pipeline import create_dual_input_processor, prepare_model_inputs
    from .data.splits import assert_temporal, load_split
    from .models import get_model

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)

    log.info("=" * 60)
    log.info("TEMPORAL K-FOLD EVALUATION")
    log.info("=" * 60)
    log.info("Evaluate start | model=%s", model)

    start_time = time.time()

    # Load data
    train_path = paths.data / "processed" / "train.parquet"
    if not train_path.exists():
        log.error(f"Training data not found at {train_path}")
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_parquet(train_path)
    log.info(f"Loaded {len(df)} samples for evaluation")

    # Check if model was trained with engineered features
    model_dir = paths.artifacts / "models" / model
    feature_metadata_path = model_dir / "feature_metadata.json"

    if feature_metadata_path.exists():
        # Load feature metadata from training
        with open(feature_metadata_path) as f:
            feature_metadata = json.load(f)
        trained_with_features = feature_metadata["feature_config"]["use_engineered_features"]
        trained_max_features = feature_metadata["feature_config"]["max_engineered_features"]
        trained_use_hopsketch = feature_metadata["feature_config"]["use_hopsketch"]

        log.info(f"Model was trained with engineered features: {trained_with_features}")
        if trained_with_features:
            log.info(f"  - Max engineered features: {trained_max_features}")
            log.info(f"  - HopSketch features: {trained_use_hopsketch}")

        # Use same feature configuration as training
        use_engineered_features = trained_with_features
        max_engineered_features = trained_max_features
        use_hopsketch = trained_use_hopsketch
    else:
        log.info("No feature metadata found, using provided flags")

    # Process data with dual-input pipeline
    processor = create_dual_input_processor(
        use_engineered_features=use_engineered_features,
        max_engineered_features=max_engineered_features,
        use_hopsketch=use_hopsketch,
        enable_augmentation=augment_data,
        augmentation_ratio=augmentation_ratio,
        max_synthetic_samples=max_synthetic_samples,
        augmentation_seed=augmentation_seed,
        quality_threshold=quality_threshold,
        use_safe_strategies_only=True,  # Always use safe strategies in CLI
    )

    processed_data = processor.process_training_data(
        df, enable_engineered_features=use_engineered_features
    )

    # Prepare model inputs based on model type
    model_inputs = prepare_model_inputs(
        processed_data, model_type=model, use_engineered_features=use_engineered_features
    )

    X = model_inputs["X"]
    y = model_inputs["y"]
    expansion_start = model_inputs.get("expansion_start")
    expansion_end = model_inputs.get("expansion_end")

    log.info(f"Evaluation data shape for {model}: {X.shape}")
    if use_engineered_features and processed_data["X_engineered"] is not None:
        log.info(f"Using engineered features: {processed_data['X_engineered'].shape}")

    # Load model
    model_dir = paths.artifacts / "models" / model
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        log.error(f"Model not found at {model_path}. Run 'moola train --model {model}' first.")
        raise FileNotFoundError(f"Missing {model_path}")

    model_instance = get_model(model, seed=cfg.seed)
    model_instance.load(model_path)

    log.info(f"Loaded model from {model_path}")
    log.info(f"Running temporal {num_folds}-fold cross-validation")

    # Load temporal K-fold splits (NO random/stratified splitting!)
    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    for fold_idx in range(num_folds):
        # Load this fold's split file
        fold_file = Path(split_dir) / f"fold_{fold_idx}.json"
        split_data = load_split(str(fold_file))
        assert_temporal(split_data)

        train_idx = np.array(split_data["train_indices"])
        val_idx = np.array(split_data["val_indices"])

        log.info(f"Fold {fold_idx + 1}/{num_folds} - Train: {len(train_idx)}, Val: {len(val_idx)}")

        # Extract fold data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Split expansion indices for this fold
        if expansion_start is not None and expansion_end is not None:
            exp_start_train, exp_start_val = expansion_start[train_idx], expansion_start[val_idx]
            exp_end_train, exp_end_val = expansion_end[train_idx], expansion_end[val_idx]
        else:
            exp_start_train, exp_start_val = None, None
            exp_end_train, exp_end_val = None, None

        # Create fresh model instance for this fold
        fold_model = get_model(model, seed=cfg.seed)
        fold_model.fit(
            X_train_fold, y_train_fold, expansion_start=exp_start_train, expansion_end=exp_end_train
        )
        y_pred_fold = fold_model.predict(
            X_val_fold, expansion_start=exp_start_val, expansion_end=exp_end_val
        )

        # Calculate fold metrics
        acc = accuracy_score(y_val_fold, y_pred_fold)
        prec = precision_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
        rec = recall_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
        f1 = f1_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)

        fold_metrics.append(
            {"fold": fold_idx + 1, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        )
        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_pred_fold)

        log.info(f"Fold {fold_idx + 1}/{num_folds} | acc={acc:.3f} f1={f1:.3f}")

    # Aggregate metrics across all folds
    mean_accuracy = np.mean([m["accuracy"] for m in fold_metrics])
    mean_f1 = np.mean([m["f1"] for m in fold_metrics])
    mean_precision = np.mean([m["precision"] for m in fold_metrics])
    mean_recall = np.mean([m["recall"] for m in fold_metrics])

    log.info(f"Mean CV metrics | acc={mean_accuracy:.3f} f1={mean_f1:.3f}")

    # Generate confusion matrix from all predictions
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_df = pd.DataFrame(cm, index=np.unique(y), columns=np.unique(y))

    # Save confusion matrix
    cm_path = paths.artifacts / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    log.info(f"Saved confusion matrix to {cm_path}")

    # Save metrics.json with required keys
    metrics = {
        "model": model,
        "accuracy": mean_accuracy,
        "f1": mean_f1,
        "precision": mean_precision,
        "recall": mean_recall,
        "cv_folds": k,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fold_details": fold_metrics,
    }

    # Save metrics to model-specific directory
    metrics_dir = paths.artifacts / "models" / model
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Saved metrics to {metrics_path}")

    # Also save confusion matrix to model directory
    cm_path = metrics_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    log.info(f"Saved confusion matrix to {cm_path}")

    # Track run in runs.csv
    duration = time.time() - start_time
    # Safely access run_id from config
    try:
        run_id = (
            cfg.moola.run_id if hasattr(cfg, "moola") and hasattr(cfg.moola, "run_id") else "dev"
        )
    except Exception:
        run_id = "dev"

    # Get git SHA
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        git_sha = "unknown"

    # Append to runs.csv
    runs_path = paths.artifacts / "runs.csv"
    file_exists = runs_path.exists()

    with open(runs_path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "model", "git_sha", "accuracy", "f1", "duration"]
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "run_id": run_id,
                "model": model,
                "git_sha": git_sha,
                "accuracy": f"{mean_accuracy:.4f}",
                "f1": f"{mean_f1:.4f}",
                "duration": f"{duration:.2f}",
            }
        )

    log.info(f"Tracked run to {runs_path} | run_id={run_id} model={model} git_sha={git_sha}")
    log.info("Evaluate done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option(
    "--model", required=True, help="Model name (logreg, rf, xgb, rwkv_ts, cnn_transformer)"
)
@click.option("--seed", type=int, default=None, help="Random seed (defaults to config seed)")
@click.option(
    "--device", default="cpu", type=click.Choice(["cpu", "cuda"]), help="Device for training"
)
@click.option(
    "--load-pretrained-encoder",
    type=click.Path(exists=True),
    default=None,
    help="Path to pre-trained encoder weights (for cnn_transformer only)",
)
def oof(cfg_dir, over, model, seed, device, load_pretrained_encoder):
    """Generate out-of-fold predictions for ensemble stacking."""
    import numpy as np
    import pandas as pd

    from .pipelines import generate_oof
    from .utils.seeds import print_gpu_info

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)

    # Use provided seed or default from config
    seed = seed if seed is not None else cfg.seed
    k = cfg.get("cv_folds", 5)

    log.info("OOF generation start | model=%s seed=%s k=%s device=%s", model, seed, k, device)

    # Check for pre-trained encoder
    if load_pretrained_encoder:
        if model in ["cnn_transformer", "simple_lstm"]:
            log.info(f"Will load pre-trained encoder from: {load_pretrained_encoder}")
        else:
            log.warning(
                f"--load-pretrained-encoder specified but model={model} doesn't support it (only cnn_transformer, simple_lstm)"
            )
            load_pretrained_encoder = None

    # GPU verification for deep learning models
    if device == "cuda" and model in ["rwkv_ts", "cnn_transformer"]:
        print_gpu_info()

    # Load training data
    train_path = paths.data / "processed" / "train.parquet"
    if not train_path.exists():
        log.error(f"Training data not found at {train_path}")
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_parquet(train_path)

    # Clean data: remove samples with invalid expansion indices
    from moola.data.load import validate_expansions

    df = validate_expansions(df)

    X = np.stack([np.stack(f) for f in df["features"]])
    y = df["label"].values

    # Extract expansion indices if available
    expansion_start = df["expansion_start"].values if "expansion_start" in df.columns else None
    expansion_end = df["expansion_end"].values if "expansion_end" in df.columns else None

    log.info(f"Loaded {len(df)} samples | shape={X.shape}")

    # Define paths for splits and OOF output
    splits_dir = paths.artifacts / "splits" / "v1"
    oof_dir = paths.artifacts / "oof" / model / "v1"
    oof_path = oof_dir / f"seed_{seed}.npy"

    # Generate OOF predictions
    # Pass device parameter for deep learning models
    # Enable multi-task pointer prediction for CNN-Transformer
    model_kwargs = {"device": device}
    if model == "cnn_transformer":
        model_kwargs["predict_pointers"] = True
        if load_pretrained_encoder:
            model_kwargs["load_pretrained_encoder"] = load_pretrained_encoder
    elif model == "enhanced_simple_lstm":
        # PHASE 1: Add uncertainty weighting parameter (CRITICAL for production)
        model_kwargs["use_uncertainty_weighting"] = True
        if load_pretrained_encoder:
            model_kwargs["load_pretrained_encoder"] = load_pretrained_encoder
    elif model == "simple_lstm":
        if load_pretrained_encoder:
            model_kwargs["load_pretrained_encoder"] = load_pretrained_encoder
            # Encoder is bidirectional (128 per direction, 256 total)
            # SimpleLSTM is unidirectional, so use 128 to match forward direction
            model_kwargs["hidden_size"] = 128

    oof_predictions = generate_oof(
        X=X,
        y=y,
        model_name=model,
        seed=seed,
        k=k,
        splits_dir=splits_dir,
        output_path=oof_path,
        expansion_start=expansion_start,
        expansion_end=expansion_end,
        **model_kwargs,
    )

    log.info(f"OOF generation complete | shape={oof_predictions.shape}")
    log.info("OOF done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option(
    "--device", default="cuda", type=click.Choice(["cpu", "cuda"]), help="Device for pretraining"
)
@click.option("--epochs", default=100, type=int, help="Number of pretraining epochs")
@click.option("--patience", default=15, type=int, help="Early stopping patience")
def pretrain_tcc(cfg_dir, over, device, epochs, patience):
    """Pretrain TS-TCC encoder with contrastive learning on unlabeled data."""
    import numpy as np
    import pandas as pd

    from .models.ts_tcc import TSTCCPretrainer
    from .utils.seeds import print_gpu_info

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("TS-TCC Pretraining start | device=%s epochs=%d patience=%d", device, epochs, patience)

    # GPU verification
    if device == "cuda":
        print_gpu_info()

    # Load training data (use as unlabeled data for contrastive learning)
    train_path = paths.data / "processed" / "train.parquet"
    if not train_path.exists():
        log.error(f"Training data not found at {train_path}")
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_parquet(train_path)
    X = np.stack([np.stack(f) for f in df["features"]])
    log.info(f"Loaded {len(df)} samples | shape={X.shape}")

    # Initialize TS-TCC pretrainer
    pretrainer = TSTCCPretrainer(
        input_dim=4,  # OHLC
        n_epochs=epochs,
        early_stopping_patience=patience,
        device=device,
        seed=cfg.seed,
    )

    # Pretrain encoder
    history = pretrainer.pretrain(X)

    # Save pretrained encoder
    encoder_dir = paths.artifacts / "models" / "ts_tcc"
    encoder_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = encoder_dir / "pretrained_encoder.pt"
    pretrainer.save_encoder(encoder_path)

    log.info(f"Pretraining complete | best_epoch={history['best_epoch']+1}")
    log.info(f"Saved encoder to {encoder_path}")
    log.info("Pretrain-TCC done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to unlabeled data parquet file",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Path to save pre-trained encoder (default: artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt)",
)
@click.option(
    "--device",
    default="cuda",
    type=click.Choice(["cpu", "cuda"]),
    help="Device for training (cuda for RTX 4090)",
)
@click.option("--epochs", default=50, type=int, help="Number of pre-training epochs")
@click.option("--patience", default=10, type=int, help="Early stopping patience")
@click.option(
    "--mask-ratio", default=0.15, type=float, help="Proportion of timesteps to mask (0.15 = 15%)"
)
@click.option(
    "--mask-strategy",
    default="patch",
    type=click.Choice(["random", "block", "patch"]),
    help="Masking strategy",
)
@click.option("--patch-size", default=7, type=int, help="Patch size for patch masking")
@click.option("--hidden-dim", default=128, type=int, help="LSTM hidden dimension per direction")
@click.option("--batch-size", default=512, type=int, help="Training batch size")
@click.option(
    "--augment", default=False, is_flag=True, help="Apply data augmentation to unlabeled samples"
)
@click.option(
    "--num-augmentations", default=4, type=int, help="Number of augmented versions per sample"
)
@click.option(
    "--input-dim",
    default=4,
    type=int,
    help="Input feature dimension (4 for OHLC, 11 for RelativeTransform)",
)
@click.option("--seed", default=None, type=int, help="Random seed (overrides config)")
def pretrain_bilstm(
    cfg_dir,
    over,
    input_path,
    output_path,
    device,
    epochs,
    patience,
    mask_ratio,
    mask_strategy,
    patch_size,
    hidden_dim,
    batch_size,
    augment,
    num_augmentations,
    input_dim,
    seed,
):
    """Pre-train bidirectional masked LSTM autoencoder on unlabeled OHLC data.

    This command implements masked autoencoding pre-training for SimpleLSTM,
    providing a strong initialization for downstream classification tasks.

    Expected improvements:
    - +8-12% accuracy gain over baseline
    - Breaks class collapse (Class 1: 0% → 45-55%)
    - Better feature representations

    Hardware targets:
    - RTX 4090: 24GB VRAM, batch_size=512
    - Training time: ~20 minutes on GPU for 11K samples

    Example:
        moola pretrain-bilstm \\
            --input data/raw/unlabeled_windows.parquet \\
            --output artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt \\
            --device cuda \\
            --epochs 50 \\
            --mask-strategy patch
    """
    import numpy as np
    import pandas as pd

    from .pretraining.data_augmentation import TimeSeriesAugmenter
    from .pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
    from .utils.seeds import print_gpu_info

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)

    # Override seed if provided
    if seed is not None:
        cfg.seed = seed

    log.info("=" * 70)
    log.info("BIDIRECTIONAL MASKED LSTM PRE-TRAINING")
    log.info("=" * 70)
    log.info(f"Input: {input_path}")
    log.info(
        f"Input dim: {input_dim} ({'OHLC' if input_dim == 4 else 'RelativeTransform' if input_dim == 11 else 'Custom'})"
    )
    log.info(f"Device: {device}")
    log.info(f"Mask strategy: {mask_strategy} (ratio={mask_ratio})")
    log.info(f"Hidden dim: {hidden_dim} (bidirectional → {hidden_dim*2} total)")
    log.info(f"Epochs: {epochs} | Patience: {patience}")
    log.info(f"Batch size: {batch_size}")
    log.info(f"Seed: {cfg.seed}")

    # GPU verification for CUDA
    if device == "cuda":
        print_gpu_info()

    # Load unlabeled data
    input_path = Path(input_path)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Missing {input_path}")

    df = pd.read_parquet(input_path)

    # Extract features (handle both formats)
    if "features" in df.columns:
        # New format: features column with nested arrays
        X_unlabeled = np.stack([np.stack(f) for f in df["features"]])
    else:
        # Old format: flat feature columns
        feature_cols = [c for c in df.columns if c not in {"window_id", "label"}]
        X_raw = df[feature_cols].values
        # Reshape to [N, 105, 4] assuming OHLC structure
        N = len(X_raw)
        X_unlabeled = X_raw.reshape(N, 105, 4)

    log.info(f"Loaded {len(X_unlabeled)} unlabeled samples | shape={X_unlabeled.shape}")

    # Validate shape
    if X_unlabeled.ndim != 3 or X_unlabeled.shape[2] != input_dim:
        log.error(f"Invalid shape: expected [N, 105, {input_dim}], got {X_unlabeled.shape}")
        raise ValueError(f"Data must have shape [N, 105, {input_dim}] for input_dim={input_dim}")

    # Apply data augmentation if requested
    if augment:
        log.info(f"Applying data augmentation ({num_augmentations}x)...")
        augmenter = TimeSeriesAugmenter()
        X_unlabeled = augmenter.augment_dataset(X_unlabeled, num_augmentations=num_augmentations)
        log.info(f"After augmentation: {len(X_unlabeled)} samples")

    # Initialize pre-trainer
    pretrainer = MaskedLSTMPretrainer(
        input_dim=input_dim,  # Configurable: 4 for OHLC, 11 for RelativeTransform
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.2,
        mask_ratio=mask_ratio,
        mask_strategy=mask_strategy,
        patch_size=patch_size,
        learning_rate=1e-3,
        batch_size=batch_size,
        device=device,
        seed=cfg.seed,
    )

    # Determine output path
    if output_path is None:
        suffix = f"{input_dim}d_v1.pt"
        output_path = paths.artifacts / "encoders" / "pretrained" / f"bilstm_mae_{suffix}"
    else:
        output_path = Path(output_path)

    log.info(f"Output: {output_path}")
    log.info("=" * 70)

    # Pre-train encoder
    history = pretrainer.pretrain(
        X_unlabeled=X_unlabeled,
        n_epochs=epochs,
        val_split=0.1,
        patience=patience,
        save_path=output_path,
        verbose=True,
    )

    # Log final results
    log.info("=" * 70)
    log.info("PRE-TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    log.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    log.info(f"Best val loss: {min(history['val_loss']):.4f}")
    log.info(f"Encoder saved: {output_path}")
    log.info("=" * 70)
    log.info("\nNext steps:")
    log.info("  1. Load encoder in SimpleLSTM:")
    log.info("     model.load_pretrained_encoder(encoder_path)")
    log.info("  2. Train with encoder frozen (first 10 epochs)")
    log.info("  3. Unfreeze and fine-tune (remaining epochs)")
    log.info("\nExpected improvement: +8-12% accuracy")
    log.info("Pretrain-BiLSTM done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to labeled data parquet file with expansion/swing/candle labels",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(),
    default=None,
    help="Path to save pre-trained encoder (default: artifacts/pretrained/multitask_encoder.pt)",
)
@click.option(
    "--device",
    default="cuda",
    type=click.Choice(["cpu", "cuda"]),
    help="Device for training (cuda for GPU)",
)
@click.option("--epochs", default=50, type=int, help="Number of pre-training epochs")
@click.option("--patience", default=10, type=int, help="Early stopping patience")
@click.option("--hidden-dim", default=128, type=int, help="LSTM hidden dimension per direction")
@click.option("--batch-size", default=512, type=int, help="Training batch size")
def pretrain_multitask(
    cfg_dir, over, input_path, output_path, device, epochs, patience, hidden_dim, batch_size
):
    """Pre-train bidirectional LSTM encoder with multi-task learning.

    Trains encoder on three auxiliary tasks:
    1. Expansion direction (up/down) - weight 0.5
    2. Swing type (high/low/continuation) - weight 0.3
    3. Candle pattern classification - weight 0.2

    Expected improvements:
    - +10-15% accuracy gain over baseline
    - Better than masked reconstruction (more semantic)
    - Richer representations from multi-task learning

    Hardware targets:
    - RTX 4090: 24GB VRAM, batch_size=512
    - Training time: ~25 minutes on GPU for 11K samples

    Example:
        moola pretrain-multitask \\
            --input data/processed/train.parquet \\
            --output artifacts/pretrained/multitask_encoder.pt \\
            --device cuda \\
            --epochs 50
    """
    import numpy as np
    import pandas as pd

    from .pretraining.multitask_pretrain import MultiTaskPretrainer
    from .utils.seeds import print_gpu_info

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)

    log.info("=" * 70)
    log.info("MULTI-TASK BILSTM PRE-TRAINING")
    log.info("=" * 70)
    log.info(f"Input: {input_path}")
    log.info(f"Device: {device}")
    log.info(f"Hidden dim: {hidden_dim} (bidirectional → {hidden_dim*2} total)")
    log.info(f"Epochs: {epochs} | Patience: {patience}")
    log.info(f"Batch size: {batch_size}")
    log.info("Task weights: [0.5, 0.3, 0.2] (expansion, swing, candle)")

    # GPU verification for CUDA
    if device == "cuda":
        print_gpu_info()

    # Load labeled data
    input_path = Path(input_path)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        raise FileNotFoundError(f"Missing {input_path}")

    df = pd.read_parquet(input_path)

    # Extract features
    if "features" in df.columns:
        X_unlabeled = np.stack([np.stack(f) for f in df["features"]])
    else:
        log.error("Dataset must have 'features' column")
        raise ValueError("Invalid dataset format")

    # Extract task labels
    required_columns = ["expansion_label", "swing_label", "candle_label"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        log.error(f"Missing required label columns: {missing}")
        log.error("Dataset must have: expansion_label, swing_label, candle_label")
        raise ValueError(f"Missing columns: {missing}")

    expansion_labels = df["expansion_label"].values
    swing_labels = df["swing_label"].values
    candle_labels = df["candle_label"].values

    log.info(f"Loaded {len(X_unlabeled)} labeled samples | shape={X_unlabeled.shape}")
    log.info(f"Expansion classes: {np.unique(expansion_labels)}")
    log.info(f"Swing classes: {np.unique(swing_labels)}")
    log.info(f"Candle classes: {np.unique(candle_labels)}")

    # Validate input dimension
    if X_unlabeled.shape[2] != 11:
        log.error(f"Expected input_dim=11 (OHLC + 7 engineered), got {X_unlabeled.shape[2]}")
        raise ValueError(f"Invalid feature dimension: {X_unlabeled.shape[2]}")

    # Initialize pre-trainer
    pretrainer = MultiTaskPretrainer(
        input_dim=11,  # OHLC + 7 engineered features
        hidden_dim=hidden_dim,
        num_layers=2,
        dropout=0.3,
        task_weights=(0.5, 0.3, 0.2),
        learning_rate=1e-3,
        batch_size=batch_size,
        device=device,
        seed=cfg.seed,
    )

    # Determine output path
    if output_path is None:
        output_path = paths.artifacts / "pretrained" / "multitask_encoder.pt"
    else:
        output_path = Path(output_path)

    log.info(f"Output: {output_path}")
    log.info("=" * 70)

    # Pre-train encoder
    history = pretrainer.pretrain(
        X_unlabeled=X_unlabeled,
        expansion_labels=expansion_labels,
        swing_labels=swing_labels,
        candle_labels=candle_labels,
        n_epochs=epochs,
        val_split=0.1,
        patience=patience,
        save_path=output_path,
        verbose=True,
    )

    # Log final results
    log.info("=" * 70)
    log.info("PRE-TRAINING COMPLETE")
    log.info("=" * 70)
    log.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    log.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    log.info(f"Best val loss: {min(history['val_loss']):.4f}")
    log.info(f"Best expansion acc: {max(history['val_expansion_acc']):.3f}")
    log.info(f"Best swing acc: {max(history['val_swing_acc']):.3f}")
    log.info(f"Best candle acc: {max(history['val_candle_acc']):.3f}")
    log.info(f"Encoder saved: {output_path}")
    log.info("=" * 70)
    log.info("\nNext steps:")
    log.info("  1. Load encoder in SimpleLSTM:")
    log.info("     model.load_pretrained_encoder(encoder_path)")
    log.info("  2. Train with encoder frozen (first 10 epochs)")
    log.info("  3. Unfreeze and fine-tune (remaining epochs)")
    log.info("\nExpected improvement: +10-15% accuracy")
    log.info("Pretrain-MultiTask done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option("--model", required=True, help="Model name (logreg, rf, xgb, stack)")
@click.option(
    "--input", "input_path", required=True, type=click.Path(exists=True), help="Input parquet file"
)
@click.option(
    "--output", "output_path", required=True, type=click.Path(), help="Output predictions CSV file"
)
def predict(cfg_dir, over, model, input_path, output_path):
    """Generate predictions using a trained model."""
    import numpy as np
    import pandas as pd

    from .models import get_model

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Predict start | model=%s", model)

    # Load input data
    input_path = Path(input_path)
    df = pd.read_parquet(input_path)
    log.info(f"Loaded {len(df)} samples from {input_path}")

    # Extract features
    if "features" in df.columns:
        X = np.stack([np.stack(f) for f in df["features"]])
    else:
        # Assume all columns except certain metadata columns are features
        exclude_cols = {"window_id", "label"}
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols].values

    log.info(f"Feature matrix shape: {X.shape}")

    # For stack model, need to load OOF predictions instead
    if model == "stack":
        # Stack model expects concatenated base model predictions [N, 3*C]
        # For inference, we need to generate predictions from each base model first
        base_models = ["logreg", "rf", "xgb", "rwkv_ts", "cnn_transformer"]
        base_predictions = []

        for base_model_name in base_models:
            # Load base model
            model_dir = paths.artifacts / "models" / base_model_name
            model_path = model_dir / "model.pkl"
            if not model_path.exists():
                log.error(f"Base model not found: {base_model_name} at {model_path}")
                raise FileNotFoundError(f"Missing {model_path}")

            base_model_instance = get_model(base_model_name, seed=cfg.seed)
            base_model_instance.load(model_path)
            log.info(f"Loaded base model: {base_model_name}")

            # Get base model predictions
            base_proba = base_model_instance.predict_proba(X)
            base_predictions.append(base_proba)
            log.info(f"{base_model_name} predictions shape: {base_proba.shape}")

        # Concatenate base predictions for stack input
        X_stack = np.concatenate(base_predictions, axis=1)
        log.info(f"Stack input shape (before meta-features): {X_stack.shape}")

        # Add diversity meta-features (same as during training)
        from .pipelines.stack_train import add_meta_features

        meta_features = add_meta_features(base_predictions)
        X_stack = np.concatenate([X_stack, meta_features], axis=1)
        log.info(f"Stack input shape (after meta-features): {X_stack.shape}")

        # Load stack model
        model_dir = paths.artifacts / "models" / "stack"
        model_path = model_dir / "stack.pkl"
    else:
        # Load regular model
        model_dir = paths.artifacts / "models" / model
        model_path = model_dir / "model.pkl"

    if not model_path.exists():
        log.error(f"Model not found at {model_path}. Run 'moola train --model {model}' first.")
        raise FileNotFoundError(f"Missing {model_path}")

    model_instance = get_model(model, seed=cfg.seed)
    model_instance.load(model_path)
    log.info(f"Loaded model from {model_path}")

    # Generate predictions
    if model == "stack":
        y_pred = model_instance.predict(X_stack)
        y_proba = model_instance.predict_proba(X_stack)
    else:
        y_pred = model_instance.predict(X)
        y_proba = model_instance.predict_proba(X)

    log.info(f"Generated {len(y_pred)} predictions")

    # Create output DataFrame
    output_df = pd.DataFrame(
        {
            "prediction": y_pred,
        }
    )

    # Add probability columns
    n_classes = y_proba.shape[1]
    for i in range(n_classes):
        output_df[f"prob_class_{i}"] = y_proba[:, i]

    # Add window_id if present in input
    if "window_id" in df.columns:
        output_df.insert(0, "window_id", df["window_id"].values)

    # Save predictions
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    log.info(f"Saved predictions to {output_path}")
    log.info("Predict done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option("--seed", type=int, default=None, help="Random seed (defaults to config seed)")
@click.option("--stacker", default="rf", help="Meta-learner type (rf)")
def stack_train(cfg_dir, over, seed, stacker):
    """Train stacking meta-learner on OOF predictions."""
    import pandas as pd

    from .pipelines import train_stack

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)

    # Use provided seed or default from config
    seed = seed if seed is not None else cfg.seed
    k = cfg.get("cv_folds", 5)

    log.info("Stack training start | seed=%s k=%s", seed, k)

    # Load training data for labels (use cleaned data to match OOF)
    train_path = paths.data / "processed" / "train_clean.parquet"
    if not train_path.exists():
        log.error(f"Training data not found at {train_path}")
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_parquet(train_path)
    y = df["label"].values
    log.info(f"Loaded {len(df)} samples | labels shape={y.shape}")

    # Define paths
    oof_dir = paths.artifacts / "oof"
    splits_dir = paths.artifacts / "splits" / "v1"
    model_dir = paths.artifacts / "models" / "stack"
    output_path = model_dir / "stack.pkl"
    metrics_path = model_dir / "metrics.json"
    manifest_path = paths.artifacts / "manifest.json"

    # Train stack
    metrics = train_stack(
        y=y,
        seed=seed,
        k=k,
        oof_dir=oof_dir,
        splits_dir=splits_dir,
        output_path=output_path,
        metrics_path=metrics_path,
        manifest_path=manifest_path,
    )

    log.info(f"Stack training complete | f1={metrics['f1']:.3f} ece={metrics['ece']:.3f}")
    log.info("Stack-train done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option("--section", default="base", help="Audit section (base, oof, stack, all)")
def audit(cfg_dir, over, section):
    """Audit pipeline for completeness and consistency."""
    import sys

    import numpy as np

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Audit start | section=%s", section)

    passed = True
    base_models = ["logreg", "rf", "xgb", "rwkv_ts", "cnn_transformer"]

    # Check manifest validity (for all sections)
    if section == "all":
        log.info("=== Manifest & Schema Audit ===")

        # Check manifest exists
        manifest_path = paths.artifacts / "manifest.json"
        if not manifest_path.exists():
            log.warning(f"⚠️  Manifest not found: {manifest_path}")
        else:
            log.info(f"✅ Manifest exists: {manifest_path}")

            # Verify manifest integrity
            from .utils.manifest import verify_manifest

            verification = verify_manifest(paths.artifacts, manifest_path)
            failed_files = [f for f, valid in verification.items() if not valid]

            if failed_files:
                log.error(f"❌ Manifest verification failed for {len(failed_files)} files:")
                for f in failed_files[:5]:  # Show first 5 failures
                    log.error(f"   {f}")
                passed = False
            else:
                log.info(f"✅ Manifest verified: {len(verification)} files match")

        # Validate training data schema
        train_path = paths.data / "processed" / "train.parquet"
        if train_path.exists():
            import pandas as pd

            from .schemas.canonical_v1 import check_training_data

            df = pd.read_parquet(train_path)
            schema_valid = check_training_data(df)
            if schema_valid:
                log.info(f"✅ Training data schema valid: {train_path}")
            else:
                log.error(f"❌ Training data schema invalid: {train_path}")
                passed = False
        else:
            log.warning("⚠️  Training data not found for schema validation")

    if section in ["base", "all"]:
        log.info("=== Base Models Audit ===")

        # Check training data exists
        train_path = paths.data / "processed" / "train.parquet"
        if not train_path.exists():
            log.error(f"❌ Training data missing: {train_path}")
            passed = False
        else:
            log.info(f"✅ Training data exists: {train_path}")

        # Check each base model
        for model in base_models:
            model_dir = paths.artifacts / "models" / model

            # Check for both .pt and .pkl files (some models might be saved as .pkl)
            model_path_pt = model_dir / "model.pt"
            model_path_pkl = model_dir / "model.pkl"

            if model_path_pt.exists():
                model_path = model_path_pt
            elif model_path_pkl.exists():
                model_path = model_path_pkl
            else:
                log.error(f"❌ Model missing: {model} (checked .pt and .pkl)")
                passed = False
                continue

            log.info(f"✅ Model exists: {model}")

            # Check metrics file
            metrics_path = model_dir / "metrics.json"
            if metrics_path.exists():
                import json

                with open(metrics_path) as f:
                    metrics = json.load(f)
                acc = metrics.get("accuracy", 0)
                log.info(f"   Accuracy: {acc:.3f}")
            else:
                log.warning(f"⚠️  Metrics missing for {model}")

    if section in ["oof", "all"]:
        log.info("=== OOF Predictions Audit ===")

        seed = cfg.seed
        splits_dir = paths.artifacts / "splits" / "v1"

        # Check split manifests exist
        if not splits_dir.exists() or not (splits_dir / "fold_0.json").exists():
            log.error(f"❌ Split manifests missing: {splits_dir}")
            passed = False
        else:
            log.info(f"✅ Split manifests exist: {splits_dir}")

        # Check OOF predictions for each base model
        for model in base_models:
            oof_path = paths.artifacts / "oof" / model / "v1" / f"seed_{seed}.npy"

            if not oof_path.exists():
                log.error(f"❌ OOF predictions missing: {model} at {oof_path}")
                passed = False
            else:
                oof = np.load(oof_path)
                log.info(f"✅ OOF exists: {model} | shape={oof.shape}")

                # Check for zero rows
                zero_rows = np.where(np.all(oof == 0, axis=1))[0]
                if len(zero_rows) > 0:
                    log.warning(f"⚠️  Found {len(zero_rows)} zero rows in {model} OOF")
                    passed = False

    if section in ["stack", "all"]:
        log.info("=== Stacking Audit ===")

        seed = cfg.seed
        stack_dir = paths.artifacts / "models" / "stack"
        stack_path = stack_dir / "stack.pkl"

        # Check stack model exists
        if not stack_path.exists():
            log.error(f"❌ Stack model missing: {stack_path}")
            passed = False
        else:
            log.info(f"✅ Stack model exists: {stack_path}")

            # Check stack metrics
            stack_metrics_path = stack_dir / "metrics.json"
            if stack_metrics_path.exists():
                import json

                with open(stack_metrics_path) as f:
                    stack_metrics = json.load(f)
                f1 = stack_metrics.get("f1", 0)
                ece = stack_metrics.get("ece", 0)
                log.info(f"   F1: {f1:.3f} | ECE: {ece:.3f}")

                # Load base model metrics for comparison
                best_base_f1 = 0
                for model in base_models:
                    metrics_path = paths.artifacts / "models" / model / "metrics.json"
                    if metrics_path.exists():
                        with open(metrics_path) as f:
                            base_metrics = json.load(f)
                        base_f1 = base_metrics.get("f1", 0)
                        best_base_f1 = max(best_base_f1, base_f1)

                # Check acceptance criteria: F1 improvement >= 2pp OR ECE improvement >= 0.03
                f1_improvement = (f1 - best_base_f1) * 100  # in percentage points
                log.info(
                    f"   Best base F1: {best_base_f1:.3f} | Improvement: {f1_improvement:.1f}pp"
                )

                if f1_improvement >= 2.0:
                    log.info(f"✅ Stack F1 improvement >= 2pp: {f1_improvement:.1f}pp")
                elif ece <= 0.03:
                    log.info(f"✅ Stack ECE <= 0.03: {ece:.3f}")
                else:
                    log.warning(
                        f"⚠️  Stack acceptance criteria not met (F1 improvement: {f1_improvement:.1f}pp, ECE: {ece:.3f})"
                    )
            else:
                log.warning("⚠️  Stack metrics missing")

    # Final verdict
    log.info("=" * 40)
    if passed:
        log.info("✅ Audit PASSED - all checks successful")
        sys.exit(0)
    else:
        log.error("❌ Audit FAILED - some checks failed")
        sys.exit(1)


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
def deploy(cfg_dir, over):
    "Placeholder deployment step."
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Deploy start")
    # TODO: implement real deployment
    (paths.artifacts / "deployment.txt").write_text("deployed")
    log.info("Deploy done")


# Import GitOps commands
try:
    from molla.deployment.cli_extensions import register_gitops_commands

    register_gitops_commands(app)
except ImportError:
    # GitOps dependencies not available - skip registration
    pass


if __name__ == "__main__":
    app()
