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
def ingest(cfg_dir, over):
    """Ingest and validate raw data, write processed/train.parquet."""
    import numpy as np
    import pandas as pd

    from .schema import TrainingDataRow

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Ingest start")

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
@click.option("--model", default="logreg", help="Model name (logreg, rf, xgb)")
def train(cfg_dir, over, model):
    """Train classifier on processed/train.parquet."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    from .models import get_model

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Train start | model=%s seed=%s", model, cfg.seed)

    # Load training data
    train_path = paths.data / "processed" / "train.parquet"
    if not train_path.exists():
        log.error(f"Training data not found at {train_path}. Run 'moola ingest' first.")
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_parquet(train_path)
    log.info(f"Loaded {len(df)} training samples from {train_path}")

    # Extract features and labels
    # Convert list of features to 2D numpy array
    X = np.array(df["features"].tolist())
    y = df["label"].values

    log.info(f"Feature shape: {X.shape} | Unique labels: {np.unique(y)}")

    # Stratified train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=cfg.seed, stratify=y
    )

    log.info(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # Get model from registry and train
    model_instance = get_model(model, seed=cfg.seed)
    model_instance.fit(X_train, y_train)

    # Calculate accuracy using model's predict method (handles label encoding)
    y_train_pred = model_instance.predict(X_train)
    y_test_pred = model_instance.predict(X_test)
    train_score = (y_train_pred == y_train).mean()
    test_score = (y_test_pred == y_test).mean()
    log.info(f"Train accuracy: {train_score:.3f} | Test accuracy: {test_score:.3f}")

    # Save model
    model_dir = paths.artifacts / "models" / model
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.pkl"
    model_instance.save(model_path)

    log.info(f"Saved model to {model_path}")
    log.info("Train done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option("--model", default="logreg", help="Model name (logreg, rf, xgb)")
def evaluate(cfg_dir, over, model):
    """Evaluate model with stratified K-fold CV, output metrics and confusion matrix."""
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
    from sklearn.model_selection import StratifiedKFold

    from .models import get_model

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Evaluate start | model=%s", model)

    start_time = time.time()

    # Load data
    train_path = paths.data / "processed" / "train.parquet"
    if not train_path.exists():
        log.error(f"Training data not found at {train_path}")
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_parquet(train_path)
    X = np.array(df["features"].tolist())
    y = df["label"].values

    # Load model
    model_dir = paths.artifacts / "models" / model
    model_path = model_dir / "model.pkl"
    if not model_path.exists():
        log.error(f"Model not found at {model_path}. Run 'moola train --model {model}' first.")
        raise FileNotFoundError(f"Missing {model_path}")

    model_instance = get_model(model, seed=cfg.seed)
    model_instance.load(model_path)

    log.info(f"Loaded model from {model_path}")
    log.info(f"Running stratified {cfg.get('cv_folds', 5)}-fold cross-validation")

    # Stratified K-fold cross-validation
    k = cfg.get("cv_folds", 5)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.seed)

    fold_metrics = []
    all_y_true = []
    all_y_pred = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Create fresh model instance for this fold
        fold_model = get_model(model, seed=cfg.seed)
        fold_model.fit(X_train_fold, y_train_fold)
        y_pred_fold = fold_model.predict(X_val_fold)

        # Calculate fold metrics
        acc = accuracy_score(y_val_fold, y_pred_fold)
        prec = precision_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
        rec = recall_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
        f1 = f1_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)

        fold_metrics.append({"fold": fold_idx, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1})
        all_y_true.extend(y_val_fold)
        all_y_pred.extend(y_pred_fold)

        log.info(f"Fold {fold_idx}/{k} | acc={acc:.3f} f1={f1:.3f}")

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
        run_id = cfg.moola.run_id if hasattr(cfg, "moola") and hasattr(cfg.moola, "run_id") else "dev"
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
        writer.writerow({
            "run_id": run_id,
            "model": model,
            "git_sha": git_sha,
            "accuracy": f"{mean_accuracy:.4f}",
            "f1": f"{mean_f1:.4f}",
            "duration": f"{duration:.2f}",
        })

    log.info(f"Tracked run to {runs_path} | run_id={run_id} model={model} git_sha={git_sha}")
    log.info("Evaluate done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option("--model", required=True, help="Model name (logreg, rf, xgb)")
@click.option("--seed", type=int, default=None, help="Random seed (defaults to config seed)")
def oof(cfg_dir, over, model, seed):
    """Generate out-of-fold predictions for ensemble stacking."""
    import numpy as np
    import pandas as pd

    from .pipelines import generate_oof

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)

    # Use provided seed or default from config
    seed = seed if seed is not None else cfg.seed
    k = cfg.get("cv_folds", 5)

    log.info("OOF generation start | model=%s seed=%s k=%s", model, seed, k)

    # Load training data
    train_path = paths.data / "processed" / "train.parquet"
    if not train_path.exists():
        log.error(f"Training data not found at {train_path}")
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_parquet(train_path)
    X = np.array(df["features"].tolist())
    y = df["label"].values

    log.info(f"Loaded {len(df)} samples | shape={X.shape}")

    # Define paths for splits and OOF output
    splits_dir = paths.artifacts / "splits" / "v1"
    oof_dir = paths.artifacts / "oof" / model / "v1"
    oof_path = oof_dir / f"seed_{seed}.npy"

    # Generate OOF predictions
    oof_predictions = generate_oof(
        X=X,
        y=y,
        model_name=model,
        seed=seed,
        k=k,
        splits_dir=splits_dir,
        output_path=oof_path,
    )

    log.info(f"OOF generation complete | shape={oof_predictions.shape}")
    log.info("OOF done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
@click.option("--model", required=True, help="Model name (logreg, rf, xgb, stack)")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True), help="Input parquet file")
@click.option("--output", "output_path", required=True, type=click.Path(), help="Output predictions CSV file")
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
        X = np.array(df["features"].tolist())
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
        base_models = ["logreg", "rf", "xgb"]
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
        log.info(f"Stack input shape: {X_stack.shape}")

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
    output_df = pd.DataFrame({
        "prediction": y_pred,
    })

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
def stack_train(cfg_dir, over, seed):
    """Train stacking meta-learner on OOF predictions."""
    import numpy as np
    import pandas as pd

    from .pipelines import train_stack

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)

    # Use provided seed or default from config
    seed = seed if seed is not None else cfg.seed
    k = cfg.get("cv_folds", 5)

    log.info("Stack training start | seed=%s k=%s", seed, k)

    # Load training data for labels
    train_path = paths.data / "processed" / "train.parquet"
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

    from .models import list_models

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Audit start | section=%s", section)

    passed = True
    base_models = ["logreg", "rf", "xgb"]

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
            log.warning(f"⚠️  Training data not found for schema validation")

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
            model_path = model_dir / "model.pkl"

            if not model_path.exists():
                log.error(f"❌ Model missing: {model} at {model_path}")
                passed = False
            else:
                log.info(f"✅ Model exists: {model}")

                # Check metrics file
                metrics_path = model_dir / "metrics.json"
                if metrics_path.exists():
                    import json

                    with open(metrics_path, "r") as f:
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

                with open(stack_metrics_path, "r") as f:
                    stack_metrics = json.load(f)
                f1 = stack_metrics.get("f1", 0)
                ece = stack_metrics.get("ece", 0)
                log.info(f"   F1: {f1:.3f} | ECE: {ece:.3f}")

                # Load base model metrics for comparison
                best_base_f1 = 0
                for model in base_models:
                    metrics_path = paths.artifacts / "models" / model / "metrics.json"
                    if metrics_path.exists():
                        with open(metrics_path, "r") as f:
                            base_metrics = json.load(f)
                        base_f1 = base_metrics.get("f1", 0)
                        best_base_f1 = max(best_base_f1, base_f1)

                # Check acceptance criteria: F1 improvement >= 2pp OR ECE improvement >= 0.03
                f1_improvement = (f1 - best_base_f1) * 100  # in percentage points
                log.info(f"   Best base F1: {best_base_f1:.3f} | Improvement: {f1_improvement:.1f}pp")

                if f1_improvement >= 2.0:
                    log.info(f"✅ Stack F1 improvement >= 2pp: {f1_improvement:.1f}pp")
                elif ece <= 0.03:
                    log.info(f"✅ Stack ECE <= 0.03: {ece:.3f}")
                else:
                    log.warning(f"⚠️  Stack acceptance criteria not met (F1 improvement: {f1_improvement:.1f}pp, ECE: {ece:.3f})")
            else:
                log.warning(f"⚠️  Stack metrics missing")

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


if __name__ == "__main__":
    app()
