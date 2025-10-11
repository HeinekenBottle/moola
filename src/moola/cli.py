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
def train(cfg_dir, over):
    """Train baseline classifier on processed/train.parquet."""
    import pickle

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Train start | seed=%s", cfg.seed)

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

    # Train LogisticRegression baseline
    model = LogisticRegression(random_state=cfg.seed, max_iter=1000)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    log.info(f"Train accuracy: {train_score:.3f} | Test accuracy: {test_score:.3f}")

    # Save model
    model_path = paths.artifacts / "model.bin"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    log.info(f"Saved model to {model_path}")
    log.info("Train done")


@app.command()
@click.option("--cfg-dir", type=click.Path(exists=True, file_okay=False), default="configs")
@click.option("--over", multiple=True)
def evaluate(cfg_dir, over):
    """Evaluate model with stratified K-fold CV, output metrics and confusion matrix."""
    import csv
    import json
    import pickle
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

    cfg = _load_cfg(Path(cfg_dir), list(over))
    paths = resolve_paths()
    log = setup_logging(paths.logs)
    log.info("Evaluate start")

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
    model_path = paths.artifacts / "model.bin"
    if not model_path.exists():
        log.error(f"Model not found at {model_path}. Run 'moola train' first.")
        raise FileNotFoundError(f"Missing {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

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

        # Clone and train model on this fold
        from sklearn.base import clone

        fold_model = clone(model)
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
        "accuracy": mean_accuracy,
        "f1": mean_f1,
        "precision": mean_precision,
        "recall": mean_recall,
        "cv_folds": k,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "fold_details": fold_metrics,
    }

    metrics_path = paths.artifacts / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info(f"Saved metrics to {metrics_path}")

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
        writer = csv.DictWriter(f, fieldnames=["run_id", "git_sha", "accuracy", "f1", "duration"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "run_id": run_id,
            "git_sha": git_sha,
            "accuracy": f"{mean_accuracy:.4f}",
            "f1": f"{mean_f1:.4f}",
            "duration": f"{duration:.2f}",
        })

    log.info(f"Tracked run to {runs_path} | run_id={run_id} git_sha={git_sha}")
    log.info("Evaluate done")


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
