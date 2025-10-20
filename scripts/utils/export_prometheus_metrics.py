#!/usr/bin/env python3
"""Export MLflow experiment metrics to Prometheus Pushgateway.

This script extracts metrics from an MLflow run and pushes them
to Prometheus Pushgateway for monitoring and alerting.
"""

import argparse
import sys
from typing import Dict, Optional

import mlflow
from loguru import logger
from prometheus_client import CollectorRegistry, Counter, Gauge, push_to_gateway


def get_run_metrics(experiment_name: str, run_name: str) -> Optional[Dict]:
    """Get metrics from an MLflow run.

    Args:
        experiment_name: MLflow experiment name
        run_name: MLflow run name

    Returns:
        Dictionary of metrics or None if run not found
    """
    client = mlflow.MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experiment '{experiment_name}' not found")
        return None

    # Search for run by name
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1,
    )

    if not runs:
        logger.error(f"Run '{run_name}' not found in experiment '{experiment_name}'")
        return None

    run = runs[0]

    # Extract metrics
    metrics = {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "accuracy": run.data.metrics.get("oof_accuracy", 0.0),
        "class_0_accuracy": run.data.metrics.get("accuracy_class_0", 0.0),
        "class_1_accuracy": run.data.metrics.get("accuracy_class_1", 0.0),
        "start_time": run.info.start_time / 1000,  # Convert to seconds
        "end_time": (run.info.end_time or 0) / 1000,
    }

    # Calculate duration
    if run.info.end_time:
        metrics["duration_seconds"] = (run.info.end_time - run.info.start_time) / 1000
    else:
        metrics["duration_seconds"] = 0

    return metrics


def push_metrics_to_prometheus(
    metrics: Dict,
    experiment_name: str,
    pushgateway_url: str
) -> None:
    """Push metrics to Prometheus Pushgateway.

    Args:
        metrics: Dictionary of metrics
        experiment_name: Experiment name for labeling
        pushgateway_url: Pushgateway URL
    """
    registry = CollectorRegistry()

    # Create Prometheus metrics
    accuracy = Gauge(
        'mlflow_oof_accuracy',
        'Out-of-fold accuracy',
        ['experiment', 'run_name'],
        registry=registry
    )

    class_0_accuracy = Gauge(
        'mlflow_class_0_accuracy',
        'Class 0 accuracy',
        ['experiment', 'run_name'],
        registry=registry
    )

    class_1_accuracy = Gauge(
        'mlflow_class_1_accuracy',
        'Class 1 accuracy',
        ['experiment', 'run_name'],
        registry=registry
    )

    duration = Gauge(
        'mlflow_experiment_duration_seconds',
        'Experiment duration in seconds',
        ['experiment', 'run_name'],
        registry=registry
    )

    experiment_start = Gauge(
        'mlflow_experiment_start_time',
        'Experiment start time (Unix timestamp)',
        ['experiment', 'run_name'],
        registry=registry
    )

    # Set metric values
    run_name = experiment_name.split('_')[-1]  # Extract run identifier

    accuracy.labels(experiment=experiment_name, run_name=run_name).set(
        metrics['accuracy']
    )
    class_0_accuracy.labels(experiment=experiment_name, run_name=run_name).set(
        metrics['class_0_accuracy']
    )
    class_1_accuracy.labels(experiment=experiment_name, run_name=run_name).set(
        metrics['class_1_accuracy']
    )
    duration.labels(experiment=experiment_name, run_name=run_name).set(
        metrics['duration_seconds']
    )
    experiment_start.labels(experiment=experiment_name, run_name=run_name).set(
        metrics['start_time']
    )

    # Push to Pushgateway
    try:
        push_to_gateway(
            pushgateway_url,
            job='mlflow_experiments',
            registry=registry
        )
        logger.success(f"Pushed metrics to Pushgateway at {pushgateway_url}")
    except Exception as e:
        logger.error(f"Failed to push metrics: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Export MLflow metrics to Prometheus Pushgateway"
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="MLflow run name (e.g., phase1_tw_0.10)",
    )
    parser.add_argument(
        "--pushgateway-url",
        default="localhost:9091",
        help="Prometheus Pushgateway URL",
    )

    args = parser.parse_args()

    logger.info(f"Exporting metrics for run: {args.experiment_name}")

    # Get metrics from MLflow
    mlflow_experiment = "lstm-optimization-2025"  # Default experiment name
    metrics = get_run_metrics(mlflow_experiment, args.experiment_name)

    if metrics is None:
        logger.error("Failed to get metrics from MLflow")
        sys.exit(1)

    logger.info(f"Retrieved metrics: {metrics}")

    # Push to Prometheus
    push_metrics_to_prometheus(
        metrics,
        args.experiment_name,
        args.pushgateway_url
    )

    logger.success("Metrics export complete")


if __name__ == "__main__":
    main()
