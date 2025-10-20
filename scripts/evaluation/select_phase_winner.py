#!/usr/bin/env python3
"""Select the winning configuration from a completed phase.

This script queries MLflow to find the best-performing experiment
from a given phase based on accuracy and class balance metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd
from loguru import logger


def get_phase_runs(experiment_name: str, phase: int) -> List[mlflow.entities.Run]:
    """Get all runs for a specific phase.

    Args:
        experiment_name: MLflow experiment name
        phase: Phase number (1, 2, or 3)

    Returns:
        List of MLflow runs for the phase
    """
    client = mlflow.MlflowClient()

    # Get experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experiment '{experiment_name}' not found")
        sys.exit(1)

    # Define phase-specific run name patterns
    phase_patterns = {
        1: "phase1_tw_",
        2: "phase2_",
        3: "phase3_pretrain_",
    }

    if phase not in phase_patterns:
        logger.error(f"Invalid phase: {phase}. Must be 1, 2, or 3")
        sys.exit(1)

    pattern = phase_patterns[phase]

    # Search for runs matching the phase pattern
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName LIKE '{pattern}%'",
        order_by=["metrics.oof_accuracy DESC"],
    )

    if not runs:
        logger.warning(f"No runs found for phase {phase} in experiment '{experiment_name}'")

    logger.info(f"Found {len(runs)} runs for phase {phase}")
    return runs


def select_winner(
    runs: List[mlflow.entities.Run],
    phase: int,
    min_class1_accuracy: float = 0.30
) -> Optional[Dict]:
    """Select the best configuration from phase runs.

    Selection criteria:
    - Phase 1: Best overall accuracy with class_1_accuracy > threshold
    - Phase 2: Best overall accuracy with class_1_accuracy > threshold
    - Phase 3: Best overall accuracy with class_1_accuracy > threshold

    Args:
        runs: List of MLflow runs
        phase: Phase number
        min_class1_accuracy: Minimum class 1 accuracy threshold

    Returns:
        Dictionary with winner details or None
    """
    if not runs:
        return None

    # Convert runs to DataFrame for easier analysis
    run_data = []
    for run in runs:
        data = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "accuracy": run.data.metrics.get("oof_accuracy", 0.0),
            "class_1_accuracy": run.data.metrics.get("accuracy_class_1", 0.0),
            "class_0_accuracy": run.data.metrics.get("accuracy_class_0", 0.0),
        }

        # Phase-specific parameters
        if phase == 1:
            # Extract time_warp_sigma from run name (e.g., "phase1_tw_0.10")
            run_name = data["run_name"]
            if "tw_" in run_name:
                sigma_str = run_name.split("tw_")[1]
                data["time_warp_sigma"] = float(sigma_str)

        elif phase == 2:
            # Extract architecture parameters
            data["hidden_size"] = run.data.params.get("model_hidden_size")
            data["num_heads"] = run.data.params.get("model_num_heads")
            data["num_layers"] = run.data.params.get("model_num_layers")

        elif phase == 3:
            # Extract pre-training epochs from run name
            run_name = data["run_name"]
            if "_e" in run_name:
                epochs_str = run_name.split("_e")[1]
                data["pretrain_epochs"] = int(epochs_str)

        run_data.append(data)

    df = pd.DataFrame(run_data)

    # Filter by minimum class 1 accuracy
    df_filtered = df[df["class_1_accuracy"] >= min_class1_accuracy]

    if len(df_filtered) == 0:
        logger.warning(
            f"No runs meet minimum class 1 accuracy threshold ({min_class1_accuracy:.1%})"
        )
        logger.warning("Selecting best run without class balance constraint")
        df_filtered = df

    # Sort by overall accuracy (descending)
    df_sorted = df_filtered.sort_values("accuracy", ascending=False)

    # Select winner
    winner_row = df_sorted.iloc[0]
    winner = winner_row.to_dict()

    # Log selection details
    logger.info(f"Phase {phase} Winner:")
    logger.info(f"  Run ID: {winner['run_id']}")
    logger.info(f"  Run Name: {winner['run_name']}")
    logger.info(f"  Overall Accuracy: {winner['accuracy']:.4f}")
    logger.info(f"  Class 1 Accuracy: {winner['class_1_accuracy']:.4f}")

    if phase == 1:
        logger.info(f"  Time Warp Sigma: {winner.get('time_warp_sigma', 'N/A')}")
    elif phase == 2:
        logger.info(f"  Hidden Size: {winner.get('hidden_size', 'N/A')}")
        logger.info(f"  Num Heads: {winner.get('num_heads', 'N/A')}")
        logger.info(f"  Num Layers: {winner.get('num_layers', 'N/A')}")
    elif phase == 3:
        logger.info(f"  Pretrain Epochs: {winner.get('pretrain_epochs', 'N/A')}")

    return winner


def main():
    parser = argparse.ArgumentParser(
        description="Select winning configuration from experiment phase"
    )
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Phase number (1=time_warp, 2=architecture, 3=depth)",
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default="phase_winner.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--min-class1-accuracy",
        type=float,
        default=0.30,
        help="Minimum class 1 accuracy threshold",
    )

    args = parser.parse_args()

    logger.info(f"Selecting Phase {args.phase} winner from '{args.experiment_name}'")

    # Get phase runs
    runs = get_phase_runs(args.experiment_name, args.phase)

    if not runs:
        logger.error("No runs found for this phase")
        sys.exit(1)

    # Select winner
    winner = select_winner(runs, args.phase, args.min_class1_accuracy)

    if winner is None:
        logger.error("Failed to select winner")
        sys.exit(1)

    # Save to file
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output_file, "w") as f:
        json.dump(winner, f, indent=2)

    logger.success(f"Saved winner details to {args.output_file}")


if __name__ == "__main__":
    main()
