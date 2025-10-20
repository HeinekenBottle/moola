#!/usr/bin/env python3
"""Select the best model across all 13 experiments.

Compares all experiments from phases 1-3 and selects the best
performing model based on overall accuracy and class balance.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd
from loguru import logger


def get_all_experiment_runs(experiment_name: str) -> List[mlflow.entities.Run]:
    """Get all runs from the experiment.

    Args:
        experiment_name: MLflow experiment name

    Returns:
        List of all MLflow runs
    """
    client = mlflow.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        logger.error(f"Experiment '{experiment_name}' not found")
        sys.exit(1)

    # Get all runs, ordered by accuracy
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.oof_accuracy DESC"],
    )

    logger.info(f"Found {len(runs)} total runs in experiment")
    return runs


def select_best_model(
    runs: List[mlflow.entities.Run],
    min_class1_accuracy: float = 0.30
) -> Optional[Dict]:
    """Select the best model across all experiments.

    Selection criteria:
    1. Class 1 accuracy must be >= min_class1_accuracy
    2. Maximum overall accuracy among qualified runs

    Args:
        runs: List of all MLflow runs
        min_class1_accuracy: Minimum class 1 accuracy threshold

    Returns:
        Dictionary with best model details
    """
    if not runs:
        return None

    # Convert runs to DataFrame
    run_data = []
    for run in runs:
        data = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "accuracy": run.data.metrics.get("oof_accuracy", 0.0),
            "class_1_accuracy": run.data.metrics.get("accuracy_class_1", 0.0),
            "class_0_accuracy": run.data.metrics.get("accuracy_class_0", 0.0),
            "start_time": run.info.start_time,
        }

        # Extract phase from run name
        run_name = data["run_name"]
        if "phase1" in run_name:
            data["phase"] = "phase1_time_warp_ablation"
        elif "phase2" in run_name:
            data["phase"] = "phase2_architecture_search"
        elif "phase3" in run_name:
            data["phase"] = "phase3_depth_search"
        else:
            data["phase"] = "unknown"

        run_data.append(data)

    df = pd.DataFrame(run_data)

    # Filter by minimum class 1 accuracy
    df_filtered = df[df["class_1_accuracy"] >= min_class1_accuracy]

    if len(df_filtered) == 0:
        logger.warning(
            f"No models meet class 1 accuracy threshold ({min_class1_accuracy:.1%})"
        )
        logger.warning("Selecting best model without class balance constraint")
        df_filtered = df

    # Sort by overall accuracy
    df_sorted = df_filtered.sort_values("accuracy", ascending=False)

    # Select best
    best_row = df_sorted.iloc[0]
    best = best_row.to_dict()

    logger.info("Best Model Selected:")
    logger.info(f"  Run ID: {best['run_id']}")
    logger.info(f"  Run Name: {best['run_name']}")
    logger.info(f"  Phase: {best['phase']}")
    logger.info(f"  Overall Accuracy: {best['accuracy']:.4f}")
    logger.info(f"  Class 0 Accuracy: {best['class_0_accuracy']:.4f}")
    logger.info(f"  Class 1 Accuracy: {best['class_1_accuracy']:.4f}")

    return best, df_sorted


def generate_comparison_report(
    df: pd.DataFrame,
    best_model: Dict,
    output_path: Path
) -> None:
    """Generate markdown comparison report.

    Args:
        df: DataFrame with all runs
        best_model: Best model details
        output_path: Output markdown file path
    """
    report = []
    report.append("# LSTM Optimization Experiment Comparison Report")
    report.append("")
    report.append(f"**Total Experiments**: {len(df)}")
    report.append("")

    # Best model summary
    report.append("## Best Model")
    report.append("")
    report.append(f"- **Run ID**: `{best_model['run_id']}`")
    report.append(f"- **Run Name**: {best_model['run_name']}")
    report.append(f"- **Phase**: {best_model['phase']}")
    report.append(f"- **Overall Accuracy**: {best_model['accuracy']:.4f}")
    report.append(f"- **Class 0 Accuracy**: {best_model['class_0_accuracy']:.4f}")
    report.append(f"- **Class 1 Accuracy**: {best_model['class_1_accuracy']:.4f}")
    report.append("")

    # Phase breakdown
    report.append("## Results by Phase")
    report.append("")

    for phase in df["phase"].unique():
        if phase == "unknown":
            continue

        phase_df = df[df["phase"] == phase].sort_values("accuracy", ascending=False)

        report.append(f"### {phase.replace('_', ' ').title()}")
        report.append("")
        report.append("| Rank | Run Name | Accuracy | Class 0 Acc | Class 1 Acc |")
        report.append("|------|----------|----------|-------------|-------------|")

        for idx, (_, row) in enumerate(phase_df.head(10).iterrows(), 1):
            report.append(
                f"| {idx} | {row['run_name']} | "
                f"{row['accuracy']:.4f} | "
                f"{row['class_0_accuracy']:.4f} | "
                f"{row['class_1_accuracy']:.4f} |"
            )

        report.append("")

    # Top 10 overall
    report.append("## Top 10 Models Overall")
    report.append("")
    report.append("| Rank | Run Name | Phase | Accuracy | Class 0 Acc | Class 1 Acc |")
    report.append("|------|----------|-------|----------|-------------|-------------|")

    for idx, (_, row) in enumerate(df.head(10).iterrows(), 1):
        report.append(
            f"| {idx} | {row['run_name']} | {row['phase']} | "
            f"{row['accuracy']:.4f} | "
            f"{row['class_0_accuracy']:.4f} | "
            f"{row['class_1_accuracy']:.4f} |"
        )

    report.append("")

    # Statistical summary
    report.append("## Statistical Summary")
    report.append("")
    report.append("| Metric | Mean | Std | Min | Max |")
    report.append("|--------|------|-----|-----|-----|")

    for metric in ["accuracy", "class_0_accuracy", "class_1_accuracy"]:
        report.append(
            f"| {metric.replace('_', ' ').title()} | "
            f"{df[metric].mean():.4f} | "
            f"{df[metric].std():.4f} | "
            f"{df[metric].min():.4f} | "
            f"{df[metric].max():.4f} |"
        )

    report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")
    report.append("1. **Deploy**: Tag the best model as `production_candidate` in MLflow")
    report.append("2. **Validate**: Run additional validation on held-out test set")
    report.append("3. **Monitor**: Track model performance in production")
    report.append("4. **Iterate**: Consider ensemble of top-3 models for improved robustness")
    report.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(report))

    logger.success(f"Generated comparison report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Select best model from all experiment phases"
    )
    parser.add_argument(
        "--experiment-name",
        required=True,
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default="comparison_report.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--min-class1-accuracy",
        type=float,
        default=0.30,
        help="Minimum class 1 accuracy threshold",
    )

    args = parser.parse_args()

    logger.info(f"Analyzing all experiments in '{args.experiment_name}'")

    # Get all runs
    runs = get_all_experiment_runs(args.experiment_name)

    if not runs:
        logger.error("No runs found")
        sys.exit(1)

    # Select best model
    best_model, df_sorted = select_best_model(runs, args.min_class1_accuracy)

    if best_model is None:
        logger.error("Failed to select best model")
        sys.exit(1)

    # Save best model details
    best_model_path = Path("best_model.json")
    with open(best_model_path, "w") as f:
        json.dump(best_model, f, indent=2)

    logger.success(f"Saved best model details to {best_model_path}")

    # Generate comparison report
    generate_comparison_report(df_sorted, best_model, args.output_report)

    logger.success("Model selection complete!")


if __name__ == "__main__":
    main()
