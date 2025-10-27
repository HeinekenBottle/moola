#!/usr/bin/env python3
"""
Cleanlab CLI Entrypoint — Consolidated Interface

This script provides a unified command-line interface for all Cleanlab functionality
in the Moola project. It consolidates the following scripts:
- cleanlab_analysis.py
- convert_to_cleanlab_format.py
- export_for_cleanlab_studio.py
- run_cleanlab_phase2.py

Usage:
    python3 scripts/cleanlab/run_cleanlab.py analyze --data data/processed/labeled/train_latest.parquet
    python3 scripts/cleanlab/run_cleanlab.py convert --input data.parquet --output cleanlab_format.csv
    python3 scripts/cleanlab/run_cleanlab.py export --data data.parquet --output studio_export/
    python3 scripts/cleanlab/run_cleanlab.py phase2 --data data.parquet --model jade

For detailed usage, see: src/moola/utils/cleanlab_utils.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import click

from moola.utils.cleanlab_utils import (
    analyze_data_quality,
    convert_to_cleanlab_format,
    export_for_studio,
    run_phase2_analysis,
)


@click.group()
def cli():
    """Cleanlab CLI for data quality analysis and label correction."""
    pass


@cli.command()
@click.option("--data", required=True, help="Path to training data (parquet)")
@click.option("--output", default="cleanlab_analysis.csv", help="Output CSV path")
@click.option("--threshold", default=0.3, type=float, help="Quality threshold")
def analyze(data, output, threshold):
    """Analyze data quality using Cleanlab."""
    click.echo(f"Analyzing data quality: {data}")
    analyze_data_quality(data, output, threshold)
    click.echo(f"✅ Analysis complete: {output}")


@cli.command()
@click.option("--input", required=True, help="Input parquet file")
@click.option("--output", required=True, help="Output CSV file")
def convert(input, output):
    """Convert data to Cleanlab format."""
    click.echo(f"Converting {input} → {output}")
    convert_to_cleanlab_format(input, output)
    click.echo("✅ Conversion complete")


@cli.command()
@click.option("--data", required=True, help="Path to training data")
@click.option("--output", required=True, help="Output directory for Studio export")
def export(data, output):
    """Export data for Cleanlab Studio."""
    click.echo(f"Exporting to Cleanlab Studio: {output}")
    export_for_studio(data, output)
    click.echo("✅ Export complete")


@cli.command()
@click.option("--data", required=True, help="Path to training data")
@click.option("--model", default="jade", help="Model to use for predictions")
@click.option("--output", default="phase2_results.csv", help="Output CSV path")
def phase2(data, model, output):
    """Run Phase 2 Cleanlab analysis with model predictions."""
    click.echo(f"Running Phase 2 analysis with {model}")
    run_phase2_analysis(data, model, output)
    click.echo(f"✅ Phase 2 complete: {output}")


if __name__ == "__main__":
    cli()
