#!/usr/bin/env python3
"""Aggregate and Analyze LSTM Optimization Experiment Results.

Generates comprehensive analysis reports:
- Phase-wise comparison tables
- Winner selection validation
- Performance improvement analysis
- Time warping impact visualization
- Architecture ablation analysis

Usage:
    python aggregate_results.py
    python aggregate_results.py --results_dir data/artifacts --output results_analysis.html
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


class ResultsAggregator:
    """Aggregates and analyzes experiment results."""

    def __init__(self, results_dir: Path):
        """Initialize aggregator.

        Args:
            results_dir: Directory containing experiment results
        """
        self.results_dir = results_dir
        self.all_results = []

    def load_all_results(self) -> List[Dict]:
        """Load results from all experiments.

        Returns:
            List of result dictionaries
        """
        results = []

        # Search for results.json in experiment subdirectories
        for exp_dir in self.results_dir.glob("exp_*"):
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    result = json.load(f)
                    result['experiment_id'] = exp_dir.name
                    results.append(result)

        print(f"[LOADED] {len(results)} experiment results")
        self.all_results = results
        return results

    def create_summary_table(self) -> pd.DataFrame:
        """Create summary table of all experiments.

        Returns:
            DataFrame with key metrics
        """
        if not self.all_results:
            self.load_all_results()

        data = []
        for result in self.all_results:
            data.append({
                'Experiment': result.get('experiment_id', 'unknown'),
                'Phase': self._extract_phase(result['experiment_id']),
                'Accuracy': result.get('accuracy', 0.0),
                'Class 0 Acc': result.get('class_0_accuracy', 0.0),
                'Class 1 Acc': result.get('class_1_accuracy', 0.0),
                'Pretrain Time (min)': result.get('pretrain_time_sec', 0) / 60,
                'Finetune Time (min)': result.get('finetune_time_sec', 0) / 60,
                'Total Time (min)': result.get('total_time_sec', 0) / 60,
                'Status': result.get('status', 'unknown'),
            })

        df = pd.DataFrame(data)
        df = df.sort_values(['Phase', 'Accuracy'], ascending=[True, False])
        return df

    def analyze_phase1_timewarp(self) -> pd.DataFrame:
        """Analyze Phase 1 time warping ablation.

        Returns:
            DataFrame with time warp analysis
        """
        phase1_results = [
            r for r in self.all_results
            if r['experiment_id'].startswith('exp_phase1')
        ]

        data = []
        for result in phase1_results:
            exp_id = result['experiment_id']
            # Extract sigma from experiment ID
            if '0.10' in exp_id:
                sigma = 0.10
            elif '0.12' in exp_id:
                sigma = 0.12
            elif '0.15' in exp_id:
                sigma = 0.15
            elif '0.20' in exp_id:
                sigma = 0.20
            else:
                sigma = None

            data.append({
                'Time Warp Sigma': sigma,
                'Accuracy': result.get('accuracy', 0.0),
                'Class 1 Accuracy': result.get('class_1_accuracy', 0.0),
                'Class Balance': self._compute_balance_score(
                    result.get('class_0_accuracy', 0.0),
                    result.get('class_1_accuracy', 0.0)
                ),
                'Total Time (min)': result.get('total_time_sec', 0) / 60,
            })

        df = pd.DataFrame(data)
        df = df.sort_values('Time Warp Sigma')
        return df

    def analyze_phase2_architecture(self) -> pd.DataFrame:
        """Analyze Phase 2 architecture search.

        Returns:
            DataFrame with architecture analysis
        """
        phase2_results = [
            r for r in self.all_results
            if r['experiment_id'].startswith('exp_phase2')
        ]

        data = []
        for result in phase2_results:
            exp_id = result['experiment_id']
            # Extract architecture from experiment ID
            if '_64_4' in exp_id:
                hidden, heads = 64, 4
            elif '_128_8' in exp_id:
                hidden, heads = 128, 8
            elif '_128_4' in exp_id:
                hidden, heads = 128, 4
            else:
                hidden, heads = None, None

            data.append({
                'Hidden Size': hidden,
                'Num Heads': heads,
                'Per-Head Dim': hidden / heads if heads else None,
                'Accuracy': result.get('accuracy', 0.0),
                'Class 1 Accuracy': result.get('class_1_accuracy', 0.0),
                'Total Time (min)': result.get('total_time_sec', 0) / 60,
            })

        df = pd.DataFrame(data)
        df = df.sort_values('Accuracy', ascending=False)
        return df

    def analyze_phase3_depth(self) -> pd.DataFrame:
        """Analyze Phase 3 depth search.

        Returns:
            DataFrame with depth analysis
        """
        phase3_results = [
            r for r in self.all_results
            if r['experiment_id'].startswith('exp_phase3')
        ]

        data = []
        for result in phase3_results:
            exp_id = result['experiment_id']
            # Extract epochs from experiment ID
            if '_50' in exp_id:
                epochs = 50
            elif '_75' in exp_id:
                epochs = 75
            elif '_100' in exp_id:
                epochs = 100
            else:
                epochs = None

            data.append({
                'Pretrain Epochs': epochs,
                'Accuracy': result.get('accuracy', 0.0),
                'Class 1 Accuracy': result.get('class_1_accuracy', 0.0),
                'Pretrain Time (min)': result.get('pretrain_time_sec', 0) / 60,
                'Total Time (min)': result.get('total_time_sec', 0) / 60,
                'Time per Epoch (min)': (result.get('pretrain_time_sec', 0) / 60) / epochs if epochs else None,
            })

        df = pd.DataFrame(data)
        df = df.sort_values('Pretrain Epochs')
        return df

    def generate_comparison_report(self) -> str:
        """Generate text comparison report.

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("LSTM OPTIMIZATION PHASE IV - RESULTS ANALYSIS")
        report.append("=" * 80)
        report.append("")

        # Overall summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 80)
        summary_df = self.create_summary_table()
        report.append(summary_df.to_string())
        report.append("")

        # Phase 1: Time Warping
        report.append("PHASE 1: TIME WARPING ABLATION")
        report.append("-" * 80)
        phase1_df = self.analyze_phase1_timewarp()
        report.append(phase1_df.to_string())
        report.append("")

        # Key finding
        if not phase1_df.empty:
            best_sigma = phase1_df.loc[phase1_df['Accuracy'].idxmax(), 'Time Warp Sigma']
            best_acc = phase1_df['Accuracy'].max()
            report.append(f"Best Time Warp Sigma: {best_sigma} (Accuracy: {best_acc:.4f})")
            report.append("")

        # Phase 2: Architecture
        report.append("PHASE 2: ARCHITECTURE SEARCH")
        report.append("-" * 80)
        phase2_df = self.analyze_phase2_architecture()
        report.append(phase2_df.to_string())
        report.append("")

        # Key finding
        if not phase2_df.empty:
            best_arch = phase2_df.iloc[0]
            report.append(f"Best Architecture: hidden={best_arch['Hidden Size']}, heads={best_arch['Num Heads']} "
                         f"(Accuracy: {best_arch['Accuracy']:.4f})")
            report.append("")

        # Phase 3: Depth
        report.append("PHASE 3: DEPTH SEARCH")
        report.append("-" * 80)
        phase3_df = self.analyze_phase3_depth()
        report.append(phase3_df.to_string())
        report.append("")

        # Key finding
        if not phase3_df.empty:
            best_depth = phase3_df.loc[phase3_df['Accuracy'].idxmax(), 'Pretrain Epochs']
            best_acc = phase3_df['Accuracy'].max()
            report.append(f"Best Pretrain Depth: {best_depth} epochs (Accuracy: {best_acc:.4f})")
            report.append("")

        # Overall best
        report.append("=" * 80)
        report.append("FINAL RECOMMENDATION")
        report.append("=" * 80)
        best_result = max(self.all_results, key=lambda r: r.get('accuracy', 0.0))
        report.append(f"Best Experiment: {best_result['experiment_id']}")
        report.append(f"  Accuracy: {best_result.get('accuracy', 0.0):.4f}")
        report.append(f"  Class 0 Accuracy: {best_result.get('class_0_accuracy', 0.0):.4f}")
        report.append(f"  Class 1 Accuracy: {best_result.get('class_1_accuracy', 0.0):.4f}")
        report.append(f"  Total Time: {best_result.get('total_time_sec', 0) / 60:.1f} minutes")
        report.append("=" * 80)

        return "\n".join(report)

    def _extract_phase(self, experiment_id: str) -> int:
        """Extract phase number from experiment ID."""
        if 'phase1' in experiment_id:
            return 1
        elif 'phase2' in experiment_id:
            return 2
        elif 'phase3' in experiment_id:
            return 3
        else:
            return 0

    def _compute_balance_score(self, class0_acc: float, class1_acc: float) -> float:
        """Compute class balance score (1.0 = perfect balance)."""
        if class0_acc == 0 or class1_acc == 0:
            return 0.0
        return min(class0_acc, class1_acc) / max(class0_acc, class1_acc)

    def save_report(self, output_path: Path):
        """Save analysis report to file.

        Args:
            output_path: Path to save report
        """
        report = self.generate_comparison_report()

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"[SAVED] Analysis report: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate and analyze LSTM optimization experiment results"
    )
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("data/artifacts"),
        help="Directory containing experiment results"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/artifacts/phase_iv_analysis.txt"),
        help="Output report path"
    )

    args = parser.parse_args()

    # Validate results directory
    if not args.results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")

    # Initialize aggregator
    aggregator = ResultsAggregator(args.results_dir)

    # Load results
    results = aggregator.load_all_results()

    if not results:
        print("[WARNING] No results found")
        return

    # Generate report
    report = aggregator.generate_comparison_report()
    print(report)

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    aggregator.save_report(args.output)


if __name__ == "__main__":
    main()
