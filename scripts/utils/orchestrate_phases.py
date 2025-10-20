#!/usr/bin/env python3
"""Phase-based Orchestration for LSTM Optimization Experiments.

Executes experiments in phases:
- Phase 1: 4 time warping experiments (parallel or sequential)
- Phase 2: 3 architecture experiments (depends on Phase 1 winner)
- Phase 3: 3 depth experiments (depends on Phase 2 winner)

Winner selection:
- Class 1 accuracy >= 30% (prevent class collapse)
- Highest overall accuracy among valid candidates

Usage:
    # Sequential execution (RTX 4090 safe)
    python orchestrate_phases.py --mode sequential

    # Parallel execution (requires multiple GPUs)
    python orchestrate_phases.py --mode parallel --num_workers 4

    # Run specific phase
    python orchestrate_phases.py --mode sequential --phase 1
"""

import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from experiment_configs import (
    PHASE_1_EXPERIMENTS,
    get_phase2_experiments,
    get_phase3_experiments,
    select_winner,
    DEFAULT_HARDWARE,
    ExperimentConfig
)


class PhaseOrchestrator:
    """Orchestrates multi-phase experiment execution."""

    def __init__(
        self,
        data_dir: Path,
        mode: str = "sequential",
        num_workers: int = 1,
        mlflow_tracking_uri: str = "./mlruns",
        mlflow_experiment_name: str = "LSTM_Optimization_Phase_IV"
    ):
        """Initialize orchestrator.

        Args:
            data_dir: Data directory containing train/test splits
            mode: 'sequential' or 'parallel'
            num_workers: Number of parallel workers (ignored if mode='sequential')
            mlflow_tracking_uri: MLflow tracking URI
            mlflow_experiment_name: MLflow experiment name
        """
        self.data_dir = data_dir
        self.mode = mode
        self.num_workers = 1 if mode == "sequential" else num_workers
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name

        # Results storage
        self.phase_results = {1: [], 2: [], 3: []}
        self.phase_winners = {}

    def run_single_experiment(self, config: ExperimentConfig) -> Dict:
        """Run a single experiment using subprocess.

        Args:
            config: Experiment configuration

        Returns:
            Experiment results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Starting: {config.experiment_id}")
        print(f"{'='*70}")

        # Build command
        cmd = [
            "python",
            str(Path(__file__).parent / "run_lstm_experiment.py"),
            "--experiment_id", config.experiment_id,
            "--data_dir", str(self.data_dir),
            "--mlflow_tracking_uri", self.mlflow_tracking_uri,
            "--mlflow_experiment_name", self.mlflow_experiment_name,
        ]

        # Execute
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
            print(result.stdout)

            # Load results
            results_path = Path(config.get_save_path()) / "results.json"
            if results_path.exists():
                with open(results_path, 'r') as f:
                    results = json.load(f)
                results['experiment_id'] = config.experiment_id
                results['status'] = 'success'
                return results
            else:
                raise FileNotFoundError(f"Results not found: {results_path}")

        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Experiment failed: {config.experiment_id}")
            print(e.stderr)
            return {
                'experiment_id': config.experiment_id,
                'status': 'failed',
                'error': str(e),
                'accuracy': 0.0,
                'class_1_accuracy': 0.0,
            }
        finally:
            elapsed = time.time() - start_time
            print(f"Completed: {config.experiment_id} ({elapsed/60:.1f} min)")

    def run_phase_sequential(self, experiments: List[ExperimentConfig]) -> List[Dict]:
        """Run experiments sequentially.

        Args:
            experiments: List of experiment configs

        Returns:
            List of result dictionaries
        """
        results = []
        for exp in experiments:
            result = self.run_single_experiment(exp)
            results.append(result)
        return results

    def run_phase_parallel(self, experiments: List[ExperimentConfig]) -> List[Dict]:
        """Run experiments in parallel.

        Args:
            experiments: List of experiment configs

        Returns:
            List of result dictionaries
        """
        print(f"\n[PARALLEL] Running {len(experiments)} experiments with {self.num_workers} workers")
        print(f"[WARNING] Ensure sufficient GPU memory for parallel execution!")

        results = []
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self.run_single_experiment, exp): exp
                for exp in experiments
            }

            for future in as_completed(futures):
                exp = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"[ERROR] {exp.experiment_id} raised exception: {e}")
                    results.append({
                        'experiment_id': exp.experiment_id,
                        'status': 'failed',
                        'error': str(e),
                        'accuracy': 0.0,
                        'class_1_accuracy': 0.0,
                    })

        return results

    def run_phase(self, phase: int, experiments: List[ExperimentConfig]) -> List[Dict]:
        """Run a single phase of experiments.

        Args:
            phase: Phase number (1, 2, or 3)
            experiments: List of experiment configs

        Returns:
            List of result dictionaries
        """
        print(f"\n{'#'*70}")
        print(f"# PHASE {phase}: {len(experiments)} EXPERIMENTS")
        print(f"{'#'*70}")

        for exp in experiments:
            print(f"  - {exp.experiment_id}: {exp.description}")
        print()

        # Estimate time
        estimated_time = DEFAULT_HARDWARE.estimate_time_minutes(
            len(experiments),
            parallel=(self.mode == "parallel")
        )
        print(f"Estimated time: {estimated_time} minutes ({estimated_time/60:.1f} hours)")
        print()

        # Run experiments
        phase_start = time.time()
        if self.mode == "sequential":
            results = self.run_phase_sequential(experiments)
        else:
            results = self.run_phase_parallel(experiments)
        phase_time = time.time() - phase_start

        print(f"\n{'='*70}")
        print(f"PHASE {phase} COMPLETE")
        print(f"{'='*70}")
        print(f"  Total time: {phase_time/60:.1f} minutes")
        print(f"  Successful: {sum(1 for r in results if r['status'] == 'success')}/{len(results)}")
        print(f"{'='*70}\n")

        return results

    def run_all_phases(self):
        """Execute all 3 phases with winner selection."""
        print(f"\n{'#'*70}")
        print(f"# LSTM OPTIMIZATION - FULL PHASE EXECUTION")
        print(f"# Mode: {self.mode.upper()}")
        print(f"# Workers: {self.num_workers}")
        print(f"{'#'*70}\n")

        # ====================================================================
        # PHASE 1: TIME WARPING ABLATION
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"PHASE 1: TIME WARPING ABLATION")
        print(f"{'='*70}")

        phase1_results = self.run_phase(1, PHASE_1_EXPERIMENTS)
        self.phase_results[1] = phase1_results

        # Select Phase 1 winner
        phase1_winner = select_winner(phase1_results)
        if phase1_winner is None:
            raise RuntimeError("Phase 1 failed - no valid winner found")

        self.phase_winners[1] = phase1_winner
        phase1_winner_config = next(
            e for e in PHASE_1_EXPERIMENTS
            if e.experiment_id == phase1_winner['experiment_id']
        )

        # ====================================================================
        # PHASE 2: ARCHITECTURE SEARCH
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"PHASE 2: ARCHITECTURE SEARCH")
        print(f"Using Phase 1 winner: time_warp_sigma={phase1_winner_config.time_warp_sigma}")
        print(f"{'='*70}")

        phase2_experiments = get_phase2_experiments(
            phase1_winner_sigma=phase1_winner_config.time_warp_sigma
        )
        phase2_results = self.run_phase(2, phase2_experiments)
        self.phase_results[2] = phase2_results

        # Select Phase 2 winner
        phase2_winner = select_winner(phase2_results)
        if phase2_winner is None:
            raise RuntimeError("Phase 2 failed - no valid winner found")

        self.phase_winners[2] = phase2_winner
        phase2_winner_config = next(
            e for e in phase2_experiments
            if e.experiment_id == phase2_winner['experiment_id']
        )

        # ====================================================================
        # PHASE 3: DEPTH SEARCH
        # ====================================================================
        print(f"\n{'='*70}")
        print(f"PHASE 3: DEPTH SEARCH")
        print(f"Using Phase 1 winner: time_warp_sigma={phase1_winner_config.time_warp_sigma}")
        print(f"Using Phase 2 winner: hidden_size={phase2_winner_config.hidden_size}, "
              f"num_heads={phase2_winner_config.num_heads}")
        print(f"{'='*70}")

        phase3_experiments = get_phase3_experiments(
            phase1_winner_sigma=phase1_winner_config.time_warp_sigma,
            phase2_winner_hidden=phase2_winner_config.hidden_size,
            phase2_winner_heads=phase2_winner_config.num_heads
        )
        phase3_results = self.run_phase(3, phase3_experiments)
        self.phase_results[3] = phase3_results

        # Select Phase 3 winner (FINAL WINNER)
        phase3_winner = select_winner(phase3_results)
        if phase3_winner is None:
            raise RuntimeError("Phase 3 failed - no valid winner found")

        self.phase_winners[3] = phase3_winner

        # ====================================================================
        # FINAL REPORT
        # ====================================================================
        self.generate_final_report()

    def run_specific_phase(self, phase: int):
        """Run a specific phase only (for testing/debugging).

        Args:
            phase: Phase number (1, 2, or 3)
        """
        print(f"\n{'='*70}")
        print(f"RUNNING PHASE {phase} ONLY")
        print(f"{'='*70}\n")

        if phase == 1:
            experiments = PHASE_1_EXPERIMENTS
        elif phase == 2:
            # Use default recommended config from Phase 1
            experiments = get_phase2_experiments(phase1_winner_sigma=0.12)
        elif phase == 3:
            # Use default recommended configs from Phase 1+2
            experiments = get_phase3_experiments(
                phase1_winner_sigma=0.12,
                phase2_winner_hidden=128,
                phase2_winner_heads=8
            )
        else:
            raise ValueError(f"Invalid phase: {phase}")

        results = self.run_phase(phase, experiments)
        winner = select_winner(results)

        if winner:
            print(f"\nPhase {phase} Winner: {winner['experiment_id']}")
            print(f"  Accuracy: {winner['accuracy']:.4f}")
            print(f"  Class 1 Accuracy: {winner['class_1_accuracy']:.4f}")

    def generate_final_report(self):
        """Generate final summary report."""
        print(f"\n{'#'*70}")
        print(f"# FINAL REPORT - LSTM OPTIMIZATION PHASE IV")
        print(f"{'#'*70}\n")

        # Phase summaries
        for phase in [1, 2, 3]:
            winner = self.phase_winners[phase]
            print(f"Phase {phase} Winner: {winner['experiment_id']}")
            print(f"  Accuracy: {winner['accuracy']:.4f}")
            print(f"  Class 0 Accuracy: {winner['class_0_accuracy']:.4f}")
            print(f"  Class 1 Accuracy: {winner['class_1_accuracy']:.4f}")
            print()

        # Final recommendation
        final_winner = self.phase_winners[3]
        print(f"{'='*70}")
        print(f"FINAL RECOMMENDATION")
        print(f"{'='*70}")
        print(f"  Best Config: {final_winner['experiment_id']}")
        print(f"  Overall Accuracy: {final_winner['accuracy']:.4f}")
        print(f"  Class 1 Accuracy: {final_winner['class_1_accuracy']:.4f}")

        # Compare to baseline
        if 'exp_phase1_timewarp_0.20' in [r['experiment_id'] for r in self.phase_results[1]]:
            baseline = next(
                r for r in self.phase_results[1]
                if r['experiment_id'] == 'exp_phase1_timewarp_0.20'
            )
            improvement = (final_winner['accuracy'] - baseline['accuracy']) * 100
            class1_improvement = (final_winner['class_1_accuracy'] - baseline['class_1_accuracy']) * 100
            print(f"  Improvement vs Baseline:")
            print(f"    - Accuracy: +{improvement:.1f}%")
            print(f"    - Class 1 Accuracy: +{class1_improvement:.1f}%")

        print(f"{'='*70}\n")

        # Save report
        report_path = Path("data/artifacts/phase_iv_final_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            'phase_winners': self.phase_winners,
            'phase_results': self.phase_results,
            'final_recommendation': final_winner,
        }

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"[SAVED] Final report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate LSTM optimization experiments across phases"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sequential", "parallel"],
        default="sequential",
        help="Execution mode (sequential is safer for single GPU)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers (ignored if mode=sequential)"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Run specific phase only (default: run all phases)"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Data directory containing train/test splits"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default="LSTM_Optimization_Phase_IV",
        help="MLflow experiment name"
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}\n"
            f"Run data pipeline first to generate train/test splits"
        )

    # Initialize orchestrator
    orchestrator = PhaseOrchestrator(
        data_dir=args.data_dir,
        mode=args.mode,
        num_workers=args.num_workers,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment_name=args.mlflow_experiment_name
    )

    # Run phases
    if args.phase is not None:
        orchestrator.run_specific_phase(args.phase)
    else:
        orchestrator.run_all_phases()


if __name__ == "__main__":
    main()
