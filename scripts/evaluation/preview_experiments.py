#!/usr/bin/env python3
"""Preview Experiment Matrix without Running.

Shows all experiments, expected performance, and time estimates.

Usage:
    python scripts/preview_experiments.py
    python scripts/preview_experiments.py --phase 1
"""

import argparse
from pathlib import Path

from experiment_configs import (
    PHASE_1_EXPERIMENTS,
    get_phase2_experiments,
    get_phase3_experiments,
    DEFAULT_HARDWARE,
    print_experiment_summary
)


def preview_phase(phase: int):
    """Preview specific phase experiments."""
    if phase == 1:
        experiments = PHASE_1_EXPERIMENTS
        title = "PHASE 1: TIME WARPING ABLATION"
    elif phase == 2:
        experiments = get_phase2_experiments(phase1_winner_sigma=0.12)
        title = "PHASE 2: ARCHITECTURE SEARCH (using recommended sigma=0.12)"
    elif phase == 3:
        experiments = get_phase3_experiments(
            phase1_winner_sigma=0.12,
            phase2_winner_hidden=128,
            phase2_winner_heads=8
        )
        title = "PHASE 3: DEPTH SEARCH (using recommended configs)"
    else:
        raise ValueError(f"Invalid phase: {phase}")

    print(f"\n{'='*70}")
    print(title)
    print(f"{'='*70}\n")

    for i, exp in enumerate(experiments, 1):
        print(f"{i}. {exp.experiment_id}")
        print(f"   Description: {exp.description}")
        print(f"   Expected Accuracy: {exp.expected_accuracy_min:.1%} - {exp.expected_accuracy_max:.1%}")
        print(f"   Expected Class 1: {exp.expected_class1_min:.1%} - {exp.expected_class1_max:.1%}")

        # Show key parameters
        if phase == 1:
            print(f"   time_warp_sigma: {exp.time_warp_sigma}")
        elif phase == 2:
            print(f"   hidden_size: {exp.hidden_size}, num_heads: {exp.num_heads}")
        elif phase == 3:
            print(f"   pretrain_epochs: {exp.pretrain_epochs}")

        print()

    # Time estimate
    estimated_time = DEFAULT_HARDWARE.estimate_time_minutes(len(experiments), parallel=False)
    print(f"{'='*70}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Estimated time (sequential): {estimated_time} minutes ({estimated_time/60:.1f} hours)")
    print(f"{'='*70}\n")


def preview_all():
    """Preview all phases."""
    print_experiment_summary()

    print("\n" + "="*70)
    print("WINNER SELECTION CRITERIA")
    print("="*70)
    print("1. Class 1 accuracy >= 30% (prevent class collapse)")
    print("2. Highest overall accuracy among valid candidates")
    print()
    print("Why Class 1 matters:")
    print("  - Class collapse (predicting only Class 0) = failure")
    print("  - 30% threshold ensures model learned retracement patterns")
    print("  - Overall accuracy alone can be misleading")
    print()

    print("="*70)
    print("EXPECTED IMPROVEMENTS")
    print("="*70)
    print("Conservative:")
    print("  Baseline:   60-63% accuracy, 15-25% Class 1")
    print("  Phase IV:   64-67% accuracy, 40-50% Class 1")
    print("  Gain:       +4-7% accuracy, +25-35% Class 1")
    print()
    print("Optimistic:")
    print("  Baseline:   57% accuracy, 0% Class 1 (collapsed)")
    print("  Phase IV:   68-72% accuracy, 48-58% Class 1")
    print("  Gain:       +11-15% accuracy, +48-58% Class 1")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preview experiment matrix without running"
    )
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Preview specific phase only (default: all phases)"
    )

    args = parser.parse_args()

    if args.phase is not None:
        preview_phase(args.phase)
    else:
        preview_all()


if __name__ == "__main__":
    main()
