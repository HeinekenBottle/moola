#!/usr/bin/env python3
"""Master Orchestrator for Gated Training Workflow.

Executes all gates sequentially with strict failure handling.

Gates:
0. Environment verification
1. Smoke test (EnhancedSimpleLSTM baseline)
2. Control test (MiniRocket)
3. Pretrain BiLSTM encoder
4. Finetune EnhancedSimpleLSTM
5. Train with augmentation
6. SimpleLSTM baseline
7. Ensemble

On any gate failure: STOP immediately, log failure reason.

Usage:
    python run_all.py [--start-gate N] [--end-gate N]

Exit codes:
- 0: All gates passed
- 1: At least one gate failed
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def log_result(message: str, status: str = "INFO"):
    """Log with timestamp and status."""
    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] [{status}] {message}")


def run_gate(gate_num: int, gate_script: Path) -> bool:
    """Run a single gate script.

    Args:
        gate_num: Gate number (0-7)
        gate_script: Path to gate script

    Returns:
        True if gate passed, False if gate failed
    """
    log_result("=" * 80)
    log_result(f"EXECUTING GATE {gate_num}: {gate_script.name}")
    log_result("=" * 80)

    start_time = datetime.now()

    try:
        result = subprocess.run(
            ["python3", str(gate_script)],
            cwd="/workspace/moola",
            capture_output=False,  # Print to stdout/stderr
            text=True,
        )

        elapsed = (datetime.now() - start_time).total_seconds()

        if result.returncode == 0:
            log_result(
                f"✓ GATE {gate_num} PASSED in {elapsed:.1f}s",
                "SUCCESS"
            )
            return True
        else:
            log_result(
                f"✗ GATE {gate_num} FAILED with exit code {result.returncode}",
                "ERROR"
            )
            return False

    except Exception as e:
        elapsed = (datetime.now() - start_time).total_seconds()
        log_result(
            f"✗ GATE {gate_num} EXCEPTION after {elapsed:.1f}s: {e}",
            "ERROR"
        )
        return False


def main():
    """Run gated workflow."""
    parser = argparse.ArgumentParser(description="Run gated training workflow")
    parser.add_argument(
        "--start-gate",
        type=int,
        default=0,
        help="Start from gate N (0-7, default: 0)"
    )
    parser.add_argument(
        "--end-gate",
        type=int,
        default=7,
        help="End at gate N (0-7, default: 7)"
    )
    args = parser.parse_args()

    log_result("=" * 80)
    log_result("GATED TRAINING WORKFLOW - MASTER ORCHESTRATOR")
    log_result("=" * 80)
    log_result(f"Start gate: {args.start_gate}")
    log_result(f"End gate: {args.end_gate}")
    log_result("=" * 80)

    workflow_start = datetime.now()

    # Define gates
    script_dir = Path(__file__).parent
    gates = [
        (0, script_dir / "0_verify_env.py"),
        (1, script_dir / "1_smoke_enhanced.py"),
        (2, script_dir / "2_control_minirocket.py"),
        (3, script_dir / "3_pretrain_bilstm.py"),
        (4, script_dir / "4_finetune_enhanced.py"),
        (5, script_dir / "5_augment_train.py"),
        (6, script_dir / "6_baseline_simplelstm.py"),
        (7, script_dir / "7_ensemble.py"),
    ]

    # Filter gates based on start/end
    gates_to_run = [
        (num, script) for num, script in gates
        if args.start_gate <= num <= args.end_gate
    ]

    if not gates_to_run:
        log_result("No gates to run with current start/end range", "ERROR")
        sys.exit(1)

    log_result(f"Running {len(gates_to_run)} gates: {[g[0] for g in gates_to_run]}")

    # Run gates sequentially
    passed_gates = []
    failed_gate = None

    for gate_num, gate_script in gates_to_run:
        if not gate_script.exists():
            log_result(f"✗ Gate script not found: {gate_script}", "ERROR")
            failed_gate = gate_num
            break

        # Execute gate
        if run_gate(gate_num, gate_script):
            passed_gates.append(gate_num)
        else:
            # GATE FAILURE - STOP IMMEDIATELY
            failed_gate = gate_num
            log_result("=" * 80)
            log_result(f"WORKFLOW ABORTED: Gate {gate_num} failed", "ERROR")
            log_result("=" * 80)
            break

    # Final summary
    workflow_elapsed = (datetime.now() - workflow_start).total_seconds()

    log_result("=" * 80)
    log_result("WORKFLOW SUMMARY")
    log_result("=" * 80)
    log_result(f"Total time: {workflow_elapsed:.1f}s ({workflow_elapsed/60:.1f} min)")
    log_result(f"Passed gates: {passed_gates}")

    if failed_gate is not None:
        log_result(f"Failed gate: {failed_gate}", "ERROR")
        log_result("=" * 80)
        log_result("WORKFLOW FAILED", "ERROR")
        log_result("=" * 80)
        log_result("Review logs and fix errors before re-running.", "ERROR")
        log_result("You can resume from failed gate using --start-gate flag.", "ERROR")
        log_result("=" * 80)
        sys.exit(1)
    else:
        log_result("=" * 80)
        log_result("ALL GATES PASSED - WORKFLOW COMPLETE", "SUCCESS")
        log_result("=" * 80)
        log_result("Results saved to: /workspace/moola/gated_workflow_results.jsonl")
        log_result("=" * 80)
        sys.exit(0)


if __name__ == "__main__":
    main()
