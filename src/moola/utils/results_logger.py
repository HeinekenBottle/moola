"""Simple results logging for experiments. No database, just JSON files."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ResultsLogger:
    """Dead-simple experiment results logger. Write once, compare locally."""

    def __init__(self, log_file: str = "experiment_results.jsonl"):
        self.log_file = Path(log_file)

    def log(
        self,
        phase: int,
        experiment_id: str,
        metrics: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log experiment results to file (append mode).

        Args:
            phase: Phase number (1, 2, 3)
            experiment_id: Unique experiment identifier
            metrics: Dict of metric names → values (accuracy, loss, etc.)
            config: Optional hyperparameter config
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "experiment_id": experiment_id,
            "metrics": metrics,
            "config": config or {},
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"✓ Logged: {experiment_id} ({metrics.get('accuracy', 'N/A')})")

    def get_phase_results(self, phase: int) -> list:
        """Get all results for a phase."""
        if not self.log_file.exists():
            return []

        results = []
        with open(self.log_file) as f:
            for line in f:
                record = json.loads(line)
                if record["phase"] == phase:
                    results.append(record)
        return results

    def get_winner(self, phase: int, metric: str = "accuracy") -> Optional[Dict]:
        """Get best result for a phase by metric."""
        results = self.get_phase_results(phase)
        if not results:
            return None

        return max(results, key=lambda x: x["metrics"].get(metric, 0))

    def summary(self, phase: int) -> None:
        """Print summary of phase results."""
        results = self.get_phase_results(phase)
        if not results:
            print(f"No results for phase {phase}")
            return

        print(f"\n=== Phase {phase} Summary ===")
        for r in results:
            acc = r["metrics"].get("accuracy", "N/A")
            exp_id = r["experiment_id"]
            print(f"  {exp_id}: {acc}")

        winner = self.get_winner(phase)
        if winner:
            print(
                f"\nWinner: {winner['experiment_id']} "
                f"({winner['metrics'].get('accuracy', 'N/A')})"
            )


# Usage example:
if __name__ == "__main__":
    logger = ResultsLogger()

    # After training in Phase 1
    logger.log(
        phase=1,
        experiment_id="phase1_time_warp_0.12",
        metrics={"accuracy": 0.87, "class_1_accuracy": 0.62},
        config={"time_warp_sigma": 0.12},
    )

    # After training in Phase 2
    logger.log(
        phase=2,
        experiment_id="phase2_hidden64_heads4",
        metrics={"accuracy": 0.89, "class_1_accuracy": 0.65},
        config={"hidden_size": 64, "num_heads": 4},
    )

    # View results
    logger.summary(1)
    logger.summary(2)

    # Get winner for next phase
    winner = logger.get_winner(1)
    print(f"\nUse this config for Phase 2: {winner['config']}")
