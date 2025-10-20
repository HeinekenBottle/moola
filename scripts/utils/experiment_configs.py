"""Experiment Configurations for LSTM Optimization Phase IV.

Defines all 13 experiments across 4 phases based on LSTM_OPTIMIZATION_ANALYSIS_PHASE_IV.md.

Usage:
    from experiment_configs import PHASE_1_EXPERIMENTS, PHASE_2_EXPERIMENTS
    for exp in PHASE_1_EXPERIMENTS:
        run_experiment(exp)
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class ExperimentConfig:
    """Single experiment configuration."""

    # Experiment metadata
    experiment_id: str
    phase: int
    description: str
    depends_on: Optional[str] = None  # Parent experiment ID (for phase sequencing)

    # Time warping (Phase 1 primary variable)
    time_warp_sigma: float = 0.12
    time_warp_prob: float = 0.3

    # Architecture (Phase 2 primary variable)
    hidden_size: int = 128
    num_heads: int = 8

    # Pre-training depth (Phase 3 primary variable)
    pretrain_epochs: int = 50

    # Fine-tuning config (consistent across experiments)
    finetune_epochs: int = 50
    finetune_freeze_epochs: int = 10
    finetune_lr: float = 5e-4

    # Other augmentation (consistent)
    jitter_prob: float = 0.5
    jitter_sigma: float = 0.05
    scaling_prob: float = 0.3
    scaling_sigma: float = 0.1

    # Expected performance (for validation)
    expected_accuracy_min: float = 0.60
    expected_accuracy_max: float = 0.70
    expected_class1_min: float = 0.30
    expected_class1_max: float = 0.60

    # Hardware
    device: str = "cuda"
    batch_size: int = 512

    def to_dict(self) -> Dict:
        """Convert to dictionary for MLflow logging."""
        return asdict(self)

    def get_save_path(self, base_dir: str = "data/artifacts") -> str:
        """Get path for saving artifacts."""
        return f"{base_dir}/{self.experiment_id}"

    def get_encoder_path(self, base_dir: str = "data/artifacts") -> str:
        """Get path for pre-trained encoder."""
        return f"{base_dir}/pretrained/{self.experiment_id}_encoder.pt"


# ============================================================================
# PHASE 1: TIME WARPING ABLATION (4 experiments)
# ============================================================================

PHASE_1_EXPERIMENTS = [
    ExperimentConfig(
        experiment_id="exp_phase1_timewarp_0.10",
        phase=1,
        description="Conservative time warping (10%) - pattern-preserving baseline",
        time_warp_sigma=0.10,
        expected_accuracy_min=0.63,
        expected_accuracy_max=0.65,
        expected_class1_min=0.35,
        expected_class1_max=0.40,
    ),
    ExperimentConfig(
        experiment_id="exp_phase1_timewarp_0.12",
        phase=1,
        description="Goldilocks time warping (12%) - RECOMMENDED for masked reconstruction",
        time_warp_sigma=0.12,
        expected_accuracy_min=0.65,
        expected_accuracy_max=0.69,
        expected_class1_min=0.45,
        expected_class1_max=0.55,
    ),
    ExperimentConfig(
        experiment_id="exp_phase1_timewarp_0.15",
        phase=1,
        description="Moderate time warping (15%) - documented in roadmap",
        time_warp_sigma=0.15,
        expected_accuracy_min=0.64,
        expected_accuracy_max=0.68,
        expected_class1_min=0.40,
        expected_class1_max=0.50,
    ),
    ExperimentConfig(
        experiment_id="exp_phase1_timewarp_0.20",
        phase=1,
        description="Aggressive time warping (20%) - CURRENT BASELINE (expected to fail)",
        time_warp_sigma=0.20,
        expected_accuracy_min=0.60,
        expected_accuracy_max=0.63,
        expected_class1_min=0.15,
        expected_class1_max=0.25,
    ),
]


# ============================================================================
# PHASE 2: ARCHITECTURE SEARCH (3 experiments)
# Depends on Phase 1 winner (to be determined dynamically)
# ============================================================================

def get_phase2_experiments(phase1_winner_sigma: float = 0.12) -> List[ExperimentConfig]:
    """Generate Phase 2 experiments based on Phase 1 winner.

    Args:
        phase1_winner_sigma: Best time_warp_sigma from Phase 1

    Returns:
        List of 3 architecture search experiments
    """
    return [
        ExperimentConfig(
            experiment_id="exp_phase2_arch_64_4",
            phase=2,
            description="Smaller architecture (64 hidden, 4 heads) - fast baseline",
            depends_on="phase1_winner",
            time_warp_sigma=phase1_winner_sigma,
            hidden_size=64,
            num_heads=4,
            expected_accuracy_min=0.62,
            expected_accuracy_max=0.66,
            expected_class1_min=0.40,
            expected_class1_max=0.50,
        ),
        ExperimentConfig(
            experiment_id="exp_phase2_arch_128_8",
            phase=2,
            description="RECOMMENDED architecture (128 hidden, 8 heads) - matches encoder",
            depends_on="phase1_winner",
            time_warp_sigma=phase1_winner_sigma,
            hidden_size=128,
            num_heads=8,
            expected_accuracy_min=0.66,
            expected_accuracy_max=0.70,
            expected_class1_min=0.45,
            expected_class1_max=0.55,
        ),
        ExperimentConfig(
            experiment_id="exp_phase2_arch_128_4",
            phase=2,
            description="Hybrid architecture (128 hidden, 4 heads) - deeper per-head attention",
            depends_on="phase1_winner",
            time_warp_sigma=phase1_winner_sigma,
            hidden_size=128,
            num_heads=4,
            expected_accuracy_min=0.64,
            expected_accuracy_max=0.68,
            expected_class1_min=0.42,
            expected_class1_max=0.52,
        ),
    ]


# ============================================================================
# PHASE 3: DEPTH SEARCH (3 experiments)
# Depends on Phase 2 winner (to be determined dynamically)
# ============================================================================

def get_phase3_experiments(
    phase1_winner_sigma: float = 0.12,
    phase2_winner_hidden: int = 128,
    phase2_winner_heads: int = 8
) -> List[ExperimentConfig]:
    """Generate Phase 3 experiments based on Phase 1+2 winners.

    Args:
        phase1_winner_sigma: Best time_warp_sigma from Phase 1
        phase2_winner_hidden: Best hidden_size from Phase 2
        phase2_winner_heads: Best num_heads from Phase 2

    Returns:
        List of 3 depth search experiments
    """
    return [
        ExperimentConfig(
            experiment_id="exp_phase3_depth_50",
            phase=3,
            description="Standard pre-training (50 epochs) - CURRENT baseline",
            depends_on="phase2_winner",
            time_warp_sigma=phase1_winner_sigma,
            hidden_size=phase2_winner_hidden,
            num_heads=phase2_winner_heads,
            pretrain_epochs=50,
            expected_accuracy_min=0.65,
            expected_accuracy_max=0.69,
            expected_class1_min=0.45,
            expected_class1_max=0.55,
        ),
        ExperimentConfig(
            experiment_id="exp_phase3_depth_75",
            phase=3,
            description="RECOMMENDED pre-training (75 epochs) - deeper convergence",
            depends_on="phase2_winner",
            time_warp_sigma=phase1_winner_sigma,
            hidden_size=phase2_winner_hidden,
            num_heads=phase2_winner_heads,
            pretrain_epochs=75,
            expected_accuracy_min=0.67,
            expected_accuracy_max=0.72,
            expected_class1_min=0.48,
            expected_class1_max=0.58,
        ),
        ExperimentConfig(
            experiment_id="exp_phase3_depth_100",
            phase=3,
            description="Deep pre-training (100 epochs) - maximum convergence test",
            depends_on="phase2_winner",
            time_warp_sigma=phase1_winner_sigma,
            hidden_size=phase2_winner_hidden,
            num_heads=phase2_winner_heads,
            pretrain_epochs=100,
            expected_accuracy_min=0.66,
            expected_accuracy_max=0.71,
            expected_class1_min=0.47,
            expected_class1_max=0.57,
        ),
    ]


# ============================================================================
# WINNER SELECTION CRITERIA
# ============================================================================

def select_winner(
    results: List[Dict],
    min_class1_accuracy: float = 0.30
) -> Optional[Dict]:
    """Select winning experiment based on criteria.

    Selection criteria (in order):
    1. Class 1 accuracy >= min_class1_accuracy (prevent class collapse)
    2. Highest overall accuracy among valid candidates

    Args:
        results: List of experiment result dictionaries with keys:
                 'experiment_id', 'accuracy', 'class_0_accuracy', 'class_1_accuracy'
        min_class1_accuracy: Minimum acceptable Class 1 accuracy (default: 30%)

    Returns:
        Winning experiment result dict, or None if no valid candidates
    """
    # Filter valid candidates (no class collapse)
    valid_results = [
        r for r in results
        if r.get('class_1_accuracy', 0.0) >= min_class1_accuracy
    ]

    if not valid_results:
        print(f"[WARNING] No experiments met min_class1_accuracy={min_class1_accuracy}")
        return None

    # Select highest accuracy among valid candidates
    winner = max(valid_results, key=lambda r: r.get('accuracy', 0.0))

    print(f"\n{'='*70}")
    print(f"WINNER SELECTED: {winner['experiment_id']}")
    print(f"{'='*70}")
    print(f"  Overall Accuracy: {winner['accuracy']:.4f}")
    print(f"  Class 0 Accuracy: {winner['class_0_accuracy']:.4f}")
    print(f"  Class 1 Accuracy: {winner['class_1_accuracy']:.4f}")
    print(f"{'='*70}\n")

    return winner


# ============================================================================
# HARDWARE RESOURCE MANAGEMENT
# ============================================================================

@dataclass
class HardwareConfig:
    """Hardware configuration for experiment execution."""

    device_name: str = "RTX 4090"
    vram_gb: int = 24
    parallel_jobs: int = 1  # 1 = sequential, >1 = parallel (NOT recommended for RTX 4090)
    expected_pretrain_time_min: int = 30
    expected_finetune_time_min: int = 3

    def can_run_parallel(self, num_jobs: int) -> bool:
        """Check if hardware can support parallel jobs."""
        if self.vram_gb < 24:
            return False
        if num_jobs > 1:
            print(f"[WARNING] Parallel execution on {self.device_name} may cause OOM!")
            print(f"          Recommended: Sequential execution (parallel_jobs=1)")
            return False
        return True

    def estimate_time_minutes(self, num_experiments: int, parallel: bool = False) -> int:
        """Estimate total execution time."""
        single_exp_time = self.expected_pretrain_time_min + self.expected_finetune_time_min
        if parallel and self.can_run_parallel(num_experiments):
            return single_exp_time
        else:
            return single_exp_time * num_experiments


# Default hardware config for RTX 4090
DEFAULT_HARDWARE = HardwareConfig()


# ============================================================================
# EXPERIMENT REGISTRY
# ============================================================================

def get_all_experiments_sequential() -> List[ExperimentConfig]:
    """Get all 13 experiments in sequential order (for preview/validation).

    Note: Phase 2 and 3 configs use default winner values.
    For actual execution, use phase-based orchestration.
    """
    phase1 = PHASE_1_EXPERIMENTS
    phase2 = get_phase2_experiments(phase1_winner_sigma=0.12)  # Default to recommended
    phase3 = get_phase3_experiments(
        phase1_winner_sigma=0.12,
        phase2_winner_hidden=128,
        phase2_winner_heads=8
    )
    return phase1 + phase2 + phase3


def print_experiment_summary():
    """Print summary of all experiments."""
    all_exps = get_all_experiments_sequential()

    print(f"\n{'='*70}")
    print(f"LSTM OPTIMIZATION EXPERIMENT MATRIX (13 experiments)")
    print(f"{'='*70}\n")

    for phase in [1, 2, 3]:
        phase_exps = [e for e in all_exps if e.phase == phase]
        print(f"PHASE {phase}: {len(phase_exps)} experiments")
        print(f"-" * 70)
        for exp in phase_exps:
            print(f"  {exp.experiment_id}")
            print(f"    {exp.description}")
            if phase == 1:
                print(f"    time_warp_sigma={exp.time_warp_sigma}")
            elif phase == 2:
                print(f"    hidden_size={exp.hidden_size}, num_heads={exp.num_heads}")
            elif phase == 3:
                print(f"    pretrain_epochs={exp.pretrain_epochs}")
            print()

    total_time = DEFAULT_HARDWARE.estimate_time_minutes(len(all_exps), parallel=False)
    print(f"{'='*70}")
    print(f"Estimated total time (sequential): {total_time} minutes ({total_time/60:.1f} hours)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    print_experiment_summary()
