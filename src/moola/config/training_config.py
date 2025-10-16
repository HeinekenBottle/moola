"""Centralized training hyperparameters for reproducibility.

All magic numbers from model files and training scripts are defined here to
ensure consistency across train/eval/oof/predict pipelines.

Philosophy:
- Single source of truth for all hyperparameters
- No scattered magic numbers in model files
- Easy to experiment with (edit one file instead of searching code)
- Enables reproducibility through version control
"""

# ============================================================================
# RANDOM SEED MANAGEMENT
# ============================================================================

DEFAULT_SEED = 1337
"""Default random seed for reproducibility across all RNGs."""

SEED_REPRODUCIBLE = True
"""Enforce deterministic CUDA operations (may reduce performance slightly)."""


# ============================================================================
# DEEP LEARNING - GENERAL SETTINGS
# ============================================================================

DEFAULT_DEVICE = "cpu"
"""Default device: 'cpu' or 'cuda'. Models will auto-fallback to CPU if CUDA unavailable."""

DEFAULT_BATCH_SIZE = 512
"""Default batch size for training. Adjust based on available VRAM."""

DEFAULT_NUM_WORKERS = 16
"""Number of DataLoader worker threads. Set to 0 for debugging, 4-16 for production."""

DEFAULT_PIN_MEMORY = True
"""Pin memory for faster GPU transfer. Safe for most systems."""

USE_AMP = True
"""Automatic mixed precision (FP16) for faster training. Requires CUDA."""

DEFAULT_EARLY_STOPPING_PATIENCE = 20
"""Epochs to wait before stopping if validation loss doesn't improve."""


# ============================================================================
# CNN-TRANSFORMER SPECIFIC HYPERPARAMETERS
# ============================================================================

# Architecture
CNNTR_CHANNELS = [64, 128, 128]
"""CNN output channels at each layer. Increased depth for better feature extraction."""

CNNTR_KERNELS = [3, 5, 9]
"""CNN kernel sizes for multi-scale feature extraction. Kernel 9 captures longer trends."""

CNNTR_TRANSFORMER_LAYERS = 3
"""Number of Transformer encoder layers for global context modeling."""

CNNTR_TRANSFORMER_HEADS = 4
"""Number of attention heads. Should divide d_model evenly."""

CNNTR_DROPOUT = 0.25
"""Dropout rate: 0.25 is balanced regularization for small dataset (~100 samples)."""

# Training
CNNTR_N_EPOCHS = 80
"""Maximum epochs. Increased to 80 for SSL transfer learning (needs 50+ epochs to converge)."""

CNNTR_LEARNING_RATE = 5e-4
"""Learning rate: 5e-4 is stable for small datasets. Higher rates (1e-3) caused gradient explosion."""

CNNTR_EARLY_STOPPING_PATIENCE = 30
"""Patience for early stopping. Increased to 30 for SSL transfer learning."""

CNNTR_VAL_SPLIT = 0.15
"""Validation split for early stopping. 15% ≈ 15 samples for ~100 sample dataset."""

# SSL Transfer Learning
CNNTR_FREEZE_EPOCHS = 10
"""Number of epochs to keep encoder frozen when using pre-trained weights."""

CNNTR_GRADUAL_UNFREEZE = True
"""Enable gradual unfreezing schedule for pre-trained encoder."""

CNNTR_UNFREEZE_SCHEDULE = {
    "stage1_epoch": 10,  # Unfreeze last transformer layer
    "stage2_epoch": 20,  # Unfreeze all transformer layers
    "stage3_epoch": 30,  # Unfreeze CNN blocks (full fine-tuning)
}
"""Gradual unfreezing schedule: epoch -> stage mapping."""

# Data augmentation
CNNTR_MIXUP_ALPHA = 0.4
"""Mixup interpolation strength. Higher = more aggressive mixing."""

CNNTR_CUTMIX_PROB = 0.5
"""Probability of CutMix vs Mixup augmentation."""

# Multi-task learning (Phase 3)
CNNTR_MULTI_TASK_ENABLED = False
"""Enable pointer start/end prediction alongside classification."""

CNNTR_LOSS_ALPHA_CLASSIFICATION = 1.0
"""Loss weight for classification task. Set to 1.0 to prioritize classification."""

CNNTR_LOSS_BETA_POINTER = 0.0
"""Loss weight for each pointer task. Set to 0.0 to disable pointer tasks initially."""

CNNTR_LOSS_PROGRESSIVE_WEIGHTING = True
"""Gradually increase pointer task weight during training (helps with small datasets)."""


# ============================================================================
# RWKV-TS SPECIFIC HYPERPARAMETERS
# ============================================================================

RWKV_N_EPOCHS = 50
"""Maximum epochs for RWKV-TS training."""

RWKV_LEARNING_RATE = 1e-3
"""Learning rate for RWKV-TS. Higher than CNN-Transformer."""

RWKV_BATCH_SIZE = 256
"""Batch size for RWKV-TS (uses less VRAM than CNN-Transformer)."""


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

FOCAL_LOSS_GAMMA = 2.0
"""Focal loss gamma parameter for handling class imbalance."""

FOCAL_LOSS_ALPHA = None
"""Focal loss alpha (class weights). None = no weighting. Set to inverse class freq if needed."""


# ============================================================================
# DATA PREPROCESSING
# ============================================================================

WINDOW_SIZE = 105
"""Total window size: 30 bars before + 45 prediction window + 30 bars after."""

INNER_WINDOW_START = 30
"""Start index of prediction window within 105-bar window."""

INNER_WINDOW_END = 75
"""End index of prediction window (exclusive). Window = [30:75] = 45 bars."""

INNER_WINDOW_SIZE = 45
"""Size of prediction window (end - start)."""

OHLC_DIMS = 4
"""Features per timestep: Open, High, Low, Close."""

# Data validation
EXPECTED_FEATURES_PER_WINDOW = OHLC_DIMS
EXPECTED_WINDOW_LENGTH = WINDOW_SIZE


# ============================================================================
# TEMPORAL AUGMENTATION
# ============================================================================

TEMPORAL_AUG_JITTER_PROB = 0.5
"""Probability of applying jitter noise to features."""

TEMPORAL_AUG_JITTER_SIGMA = 0.05
"""Jitter magnitude: 5% of feature range."""

TEMPORAL_AUG_SCALING_PROB = 0.3
"""Probability of scaling features (magnitude variation)."""

TEMPORAL_AUG_SCALING_SIGMA = 0.1
"""Scaling magnitude: 10% variation in amplitudes."""

TEMPORAL_AUG_TIME_WARP_PROB = 0.3
"""Probability of time warping (temporal distortion)."""

TEMPORAL_AUG_TIME_WARP_SIGMA = 0.2
"""Time warp magnitude: 20% temporal distortion."""


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

DEFAULT_CV_FOLDS = 5
"""Number of folds for K-fold cross-validation. Use 5 for standard ML."""

STRATIFIED_SPLIT = True
"""Ensure class balance preserved in train/val splits."""


# ============================================================================
# SMOTE FOR DATA AUGMENTATION
# ============================================================================

SMOTE_TARGET_COUNT = 150
"""Target samples per class after SMOTE augmentation."""

SMOTE_K_NEIGHBORS = 5
"""Number of nearest neighbors for SMOTE. Increase for smoother synthetic samples."""


# ============================================================================
# EXPORT ALL CONSTANTS
# ============================================================================

__all__ = [
    # Seed management
    "DEFAULT_SEED",
    "SEED_REPRODUCIBLE",
    # General DL settings
    "DEFAULT_DEVICE",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_NUM_WORKERS",
    "DEFAULT_PIN_MEMORY",
    "USE_AMP",
    "DEFAULT_EARLY_STOPPING_PATIENCE",
    # CNN-Transformer
    "CNNTR_CHANNELS",
    "CNNTR_KERNELS",
    "CNNTR_TRANSFORMER_LAYERS",
    "CNNTR_TRANSFORMER_HEADS",
    "CNNTR_DROPOUT",
    "CNNTR_N_EPOCHS",
    "CNNTR_LEARNING_RATE",
    "CNNTR_EARLY_STOPPING_PATIENCE",
    "CNNTR_VAL_SPLIT",
    "CNNTR_MIXUP_ALPHA",
    "CNNTR_CUTMIX_PROB",
    "CNNTR_MULTI_TASK_ENABLED",
    "CNNTR_LOSS_ALPHA_CLASSIFICATION",
    "CNNTR_LOSS_BETA_POINTER",
    "CNNTR_LOSS_PROGRESSIVE_WEIGHTING",
    "CNNTR_FREEZE_EPOCHS",
    "CNNTR_GRADUAL_UNFREEZE",
    "CNNTR_UNFREEZE_SCHEDULE",
    # RWKV-TS
    "RWKV_N_EPOCHS",
    "RWKV_LEARNING_RATE",
    "RWKV_BATCH_SIZE",
    # Loss functions
    "FOCAL_LOSS_GAMMA",
    "FOCAL_LOSS_ALPHA",
    # Data preprocessing
    "WINDOW_SIZE",
    "INNER_WINDOW_START",
    "INNER_WINDOW_END",
    "INNER_WINDOW_SIZE",
    "OHLC_DIMS",
    "EXPECTED_FEATURES_PER_WINDOW",
    "EXPECTED_WINDOW_LENGTH",
    # Temporal augmentation
    "TEMPORAL_AUG_JITTER_PROB",
    "TEMPORAL_AUG_JITTER_SIGMA",
    "TEMPORAL_AUG_SCALING_PROB",
    "TEMPORAL_AUG_SCALING_SIGMA",
    "TEMPORAL_AUG_TIME_WARP_PROB",
    "TEMPORAL_AUG_TIME_WARP_SIGMA",
    # Cross-validation
    "DEFAULT_CV_FOLDS",
    "STRATIFIED_SPLIT",
    # SMOTE
    "SMOTE_TARGET_COUNT",
    "SMOTE_K_NEIGHBORS",
]
