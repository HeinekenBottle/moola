"""Data pipeline utilities for integrating raw OHLC and engineered features.

This module provides utilities for:
- Dual-input data processing (raw OHLC + engineered features)
- Feature extraction and integration
- Proper alignment between OHLC sequences and feature vectors
- Backward compatibility with existing models
- Efficient processing for small datasets

Key Features:
- Seamless integration of multiple feature types
- Configurable feature selection
- Proper handling of expansion indices
- Backward compatibility with existing data pipeline
- Error handling and validation
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any, Sequence
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

from ..features.small_dataset_features import (
    SmallDatasetFeatureEngineer,
    create_small_dataset_feature_engineer,
    extract_optimized_features
)
from ..features.price_action_features import (
    engineer_multiscale_features,
    extract_hopsketch_features
)
from ..features.relative_transform import RelativeFeatureTransform
from ..utils.pseudo_sample_generation import (
    PseudoSampleGenerationPipeline,
    TemporalAugmentationGenerator,
    PatternBasedSynthesisGenerator
)


@dataclass
class FeatureConfig:
    """Configuration for dual-input feature processing."""

    # Raw OHLC features
    use_raw_ohlc: bool = True  # Always include raw OHLC (105×4)
    raw_ohlc_scaling: bool = False  # Keep raw OHLC unnormalized for LSTM models
    
    # 11D Relative Features
    use_relative_features: bool = False  # Enable 11D relative features
    relative_features_eps: float = 1e-8  # Numerical stability constant
    auto_detect_features: bool = True  # Auto-detect 4D vs 11D from data shape

    # Engineered features
    use_small_dataset_features: bool = True
    small_dataset_max_features: int = 25
    small_dataset_scaling: bool = True

    use_price_action_features: bool = True
    use_multiscale_features: bool = True  # 21 features
    use_hopsketch_features: bool = False  # 1575 features (optional for XGBoost)

    # Feature selection
    max_total_engineered_features: int = 50  # Limit to avoid overfitting
    feature_selection_threshold: float = 0.1

    # Processing options
    cache_features: bool = True  # Cache engineered features for efficiency
    handle_missing_features: str = "zero"  # "zero", "mean", "drop"

    # Validation
    validate_expansion_indices: bool = True
    min_pattern_length: int = 5

    # Pseudo-sample augmentation
    enable_augmentation: bool = False  # Enable pseudo-sample generation
    augmentation_ratio: float = 2.0  # Target ratio of synthetic:real samples (2.0 = 2:1)
    max_synthetic_samples: int = 210  # Maximum number of synthetic samples to generate
    augmentation_seed: int = 1337  # Random seed for reproducible augmentation
    quality_threshold: float = 0.7  # Minimum quality score for sample acceptance
    use_safe_strategies_only: bool = True  # Use only temporal and pattern-based strategies


class DualInputDataProcessor:
    """Handles dual-input data processing for raw OHLC and engineered features."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the dual-input data processor.

        Args:
            config: Feature processing configuration
        """
        self.config = config or FeatureConfig()
        self.feature_cache: Dict[str, np.ndarray] = {}
        self.feature_names: List[str] = []
        self._setup_feature_engineers()
        self._setup_augmentation_pipeline()

    def _setup_feature_engineers(self):
        """Setup feature engineering components."""
        if self.config.use_small_dataset_features:
            self.small_dataset_engineer = create_small_dataset_feature_engineer(
                max_features_per_category=self.config.small_dataset_max_features // 5,
                robust_scaling=self.config.small_dataset_scaling,
                feature_selection=True
            )

    def _setup_augmentation_pipeline(self):
        """Setup pseudo-sample augmentation pipeline."""
        if self.config.enable_augmentation:
            # Configure strategy weights for safe methods only
            if self.config.use_safe_strategies_only:
                strategy_weights = {
                    'temporal_augmentation': 0.6,
                    'pattern_synthesis': 0.4
                }
            else:
                strategy_weights = {
                    'temporal_augmentation': 0.25,
                    'pattern_synthesis': 0.25,
                    'statistical_simulation': 0.2,
                    'market_condition': 0.3
                }

            self.augmentation_pipeline = PseudoSampleGenerationPipeline(
                seed=self.config.augmentation_seed,
                strategy_weights=strategy_weights,
                validation_threshold=self.config.quality_threshold
            )
            logger.info(f"Augmentation pipeline initialized with safe strategies only: {self.config.use_safe_strategies_only}")
        else:
            self.augmentation_pipeline = None

    def process_training_data(
        self,
        df: pd.DataFrame,
        enable_engineered_features: bool = True
    ) -> Dict[str, Any]:
        """Process training data with dual-input support.

        Args:
            df: Training dataframe with OHLC data in 'features' column
            enable_engineered_features: Whether to extract engineered features

        Returns:
            Dictionary containing processed data:
            - X_ohlc: Raw OHLC data [N, 105, D] where D=4 for OHLC or 11 for RelativeTransform
            - X_engineered: Engineered features [N, F] (if enabled)
            - y: Labels
            - expansion_start: Expansion start indices
            - expansion_end: Expansion end indices
            - feature_names: Names of engineered features
            - metadata: Processing metadata
        """
        logger.info(f"Processing {len(df)} samples with dual-input pipeline")

        # Validate input data
        self._validate_input_data(df)

        # Extract raw OHLC data
        X_ohlc = self._extract_raw_ohlc(df)
        logger.info(f"Extracted raw OHLC: {X_ohlc.shape}")

        # Extract labels and indices
        y = df["label"].values
        expansion_start = df["expansion_start"].values if "expansion_start" in df.columns else None
        expansion_end = df["expansion_end"].values if "expansion_end" in df.columns else None

        # Apply augmentation if enabled
        if self.config.enable_augmentation and self.augmentation_pipeline is not None:
            X_ohlc, y, augmentation_metadata = self._apply_augmentation(
                X_ohlc, np.asarray(y), 
                np.asarray(expansion_start) if expansion_start is not None else None,
                np.asarray(expansion_end) if expansion_end is not None else None
            )
        else:
            augmentation_metadata = {}

        # Get feature dimension for model compatibility
        _, _, ohlc_dim = X_ohlc.shape
        
        result = {
            "X_ohlc": X_ohlc,
            "y": y,
            "expansion_start": expansion_start,
            "expansion_end": expansion_end,
            "feature_names": [],
            "metadata": {
                "n_samples": len(df),
                "ohlc_shape": X_ohlc.shape,
                "ohlc_dim": ohlc_dim,
                "use_engineered_features": enable_engineered_features,
                "augmentation_metadata": augmentation_metadata
            }
        }

        # Extract engineered features if enabled
        if enable_engineered_features:
            X_engineered, feature_names = self._extract_engineered_features(
                X_ohlc, 
                np.asarray(expansion_start) if expansion_start is not None else None,
                np.asarray(expansion_end) if expansion_end is not None else None,
                np.asarray(y) if y is not None else None
            )
            result["X_engineered"] = X_engineered
            result["feature_names"] = feature_names
            result["metadata"]["engineered_shape"] = X_engineered.shape
            result["metadata"]["n_engineered_features"] = X_engineered.shape[1]
            logger.info(f"Extracted engineered features: {X_engineered.shape}")
        else:
            result["X_engineered"] = None
            logger.info("Engineered features disabled")

        return result

    def _validate_input_data(self, df: pd.DataFrame):
        """Validate input dataframe structure."""
        required_columns = ["features", "label"]
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Validate features column structure
        if not all(isinstance(f, np.ndarray) for f in df["features"]):
            raise ValueError("All entries in 'features' column must be numpy arrays")

        # Check expansion indices if present
        if self.config.validate_expansion_indices:
            if "expansion_start" in df.columns and "expansion_end" in df.columns:
                self._validate_expansion_indices(df)

    def _validate_expansion_indices(self, df: pd.DataFrame):
        """Validate expansion start/end indices."""
        for i, (_, row) in enumerate(df.iterrows()):
            start = row["expansion_start"]
            end = row["expansion_end"]

            if not (0 <= start <= end < 105):
                logger.warning(f"Invalid expansion indices for sample {i}: start={start}, end={end}")

            pattern_length = end - start + 1
            if pattern_length < self.config.min_pattern_length:
                logger.warning(f"Short pattern for sample {i}: length={pattern_length}")

    def _extract_raw_ohlc(self, df: pd.DataFrame) -> np.ndarray:
        """Extract raw OHLC data from features column.

        Returns:
            OHLC data [N, 105, 4]
        """
        # Convert features column to 3D array
        X_ohlc = np.stack([np.stack(f) for f in df["features"]])

        # Validate shape
        N_actual, T, D = X_ohlc.shape
        if T != 105:
            raise ValueError(f"Expected 105 timesteps, got {T}")
        # Auto-detect feature dimension (4=OHLC, 11=RelativeTransform)

        return X_ohlc.astype(np.float32)

    def _extract_engineered_features(
        self,
        X_ohlc: np.ndarray,
        expansion_start: Optional[Union[np.ndarray, Sequence]],
        expansion_end: Optional[Union[np.ndarray, Sequence]],
        y: Optional[Union[np.ndarray, Sequence]] = None
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract engineered features from OHLC data.

        Args:
            X_ohlc: Raw OHLC data [N, 105, 4]
            expansion_start: Pattern start indices [N]
            expansion_end: Pattern end indices [N]
            y: Optional labels for supervised feature selection

        Returns:
            Tuple of (engineered_features [N, F], feature_names [F])
        """
        # Generate cache key
        cache_key = self._generate_cache_key(X_ohlc.shape, expansion_start, expansion_end)

        # Check cache
        if self.config.cache_features and cache_key in self.feature_cache:
            logger.info(f"Using cached engineered features: {cache_key}")
            cached_data = self.feature_cache[cache_key]
            return cached_data["features"], cached_data["names"]

        all_features = []
        all_feature_names = []

        # 1. Small dataset optimized features
        if self.config.use_small_dataset_features:
            logger.info("Extracting small dataset optimized features...")
            small_features, small_names = extract_optimized_features(
                X_ohlc, expansion_start, expansion_end, y,
                max_total_features=self.config.small_dataset_max_features
            )
            all_features.append(small_features)
            all_feature_names.extend(small_names)
            logger.info(f"Small dataset features: {small_features.shape}")

        # 2. Multi-scale price action features
        if self.config.use_price_action_features and self.config.use_multiscale_features:
            logger.info("Extracting multi-scale price action features...")

            # Validate expansion indices for multi-scale features
            if expansion_start is None or expansion_end is None:
                logger.warning("Multi-scale features require expansion indices, using defaults")
                expansion_start = np.full(X_ohlc.shape[0], 30)
                expansion_end = np.full(X_ohlc.shape[0], 74)

            multiscale_features = engineer_multiscale_features(
                X_ohlc, expansion_start, expansion_end
            )

            # Add feature names for multi-scale features
            multiscale_names = [f"multiscale_{i}" for i in range(multiscale_features.shape[1])]
            all_features.append(multiscale_features)
            all_feature_names.extend(multiscale_names)
            logger.info(f"Multi-scale features: {multiscale_features.shape}")

        # 3. HopSketch features (optional, for XGBoost models)
        if self.config.use_price_action_features and self.config.use_hopsketch_features:
            logger.info("Extracting HopSketch features...")
            hopsketch_features = extract_hopsketch_features(X_ohlc)
            hopsketch_names = [f"hopsketch_{i}" for i in range(hopsketch_features.shape[1])]
            all_features.append(hopsketch_features)
            all_feature_names.extend(hopsketch_names)
            logger.info(f"HopSketch features: {hopsketch_features.shape}")

        # Concatenate all engineered features
        if all_features:
            X_engineered = np.hstack(all_features)
            logger.info(f"Total engineered features: {X_engineered.shape}")
        else:
            X_engineered = np.zeros((X_ohlc.shape[0], 0))
            logger.warning("No engineered features extracted")

        # Limit features if necessary
        if X_engineered.shape[1] > self.config.max_total_engineered_features:
            logger.info(f"Limiting features from {X_engineered.shape[1]} to {self.config.max_total_engineered_features}")
            X_engineered, all_feature_names = self._select_top_features(
                X_engineered, all_feature_names, y, self.config.max_total_engineered_features
            )

        # Handle missing values
        X_engineered = self._handle_missing_values(X_engineered)

        # Cache results
        if self.config.cache_features:
            self.feature_cache[cache_key] = {
                "features": X_engineered,
                "names": all_feature_names
            }

        return X_engineered, all_feature_names

    def _generate_cache_key(
        self,
        shape: Tuple[int, ...],
        expansion_start: Optional[Union[np.ndarray, Sequence]],
        expansion_end: Optional[Union[np.ndarray, Sequence]]
    ) -> str:
        """Generate cache key for feature extraction."""
        key_parts = [f"shape_{shape}"]

        if expansion_start is not None:
            key_parts.append(f"start_{expansion_start.tobytes().hex()[:16]}")
        if expansion_end is not None:
            key_parts.append(f"end_{expansion_end.tobytes().hex()[:16]}")

        return "_".join(key_parts)

    def _select_top_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
        y: Optional[np.ndarray],
        n_features: int
    ) -> Tuple[np.ndarray, List[str]]:
        """Select top N features using variance or mutual information."""
        if features.shape[1] <= n_features:
            return features, feature_names

        if y is not None:
            # Use mutual information if labels are available
            try:
                from sklearn.feature_selection import mutual_info_classif
                mi_scores = mutual_info_classif(features, y, random_state=42)
                top_indices = np.argsort(mi_scores)[-n_features:]
                logger.info(f"Selected top {n_features} features by mutual information")
            except Exception as e:
                logger.warning(f"Mutual information failed: {e}, using variance")
                feature_variances = np.var(features, axis=0)
                top_indices = np.argsort(feature_variances)[-n_features:]
        else:
            # Use variance if no labels available
            feature_variances = np.var(features, axis=0)
            top_indices = np.argsort(feature_variances)[-n_features:]
            logger.info(f"Selected top {n_features} features by variance")

        selected_features = features[:, top_indices]
        selected_names = [feature_names[i] for i in top_indices]

        return selected_features, selected_names

    def _handle_missing_values(self, features: np.ndarray) -> np.ndarray:
        """Handle missing values in engineered features."""
        if self.config.handle_missing_features == "zero":
            features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        elif self.config.handle_missing_features == "mean":
            # Replace NaN with column mean
            col_means = np.nanmean(features, axis=0)
            nan_indices = np.isnan(features)
            features[nan_indices] = np.take(col_means, np.where(nan_indices)[1])
        elif self.config.handle_missing_features == "drop":
            # Remove features with too many NaN values
            nan_ratio = np.isnan(features).mean(axis=0)
            valid_features = nan_ratio < 0.1  # Keep features with <10% NaN
            features = features[:, valid_features]
            logger.warning(f"Dropped {(~valid_features).sum()} features due to NaN values")

        return features

    def _apply_augmentation(
        self,
        X_ohlc: np.ndarray,
        y: np.ndarray,
        expansion_start: Optional[np.ndarray],
        expansion_end: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Apply pseudo-sample augmentation to training data.

        Args:
            X_ohlc: Original OHLC data [N, T, 4]
            y: Original labels [N]
            expansion_start: Expansion start indices [N]
            expansion_end: Expansion end indices [N]

        Returns:
            Tuple of (augmented_X_ohlc, augmented_y, augmentation_metadata)
        """
        n_original = len(X_ohlc)

        # Calculate number of synthetic samples to generate
        target_synthetic = min(
            int(n_original * self.config.augmentation_ratio),
            self.config.max_synthetic_samples
        )

        # Start with conservative approach for Week 1 (50 samples max)
        max_week1_samples = min(50, target_synthetic)
        n_synthetic = max_week1_samples

        logger.info(f"Augmentation enabled: generating {n_synthetic} synthetic samples "
                   f"(ratio: {self.config.augmentation_ratio:.1f}, max: {self.config.max_synthetic_samples})")

        # Generate pseudo-samples
        try:
            X_synthetic, y_synthetic, generation_metadata = self.augmentation_pipeline.generate_samples(
                data=X_ohlc,
                labels=y,
                n_samples=n_synthetic,
                quality_check=True
            )

            # Validate OHLC integrity
            ohlc_violations = self._validate_ohlc_integrity(X_synthetic)
            if ohlc_violations > 0:
                logger.warning(f"Found {ohlc_violations} OHLC violations in synthetic samples")
                # Filter out samples with violations
                valid_mask = self._filter_ohlc_valid_samples(X_synthetic)
                X_synthetic = X_synthetic[valid_mask]
                y_synthetic = y_synthetic[valid_mask]
                logger.info(f"Filtered to {len(X_synthetic)} samples with valid OHLC relationships")

            # Combine original and synthetic data
            X_augmented = np.concatenate([X_ohlc, X_synthetic], axis=0)
            y_augmented = np.concatenate([y, y_synthetic], axis=0)

            # Handle expansion indices for synthetic samples
            if expansion_start is not None and expansion_end is not None:
                # Generate reasonable expansion indices for synthetic samples
                synthetic_expansion_start = self._generate_synthetic_expansion_indices(
                    X_synthetic, expansion_start, expansion_end
                )
                synthetic_expansion_end = self._generate_synthetic_expansion_indices(
                    X_synthetic, expansion_start, expansion_end
                )

                expansion_start = np.concatenate([expansion_start, synthetic_expansion_start], axis=0)
                expansion_end = np.concatenate([expansion_end, synthetic_expansion_end], axis=0)

            # Log augmentation results
            augmentation_metadata = {
                "n_original": n_original,
                "n_synthetic_generated": n_synthetic,
                "n_synthetic_accepted": len(X_synthetic),
                "n_total": len(X_augmented),
                "synthetic_ratio": len(X_synthetic) / n_original,
                "generation_metadata": generation_metadata,
                "ohlc_integrity_rate": 1.0 - (ohlc_violations / (len(X_synthetic) * X_synthetic.shape[1])) if len(X_synthetic) > 0 else 0.0
            }

            logger.info(f"Augmentation complete: {n_original} → {len(X_augmented)} total samples "
                       f"(synthetic ratio: {augmentation_metadata['synthetic_ratio']:.2f})")

            return X_augmented, y_augmented, augmentation_metadata

        except Exception as e:
            logger.error(f"Augmentation failed: {e}")
            logger.info("Proceeding with original data without augmentation")
            return X_ohlc, y, {"error": str(e), "n_original": n_original}

    def _validate_ohlc_integrity(self, ohlc_data: np.ndarray) -> int:
        """Validate OHLC relationships in synthetic data.

        Args:
            ohlc_data: OHLC data [N, T, 4]

        Returns:
            Number of OHLC violations found
        """
        violations = 0
        for sample in ohlc_data:
            for t in range(sample.shape[0]):
                o, h, l, c = sample[t]
                # Check OHLC relationships: O <= H >= L <= C, and H >= L
                if not (o <= h + 1e-8 and l <= h + 1e-8 and o >= l - 1e-8 and c >= l - 1e-8 and h >= l - 1e-8):
                    violations += 1
        return violations

    def _filter_ohlc_valid_samples(self, ohlc_data: np.ndarray) -> np.ndarray:
        """Filter for samples with valid OHLC relationships.

        Args:
            ohlc_data: OHLC data [N, T, 4]

        Returns:
            Boolean mask of valid samples
        """
        valid_mask = []
        for sample in ohlc_data:
            sample_valid = True
            for t in range(sample.shape[0]):
                o, h, l, c = sample[t]
                if not (o <= h + 1e-8 and l <= h + 1e-8 and o >= l - 1e-8 and c >= l - 1e-8 and h >= l - 1e-8):
                    sample_valid = False
                    break
            valid_mask.append(sample_valid)
        return np.array(valid_mask)

    def _generate_synthetic_expansion_indices(
        self,
        X_synthetic: np.ndarray,
        original_start: np.ndarray,
        original_end: np.ndarray
    ) -> np.ndarray:
        """Generate reasonable expansion indices for synthetic samples.

        Args:
            X_synthetic: Synthetic OHLC data [M, T, 4]
            original_start: Original expansion start indices [N]
            original_end: Original expansion end indices [N]

        Returns:
            Synthetic expansion start indices [M]
        """
        n_synthetic = len(X_synthetic)

        # Use distribution of original expansion indices
        start_mean, start_std = np.mean(original_start), np.std(original_start)
        end_mean, end_std = np.mean(original_end), np.std(original_end)

        # Generate synthetic indices with some randomness but within reasonable bounds
        synthetic_start = np.clip(
            np.random.normal(start_mean, start_std, n_synthetic),
            5, 45  # Reasonable bounds for 105-timestep sequences
        ).astype(int)

        return synthetic_start

    def get_feature_statistics(self, X_engineered: np.ndarray) -> Dict[str, Any]:
        """Get statistics for engineered features."""
        if X_engineered is None or X_engineered.size == 0:
            return {}

        return {
            "n_features": X_engineered.shape[1],
            "mean": float(np.mean(X_engineered)),
            "std": float(np.std(X_engineered)),
            "min": float(np.min(X_engineered)),
            "max": float(np.max(X_engineered)),
            "nan_count": int(np.isnan(X_engineered).sum()),
            "inf_count": int(np.isinf(X_engineered).sum())
        }


def create_dual_input_processor(
    use_engineered_features: bool = True,
    max_engineered_features: int = 50,
    use_hopsketch: bool = False,
    enable_augmentation: bool = False,
    augmentation_ratio: float = 2.0,
    max_synthetic_samples: int = 210,
    augmentation_seed: int = 1337,
    quality_threshold: float = 0.7,
    use_safe_strategies_only: bool = True
) -> DualInputDataProcessor:
    """Create a dual-input data processor with sensible defaults.

    Args:
        use_engineered_features: Whether to enable engineered features
        max_engineered_features: Maximum number of engineered features
        use_hopsketch: Whether to include HopSketch features (for XGBoost)
        enable_augmentation: Whether to enable pseudo-sample augmentation
        augmentation_ratio: Target ratio of synthetic:real samples
        max_synthetic_samples: Maximum number of synthetic samples to generate
        augmentation_seed: Random seed for reproducible augmentation
        quality_threshold: Minimum quality score for sample acceptance
        use_safe_strategies_only: Use only temporal and pattern-based strategies

    Returns:
        Configured DualInputDataProcessor instance
    """
    config = FeatureConfig(
        use_raw_ohlc=True,
        use_small_dataset_features=use_engineered_features,
        small_dataset_max_features=min(25, max_engineered_features // 2),
        use_price_action_features=use_engineered_features,
        use_multiscale_features=use_engineered_features,
        use_hopsketch_features=use_hopsketch,
        max_total_engineered_features=max_engineered_features,
        cache_features=True,
        handle_missing_features="zero",
        enable_augmentation=enable_augmentation,
        augmentation_ratio=augmentation_ratio,
        max_synthetic_samples=max_synthetic_samples,
        augmentation_seed=augmentation_seed,
        quality_threshold=quality_threshold,
        use_safe_strategies_only=use_safe_strategies_only
    )

    return DualInputDataProcessor(config)


def prepare_model_inputs(
    processed_data: Dict[str, Any],
    model_type: str = "lstm",
    use_engineered_features: bool = True
) -> Dict[str, Any]:
    """Prepare model inputs based on model type and available features.

    Args:
        processed_data: Processed data from DualInputDataProcessor
        model_type: Type of model ("lstm", "transformer", "xgboost", "ensemble")
        use_engineered_features: Whether to use engineered features if available

    Returns:
        Model-specific inputs dictionary
    """
    X_ohlc = processed_data["X_ohlc"]
    X_engineered = processed_data.get("X_engineered")

    if model_type in ["lstm", "transformer", "cnn_transformer"]:
        # Deep learning models use raw OHLC sequences
        inputs = {
            "X": X_ohlc,  # [N, 105, 4]
            "y": processed_data["y"],
            "expansion_start": processed_data["expansion_start"],
            "expansion_end": processed_data["expansion_end"]
        }

        # Some models can also use engineered features as additional context
        if use_engineered_features and X_engineered is not None:
            # Could be used as additional input channels or attention context
            inputs["X_engineered"] = X_engineered
            logger.info(f"Model {model_type} has engineered features available: {X_engineered.shape}")

    elif model_type in ["xgboost", "rf", "logreg"]:
        # Tree-based models use engineered features or flattened OHLC
        if use_engineered_features and X_engineered is not None:
            inputs = {
                "X": X_engineered,  # [N, F]
                "y": processed_data["y"]
            }
            logger.info(f"Model {model_type} using engineered features: {X_engineered.shape}")
        else:
            # Flatten OHLC for tree-based models
            X_flat = X_ohlc.reshape(X_ohlc.shape[0], -1)  # [N, 420]
            inputs = {
                "X": X_flat,
                "y": processed_data["y"]
            }
            logger.info(f"Model {model_type} using flattened OHLC: {X_flat.shape}")

    else:
        # Default: use raw OHLC
        inputs = {
            "X": X_ohlc,
            "y": processed_data["y"],
            "expansion_start": processed_data["expansion_start"],
            "expansion_end": processed_data["expansion_end"]
        }

    return inputs