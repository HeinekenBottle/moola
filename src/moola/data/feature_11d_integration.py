"""11-dimensional feature integration for Moola data pipeline.

Provides backward-compatible integration of 11D relative features with existing 4D OHLC data.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
from loguru import logger

from ..features.relative_transform import RelativeFeatureTransform


class Feature11DIntegrator:
    """Integrates 11D relative features with existing Moola pipeline."""
    
    def __init__(self, eps: float = 1e-8):
        """Initialize the 11D feature integrator.
        
        Args:
            eps: Small constant for numerical stability
        """
        self.transformer = RelativeFeatureTransform(eps=eps)
        self.eps = eps
        
    def load_and_transform_data(self, data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load 4D OHLC data and transform to 11D features.
        
        Args:
            data_path: Path to parquet file with 4D OHLC data
            
        Returns:
            Tuple of (ohlc_data, relative_11d_data)
            - ohlc_data: [N, 105, 4] original OHLC data
            - relative_11d_data: [N, 105, 11] relative features
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load parquet data
        df = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Extract OHLC features
        if 'features' in df.columns:
            features_list = df['features'].tolist()
            
            # Convert to numpy array [N, 105, 4]
            ohlc_data = np.array(features_list, dtype=np.float32)
            logger.info(f"OHLC data shape: {ohlc_data.shape}")
            
            # Validate shape
            if ohlc_data.shape[1:] != (105, 4):
                raise ValueError(f"Expected OHLC shape [N, 105, 4], got {ohlc_data.shape}")
            
            # Transform to 11D relative features
            logger.info("Transforming to 11D relative features...")
            relative_11d_data = self.transformer.transform(ohlc_data)
            logger.info(f"Generated 11D features: {relative_11d_data.shape}")
            
            return ohlc_data, relative_11d_data
        else:
            raise ValueError("Data must contain 'features' column with OHLC data")
    
    def create_dual_input_dataset(self, 
                                 data_path: Path,
                                 save_path: Optional[Path] = None) -> pd.DataFrame:
        """Create dual-input dataset with both 4D OHLC and 11D relative features.
        
        Args:
            data_path: Path to original 4D OHLC data
            save_path: Optional path to save enhanced dataset
            
        Returns:
            DataFrame with both OHLC and relative features
        """
        # Load and transform data
        ohlc_data, relative_11d_data = self.load_and_transform_data(data_path)
        
        # Create enhanced dataset
        N = ohlc_data.shape[0]
        enhanced_data = []
        
        for i in range(N):
            # Combine OHLC and relative features into 15D array
            enhanced_features = np.concatenate([
                ohlc_data[i],      # [105, 4] OHLC
                relative_11d_data[i]  # [105, 11] relative
            ], axis=1)  # [105, 15]
            
            enhanced_data.append({
                'window_id': f"window_{i:06d}",
                'features_ohlc': ohlc_data[i].tolist(),
                'features_relative': relative_11d_data[i].tolist(),
                'features_enhanced': enhanced_features.tolist(),
                'feature_dimension': 'dual_input'
            })
        
        df_enhanced = pd.DataFrame(enhanced_data)
        
        # Save if path provided
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df_enhanced.to_parquet(save_path, index=False)
            logger.info(f"Saved enhanced dataset to {save_path}")
        
        return df_enhanced
    
    def validate_feature_quality(self, 
                                ohlc_data: np.ndarray, 
                                relative_data: np.ndarray) -> Dict[str, Any]:
        """Validate quality of 11D features.
        
        Args:
            ohlc_data: [N, 105, 4] OHLC data
            relative_data: [N, 105, 11] relative features
            
        Returns:
            Quality validation report
        """
        report = {
            'ohlc_quality': {},
            'relative_quality': {},
            'cross_validation': {}
        }
        
        # OHLC quality checks
        report['ohlc_quality'] = {
            'shape': ohlc_data.shape,
            'missing_values': int(np.isnan(ohlc_data).sum()),
            'inf_values': int(np.isinf(ohlc_data).sum()),
            'price_range': {
                'min': float(ohlc_data.min()),
                'max': float(ohlc_data.max()),
                'mean': float(ohlc_data.mean()),
                'std': float(ohlc_data.std())
            }
        }
        
        # Relative features quality checks
        report['relative_quality'] = {
            'shape': relative_data.shape,
            'missing_values': int(np.isnan(relative_data).sum()),
            'inf_values': int(np.isinf(relative_data).sum()),
            'feature_stats': {
                'min': float(relative_data.min()),
                'max': float(relative_data.max()),
                'mean': float(relative_data.mean()),
                'std': float(relative_data.std())
            }
        }
        
        # Cross-validation checks
        report['cross_validation'] = {
            'consistent_samples': ohlc_data.shape[0] == relative_data.shape[0],
            'consistent_timesteps': ohlc_data.shape[1] == relative_data.shape[1],
            'expected_ohlc_dims': ohlc_data.shape[2] == 4,
            'expected_relative_dims': relative_data.shape[2] == 11
        }
        
        # Feature-specific validation for relative features
        feature_names = self.transformer.get_feature_names()
        for i, name in enumerate(feature_names):
            feature_data = relative_data[:, :, i]
            report[f'feature_{name}'] = {
                'mean': float(feature_data.mean()),
                'std': float(feature_data.std()),
                'min': float(feature_data.min()),
                'max': float(feature_data.max()),
                'missing': int(np.isnan(feature_data).sum())
            }
        
        return report
    
    def get_feature_importance_analysis(self, 
                                      relative_data: np.ndarray,
                                      labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Analyze feature importance for 11D relative features.
        
        Args:
            relative_data: [N, 105, 11] relative features
            labels: Optional [N] labels for supervised analysis
            
        Returns:
            Feature importance analysis
        """
        # Flatten features for analysis
        N, T, F = relative_data.shape
        flattened_data = relative_data.reshape(N, T * F)  # [N, 105*11]
        
        analysis = {
            'feature_variance': {},
            'feature_correlation': {},
            'temporal_stability': {}
        }
        
        # Feature variance analysis
        feature_names = self.transformer.get_feature_names()
        for i, name in enumerate(feature_names):
            feature_data = relative_data[:, :, i].flatten()
            analysis['feature_variance'][name] = {
                'variance': float(np.var(feature_data)),
                'std': float(np.std(feature_data)),
                'range': float(np.max(feature_data) - np.min(feature_data))
            }
        
        # Temporal stability (how features change over time)
        for i, name in enumerate(feature_names):
            # Compute temporal autocorrelation for each feature
            temporal_autocorr = []
            for t in range(1, T):
                corr = np.corrcoef(
                    relative_data[:, t-1, i].flatten(),
                    relative_data[:, t, i].flatten()
                )[0, 1]
                if not np.isnan(corr):
                    temporal_autocorr.append(corr)
            
            analysis['temporal_stability'][name] = {
                'mean_autocorrelation': float(np.mean(temporal_autocorr)) if temporal_autocorr else 0.0,
                'autocorr_std': float(np.std(temporal_autocorr)) if temporal_autocorr else 0.0
            }
        
        # Supervised analysis if labels provided
        if labels is not None:
            from scipy.stats import pointbiserialr
            
            analysis['supervised_importance'] = {}
            for i, name in enumerate(feature_names):
                # Average feature value per sample
                feature_avg = relative_data[:, :, i].mean(axis=1)
                
                # Point-biserial correlation with binary labels
                if len(np.unique(labels)) == 2:
                    corr, p_value = pointbiserialr(feature_avg, labels)
                    analysis['supervised_importance'][name] = {
                        'correlation': float(corr) if not np.isnan(corr) else 0.0,
                        'p_value': float(p_value) if not np.isnan(p_value) else 1.0
                    }
        
        return analysis


# Convenience functions for backward compatibility
def load_11d_features(data_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load 4D OHLC data and generate 11D relative features.
    
    Args:
        data_path: Path to parquet file with 4D OHLC data
        
    Returns:
        Tuple of (ohlc_data, relative_11d_data)
    """
    integrator = Feature11DIntegrator()
    return integrator.load_and_transform_data(data_path)


def create_enhanced_dataset(input_path: Path, output_path: Path) -> pd.DataFrame:
    """Create enhanced dataset with 11D features.
    
    Args:
        input_path: Path to original 4D data
        output_path: Path to save enhanced dataset
        
    Returns:
        Enhanced DataFrame with both 4D and 11D features
    """
    integrator = Feature11DIntegrator()
    return integrator.create_dual_input_dataset(input_path, output_path)