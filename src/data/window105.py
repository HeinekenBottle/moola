"""Window105 Dataset - Adapter for Moola annotation system.

This module provides a Window105Dataset interface that wraps Moola's existing
training data structure for use with the Candlesticks annotation system.

The adapter presents Moola's 115 training samples as a fixed, non-hopping
window set with sequential indexing (0-114) while maintaining the original
window_id mapping for data integrity.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class Window105Dataset:
    """
    Dataset adapter that presents Moola training data in Window105 format.
    
    Provides sequential access to Moola's 115 training samples with:
    - Fixed window indexing (0-114) to prevent hopping
    - 105-bar OHLC data for candlestick visualization  
    - Original window_id mapping for traceability
    - Label and expansion information from CleanLab analysis
    """
    
    def __init__(self, load_in_memory: bool = True):
        """
        Initialize the Window105 dataset.
        
        Args:
            load_in_memory: Whether to load all data into memory (default: True)
        """
        self.load_in_memory = load_in_memory
        self.data_root = Path("/Users/jack/projects/moola/data")
        
        # Load training data
        self._load_training_data()
        
        # Build index mapping from sequential to original window_ids
        self._build_index_mapping()
        
        # Load OHLC features for visualization
        self._load_ohlc_features()
        
        # Load CleanLab priority samples for review workflow
        self._load_priority_samples()
        
        logger.info(f"Window105Dataset initialized: {len(self.samples)} samples")
    
    def _load_training_data(self) -> None:
        """Load the main training parquet file."""
        # UPDATED: Load from batch_204_annotation_ready.parquet (latest batch for annotation)
        train_path = self.data_root / "batches" / "batch_204_annotation_ready.parquet"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")

        self.samples = pd.read_parquet(train_path)
        logger.info(f"Loaded {len(self.samples)} training samples from batch_204_annotation_ready.parquet")

        # Validate expected columns
        # NOTE: batch_204 doesn't have 'label', 'expansion_start', 'expansion_end' yet (to be annotated)
        required_cols = ['window_id', 'features']
        missing_cols = [col for col in required_cols if col not in self.samples.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in training data: {missing_cols}")
    
    def _build_index_mapping(self) -> None:
        """Build mapping between sequential indices and original window_ids."""
        # Create sequential index mapping (0, 1, 2, ...) 
        self.index_to_window_id = dict(enumerate(self.samples['window_id']))
        self.window_id_to_index = {wid: idx for idx, wid in self.index_to_window_id.items()}
        
        logger.info(f"Built index mapping for {len(self.index_to_window_id)} windows")
    
    def _load_ohlc_features(self) -> None:
        """Load OHLC features for candlestick visualization."""
        features_path = self.data_root / "corrections" / "moola_features_for_viz.parquet"
        
        if not features_path.exists():
            logger.warning(f"OHLC features not found at {features_path}, will generate from training data")
            self.ohlc_features = None
            return
        
        # Features stored as long format: window_id, bar_index, open, high, low, close
        self.ohlc_features = pd.read_parquet(features_path)
        logger.info(f"Loaded OHLC features for {self.ohlc_features['window_id'].nunique()} windows")
    
    def _load_priority_samples(self) -> None:
        """Load CleanLab priority samples for review workflow."""
        priority_path = self.data_root / "corrections" / "cleanlab_studio_priority_review.csv"
        
        if priority_path.exists():
            self.priority_samples = pd.read_csv(priority_path)
            logger.info(f"Loaded {len(self.priority_samples)} priority samples for review")
        else:
            self.priority_samples = pd.DataFrame()
            logger.info("No priority samples found")
    
    def __len__(self) -> int:
        """Return the number of windows in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get window data by sequential index.
        
        Args:
            idx: Sequential index (0-114)
            
        Returns:
            Dictionary with window data in expected format
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.samples)}")
        
        # Get sample by index
        sample = self.samples.iloc[idx]
        
        # Extract OHLC data
        ohlc_data = self._get_ohlc_data(sample['window_id'])
        
        # Build window response dict
        window_data = {
            'window_id': sample['window_id'],
            'sequential_index': idx,  # For Candlesticks internal use
            'label': sample.get('label', None),  # May not exist for unannotated batches
            'expansion_start': int(sample['expansion_start']) if 'expansion_start' in sample else None,
            'expansion_end': int(sample['expansion_end']) if 'expansion_end' in sample else None,
            'ohlc': ohlc_data,
            'features': sample['features'],  # Raw features array (105 bars × 4 OHLC)
            'center_timestamp': self._extract_timestamp(sample),
        }
        
        # Add CleanLab priority info if applicable
        priority_info = self._get_priority_info(sample['window_id'])
        if priority_info:
            window_data.update(priority_info)
        
        return window_data
    
    def _get_ohlc_data(self, window_id: str) -> np.ndarray:
        """
        Get OHLC data for a specific window_id.
        
        Args:
            window_id: Original window identifier
            
        Returns:
            OHLC data as (105, 4) numpy array
        """
        if self.ohlc_features is not None:
            # Use pre-computed features from parquet
            window_ohlc = self.ohlc_features[self.ohlc_features['window_id'] == window_id].sort_values('bar_index')
            if len(window_ohlc) == 105:
                return window_ohlc[['open', 'high', 'low', 'close']].values
            else:
                logger.warning(f"Found {len(window_ohlc)} bars for window {window_id}, expected 105")
        
        # Fallback: extract from features array in training data
        sample = self.samples[self.samples['window_id'] == window_id].iloc[0]
        features = sample['features']

        # Handle features stored as list (from batch_200.parquet)
        if isinstance(features, list) and len(features) == 105:
            # Convert list of lists to numpy array
            return np.array(features, dtype=np.float64)

        # Handle features as numpy array
        if isinstance(features, np.ndarray) and len(features) == 105:
            # Check if it's already OHLC data
            if features.shape == (105, 4):
                return features
            # Check if it's structured as [ [open, high, low, close], ... ]
            elif len(features[0]) == 4:
                return np.array([list(bar) for bar in features])

        # Last resort: generate synthetic OHLC data (shouldn't happen in normal operation)
        logger.error(f"Could not extract OHLC data for window {window_id}, using zeros")
        return np.zeros((105, 4))
    
    def _extract_timestamp(self, sample: pd.Series) -> str:
        """
        Extract/estimate timestamp from sample data.

        Args:
            sample: Sample row from the dataset

        Returns:
            Timestamp string for center of window
        """
        # First priority: Use start_ts from parquet if available
        if 'start_ts' in sample and pd.notna(sample['start_ts']):
            # start_ts is the beginning of the window, compute center (52 bars into 105-bar window)
            start_time = pd.to_datetime(sample['start_ts'])
            center_time = start_time + timedelta(minutes=52)
            return center_time.isoformat()

        # Second priority: Parse window_id if it contains timestamp
        window_id = sample['window_id']

        # New format: batch_YYYYMMDDHHMM_NNN (e.g., batch_202510182107_001)
        if window_id.startswith('batch_'):
            try:
                parts = window_id.split('_')
                if len(parts) >= 2:
                    timestamp_str = parts[1]  # YYYYMMDDHHMM
                    # Parse: YYYYMMDDHHMM
                    year = int(timestamp_str[0:4])
                    month = int(timestamp_str[4:6])
                    day = int(timestamp_str[6:8])
                    hour = int(timestamp_str[8:10])
                    minute = int(timestamp_str[10:12])
                    dt = datetime(year, month, day, hour, minute)
                    return dt.isoformat()
            except (ValueError, IndexError):
                pass

        # Old format: "104_exp_1", "197_exp_1" etc.
        try:
            base_num = int(window_id.split('_')[0])
            # Convert to a reasonable datetime format
            base_date = datetime(2020, 1, 1)  # Arbitrary base date
            timestamp = base_date + timedelta(days=base_num)
            return timestamp.isoformat()
        except (ValueError, IndexError):
            # Fallback: current time
            return datetime.now().isoformat()
    
    def _get_priority_info(self, window_id: str) -> Optional[Dict[str, Any]]:
        """
        Get CleanLab priority information for a window.
        
        Args:
            window_id: Original window identifier
            
        Returns:
            Priority info dict or None if not a priority sample
        """
        if self.priority_samples.empty:
            return None
        
        priority_row = self.priority_samples[self.priority_samples['id'] == window_id]
        if priority_row.empty:
            return None
        
        priority = priority_row.iloc[0]
        return {
            'is_cleanlab_priority': True,
            'label_quality_score': float(priority['label_quality_score']),
            'prob_consolidation': float(priority['prob_consolidation']),
            'prob_retracement': float(priority['prob_retracement']),
            'is_cleanlab_issue': bool(priority['is_cleanlab_issue']),
            'review_priority': int(priority['review_priority']),
            'issue_type': str(priority['issue_type']),
        }
    
    def get_window_id_by_index(self, idx: int) -> str:
        """Get original window_id by sequential index."""
        return self.index_to_window_id.get(idx, "")
    
    def get_index_by_window_id(self, window_id: str) -> Optional[int]:
        """Get sequential index by original window_id."""
        return self.window_id_to_index.get(window_id)
    
    def get_available_window_ids(self) -> List[str]:
        """Get list of all available window_ids."""
        return list(self.index_to_window_id.values())
    
    def is_priority_window(self, window_id: str) -> bool:
        """Check if window is flagged for priority review."""
        if self.priority_samples.empty:
            return False
        return window_id in set(self.priority_samples['id'])
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get dataset summary statistics."""
        label_counts = self.samples['label'].value_counts().to_dict()
        
        stats = {
            'total_windows': len(self.samples),
            'label_distribution': label_counts,
            'expansion_stats': {
                'avg_start': float(self.samples['expansion_start'].mean()),
                'avg_end': float(self.samples['expansion_end'].mean()),
                'avg_length': float((self.samples['expansion_end'] - self.samples['expansion_start']).mean()),
            },
            'priority_windows': len(self.priority_samples) if not self.priority_samples.empty else 0,
        }
        
        return stats


# Compatibility with existing imports
def get_dataset() -> Window105Dataset:
    """Get the default Window105Dataset instance."""
    return Window105Dataset()


# Test function to verify the adapter works
def test_integration():
    """Test the Window105Dataset integration."""
    try:
        dataset = Window105Dataset()
        print(f"✓ Dataset loaded: {len(dataset)} samples")
        
        # Test first few samples
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"✓ Sample {i}: {sample['window_id']} ({sample['label']}) - OHLC shape: {sample['ohlc'].shape}")
        
        # Test window_id mapping
        first_window_id = dataset.get_window_id_by_index(0)
        mapped_index = dataset.get_index_by_window_id(first_window_id)
        print(f"✓ Window ID mapping: {first_window_id} -> index {mapped_index}")
        
        # Test summary stats
        stats = dataset.get_summary_stats()
        print(f"✓ Summary: {stats}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_integration()
