"""Storage architecture for 11-dimensional features with backward compatibility.

Defines storage formats and migration strategies for 4D→11D feature integration.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from loguru import logger


class Storage11DArchitecture:
    """Storage architecture for 11D features maintaining backward compatibility."""
    
    def __init__(self, base_path: Path):
        """Initialize storage architecture.
        
        Args:
            base_path: Base data directory path
        """
        self.base_path = Path(base_path)
        self.raw_path = self.base_path / "raw"
        self.batches_path = self.base_path / "batches"
        self.processed_path = self.base_path / "processed"
        self.enhanced_path = self.base_path / "enhanced_11d"
        
        # Create directories
        for path in [self.raw_path, self.batches_path, self.processed_path, self.enhanced_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def get_storage_schema(self) -> Dict[str, Any]:
        """Get recommended storage schema for 11D features."""
        return {
            "raw_4d_ohlc": {
                "path": "raw/unlabeled_windows.parquet",
                "format": "parquet",
                "schema": "[N, 105, 4] OHLC data",
                "description": "Original 4D OHLC data (unchanged)"
            },
            "enhanced_11d": {
                "path": "enhanced_11d/windows_enhanced.parquet",
                "format": "parquet",
                "schema": {
                    "window_id": "string",
                    "features_ohlc": "array[105, 4]",
                    "features_relative": "array[105, 11]",
                    "features_enhanced": "array[105, 15]",
                    "feature_dimension": "string",
                    "metadata": "dict"
                },
                "description": "Enhanced dataset with both 4D and 11D features"
            },
            "batches_11d": {
                "path": "batches/batch_{id}_11d.parquet",
                "format": "parquet",
                "schema": "Same as enhanced_11d but for annotation batches",
                "description": "Annotation batches with 11D features"
            },
            "processed_11d": {
                "path": "processed/train_enhanced.parquet",
                "format": "parquet",
                "schema": {
                    "window_id": "string",
                    "features_ohlc": "array[105, 4]",
                    "features_relative": "array[105, 11]",
                    "label": "string",
                    "expansion_start": "int",
                    "expansion_end": "int",
                    "confidence": "float"
                },
                "description": "Final training dataset with labels"
            }
        }
    
    def migrate_existing_data(self, 
                             input_path: Path,
                             output_path: Optional[Path] = None) -> Path:
        """Migrate existing 4D data to enhanced 11D format.
        
        Args:
            input_path: Path to existing 4D parquet file
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            Path to migrated enhanced dataset
        """
        if output_path is None:
            # Generate output path in enhanced_11d directory
            filename = input_path.stem + "_enhanced" + input_path.suffix
            output_path = self.enhanced_path / filename
        
        logger.info(f"Migrating {input_path} to {output_path}")
        
        # Load existing 4D data
        df_4d = pd.read_parquet(input_path)
        logger.info(f"Loaded {len(df_4d)} samples from {input_path}")
        
        # Import here to avoid circular imports
        from .feature_11d_integration import Feature11DIntegrator
        integrator = Feature11DIntegrator()
        
        # Create enhanced dataset
        df_enhanced = integrator.create_dual_input_dataset(input_path, output_path)
        
        logger.info(f"Successfully migrated {len(df_enhanced)} samples to 11D format")
        return output_path
    
    def create_annotation_batch_11d(self,
                                   window_ids: list,
                                   output_path: Optional[Path] = None) -> Path:
        """Create annotation batch with 11D features.
        
        Args:
            window_ids: List of window IDs to include in batch
            output_path: Optional output path
            
        Returns:
            Path to created batch file
        """
        if output_path is None:
            batch_id = f"batch_11d_{len(window_ids)}_samples"
            output_path = self.batches_path / f"{batch_id}.parquet"
        
        logger.info(f"Creating 11D annotation batch with {len(window_ids)} windows")
        
        # Load enhanced dataset
        enhanced_path = self.enhanced_path / "windows_enhanced.parquet"
        if not enhanced_path.exists():
            raise FileNotFoundError(f"Enhanced dataset not found at {enhanced_path}")
        
        df_enhanced = pd.read_parquet(enhanced_path)
        
        # Filter by window IDs
        batch_data = df_enhanced[df_enhanced['window_id'].isin(window_ids)].copy()
        
        # Reset index and add batch metadata
        batch_data = batch_data.reset_index(drop=True)
        batch_data['batch_id'] = output_path.stem
        batch_data['created_at'] = pd.Timestamp.now()
        
        # Save batch
        batch_data.to_parquet(output_path, index=False)
        logger.info(f"Created annotation batch: {output_path}")
        
        return output_path
    
    def validate_storage_integrity(self) -> Dict[str, Any]:
        """Validate storage architecture integrity.
        
        Returns:
            Validation report
        """
        report = {
            "directories_exist": {},
            "files_exist": {},
            "schema_validation": {},
            "data_quality": {}
        }
        
        # Check directories
        for name, path in [
            ("raw", self.raw_path),
            ("batches", self.batches_path),
            ("processed", self.processed_path),
            ("enhanced_11d", self.enhanced_path)
        ]:
            report["directories_exist"][name] = path.exists()
        
        # Check key files
        schema = self.get_storage_schema()
        for key, config in schema.items():
            file_path = self.base_path / config["path"]
            report["files_exist"][key] = file_path.exists()
            
            if file_path.exists():
                try:
                    # Basic schema validation
                    df = pd.read_parquet(file_path)
                    report["schema_validation"][key] = {
                        "rows": len(df),
                        "columns": list(df.columns),
                        "valid": True
                    }
                except Exception as e:
                    report["schema_validation"][key] = {
                        "error": str(e),
                        "valid": False
                    }
        
        return report
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Storage statistics
        """
        stats = {
            "total_size_mb": 0,
            "file_counts": {},
            "sample_counts": {}
        }
        
        # Calculate total size and file counts
        for path in self.base_path.rglob("*.parquet"):
            size_mb = path.stat().st_size / (1024 * 1024)
            stats["total_size_mb"] += size_mb
            
            # Categorize file
            if "enhanced_11d" in str(path):
                category = "enhanced_11d"
            elif "batches" in str(path):
                category = "batches"
            elif "processed" in str(path):
                category = "processed"
            elif "raw" in str(path):
                category = "raw"
            else:
                category = "other"
            
            stats["file_counts"][category] = stats["file_counts"].get(category, 0) + 1
        
        # Sample counts for key datasets
        schema = self.get_storage_schema()
        for key, config in schema.items():
            file_path = self.base_path / config["path"]
            if file_path.exists():
                try:
                    df = pd.read_parquet(file_path)
                    stats["sample_counts"][key] = len(df)
                except Exception:
                    stats["sample_counts"][key] = "error"
        
        return stats


# Global storage instance
_storage_instance = None

def get_storage_architecture(base_path: Optional[Path] = None) -> Storage11DArchitecture:
    """Get global storage architecture instance.
    
    Args:
        base_path: Optional base path (uses default if None)
        
    Returns:
        Storage11DArchitecture instance
    """
    global _storage_instance
    
    if _storage_instance is None:
        if base_path is None:
            base_path = Path("/Users/jack/projects/moola/data")
        _storage_instance = Storage11DArchitecture(base_path)
    
    return _storage_instance


# Convenience functions
def migrate_to_11d(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Migrate existing data to 11D format.
    
    Args:
        input_path: Path to existing 4D data
        output_path: Optional output path
        
    Returns:
        Path to migrated data
    """
    storage = get_storage_architecture()
    return storage.migrate_existing_data(input_path, output_path)


def create_11d_batch(window_ids: list, output_path: Optional[Path] = None) -> Path:
    """Create 11D annotation batch.
    
    Args:
        window_ids: List of window IDs
        output_path: Optional output path
        
    Returns:
        Path to created batch
    """
    storage = get_storage_architecture()
    return storage.create_annotation_batch_11d(window_ids, output_path)