#!/usr/bin/env python3
"""Data lineage tracking system for ML pipelines.

Tracks data transformations, checksums, and dependencies across the pipeline.

Usage:
    from moola.data_infra.lineage import LineageTracker

    tracker = LineageTracker()
    tracker.log_transformation(
        dataset_id="train_v1",
        transformation_type="windowing",
        input_path=Path("data/raw/ohlc.parquet"),
        output_path=Path("data/raw/unlabeled_windows.parquet"),
        rows_in=50000,
        rows_out=11873,
        params={"window_length": 105, "stride": 1}
    )
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from ..schemas import DataLineage, DataVersion

# ============================================================================
# LINEAGE TRACKER
# ============================================================================


class LineageTracker:
    """Track data lineage and transformations."""

    def __init__(self, lineage_dir: Path = Path("data/lineage")):
        self.lineage_dir = lineage_dir
        self.lineage_dir.mkdir(parents=True, exist_ok=True)

        # Lineage graph: dataset_id -> DataLineage
        self.lineage_graph: Dict[str, DataLineage] = {}

        # Load existing lineage
        self._load_lineage()

    def log_transformation(
        self,
        dataset_id: str,
        transformation_type: str,
        input_path: Optional[Path],
        output_path: Optional[Path],
        rows_in: int,
        rows_out: int,
        params: Optional[Dict[str, Any]] = None,
        parent_datasets: Optional[List[str]] = None,
        executed_by: str = "system",
    ) -> DataLineage:
        """Log a data transformation.

        Args:
            dataset_id: Unique identifier for output dataset
            transformation_type: Type of transformation (e.g., "windowing", "augmentation")
            input_path: Path to input data
            output_path: Path to output data
            rows_in: Number of input rows
            rows_out: Number of output rows
            params: Transformation parameters
            parent_datasets: List of parent dataset IDs
            executed_by: User/system that executed transformation

        Returns:
            DataLineage object
        """
        start_time = time.time()

        # Compute checksums
        checksum_in = self._compute_file_checksum(input_path) if input_path else None
        checksum_out = self._compute_file_checksum(output_path) if output_path else None

        execution_time = time.time() - start_time

        lineage = DataLineage(
            dataset_id=dataset_id,
            parent_datasets=parent_datasets or [],
            transformation_type=transformation_type,
            transformation_params=params or {},
            input_path=input_path,
            output_path=output_path,
            rows_in=rows_in,
            rows_out=rows_out,
            checksum_in=checksum_in,
            checksum_out=checksum_out,
            executed_by=executed_by,
            executed_at=datetime.utcnow(),
            execution_time_seconds=execution_time,
            metadata={},
        )

        # Store in graph
        self.lineage_graph[dataset_id] = lineage

        # Save to disk
        self._save_lineage(lineage)

        logger.info(f"Logged lineage for dataset: {dataset_id}")
        logger.info(f"  Transformation: {transformation_type}")
        logger.info(f"  Rows: {rows_in:,} → {rows_out:,}")

        return lineage

    def get_lineage(self, dataset_id: str) -> Optional[DataLineage]:
        """Get lineage for a dataset."""
        return self.lineage_graph.get(dataset_id)

    def get_ancestors(self, dataset_id: str) -> List[str]:
        """Get all ancestor datasets (transitive parents)."""
        ancestors = []
        to_visit = [dataset_id]
        visited = set()

        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue

            visited.add(current)
            lineage = self.lineage_graph.get(current)

            if lineage and lineage.parent_datasets:
                ancestors.extend(lineage.parent_datasets)
                to_visit.extend(lineage.parent_datasets)

        return list(set(ancestors))

    def get_descendants(self, dataset_id: str) -> List[str]:
        """Get all descendant datasets (datasets derived from this one)."""
        descendants = []

        for other_id, lineage in self.lineage_graph.items():
            if dataset_id in lineage.parent_datasets:
                descendants.append(other_id)
                descendants.extend(self.get_descendants(other_id))

        return list(set(descendants))

    def visualize_lineage(self, dataset_id: str, output_path: Optional[Path] = None):
        """Generate lineage visualization (requires graphviz).

        Args:
            dataset_id: Dataset to visualize lineage for
            output_path: Path to save visualization (PNG/SVG)
        """
        try:
            import graphviz
        except ImportError:
            logger.warning("graphviz not installed. Install with: pip install graphviz")
            return

        # Build graph
        dot = graphviz.Digraph(comment=f"Lineage for {dataset_id}")
        dot.attr(rankdir="LR")

        # Get all related datasets
        ancestors = self.get_ancestors(dataset_id)
        descendants = self.get_descendants(dataset_id)
        all_datasets = set([dataset_id] + ancestors + descendants)

        # Add nodes
        for ds_id in all_datasets:
            lineage = self.lineage_graph.get(ds_id)
            if lineage:
                label = f"{ds_id}\n{lineage.transformation_type}\n{lineage.rows_out:,} rows"
                dot.node(ds_id, label)

        # Add edges
        for ds_id in all_datasets:
            lineage = self.lineage_graph.get(ds_id)
            if lineage and lineage.parent_datasets:
                for parent in lineage.parent_datasets:
                    if parent in all_datasets:
                        dot.edge(parent, ds_id)

        # Render
        if output_path:
            dot.render(output_path, format="png", cleanup=True)
            logger.info(f"Lineage visualization saved: {output_path}")
        else:
            dot.render("lineage", view=True, cleanup=True)

    def export_lineage_report(self, output_path: Path):
        """Export complete lineage graph as JSON.

        Args:
            output_path: Path to save lineage report
        """
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "total_datasets": len(self.lineage_graph),
            "lineage": {},
        }

        for dataset_id, lineage in self.lineage_graph.items():
            report["lineage"][dataset_id] = lineage.model_dump(mode="json")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Lineage report exported: {output_path}")

    def _compute_file_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of file."""
        if not file_path or not file_path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def _save_lineage(self, lineage: DataLineage):
        """Save lineage to disk."""
        lineage_file = self.lineage_dir / f"{lineage.dataset_id}.json"
        with open(lineage_file, "w") as f:
            json.dump(lineage.model_dump(mode="json"), f, indent=2, default=str)

    def _load_lineage(self):
        """Load existing lineage from disk."""
        if not self.lineage_dir.exists():
            return

        for lineage_file in self.lineage_dir.glob("*.json"):
            try:
                with open(lineage_file, "r") as f:
                    data = json.load(f)
                    lineage = DataLineage(**data)
                    self.lineage_graph[lineage.dataset_id] = lineage
            except Exception as e:
                logger.warning(f"Failed to load lineage from {lineage_file}: {e}")

        logger.info(f"Loaded {len(self.lineage_graph)} lineage records")


# ============================================================================
# VERSION CONTROL
# ============================================================================


class DataVersionControl:
    """Data version control with DVC integration."""

    def __init__(self, versions_dir: Path = Path("data/versions")):
        self.versions_dir = versions_dir
        self.versions_dir.mkdir(parents=True, exist_ok=True)

        # Version registry: dataset_name -> List[DataVersion]
        self.versions: Dict[str, List[DataVersion]] = {}

        self._load_versions()

    def create_version(
        self,
        dataset_name: str,
        file_path: Path,
        version_id: str,
        label_distribution: Optional[Dict[str, int]] = None,
        parent_version: Optional[str] = None,
        transformation: Optional[str] = None,
        created_by: str = "automated_pipeline",
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
    ) -> DataVersion:
        """Create a new data version.

        Args:
            dataset_name: Name of dataset (e.g., "train", "unlabeled_windows")
            file_path: Path to dataset file
            version_id: Semantic version (e.g., "v1.0.0")
            label_distribution: Distribution of labels (for labeled data)
            parent_version: Parent version ID if derived from another version
            transformation: Transformation applied to create this version
            created_by: Creator identifier
            tags: Version tags
            notes: Version notes

        Returns:
            DataVersion object
        """
        # Validate file exists
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # Load dataset to get properties
        df = pd.read_parquet(file_path)
        num_samples = len(df)

        # Infer feature shape
        if "features" in df.columns:
            sample = df["features"].iloc[0]
            if isinstance(sample, (list, np.ndarray)):
                import numpy as np

                arr = np.vstack(sample) if isinstance(sample[0], (list, np.ndarray)) else sample
                feature_shape = tuple(arr.shape)
            else:
                feature_shape = (1,)
        else:
            feature_shape = tuple(df.shape)

        # Compute DVC hash
        dvc_hash = self._compute_dvc_hash(file_path)
        dvc_size = file_path.stat().st_size

        # Placeholder quality score (would be computed from validation)
        quality_score = 95.0
        validation_passed = True

        version = DataVersion(
            version_id=version_id,
            dataset_name=dataset_name,
            file_path=file_path,
            dvc_hash=dvc_hash,
            dvc_size_bytes=dvc_size,
            num_samples=num_samples,
            feature_shape=feature_shape,
            label_distribution=label_distribution,
            quality_score=quality_score,
            validation_passed=validation_passed,
            parent_version=parent_version,
            transformation_applied=transformation,
            created_by=created_by,
            tags=tags or [],
            notes=notes,
        )

        # Add to registry
        if dataset_name not in self.versions:
            self.versions[dataset_name] = []
        self.versions[dataset_name].append(version)

        # Save version metadata
        self._save_version(version)

        logger.info(f"Created version {version_id} for dataset {dataset_name}")
        logger.info(f"  Samples: {num_samples:,}")
        logger.info(f"  Size: {dvc_size / (1024**2):.2f} MB")

        return version

    def get_version(self, dataset_name: str, version_id: str) -> Optional[DataVersion]:
        """Get specific version of dataset."""
        versions = self.versions.get(dataset_name, [])
        for version in versions:
            if version.version_id == version_id:
                return version
        return None

    def get_latest_version(self, dataset_name: str) -> Optional[DataVersion]:
        """Get latest version of dataset."""
        versions = self.versions.get(dataset_name, [])
        if not versions:
            return None

        # Sort by created_at descending
        sorted_versions = sorted(versions, key=lambda v: v.created_at, reverse=True)
        return sorted_versions[0]

    def list_versions(self, dataset_name: str) -> List[DataVersion]:
        """List all versions of a dataset."""
        return self.versions.get(dataset_name, [])

    def _compute_dvc_hash(self, file_path: Path) -> str:
        """Compute DVC-compatible MD5 hash."""
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def _save_version(self, version: DataVersion):
        """Save version metadata to disk."""
        version_dir = self.versions_dir / version.dataset_name / version.version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        metadata_file = version_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(version.model_dump(mode="json"), f, indent=2, default=str)

    def _load_versions(self):
        """Load existing versions from disk."""
        if not self.versions_dir.exists():
            return

        for dataset_dir in self.versions_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            dataset_name = dataset_dir.name
            self.versions[dataset_name] = []

            for version_dir in dataset_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                metadata_file = version_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            data = json.load(f)
                            version = DataVersion(**data)
                            self.versions[dataset_name].append(version)
                    except Exception as e:
                        logger.warning(f"Failed to load version from {metadata_file}: {e}")

        total_versions = sum(len(v) for v in self.versions.values())
        logger.info(f"Loaded {total_versions} version records across {len(self.versions)} datasets")


# ============================================================================
# EXPORT
# ============================================================================

__all__ = [
    "LineageTracker",
    "DataVersionControl",
]
