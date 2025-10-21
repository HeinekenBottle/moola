"""Manifest tracking utilities for artifact lineage and integrity."""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .hashing import compute_sha256


def get_git_sha() -> str:
    """Get current git commit SHA.

    Returns:
        Short git commit SHA (7 characters)
        Returns "unknown" if git command fails or not in a git repository
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha
    except Exception:
        return "unknown"


def compute_artifact_hashes(artifacts_dir: Path, patterns: list[str] = None) -> dict[str, str]:
    """Compute SHA256 hashes for all artifacts matching patterns.

    Args:
        artifacts_dir: Root artifacts directory
        patterns: List of glob patterns to match (default: ["**/*.pkl", "**/*.npy", "**/*.json"])

    Returns:
        Dictionary mapping relative file paths to SHA256 hashes
    """
    if patterns is None:
        patterns = ["**/*.pkl", "**/*.npy", "**/*.json"]

    hashes = {}
    for pattern in patterns:
        for file_path in artifacts_dir.glob(pattern):
            if file_path.is_file():
                # Skip manifest.json itself
                if file_path.name == "manifest.json":
                    continue
                relative_path = str(file_path.relative_to(artifacts_dir))
                try:
                    file_hash = compute_sha256(file_path)
                    hashes[relative_path] = file_hash
                except Exception as e:
                    # Log error but continue processing other files
                    hashes[relative_path] = f"error: {str(e)}"

    return hashes


def create_manifest(
    artifacts_dir: Path,
    models: list[str] | None = None,
    additional_metadata: dict | None = None,
) -> dict:
    """Create manifest with artifact hashes and metadata.

    Args:
        artifacts_dir: Root artifacts directory
        models: List of model names included in this manifest
        additional_metadata: Additional metadata to include

    Returns:
        Manifest dictionary containing:
            - created_at: ISO 8601 timestamp
            - git_sha: Current git commit SHA
            - models: List of model names
            - artifacts: Dict of {file_path: sha256_hash}
            - Additional metadata if provided
    """
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": get_git_sha(),
        "models": models or [],
        "artifacts": compute_artifact_hashes(artifacts_dir),
    }

    if additional_metadata:
        manifest.update(additional_metadata)

    return manifest


def write_manifest(manifest_path: Path, manifest: dict) -> None:
    """Write manifest to JSON file.

    Args:
        manifest_path: Path to manifest.json file
        manifest: Manifest dictionary to write
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)


def read_manifest(manifest_path: Path) -> dict | None:
    """Read manifest from JSON file.

    Args:
        manifest_path: Path to manifest.json file

    Returns:
        Manifest dictionary if file exists, None otherwise
    """
    if not manifest_path.exists():
        return None

    with open(manifest_path, "r") as f:
        return json.load(f)


def verify_manifest(artifacts_dir: Path, manifest_path: Path) -> dict[str, bool]:
    """Verify artifact hashes match manifest.

    Args:
        artifacts_dir: Root artifacts directory
        manifest_path: Path to manifest.json file

    Returns:
        Dictionary mapping file paths to verification status (True if valid, False if mismatch)
        Returns empty dict if manifest doesn't exist
    """
    manifest = read_manifest(manifest_path)
    if manifest is None:
        return {}

    verification = {}
    artifact_hashes = manifest.get("artifacts", {})

    for relative_path, expected_hash in artifact_hashes.items():
        file_path = artifacts_dir / relative_path
        if not file_path.exists():
            verification[relative_path] = False
        else:
            try:
                actual_hash = compute_sha256(file_path)
                verification[relative_path] = actual_hash == expected_hash
            except Exception:
                verification[relative_path] = False

    return verification
