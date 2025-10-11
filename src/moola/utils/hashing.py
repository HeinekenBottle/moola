"""File hashing utilities for artifact tracking and integrity verification."""

import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of a file.

    Args:
        file_path: Path to file to hash
        chunk_size: Size of chunks to read (default 8192 bytes)

    Returns:
        Hexadecimal SHA256 hash string

    Raises:
        FileNotFoundError: If file does not exist
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read file in chunks to handle large files efficiently
        while chunk := f.read(chunk_size):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def compute_sha256_bytes(data: bytes) -> str:
    """Compute SHA256 hash of byte data.

    Args:
        data: Bytes to hash

    Returns:
        Hexadecimal SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(data)
    return sha256_hash.hexdigest()


def verify_sha256(file_path: Union[str, Path], expected_hash: str) -> bool:
    """Verify file SHA256 hash matches expected value.

    Args:
        file_path: Path to file to verify
        expected_hash: Expected hexadecimal SHA256 hash

    Returns:
        True if hash matches, False otherwise
    """
    try:
        actual_hash = compute_sha256(file_path)
        return actual_hash.lower() == expected_hash.lower()
    except FileNotFoundError:
        return False
