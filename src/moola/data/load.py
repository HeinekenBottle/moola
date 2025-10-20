"""Data loading utilities with validation."""
import pandas as pd


def validate_expansions(df):
    """Remove samples with invalid expansion indices.

    Filters out samples where:
    - expansion_start >= expansion_end (zero-length)
    - expansion_start or expansion_end outside valid range [30, 74]

    Args:
        df: DataFrame with 'expansion_start' and 'expansion_end' columns

    Returns:
        Cleaned DataFrame with valid expansion indices
    """
    valid = (
        (df['expansion_start'] < df['expansion_end']) &
        (df['expansion_start'] >= 30) & (df['expansion_start'] <= 74) &
        (df['expansion_end'] >= 30) & (df['expansion_end'] <= 74)
    )

    n_invalid = (~valid).sum()
    print(f"[DATA CLEAN] Removed {n_invalid}/115 invalid samples")
    print(f"[DATA CLEAN] Clean dataset: {valid.sum()} samples")

    return df[valid].reset_index(drop=True)
