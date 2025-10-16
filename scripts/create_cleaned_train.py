"""Create cleaned train.parquet matching OOF validation."""
import pandas as pd
from pathlib import Path


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
    n_valid = valid.sum()

    print(f"[DATA CLEAN] Removed {n_invalid} invalid samples from {len(df)} total")
    print(f"[DATA CLEAN] Clean dataset: {n_valid} samples")

    return df[valid].reset_index(drop=True)


if __name__ == '__main__':
    # Load original train.parquet
    train_path = Path('/Users/jack/projects/moola/data/processed/train.parquet')
    output_path = Path('/Users/jack/projects/moola/data/processed/train_clean.parquet')

    print(f"Loading {train_path}")
    df = pd.read_parquet(train_path)
    print(f"Original dataset: {len(df)} samples")

    # Apply validation (same as OOF generation)
    df_clean = validate_expansions(df)

    # Save cleaned version
    df_clean.to_parquet(output_path, index=False)
    print(f"\nSaved cleaned dataset to {output_path}")
    print(f"Shape: {df_clean.shape}")
    print(f"Columns: {list(df_clean.columns)}")
