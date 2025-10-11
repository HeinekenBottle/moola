"""Canonical v1 schema for training data using Pandera.

This schema defines the expected structure and types for processed/train.parquet.
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema


# Define the canonical training data schema
TrainingDataSchema = DataFrameSchema(
    {
        "window_id": Column(
            int,
            required=True,
            nullable=False,
            description="Unique identifier for the data window",
        ),
        "label": Column(
            str,
            required=True,
            nullable=False,
            description="Target label for classification",
        ),
        "features": Column(
            object,  # Lists are stored as objects in pandas
            required=True,
            nullable=False,
            description="Feature vector stored as list or array",
        ),
    },
    strict=False,  # Allow additional columns
    coerce=True,   # Coerce column types if possible
)


def validate_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate training data against canonical v1 schema.

    Args:
        df: DataFrame to validate

    Returns:
        Validated DataFrame with coerced types

    Raises:
        pandera.errors.SchemaError: If validation fails
    """
    return TrainingDataSchema.validate(df)


def check_training_data(df: pd.DataFrame) -> bool:
    """Check if training data conforms to schema without raising errors.

    Args:
        df: DataFrame to check

    Returns:
        True if valid, False otherwise
    """
    try:
        TrainingDataSchema.validate(df)
        return True
    except pa.errors.SchemaError:
        return False
