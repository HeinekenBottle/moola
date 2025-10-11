"""Data validation schemas using Pydantic."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class DatasetSchema(BaseModel):
    """Schema for dataset validation."""

    name: str = Field(..., description="Dataset name")
    path: Path = Field(..., description="Path to dataset file")
    format: str = Field(default="csv", description="Dataset format (csv, parquet, json)")
    num_rows: Optional[int] = Field(None, description="Expected number of rows")
    num_cols: Optional[int] = Field(None, description="Expected number of columns")
    columns: Optional[List[str]] = Field(None, description="Expected column names")

    @field_validator("format")
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Validate dataset format."""
        allowed = {"csv", "parquet", "json", "feather"}
        if v not in allowed:
            raise ValueError(f"Format must be one of {allowed}, got {v}")
        return v


class ModelConfig(BaseModel):
    """Schema for model configuration."""

    model_type: str = Field(..., description="Type of model (e.g., 'random_forest', 'xgboost')")
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict, description="Model hyperparameters"
    )
    seed: int = Field(default=42, description="Random seed for reproducibility")
    cv_folds: int = Field(default=5, ge=2, description="Number of cross-validation folds")


class TrainingMetrics(BaseModel):
    """Schema for training metrics."""

    train_loss: Optional[float] = Field(None, description="Training loss")
    train_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Training accuracy")
    val_loss: Optional[float] = Field(None, description="Validation loss")
    val_accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Validation accuracy")
    epoch: Optional[int] = Field(None, description="Training epoch number")


class EvaluationMetrics(BaseModel):
    """Schema for evaluation metrics."""

    accuracy: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model accuracy")
    precision: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model precision")
    recall: Optional[float] = Field(None, ge=0.0, le=1.0, description="Model recall")
    f1_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="F1 score")
    auc: Optional[float] = Field(None, ge=0.0, le=1.0, description="Area under ROC curve")
    custom_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Additional custom metrics"
    )


class ModelArtifact(BaseModel):
    """Schema for model artifact metadata."""

    model_path: Path = Field(..., description="Path to serialized model")
    model_version: str = Field(..., description="Model version identifier")
    created_at: str = Field(..., description="Timestamp of model creation")
    metrics: EvaluationMetrics = Field(..., description="Model evaluation metrics")
    config: ModelConfig = Field(..., description="Model configuration used")


class DeploymentInfo(BaseModel):
    """Schema for deployment information."""

    deployment_id: str = Field(..., description="Unique deployment identifier")
    model_version: str = Field(..., description="Version of model being deployed")
    environment: str = Field(..., description="Deployment environment (dev, staging, prod)")
    endpoint_url: Optional[str] = Field(None, description="API endpoint URL if applicable")
    deployed_at: str = Field(..., description="Deployment timestamp")
    status: str = Field(default="pending", description="Deployment status")

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate deployment environment."""
        allowed = {"dev", "staging", "prod", "test"}
        if v not in allowed:
            raise ValueError(f"Environment must be one of {allowed}, got {v}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Validate deployment status."""
        allowed = {"pending", "running", "completed", "failed"}
        if v not in allowed:
            raise ValueError(f"Status must be one of {allowed}, got {v}")
        return v
