"""FastAPI serving endpoint for Moola pattern recognition model.

Production-ready API for serving the stacking ensemble model with:
- Health checks
- Input validation
- Confidence thresholds
- Prometheus metrics
- Error handling

Usage:
    # Development
    uvicorn moola.api.serve:app --reload --port 8000

    # Production
    gunicorn -w 4 -k uvicorn.workers.UvicornWorker moola.api.serve:app --bind 0.0.0.0:8000

Endpoints:
    GET  /health         - Health check
    GET  /metrics        - Prometheus metrics
    POST /predict        - Generate predictions for OHLC window
    POST /predict/batch  - Batch prediction for multiple windows
"""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, field_validator

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "predictions_total",
    "Total number of predictions made",
    ["model_type", "pattern"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)
CONFIDENCE_GAUGE = Gauge(
    "prediction_confidence",
    "Average prediction confidence (last 100 predictions)"
)
ERROR_COUNTER = Counter(
    "prediction_errors_total",
    "Total number of prediction errors",
    ["error_type"]
)
MODEL_LOAD_TIME = Gauge(
    "model_load_time_seconds",
    "Time taken to load model at startup"
)

# FastAPI app
app = FastAPI(
    title="Moola Pattern Recognition API",
    description="Production API for NQ 1-min pattern classification",
    version="1.0.0",
)

# Global model instance (loaded at startup)
model = None
model_metadata = {}
recent_confidences = []


class PredictionRequest(BaseModel):
    """Single prediction request for one OHLC window.

    Expected input: 105-bar window with OHLC values
    """
    features: List[List[float]] = Field(
        ...,
        description="105-bar OHLC window as list of [open, high, low, close]",
        min_length=105,
        max_length=105
    )

    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        """Validate feature dimensions."""
        if len(v) != 105:
            raise ValueError(f"Expected 105 bars, got {len(v)}")

        for i, bar in enumerate(v):
            if len(bar) != 4:
                raise ValueError(f"Bar {i} has {len(bar)} features, expected 4 (OHLC)")
            if not all(isinstance(x, (int, float)) for x in bar):
                raise ValueError(f"Bar {i} contains non-numeric values")

        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "features": [
                        [4500.0, 4505.0, 4498.0, 4502.0],  # Bar 0: OHLC
                        [4502.0, 4508.0, 4501.0, 4506.0],  # Bar 1: OHLC
                        # ... (103 more bars)
                    ]
                }
            ]
        }
    }


class BatchPredictionRequest(BaseModel):
    """Batch prediction request for multiple windows."""
    windows: List[List[List[float]]] = Field(
        ...,
        description="List of 105-bar OHLC windows",
        min_length=1,
        max_length=100  # Limit batch size
    )


class PredictionResponse(BaseModel):
    """Response for single prediction."""
    prediction: str = Field(..., description="Predicted pattern (consolidation or retracement)")
    confidence: float = Field(..., description="Confidence score (0-1)")
    probabilities: dict = Field(..., description="Class probabilities")
    timestamp: str = Field(..., description="Prediction timestamp (ISO 8601)")
    model_version: str = Field(..., description="Model version/hash")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prediction": "consolidation",
                    "confidence": 0.87,
                    "probabilities": {
                        "consolidation": 0.87,
                        "retracement": 0.13
                    },
                    "timestamp": "2025-10-16T12:34:56Z",
                    "model_version": "v1.0.0-abc123"
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    """Response for batch prediction."""
    predictions: List[PredictionResponse]
    batch_size: int
    total_latency: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str
    version: str


@app.on_event("startup")
async def load_model():
    """Load production model at startup."""
    global model, model_metadata

    start_time = time.time()

    try:
        # Load from artifacts directory
        # Assumes model is saved at: data/artifacts/models/stack/stack.pkl
        model_path = Path("data/artifacts/models/stack/stack.pkl")

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")

        model = joblib.load(model_path)

        # Load metadata (if available)
        metadata_path = model_path.parent / "metrics.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)

        load_time = time.time() - start_time
        MODEL_LOAD_TIME.set(load_time)

        print(f"✓ Model loaded in {load_time:.2f}s")
        print(f"  Path: {model_path}")
        print(f"  Metadata: {model_metadata}")

    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0"
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Generate prediction for single OHLC window.

    Args:
        request: Prediction request with 105-bar OHLC window

    Returns:
        PredictionResponse with pattern prediction and confidence

    Raises:
        HTTPException: If model not loaded or prediction fails
    """
    if model is None:
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    start_time = time.time()

    try:
        # Reshape input to [1, 105, 4]
        X = np.array(request.features).reshape(1, 105, 4)

        # Predict
        proba = model.predict_proba(X)[0]
        pred_class_idx = int(proba.argmax())
        confidence = float(proba.max())

        # Get class labels (assumes binary classification)
        class_labels = ["consolidation", "retracement"]
        prediction = class_labels[pred_class_idx]

        # Build probability dict
        probabilities = {
            label: float(proba[i])
            for i, label in enumerate(class_labels)
        }

        # Track metrics
        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)
        PREDICTION_COUNTER.labels(model_type="stack", pattern=prediction).inc()

        # Track confidence (rolling window of 100)
        recent_confidences.append(confidence)
        if len(recent_confidences) > 100:
            recent_confidences.pop(0)
        CONFIDENCE_GAUGE.set(sum(recent_confidences) / len(recent_confidences))

        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=datetime.utcnow().isoformat() + "Z",
            model_version=model_metadata.get("model", "unknown")
        )

    except Exception as e:
        ERROR_COUNTER.labels(error_type="prediction_error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """Generate predictions for multiple OHLC windows.

    Args:
        request: Batch prediction request with list of windows

    Returns:
        BatchPredictionResponse with list of predictions

    Raises:
        HTTPException: If model not loaded or batch prediction fails
    """
    if model is None:
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )

    start_time = time.time()

    try:
        # Process each window
        predictions = []
        for window in request.windows:
            window_request = PredictionRequest(features=window)
            prediction = await predict(window_request)
            predictions.append(prediction)

        total_latency = time.time() - start_time

        return BatchPredictionResponse(
            predictions=predictions,
            batch_size=len(predictions),
            total_latency=total_latency
        )

    except Exception as e:
        ERROR_COUNTER.labels(error_type="batch_prediction_error").inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
