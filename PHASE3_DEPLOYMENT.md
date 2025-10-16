# Phase 3: SSL Pre-training and Production Deployment

This document describes the Phase 3 implementation for the Moola ML pipeline, which includes:
1. TS-TCC self-supervised pre-training
2. End-to-end training automation
3. Production API deployment
4. Monitoring and observability

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [SSL Pre-training](#ssl-pre-training)
- [Training Automation](#training-automation)
- [API Deployment](#api-deployment)
- [Monitoring](#monitoring)
- [Expected Performance](#expected-performance)
- [Usage Guide](#usage-guide)

---

## Architecture Overview

### Phase 3 Enhancements

**SSL Pre-training Pipeline**:
```
Raw 1-min data (unlabeled)
  ↓
Extract non-overlapping windows (118k samples)
  ↓
TS-TCC contrastive pre-training (InfoNCE loss)
  ↓
Pre-trained encoder weights
  ↓
Fine-tune on labeled data (98 samples)
```

**Production Deployment Stack**:
```
FastAPI (model serving)
  ↓
Prometheus (metrics collection)
  ↓
Grafana (visualization)
  ↓
Docker/Docker Compose (containerization)
```

---

## SSL Pre-training

### TS-TCC Architecture

**Encoder Structure**:
- Multi-scale CNN blocks (kernels: 3, 5, 9)
- Transformer encoder (3 layers, 4 heads)
- Relative positional encoding
- Projection head for contrastive learning

**Training Configuration**:
- Loss: InfoNCE (temperature=0.5)
- Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
- Batch size: 512 (RTX 4090 optimized)
- Epochs: 100 with early stopping (patience=15)
- Augmentation: Temporal jitter, scaling, time warp
- Mixed precision: FP16 for GPU acceleration

### Pre-training Workflow

**1. Unlabeled Data Extraction** (Already Done):
```bash
# Unlabeled windows already available at:
data/raw/unlabeled_windows.parquet
```

**2. Pre-train TS-TCC Encoder**:
```bash
python -m moola.cli pretrain-tcc \
    --device cuda \
    --epochs 100 \
    --patience 15
```

Output:
- `data/artifacts/models/ts_tcc/pretrained_encoder.pt`
- Contains: encoder weights + hyperparameters

**3. Fine-tune with Pre-trained Encoder**:

The CNN-Transformer model supports loading pre-trained weights:

```python
from moola.models.cnn_transformer import CnnTransformerModel

# Initialize model
model = CnnTransformerModel(device="cuda")

# Load pre-trained encoder
model.load_pretrained_encoder("data/artifacts/models/ts_tcc/pretrained_encoder.pt")

# Fine-tune on labeled data
model.fit(X_train, y_train)
```

**Note**: SimpleLSTM uses LSTM architecture and cannot use CNN encoder weights.

### Expected SSL Improvements

| Metric | Without SSL | With SSL | Gain |
|--------|-------------|----------|------|
| Accuracy | 65-74% | 75-82% | +8-12% |
| Training time | 5 min | 11 min | +6 min |
| GPU cost | $2 | $8 | +$6 |

---

## Training Automation

### End-to-End Pipeline Script

**Location**: `scripts/train_full_pipeline.py`

**Features**:
- Phase 1 verification (SimpleLSTM, data checks)
- Phase 2 OOF generation (all base models)
- Phase 3 SSL pre-training (TS-TCC)
- Phase 4 stack ensemble training
- Comprehensive logging and error handling

**Usage**:

```bash
# Full pipeline (all phases)
python scripts/train_full_pipeline.py --device cuda

# Skip specific phases
python scripts/train_full_pipeline.py \
    --device cuda \
    --skip-phase2 \
    --skip-phase3

# Custom SSL configuration
python scripts/train_full_pipeline.py \
    --device cuda \
    --ssl-epochs 200 \
    --ssl-patience 20
```

**Command-line Options**:
```
--device {cpu,cuda}       Device for training (default: cuda)
--seed INT                Random seed (default: 1337)
--mlflow-experiment NAME  MLflow experiment name
--skip-phase1             Skip Phase 1 verification
--skip-phase2             Skip Phase 2 OOF generation
--skip-phase3             Skip Phase 3 SSL pre-training
--skip-phase4             Skip Phase 4 ensemble training
--ssl-epochs INT          SSL pre-training epochs (default: 100)
--ssl-patience INT        SSL early stopping patience (default: 15)
```

**Output**:
```
data/artifacts/models/
├── logreg/
│   ├── model.pkl
│   └── metrics.json
├── rf/
│   ├── model.pkl
│   └── metrics.json
├── xgb/
│   ├── model.pkl
│   └── metrics.json
├── simple_lstm/
│   ├── model.pkl
│   └── metrics.json
├── cnn_transformer/
│   ├── model.pkl
│   └── metrics.json
├── stack/
│   ├── stack.pkl
│   └── metrics.json
└── ts_tcc/
    └── pretrained_encoder.pt
```

---

## API Deployment

### FastAPI Serving

**Location**: `src/moola/api/serve.py`

**Features**:
- Production-ready REST API
- Input validation (Pydantic)
- Prometheus metrics integration
- Health checks
- Batch prediction support
- Error handling and logging

### Endpoints

**1. Health Check** (`GET /health`):
```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-16T12:34:56Z",
  "version": "1.0.0"
}
```

**2. Single Prediction** (`POST /predict`):
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [4500.0, 4505.0, 4498.0, 4502.0],
      [4502.0, 4508.0, 4501.0, 4506.0],
      ... (105 bars total)
    ]
  }'
```

Response:
```json
{
  "prediction": "consolidation",
  "confidence": 0.87,
  "probabilities": {
    "consolidation": 0.87,
    "retracement": 0.13
  },
  "timestamp": "2025-10-16T12:34:56Z",
  "model_version": "v1.0.0"
}
```

**3. Batch Prediction** (`POST /predict/batch`):
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "windows": [
      [[4500.0, 4505.0, 4498.0, 4502.0], ...],
      [[4510.0, 4515.0, 4508.0, 4512.0], ...]
    ]
  }'
```

**4. Prometheus Metrics** (`GET /metrics`):
```bash
curl http://localhost:8000/metrics
```

### Running the API

**Development**:
```bash
# Start development server
uvicorn moola.api.serve:app --reload --port 8000

# Test API
curl http://localhost:8000/health
```

**Production (Docker)**:
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

**Production (Kubernetes)** (TODO):
```bash
# Deploy to k8s cluster
kubectl apply -f k8s/deployment.yml
kubectl apply -f k8s/service.yml
kubectl apply -f k8s/ingress.yml
```

---

## Monitoring

### Stack Overview

1. **Prometheus**: Metrics collection and storage
2. **Grafana**: Visualization and dashboards
3. **FastAPI**: Application metrics exposure

### Metrics Collected

**Prediction Metrics**:
- `predictions_total` (counter): Total predictions by pattern type
- `prediction_latency_seconds` (histogram): Prediction latency distribution
- `prediction_confidence` (gauge): Average confidence (rolling window)

**Error Metrics**:
- `prediction_errors_total` (counter): Total errors by error type

**System Metrics**:
- `model_load_time_seconds` (gauge): Model initialization time

### Grafana Dashboard

**Access**: http://localhost:3000 (admin/admin)

**Panels**:
1. Predictions per second (time series)
2. Prediction latency (p50, p95)
3. Average confidence (gauge)
4. Error rate (time series)
5. Pattern distribution (pie chart)
6. Model load time (stat)
7. API health status (stat)

### Setting Up Monitoring

**1. Start monitoring stack**:
```bash
docker-compose up -d prometheus grafana
```

**2. Access Grafana**:
- URL: http://localhost:3000
- Username: `admin`
- Password: `admin`

**3. Import dashboard**:
- Navigate to Dashboards → Import
- Upload `monitoring/grafana/dashboards/moola-ml-dashboard.json`

**4. View metrics**:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

---

## Expected Performance

### Accuracy Targets

| Configuration | Accuracy | Notes |
|---------------|----------|-------|
| Phase 2 baseline | 65-74% | No SSL |
| Phase 3 with SSL | 75-82% | +8-12% improvement |
| Production ensemble | 77-85% | Stack meta-learner |

### Latency Benchmarks

| Operation | Latency | Notes |
|-----------|---------|-------|
| Model load | 2-5s | At startup |
| Single prediction | 5-15ms | GPU |
| Batch prediction (10) | 25-50ms | GPU |
| API overhead | 1-2ms | FastAPI |

### Resource Requirements

**Training** (full pipeline):
- GPU: RTX 4090 (24GB VRAM)
- Time: ~15 minutes
- Cost: ~$8 (RunPod)

**Inference** (production):
- CPU: 2 cores
- RAM: 2GB
- GPU: Optional (10x speedup)
- Latency: <50ms (p95)

---

## Usage Guide

### Complete Workflow

**1. Prepare Environment**:
```bash
# Install dependencies
pip install -r requirements.txt

# Verify data
ls data/processed/train_clean.parquet
ls data/raw/unlabeled_windows.parquet
```

**2. Train Models**:
```bash
# Full pipeline (Phase 1-4)
python scripts/train_full_pipeline.py --device cuda

# Or step-by-step:
# Step 1: Pre-train TS-TCC
python -m moola.cli pretrain-tcc --device cuda

# Step 2: Generate OOF predictions
python -m moola.cli oof --model cnn_transformer --device cuda
python -m moola.cli oof --model simple_lstm --device cuda
# ... (other models)

# Step 3: Train stack ensemble
python -m moola.cli stack-train --seed 1337
```

**3. Deploy API**:
```bash
# Start all services (API + monitoring)
docker-compose up -d

# Check health
curl http://localhost:8000/health

# View metrics
open http://localhost:9090  # Prometheus
open http://localhost:3000  # Grafana
```

**4. Make Predictions**:
```python
import requests
import numpy as np

# Generate test window
window = np.random.randn(105, 4).tolist()

# Predict
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": window}
)

print(response.json())
# {
#   "prediction": "consolidation",
#   "confidence": 0.87,
#   "probabilities": {"consolidation": 0.87, "retracement": 0.13},
#   ...
# }
```

**5. Monitor Production**:
```bash
# View Grafana dashboard
open http://localhost:3000

# Query Prometheus
curl http://localhost:9090/api/v1/query?query=predictions_total

# View API logs
docker-compose logs -f api
```

### Troubleshooting

**Model not loading**:
```bash
# Check model exists
ls data/artifacts/models/stack/stack.pkl

# Check logs
docker-compose logs api

# Rebuild container
docker-compose build api
docker-compose up -d api
```

**GPU not detected**:
```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU fallback
python scripts/train_full_pipeline.py --device cpu
```

**Metrics not appearing**:
```bash
# Restart Prometheus
docker-compose restart prometheus

# Check Prometheus targets
open http://localhost:9090/targets
```

---

## Next Steps

1. **Production Hardening**:
   - Add authentication (JWT/API keys)
   - Implement rate limiting
   - Add request/response logging
   - Set up alerting (AlertManager)

2. **Performance Optimization**:
   - Model quantization (FP16/INT8)
   - Batch prediction optimization
   - Model caching strategies
   - GPU memory optimization

3. **MLOps Integration**:
   - Model versioning (MLflow Model Registry)
   - A/B testing framework
   - Continuous retraining pipeline
   - Data drift detection

4. **Infrastructure**:
   - Kubernetes deployment
   - Auto-scaling configuration
   - Multi-region deployment
   - Disaster recovery plan

---

## References

- TS-TCC Paper: https://arxiv.org/abs/2106.14112
- FastAPI Docs: https://fastapi.tiangolo.com
- Prometheus Guide: https://prometheus.io/docs
- Grafana Tutorials: https://grafana.com/tutorials
- Docker Compose: https://docs.docker.com/compose

---

**Generated**: 2025-10-16
**Author**: Claude Code (Anthropic)
**Version**: 1.0.0
