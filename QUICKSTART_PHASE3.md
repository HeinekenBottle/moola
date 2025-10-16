# Phase 3 Quick Start Guide

Get the Moola ML pipeline running in production in 5 minutes.

---

## Prerequisites

- Python 3.10+
- Docker & Docker Compose
- CUDA-capable GPU (recommended)
- 16GB RAM minimum

---

## Step 1: Verify Installation

```bash
cd /Users/jack/projects/moola

# Check data availability
ls data/processed/train_clean.parquet
ls data/raw/unlabeled_windows.parquet

# Install dependencies
pip install -r requirements.txt
```

---

## Step 2: Train Models (Full Pipeline)

```bash
# Run complete training pipeline
python scripts/train_full_pipeline.py --device cuda

# This will:
# 1. Verify Phase 1 fixes ✓
# 2. Generate OOF predictions ✓
# 3. Pre-train TS-TCC encoder ✓
# 4. Train stack ensemble ✓
```

**Expected Output**:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    MOOLA PRODUCTION PIPELINE                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

PHASE 1: Verifying critical fixes
✓ SimpleLSTM loaded (72,066 params)
✓ Training data exists

PHASE 2: Generating OOF predictions with augmentation
✓ OOF generation for logreg
✓ OOF generation for rf
✓ OOF generation for xgb
✓ OOF generation for simple_lstm
✓ OOF generation for cnn_transformer

PHASE 3: SSL pre-training with TS-TCC
✓ TS-TCC encoder pre-training

PHASE 4: Training stack ensemble
✓ Stack ensemble training

🎉 PIPELINE COMPLETE!
```

**Time**: ~20-30 minutes on RTX 4090

---

## Step 3: Deploy API

```bash
# Start all services (API + Prometheus + Grafana)
docker-compose up -d

# View logs
docker-compose logs -f api

# Wait for startup (check health)
curl http://localhost:8000/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-10-16T12:34:56Z",
  "version": "1.0.0"
}
```

---

## Step 4: Test Predictions

```bash
# Run deployment tests
python scripts/test_deployment.py

# Expected:
# ✓ PASS | health
# ✓ PASS | single_prediction
# ✓ PASS | batch_prediction
# ✓ PASS | metrics
# ✓ PASS | latency_benchmark
#
# 🎉 All tests passed! Deployment is healthy.
```

---

## Step 5: Access Monitoring

**Grafana Dashboard**: http://localhost:3000
- Username: `admin`
- Password: `admin`
- Navigate to: Dashboards → Moola ML - Production Monitoring

**Prometheus**: http://localhost:9090
- Query: `predictions_total`
- Query: `rate(predictions_total[1m])`

---

## Making Predictions

### Python Client

```python
import requests
import numpy as np

# Generate test window (105 bars × 4 OHLC)
window = np.random.randn(105, 4).tolist()

# Predict
response = requests.post(
    "http://localhost:8000/predict",
    json={"features": window}
)

result = response.json()
print(f"Pattern: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      [4500.0, 4505.0, 4498.0, 4502.0],
      ... (105 bars total)
    ]
  }'
```

---

## Troubleshooting

### Model not loading

```bash
# Check model exists
ls data/artifacts/models/stack/stack.pkl

# If missing, retrain
python scripts/train_full_pipeline.py --skip-phase3 --device cuda
```

### Docker issues

```bash
# Rebuild containers
docker-compose down
docker-compose build --no-cache
docker-compose up -d

# Check logs
docker-compose logs api
```

### GPU not detected

```bash
# Verify CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Use CPU fallback
python scripts/train_full_pipeline.py --device cpu
```

---

## Stopping Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Next Steps

1. **Review Metrics**: Check Grafana dashboard for predictions
2. **Load Test**: Run `scripts/test_deployment.py --benchmark-requests 1000`
3. **Production Tuning**: See `PHASE3_DEPLOYMENT.md` for advanced configuration
4. **Add Authentication**: Implement API key auth (see TODO in `serve.py`)

---

## File Locations

**Models**:
- Stack ensemble: `data/artifacts/models/stack/stack.pkl`
- Pre-trained encoder: `data/artifacts/models/ts_tcc/pretrained_encoder.pt`
- Base models: `data/artifacts/models/{logreg,rf,xgb,simple_lstm,cnn_transformer}/`

**Logs**:
- Docker logs: `docker-compose logs api`
- Training logs: `data/logs/`

**Configs**:
- API config: `src/moola/api/serve.py`
- Prometheus: `monitoring/prometheus.yml`
- Grafana: `monitoring/grafana/dashboards/`

---

## Support

For detailed documentation, see:
- **PHASE3_DEPLOYMENT.md** - Complete deployment guide
- **IMPLEMENTATION_SUMMARY.md** - Implementation details
- **src/moola/api/serve.py** - API documentation
- **scripts/train_full_pipeline.py** - Training pipeline help

---

**Quick Reference**:

```bash
# Train
python scripts/train_full_pipeline.py --device cuda

# Deploy
docker-compose up -d

# Test
python scripts/test_deployment.py

# Monitor
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus

# Stop
docker-compose down
```

---

**Status**: ✅ Ready for production
**Expected Accuracy**: 75-82%
**API Latency**: <20ms (p95, GPU)
