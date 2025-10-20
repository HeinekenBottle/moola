# LSTM Optimization Experiments - MLOps Runbook

## Table of Contents

1. [Overview](#overview)
2. [Infrastructure Setup](#infrastructure-setup)
3. [Running Experiments](#running-experiments)
4. [Monitoring & Troubleshooting](#monitoring--troubleshooting)
5. [Model Promotion](#model-promotion)
6. [Cost Optimization](#cost-optimization)
7. [Disaster Recovery](#disaster-recovery)

---

## Overview

This runbook provides step-by-step instructions for executing the 13 LSTM optimization experiments, monitoring progress, and promoting the best model to production.

### Experiment Pipeline

```
Phase 1: Time Warp Ablation (4 experiments)
├── sigma=0.10 → Pre-train Masked LSTM → Fine-tune SimpleLSTM
├── sigma=0.12 → Pre-train Masked LSTM → Fine-tune SimpleLSTM
├── sigma=0.15 → Pre-train Masked LSTM → Fine-tune SimpleLSTM
└── sigma=0.20 → Pre-train Masked LSTM → Fine-tune SimpleLSTM
                 ↓ Select winner
Phase 2: Architecture Search (8 experiments)
├── hidden=64,  heads=4, layers=1
├── hidden=64,  heads=4, layers=2
├── hidden=64,  heads=8, layers=1
├── hidden=64,  heads=8, layers=2
├── hidden=128, heads=4, layers=1
├── hidden=128, heads=4, layers=2
├── hidden=128, heads=8, layers=1
└── hidden=128, heads=8, layers=2
                 ↓ Select winner
Phase 3: Pre-training Depth (3 experiments)
├── pretrain_epochs=50
├── pretrain_epochs=75
└── pretrain_epochs=100
                 ↓
Final: Compare all 13 experiments → Promote best model
```

### Infrastructure Components

- **Compute**: AWS g5.xlarge GPU instances (A10G, 24GB VRAM)
- **Storage**: S3 for artifacts, PostgreSQL for MLflow backend
- **Monitoring**: Prometheus + Grafana + CloudWatch
- **Orchestration**: GitHub Actions or local script execution
- **Tracking**: MLflow for experiment management

---

## Infrastructure Setup

### Option 1: Local Execution (RTX 4090 / High-end GPU)

**Prerequisites:**
- NVIDIA GPU with 24GB+ VRAM
- CUDA 11.8 installed
- Docker with nvidia-docker runtime
- 500GB+ free disk space

**Setup:**

```bash
# 1. Clone repository
git clone https://github.com/your-org/moola.git
cd moola

# 2. Create environment file
cat > .env <<EOF
MLFLOW_TRACKING_URI=http://localhost:5000
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
EOF

# 3. Start MLflow and monitoring stack
docker-compose -f deploy/docker-compose.experiments.yml up -d mlflow postgres prometheus grafana

# 4. Verify services
docker ps
# Expected: mlflow-server, postgres, prometheus, grafana running

# 5. Access UIs
# MLflow: http://localhost:5000
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### Option 2: Cloud Execution (AWS)

**Prerequisites:**
- AWS CLI configured with credentials
- Terraform >= 1.5.0
- SSH key pair for instance access

**Setup:**

```bash
# 1. Navigate to Terraform directory
cd terraform

# 2. Create terraform.tfvars
cat > terraform.tfvars <<EOF
aws_region                = "us-east-1"
instance_type             = "g5.xlarge"
desired_workers           = 1
max_workers               = 4
allowed_ssh_cidr_blocks   = ["YOUR_IP/32"]
mlflow_tracking_uri       = "http://mlflow.example.com:5000"
enable_auto_shutdown      = true
spot_instance_enabled     = true
spot_max_price            = "0.50"
EOF

# 3. Initialize Terraform
terraform init

# 4. Plan infrastructure
terraform plan -out=tfplan

# 5. Apply infrastructure
terraform apply tfplan

# 6. Get outputs
terraform output -json > ../outputs.json

# 7. SSH into GPU worker
INSTANCE_IP=$(cat ../outputs.json | jq -r '.worker_public_ip.value')
ssh -i ~/.ssh/your-key.pem ubuntu@$INSTANCE_IP
```

### Option 3: Hybrid (Local Development + Cloud Training)

Use GitHub Actions workflow for cloud execution:

```bash
# Trigger workflow manually
gh workflow run lstm_optimization.yml \
  --field phase=all \
  --field run_on_cloud=true \
  --field mlflow_experiment=lstm-optimization-2025
```

---

## Running Experiments

### Method 1: All-in-One Script (Recommended)

```bash
# Run all 13 experiments sequentially
./scripts/run_all_phases.sh --device cuda

# Skip specific phases
./scripts/run_all_phases.sh --device cuda --skip-phase phase1

# Custom MLflow URI
./scripts/run_all_phases.sh \
  --device cuda \
  --mlflow-uri http://your-mlflow-server:5000
```

**Expected Duration:**
- Phase 1: ~4-6 hours (4 experiments)
- Phase 2: ~8-12 hours (8 experiments)
- Phase 3: ~4-6 hours (3 experiments)
- **Total: 16-24 hours**

### Method 2: GitHub Actions Workflow

**Trigger via GitHub UI:**
1. Go to Actions tab
2. Select "LSTM Optimization Pipeline"
3. Click "Run workflow"
4. Choose phase and options
5. Click "Run workflow"

**Trigger via CLI:**

```bash
# Run all phases
gh workflow run lstm_optimization.yml \
  --field phase=all \
  --field run_on_cloud=false

# Run specific phase
gh workflow run lstm_optimization.yml \
  --field phase=phase2_architecture_search \
  --field run_on_cloud=false
```

### Method 3: Docker Container Execution

```bash
# Start experiment runner container
docker-compose -f deploy/docker-compose.experiments.yml run \
  --rm \
  experiment-runner \
  bash -c "./scripts/run_all_phases.sh --device cuda"
```

### Method 4: Individual Experiments (for testing)

```bash
# Phase 1: Single experiment
python scripts/generate_unlabeled_data.py \
  --output data/raw/unlabeled_tw_0.10.parquet \
  --time-warp-sigma 0.10 \
  --num-samples 10000 \
  --seed 1337

python scripts/pretrain_masked_lstm.py \
  --unlabeled-data data/raw/unlabeled_tw_0.10.parquet \
  --output-dir data/artifacts/pretrained/test \
  --epochs 50 \
  --batch-size 256 \
  --device cuda

python -m moola.cli oof \
  --model simple_lstm \
  --device cuda \
  --seed 1337 \
  --load-pretrained-encoder data/artifacts/pretrained/test/encoder.pt \
  --mlflow-tracking
```

---

## Monitoring & Troubleshooting

### Real-time Monitoring

**1. MLflow UI** (http://localhost:5000)
- View all experiments
- Compare metrics across runs
- Download artifacts

**2. Grafana Dashboards** (http://localhost:3000)
- GPU utilization, temperature, memory
- Training progress
- System resources

**3. Prometheus Metrics** (http://localhost:9090)
- Query custom metrics
- Check alerting rules

### Key Metrics to Monitor

| Metric | Target | Alert Threshold |
|--------|--------|-----------------|
| GPU Utilization | 80-95% | < 20% (idle), > 95% (bottleneck) |
| GPU Memory | < 90% | > 90% (OOM risk) |
| GPU Temperature | < 80°C | > 85°C |
| Training Accuracy | > 75% | N/A |
| Class 1 Accuracy | > 30% | < 30% (imbalance) |
| Experiment Duration | ~60-90min | > 120min |

### Common Issues

#### 1. OOM (Out of Memory) Error

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX MiB
```

**Solutions:**
```bash
# Reduce batch size
# Edit configs/simple_lstm.yaml
training:
  batch_size: 512  # Instead of 1024

# Or use gradient accumulation
python -m moola.cli oof \
  --model simple_lstm \
  --batch-size 256 \
  --gradient-accumulation-steps 4
```

#### 2. Training Stalled

**Symptoms:**
- GPU utilization drops to 0%
- No progress in logs for >15 minutes

**Diagnosis:**
```bash
# Check process status
nvidia-smi

# Check logs
docker logs -f lstm-experiment-runner

# Check for deadlocks
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Current device: {torch.cuda.current_device()}')
"
```

**Solutions:**
```bash
# Restart container
docker-compose -f deploy/docker-compose.experiments.yml restart experiment-runner

# Check dataloader workers
# Set num_workers=0 in config if multiprocessing issues
```

#### 3. MLflow Connection Errors

**Symptoms:**
```
ConnectionError: HTTPConnectionPool(host='mlflow', port=5000)
```

**Solutions:**
```bash
# Verify MLflow server
curl http://localhost:5000/health

# Restart MLflow
docker-compose -f deploy/docker-compose.experiments.yml restart mlflow

# Check network connectivity
docker network inspect deploy_experiment-network
```

#### 4. Low Class 1 Accuracy

**Symptoms:**
- Overall accuracy > 75% but class_1_accuracy < 30%

**Solutions:**
- Increase SMOTE augmentation target
- Adjust class weights in loss function
- Try different time_warp_sigma values
- Increase pre-training epochs

### Log Locations

```bash
# Docker logs
docker logs -f lstm-experiment-runner

# MLflow logs
docker logs -f mlflow-server

# Prometheus logs
docker logs -f prometheus-lstm

# Application logs (if running locally)
tail -f /var/log/moola/training.log
```

---

## Model Promotion

### Step 1: Review Comparison Report

```bash
# After all experiments complete
cat comparison_report.md
```

**Look for:**
- Best overall accuracy
- Class balance (class_1_accuracy >= 30%)
- Consistent performance across folds

### Step 2: Validate Best Model

```bash
# Extract best model from MLflow
BEST_RUN_ID=$(jq -r '.run_id' best_model.json)

# Download model artifacts
mlflow artifacts download \
  --run-id $BEST_RUN_ID \
  --dst-path data/artifacts/best_model

# Run validation tests
python scripts/validate_best_model.py \
  --model-path data/artifacts/best_model \
  --test-data data/processed/test.parquet
```

### Step 3: Tag in MLflow

```python
# Via Python
import mlflow

client = mlflow.MlflowClient()

# Tag as production candidate
client.set_tag(run_id, "stage", "staging")
client.set_tag(run_id, "validated_by", "your_name")
client.set_tag(run_id, "validated_at", "2025-10-16T22:00:00Z")

# Transition to production (after stakeholder approval)
model_version = client.create_model_version(
    name="SimpleLSTM",
    source=f"runs:/{run_id}/model",
    run_id=run_id
)

client.transition_model_version_stage(
    name="SimpleLSTM",
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True
)
```

### Step 4: Deploy to Production

```bash
# Update production configuration
cp data/artifacts/best_model/config.yaml configs/production.yaml

# Build production Docker image
docker build -t moola-ml:latest -f Dockerfile .

# Deploy (example: Kubernetes)
kubectl apply -f deploy/kubernetes/lstm-inference.yaml

# Or Docker Compose
docker-compose -f docker-compose.yml up -d api
```

### Step 5: Monitor Production Performance

```bash
# Set up model monitoring
python scripts/setup_production_monitoring.py \
  --model-version $(cat best_model.json | jq -r '.version')

# Check production metrics
curl http://api.moola.com/metrics
```

---

## Cost Optimization

### Estimated Costs (AWS g5.xlarge)

| Configuration | Duration | Cost |
|--------------|----------|------|
| All 13 experiments (on-demand) | 20 hours | $20.00 |
| All 13 experiments (spot) | 20 hours | $6.00 |
| MLflow + monitoring (t3.small) | 24/7 | $17/month |
| S3 storage (100GB artifacts) | - | $2.30/month |
| **Total (one-time)** | - | **$6-20** |

### Cost-Saving Strategies

#### 1. Use Spot Instances (70% savings)

```bash
# Already configured in Terraform
terraform apply -var="spot_instance_enabled=true"
```

#### 2. Auto-Shutdown After Completion

Lambda function automatically terminates instances after 4 hours of inactivity:

```python
# terraform/lambda/auto_shutdown.py (already created)
# Triggers every 4 hours via EventBridge
```

#### 3. S3 Lifecycle Policies

Artifacts older than 90 days are automatically deleted:

```bash
# Already configured in Terraform
# See: terraform/main.tf resource "aws_s3_bucket_lifecycle_configuration"
```

#### 4. Cleanup Intermediate Files

```bash
# During experiment execution
rm -f data/raw/unlabeled_windows_*.parquet  # 2-5GB per file
rm -rf data/artifacts/pretrained/phase1_*/  # Keep only winner
```

#### 5. Reserved Instances (if running regularly)

For monthly experiments:
- g5.xlarge 1-year reserved: $0.62/hr (38% savings)
- 3-year reserved: $0.40/hr (60% savings)

---

## Disaster Recovery

### Backup Strategy

**1. MLflow Backend (PostgreSQL)**

```bash
# Automated daily backups
docker exec mlflow-postgres pg_dump -U mlflow mlflow > backup_$(date +%Y%m%d).sql

# Upload to S3
aws s3 cp backup_$(date +%Y%m%d).sql s3://moola-lstm-experiments/backups/
```

**2. Model Artifacts (S3)**

Already versioned and backed up via S3 versioning (enabled in Terraform).

**3. Configuration Files**

Store in Git repository (already done).

### Recovery Procedures

#### Scenario 1: Experiment Failure Mid-Pipeline

```bash
# Resume from specific phase
./scripts/run_all_phases.sh --device cuda --skip-phase phase1
```

#### Scenario 2: MLflow Data Loss

```bash
# Restore from backup
docker exec -i mlflow-postgres psql -U mlflow mlflow < backup_20251016.sql

# Verify
curl http://localhost:5000/api/2.0/mlflow/experiments/list
```

#### Scenario 3: S3 Artifact Corruption

```bash
# Restore from S3 version
aws s3api list-object-versions \
  --bucket moola-lstm-experiments \
  --prefix phase1/

# Download specific version
aws s3api get-object \
  --bucket moola-lstm-experiments \
  --key phase1/tw_0.10/encoder.pt \
  --version-id VERSION_ID \
  encoder.pt
```

---

## Appendix

### A. Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URL | `http://mlflow:5000` |
| `AWS_ACCESS_KEY_ID` | AWS credentials | `AKIAIOSFODNN7EXAMPLE` |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials | `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY` |
| `AWS_REGION` | AWS region | `us-east-1` |
| `SLACK_WEBHOOK_URL` | Slack notifications | `https://hooks.slack.com/...` |
| `CUDA_VISIBLE_DEVICES` | GPU selection | `0` |

### B. Useful Commands

```bash
# Check GPU status
nvidia-smi -l 1  # Live monitoring

# View MLflow experiments
mlflow experiments list

# Download best model
mlflow artifacts download --run-id <RUN_ID>

# Push metrics to Prometheus
python scripts/export_prometheus_metrics.py \
  --experiment-name phase1_tw_0.10 \
  --pushgateway-url localhost:9091

# Send Slack notification
python scripts/send_slack_notification.py \
  --webhook-url "$SLACK_WEBHOOK_URL" \
  --channel "#ml-experiments" \
  --title "Test" \
  --message "Hello from runbook"
```

### C. Contact & Support

- **ML Team Lead**: ml-team@example.com
- **DevOps Support**: devops@example.com
- **Slack**: #ml-experiments, #ml-ops
- **On-call**: PagerDuty rotation

---

**Last Updated**: 2025-10-16
**Version**: 1.0
**Authors**: MLOps Team
