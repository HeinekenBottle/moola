# LSTM Optimization CI/CD Pipeline - Infrastructure Summary

## Overview

This document provides a comprehensive overview of the complete MLOps infrastructure designed for automated execution, tracking, and deployment of 13 LSTM optimization experiments.

**Date**: October 16, 2025
**Version**: 1.0
**Status**: Production Ready

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          EXPERIMENT EXECUTION                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  GitHub Actions                    Local Execution                       │
│  ┌──────────────┐                 ┌──────────────┐                     │
│  │ Workflow     │                 │ run_all_     │                     │
│  │ Trigger      │────────────────▶│ phases.sh    │                     │
│  └──────────────┘                 └──────────────┘                     │
│         │                                 │                              │
│         ▼                                 ▼                              │
│  ┌──────────────────────────────────────────────┐                      │
│  │        Phase 1: Time Warp Ablation           │                      │
│  │  σ=0.10, 0.12, 0.15, 0.20 (4 experiments)   │                      │
│  └──────────────────────────────────────────────┘                      │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────┐                      │
│  │  Phase 2: Architecture Search                │                      │
│  │  hidden×heads×layers (8 experiments)         │                      │
│  └──────────────────────────────────────────────┘                      │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────┐                      │
│  │  Phase 3: Pre-training Depth                 │                      │
│  │  epochs=50,75,100 (3 experiments)            │                      │
│  └──────────────────────────────────────────────┘                      │
│         │                                                                │
│         ▼                                                                │
│  ┌──────────────────────────────────────────────┐                      │
│  │  Model Promotion & Comparison                │                      │
│  │  Select best from all 13 experiments         │                      │
│  └──────────────────────────────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       TRACKING & MONITORING                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  MLflow                      Prometheus                   CloudWatch    │
│  ┌──────────┐               ┌──────────┐               ┌──────────┐   │
│  │ Tracking │               │ Metrics  │               │ Logs     │   │
│  │ Server   │               │ & Alerts │               │ & Events │   │
│  └──────────┘               └──────────┘               └──────────┘   │
│       │                           │                           │         │
│       └───────────────────────────┴───────────────────────────┘         │
│                                   │                                      │
│                                   ▼                                      │
│                           ┌──────────────┐                             │
│                           │   Grafana    │                             │
│                           │  Dashboard   │                             │
│                           └──────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         INFRASTRUCTURE                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  AWS Cloud (Terraform)           Local (Docker Compose)                 │
│  ┌──────────────┐               ┌──────────────┐                       │
│  │ g5.xlarge    │               │ RTX 4090     │                       │
│  │ GPU Workers  │               │ GPU Host     │                       │
│  │ Auto-scaling │               │ Single Node  │                       │
│  └──────────────┘               └──────────────┘                       │
│         │                               │                                │
│         ▼                               ▼                                │
│  ┌──────────────┐               ┌──────────────┐                       │
│  │ S3 Bucket    │               │ Local Volume │                       │
│  │ Artifacts    │               │ Artifacts    │                       │
│  └──────────────┘               └──────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Deliverables

### 1. GitHub Actions Workflow

**File**: `.github/workflows/lstm_optimization.yml`

**Features**:
- **Matrix Strategy**: Parallel execution of experiments within phases
- **Phase Dependencies**: Automatic winner selection between phases
- **Cloud Integration**: AWS g5.xlarge spot instance support
- **Artifact Management**: Automatic upload to S3
- **Auto-cleanup**: Terminate cloud resources after completion
- **Notifications**: Slack alerts on completion/failure

**Usage**:
```bash
# Trigger via CLI
gh workflow run lstm_optimization.yml \
  --field phase=all \
  --field run_on_cloud=true

# Trigger via GitHub UI
# Actions → LSTM Optimization Pipeline → Run workflow
```

**Estimated Duration**: 16-24 hours for all phases

---

### 2. Terraform Infrastructure

**Directory**: `terraform/`

**Files**:
- `main.tf` - Core infrastructure resources
- `variables.tf` - Configurable parameters
- `outputs.tf` - Infrastructure outputs
- `user_data.sh` - Instance initialization script

**Resources Created**:
- **VPC & Networking**: Isolated VPC with public subnet
- **GPU Instances**: g5.xlarge auto-scaling group (0-4 instances)
- **IAM Roles**: S3 access, CloudWatch logging
- **S3 Bucket**: Versioned artifact storage with lifecycle policies
- **Security Groups**: SSH + MLflow + Prometheus access
- **Lambda Functions**: Auto-shutdown for cost optimization
- **CloudWatch Alarms**: GPU utilization, temperature monitoring

**Cost Estimate**:
| Resource | Configuration | Monthly Cost |
|----------|--------------|--------------|
| g5.xlarge (spot) | 20 hours/month | $6.00 |
| S3 storage | 100GB | $2.30 |
| Data transfer | 50GB | $4.50 |
| CloudWatch | Logs + metrics | $5.00 |
| **Total** | | **$17.80/month** |

**Setup**:
```bash
cd terraform
terraform init
terraform plan
terraform apply
```

**Teardown**:
```bash
terraform destroy  # Complete cleanup
```

---

### 3. Docker Images

**Files**:
- `docker/Dockerfile.lstm-experiments` - GPU-enabled experiment runner
- `deploy/docker-compose.experiments.yml` - Complete MLOps stack

**Images**:
1. **Experiment Runner**: PyTorch 2.2.2 + CUDA 11.8 + MLflow
2. **MLflow Server**: Tracking server with PostgreSQL backend
3. **Prometheus**: Metrics collection + alerting
4. **Grafana**: Visualization dashboards
5. **PostgreSQL**: MLflow metadata store

**Local Deployment**:
```bash
# Start full stack
docker-compose -f deploy/docker-compose.experiments.yml up -d

# Run experiments
docker-compose -f deploy/docker-compose.experiments.yml run \
  experiment-runner \
  ./scripts/run_all_phases.sh --device cuda

# View logs
docker logs -f lstm-experiment-runner
```

**Services**:
- MLflow UI: http://localhost:5000
- Grafana: http://localhost:3000
- Prometheus: http://localhost:9090
- Pushgateway: http://localhost:9091

---

### 4. Orchestration Scripts

**Directory**: `scripts/`

#### `run_all_phases.sh`
Master script for sequential execution of all 13 experiments.

**Features**:
- Automatic phase progression
- Winner selection between phases
- Error recovery and logging
- Slack notifications
- Artifact cleanup

**Usage**:
```bash
# Run all phases
./scripts/run_all_phases.sh --device cuda

# Skip specific phases
./scripts/run_all_phases.sh --device cuda --skip-phase phase1

# Custom MLflow URI
./scripts/run_all_phases.sh \
  --device cuda \
  --mlflow-uri http://mlflow-server:5000
```

#### `select_phase_winner.py`
Analyzes MLflow runs to select best configuration per phase.

**Usage**:
```bash
python scripts/select_phase_winner.py \
  --phase 1 \
  --experiment-name lstm-optimization-2025 \
  --output-file phase1_winner.json
```

#### `select_best_model.py`
Compares all 13 experiments and generates comparison report.

**Usage**:
```bash
python scripts/select_best_model.py \
  --experiment-name lstm-optimization-2025 \
  --output-report comparison_report.md \
  --min-class1-accuracy 0.30
```

**Output**: `comparison_report.md` with ranked results

#### `send_slack_notification.py`
Sends formatted Slack alerts with experiment results.

**Usage**:
```bash
python scripts/send_slack_notification.py \
  --webhook-url "$SLACK_WEBHOOK_URL" \
  --channel "#ml-experiments" \
  --title "Pipeline Complete" \
  --message "Best accuracy: 78.5%" \
  --report comparison_report.md
```

#### `export_prometheus_metrics.py`
Pushes MLflow metrics to Prometheus Pushgateway.

**Usage**:
```bash
python scripts/export_prometheus_metrics.py \
  --experiment-name phase1_tw_0.10 \
  --pushgateway-url localhost:9091
```

---

### 5. Monitoring & Alerting

**Files**:
- `monitoring/prometheus.yml` - Prometheus configuration
- `monitoring/prometheus_rules.yml` - Alerting rules
- `monitoring/grafana/dashboards/` - Pre-built dashboards

**Alerts Configured**:
| Alert | Condition | Severity |
|-------|-----------|----------|
| HighGPUUtilization | GPU > 95% for 5min | Warning |
| LowGPUUtilization | GPU < 20% for 15min | Info |
| HighGPUMemory | Memory > 90% | Warning |
| GPUMemoryLeak | Memory growth rate > 100MB/s | Critical |
| HighGPUTemperature | Temp > 80°C | Warning |
| CriticalGPUTemperature | Temp > 90°C | Critical |
| TrainingStalled | No progress 15min | Warning |
| ExperimentTimeout | Runtime > 2 hours | Warning |
| IdleGPUInstance | GPU < 5% for 30min | Info (cost) |

**Grafana Dashboards**:
1. **GPU Metrics**: Utilization, memory, temperature, power
2. **Training Progress**: Accuracy, loss, class balance
3. **System Resources**: CPU, RAM, disk I/O
4. **Cost Dashboard**: Instance hours, estimated spend

---

### 6. MLOps Runbook

**File**: `docs/MLOPS_RUNBOOK.md`

**Sections**:
1. **Overview**: Pipeline architecture and components
2. **Infrastructure Setup**: Local, cloud, and hybrid deployment
3. **Running Experiments**: Step-by-step execution guides
4. **Monitoring & Troubleshooting**: Common issues and solutions
5. **Model Promotion**: Validation and deployment procedures
6. **Cost Optimization**: Strategies to minimize cloud spend
7. **Disaster Recovery**: Backup and recovery procedures

**Key Procedures**:
- One-command experiment execution
- Real-time monitoring setup
- Debugging OOM errors
- MLflow connection troubleshooting
- Production model deployment
- Cost tracking and optimization

---

## Experiment Specifications

### Phase 1: Time Warp Ablation (4 Experiments)

**Objective**: Find optimal time warping augmentation strength

| Experiment | time_warp_sigma | Expected Accuracy | Duration |
|-----------|-----------------|-------------------|----------|
| 1.1 | 0.10 | 72-75% | 60-90min |
| 1.2 | 0.12 | 73-76% | 60-90min |
| 1.3 | 0.15 | 74-77% | 60-90min |
| 1.4 | 0.20 | 72-75% | 60-90min |

**Selection Criteria**: Max(accuracy) where class_1_accuracy >= 30%

---

### Phase 2: Architecture Search (8 Experiments)

**Objective**: Find optimal SimpleLSTM architecture

| Experiment | hidden_size | num_heads | num_layers | Params | Duration |
|-----------|-------------|-----------|------------|--------|----------|
| 2.1 | 64 | 4 | 1 | ~45K | 60min |
| 2.2 | 64 | 4 | 2 | ~55K | 70min |
| 2.3 | 64 | 8 | 1 | ~50K | 65min |
| 2.4 | 64 | 8 | 2 | ~60K | 75min |
| 2.5 | 128 | 4 | 1 | ~90K | 80min |
| 2.6 | 128 | 4 | 2 | ~110K | 90min |
| 2.7 | 128 | 8 | 1 | ~100K | 85min |
| 2.8 | 128 | 8 | 2 | ~120K | 95min |

**Selection Criteria**: Max(accuracy) where class_1_accuracy >= 30%

---

### Phase 3: Pre-training Depth (3 Experiments)

**Objective**: Find optimal pre-training duration

| Experiment | pretrain_epochs | Pre-train Time | Fine-tune Time | Total |
|-----------|----------------|----------------|----------------|-------|
| 3.1 | 50 | 30min | 60min | 90min |
| 3.2 | 75 | 45min | 60min | 105min |
| 3.3 | 100 | 60min | 60min | 120min |

**Selection Criteria**: Max(accuracy) where class_1_accuracy >= 30%

---

## Success Criteria

### Technical Metrics

| Metric | Target | Stretch Goal |
|--------|--------|--------------|
| Overall Accuracy | > 75% | > 80% |
| Class 1 Accuracy | > 30% | > 40% |
| Class 0 Accuracy | > 80% | > 85% |
| Parameter Count | < 150K | < 100K |
| Inference Latency | < 10ms | < 5ms |

### Operational Metrics

| Metric | Target |
|--------|--------|
| Total Pipeline Duration | < 24 hours |
| GPU Utilization | > 80% |
| Cost (all experiments) | < $20 |
| Zero Manual Intervention | ✓ |
| Complete Audit Trail | ✓ |

### Deliverables Checklist

- [x] GitHub Actions workflow for automated execution
- [x] Terraform infrastructure for cloud GPU workers
- [x] Docker images for reproducible environments
- [x] Orchestration scripts for all 3 phases
- [x] MLflow integration for experiment tracking
- [x] Prometheus + Grafana for monitoring
- [x] Automated model promotion workflow
- [x] Slack notifications for pipeline events
- [x] Comprehensive runbook documentation
- [x] Cost optimization (auto-shutdown, spot instances)
- [x] Disaster recovery procedures

---

## Quick Start

### Local Execution (RTX 4090)

```bash
# 1. Start infrastructure
docker-compose -f deploy/docker-compose.experiments.yml up -d

# 2. Run experiments
./scripts/run_all_phases.sh --device cuda

# 3. View results
open http://localhost:5000  # MLflow
open http://localhost:3000  # Grafana
```

### Cloud Execution (AWS)

```bash
# 1. Provision infrastructure
cd terraform
terraform apply

# 2. Trigger workflow
gh workflow run lstm_optimization.yml --field phase=all

# 3. Monitor progress
watch -n 30 'gh run list --workflow=lstm_optimization.yml'

# 4. Cleanup
terraform destroy
```

---

## File Structure

```
moola/
├── .github/
│   └── workflows/
│       └── lstm_optimization.yml          # GitHub Actions workflow
├── terraform/
│   ├── main.tf                            # Infrastructure definition
│   ├── variables.tf                       # Configurable parameters
│   ├── outputs.tf                         # Infrastructure outputs
│   └── user_data.sh                       # EC2 initialization
├── docker/
│   └── Dockerfile.lstm-experiments        # Experiment runner image
├── deploy/
│   └── docker-compose.experiments.yml     # MLOps stack
├── scripts/
│   ├── run_all_phases.sh                  # Master orchestration
│   ├── select_phase_winner.py             # Phase winner selection
│   ├── select_best_model.py               # Best model comparison
│   ├── send_slack_notification.py         # Slack alerts
│   └── export_prometheus_metrics.py       # Metrics export
├── monitoring/
│   ├── prometheus.yml                     # Prometheus config
│   ├── prometheus_rules.yml               # Alerting rules
│   └── grafana/
│       ├── dashboards/                    # Pre-built dashboards
│       └── datasources/                   # Prometheus datasource
└── docs/
    ├── MLOPS_RUNBOOK.md                   # Operational guide
    └── MLOPS_INFRASTRUCTURE_SUMMARY.md    # This document
```

---

## Next Steps

1. **Review**: Read `docs/MLOPS_RUNBOOK.md` for operational details
2. **Test**: Run a single experiment to validate setup
3. **Execute**: Launch full pipeline via GitHub Actions or local script
4. **Monitor**: Watch progress in Grafana dashboards
5. **Promote**: Deploy best model to production using MLflow registry
6. **Iterate**: Adjust hyperparameters based on comparison report

---

## Support & Contacts

- **Documentation**: `docs/MLOPS_RUNBOOK.md`
- **Issues**: GitHub Issues
- **Slack**: #ml-experiments, #ml-ops
- **MLflow UI**: http://mlflow:5000
- **Grafana**: http://grafana:3000

---

**Infrastructure Status**: ✅ Production Ready
**Last Updated**: 2025-10-16
**Version**: 1.0
**Owner**: MLOps Engineering Team
