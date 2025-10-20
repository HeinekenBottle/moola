# ML Workflow Orchestrator Commands

## Overview

The ML Workflow Orchestrator provides sophisticated command chaining with conditional logic and parallel droid coordination for ML operations.

## Command Syntax

### Basic Commands
```bash
/ml-train                    # Train model (SimpleLSTM default)
/ml-pretrain                # Run pre-training pipeline
/ml-deploy                   # Deploy to production
/ml-monitor                  # Setup monitoring
```

### Command Chaining
```bash
/init → /data-validate → /ml-train → /ml-evaluate → /if:acc>90 → /ml-deploy
```

### Parallel Execution
```bash
/runpod-experiment parallel[sigma-0.10 | sigma-0.12 | sigma-0.15] → /results:aggregate
```

### State Management
```bash
/analyze-data → $data_quality=<result> → /if:$data_quality>0.8 → /ml-train → /else → /data-cleanup
```

## ML Workflows

### 1. Complete Training Pipeline
```bash
# Full pipeline from data preparation to deployment
/ml-pipeline-full

# Equivalent command chain:
/init → /data-validate → /feature-engineer → /ml-train → /ml-evaluate → /if:accuracy>0.87 → /ml-deploy → /else → /hyperparameter-tune → /retrain
```

### 2. RunPod Parallel Experiments
```bash
# Run multiple hyperparameter experiments simultaneously on RunPod
/runpod-hpo-parallel

# Expands to:
/runpod-setup → parallel[
  /train:sigma-0.10 |
  /train:sigma-0.12 |
  /train:sigma-0.15 |
  /train:sigma-0.20
] → /results:aggregate → /best-model:select → /if:best-acc>0.90 → /deploy → /else → /analyze-failures
```

### 3. Production Deployment Workflow
```bash
# Production deployment with quality gates
/deploy-production

# Command chain:
/code-review → /security-scan → /performance-test → /runpod-staging → /if:tests-pass → /runpod-production → /monitor-setup → /else → /rollback
```

### 4. Continuous Training Loop
```bash
# Automated retraining based on performance
/continuous-training

# Implementation:
/monitor-performance → /if:drift-detected → /data-refresh → /retrain → /evaluate → /if:better-performance → /deploy → /monitor → /else → /continue-monitoring
```

## State Variables

### Common Variables
- `$model_performance` - Model accuracy/metrics from training
- `$data_quality` - Data quality score from validation
- `$deployment_status` - Current deployment status
- `$drift_detected` - Boolean for data drift detection
- `$resource_usage` - GPU/memory utilization

### Variable Examples
```bash
/train → $model_performance=0.87 → /if:$model_performance>0.85 → /deploy → /else → /optimize

/data-validate → $data_quality=0.82 → /if:$data_quality>0.80 → /continue → /else → /data-cleanup
```

## Error Handling

### Try/Catch Patterns
```bash
/try:/train → /catch:training-failed → /fallback:use-backup-model → /else → /continue

/try:/deploy → /catch:deployment-failed → /rollback → /notify-team → /retry-deployment
```

### Retry Logic
```bash
/train-with-retry:3:backoff=300 → /if:success → /deploy → /else → /escalate
```

## Droid Coordination

### Multi-Agent Workflows
```bash
# Launch 3 specialized droids simultaneously for comprehensive analysis
/parallel-analysis

# Internally coordinates:
parallel[
  ml-production-engineer:analyze-model-architecture |
  data-infrastructure-engineer:validate-data-pipeline |
  ml-observability-monitoring-engineer:setup-monitoring
] → /aggregate:analysis-results → /plan:next-steps
```

### Sequential Droid Handoffs
```bash
/data-infrastructure-engineer:data-quality-check → /ml-production-engineer:model-optimization → /ml-observability-monitoring-engineer:monitoring-setup
```

## Moola-Specific Workflows

### LSTM Pre-training Pipeline
```bash
/lstm-pretraining-full

# Command chain:
/data-validate → /preprocess-time-series → /pretrain-bilstm → /fine-tune-simplelstm → /evaluate → /if:acc>0.87 → /deploy → /benchmark
```

### SSH/SCP RunPod Integration
```bash
/runpod-deploy-full

# Implementation:
/code-package → /scp:deploy-to-runpod → /ssh:run-training → /scp:retrieve-results → /analyze → /store-results
```

### A/B Testing Framework
```bash
/setup-ab-test

# Workflow:
/create-variants → /deploy-variant-a → /deploy-variant-b → /collect-metrics → /statistical-test → /select-winner → /deploy-loser -> archive-variant
```

## Usage Examples

### Quick Training
```bash
/ml-train --model simple_lstm --device cuda --epochs 60
```

### Parallel Experiments
```bash
/runpod-hpo --parameters sigma:0.10,0.12,0.15 --parallel 4
```

### Production Deployment
```bash
/deploy --model simple_lstm --environment production --rollback-on-failure
```

### Monitoring Setup
```bash
/setup-monitoring --model simple_lstm --alerts performance,drift --dashboard
```

## Integration with Existing Systems

The workflow orchestration integrates seamlessly with:

- **SSH/SCP RunPod workflow** for distributed training
- **Results logging system** via experiment_results.jsonl
- **Git workflow** with automatic pre-commit hooks
- **DVC** for data versioning and tracking
- **Pydantic** for configuration validation

This command-based workflow system provides the sophistication of Claude Code plugins adapted for Droid's parallel execution capabilities.
