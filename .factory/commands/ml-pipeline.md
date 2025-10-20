# Machine Learning Pipeline - Multi-Agent MLOps Orchestration

Design and implement a complete ML pipeline for: $ARGUMENTS

## Thinking

This workflow orchestrates multiple specialized agents to build a production-ready ML pipeline following modern MLOps best practices. The approach emphasizes:

- **Phase-based coordination**: Each phase builds upon previous outputs, with clear handoffs between agents
- **Modern tooling integration**: MLflow/W&B for experiments, Feast/Tecton for features, KServe/Seldon for serving
- **Production-first mindset**: Every component designed for scale, monitoring, and reliability
- **Reproducibility**: Version control for data, models, and infrastructure
- **Continuous improvement**: Automated retraining, A/B testing, and drift detection

The multi-agent approach ensures each aspect is handled by domain experts:

- Data engineers handle ingestion and quality
- Data scientists design features and experiments  
- ML engineers implement training pipelines
- MLOps engineers handle production deployment
- Observability engineers ensure monitoring

## Phase 1: Data & Requirements Analysis

<Task>
subagent_type: data-infrastructure-engineer
prompt: |
  Analyze and design data pipeline for ML system with requirements: $ARGUMENTS

  Deliverables:

  1. Data source audit and ingestion strategy:
     - Source systems and connection patterns
     - Schema validation using Pydantic/Great Expectations
     - Data versioning with DVC or lakeFS
     - Incremental loading and CDC strategies

  2. Data quality framework:
     - Profiling and statistics generation
     - Anomaly detection rules
     - Data lineage tracking
     - Quality gates and SLAs

  3. Storage architecture:
     - Raw/processed/feature layers
     - Partitioning strategy
     - Retention policies
     - Cost optimization

  Provide implementation code for critical components and integration patterns.
</Task>

<Task>
subagent_type: ml-production-engineer  
prompt: |
  Design feature engineering and model requirements for: $ARGUMENTS
  Using data architecture from: {phase1.data-infrastructure-engineer.output}

  Deliverables:

  1. Feature engineering pipeline:
     - Transformation specifications
     - Feature store schema (Feast/Tecton)
     - Statistical validation rules
     - Handling strategies for missing data/outliers

  2. Model requirements:
     - Algorithm selection rationale
     - Performance metrics and baselines
     - Training data requirements
     - Evaluation criteria and thresholds

  3. Experiment design:
     - Hypothesis and success metrics
     - A/B testing methodology
     - Sample size calculations
     - Bias detection approach

  Include feature transformation code and statistical validation logic.
</Task>

## Phase 2: Model Development & Training

<Task>
subagent_type: ml-production-engineer
prompt: |
  Implement training pipeline based on requirements: {phase1.ml-production-engineer.output}
  Using data pipeline: {phase1.data-infrastructure-engineer.output}

  Build comprehensive training system:

  1. Training pipeline implementation:
     - Modular training code with clear interfaces
     - Hyperparameter optimization (Optuna/Ray Tune)
     - Distributed training support (Horovod/PyTorch DDP)
     - Cross-validation and ensemble strategies

  2. Experiment tracking setup:
     - MLflow/Weights & Biases integration
     - Metric logging and visualization
     - Artifact management (models, plots, data samples)
     - Experiment comparison and analysis tools

  3. Model registry integration:
     - Version control and tagging strategy
     - Model metadata and lineage
     - Promotion workflows (dev → staging → prod)
     - Rollback procedures

  Provide complete training code with configuration management.
</Task>

<Task>
subagent_type: ml-observability-monitoring-engineer
prompt: |
  Set up monitoring and observability for ML system implemented in: {phase2.ml-production-engineer.output}
  Using infrastructure: {phase1.data-infrastructure-engineer.output}

  Monitoring framework:

  1. Model performance monitoring:
     - Prediction accuracy tracking
     - Latency and throughput metrics
     - Feature importance shifts
     - Business KPI correlation

  2. Data and model drift detection:
     - Statistical drift detection (KS test, PSI)
     - Concept drift monitoring
     - Feature distribution tracking
     - Automated drift alerts and reports

  3. System observability:
     - Prometheus metrics for all components
     - Grafana dashboards for visualization
     - Distributed tracing with Jaeger/Zipkin
     - Log aggregation with ELK/Loki

  4. Alerting and automation:
     - PagerDuty/Opsgenie integration
     - Automated retraining triggers
     - Performance degradation workflows
     - Incident response runbooks

  5. Cost tracking:
     - Resource utilization metrics
     - Cost allocation by model/experiment
     - Optimization recommendations
     - Budget alerts and controls

  Deliver monitoring configuration, dashboards, and alert rules.
</Task>

## Phase 3: Production Deployment & Serving

<Task>
subagent_type: ml-production-engineer
prompt: |
  Design production deployment for models from: {phase2.ml-production-engineer.output}
  With optimized code from previous phases

  Implementation requirements:

  1. Model serving infrastructure:
     - REST/gRPC APIs with FastAPI/TorchServe
     - Batch prediction pipelines (Airflow/Kubeflow)
     - Stream processing (Kafka/Kinesis integration)
     - Model serving platforms (KServe/Seldon Core)

  2. Deployment strategies:
     - Blue-green deployments for zero downtime
     - Canary releases with traffic splitting  
     - Shadow deployments for validation
     - A/B testing infrastructure

  3. CI/CD pipeline:
     - GitHub Actions/GitLab CI workflows
     - Automated testing gates
     - Model validation before deployment
     - ArgoCD for GitOps deployment

  4. Infrastructure as Code:
     - Terraform modules for cloud resources
     - Helm charts for Kubernetes deployments  
     - Docker multi-stage builds for optimization
     - Secret management with Vault/Secrets Manager

  Provide complete deployment configuration and automation scripts.
</Task>

## Configuration Options

- **experiment_tracking**: mlflow | wandb | neptune | clearml
- **feature_store**: feast | tecton | databricks | custom
- **serving_platform**: kserve | seldon | torchserve | triton
- **orchestration**: kubeflow | airflow | prefect | dagster
- **cloud_provider**: aws | azure | gcp | multi-cloud
- **deployment_mode**: realtime | batch | streaming | hybrid
- **monitoring_stack**: prometheus | datadog | newrelic | custom

## Success Criteria

### Data Pipeline Success
- < 0.1% data quality issues in production
- Automated data validation passing 99.9% of time
- Complete data lineage tracking
- Sub-second feature serving latency

### Model Performance
- Meeting or exceeding baseline metrics
- < 5% performance degradation before retraining
- Successful A/B tests with statistical significance
- No undetected model drift > 24 hours

### Operational Excellence
- 99.9% uptime for model serving
- < 200ms p99 inference latency
- Automated rollback within 5 minutes
- Complete observability with < 1 minute alert time

### Development Velocity
- < 1 hour from commit to production
- Parallel experiment execution
- Reproducible training runs
- Self-service model deployment

### Cost Efficiency
- < 20% infrastructure waste
- Optimized resource allocation
- Automatic scaling based on load
- Spot instance utilization > 60%

## Final Deliverables

Upon completion, the orchestrated pipeline will provide:

- End-to-end ML pipeline with full automation
- Comprehensive documentation and runbooks
- Production-ready infrastructure as code
- Complete monitoring and alerting system
- CI/CD pipelines for continuous improvement
- Cost optimization and scaling strategies
- Disaster recovery and rollback procedures
