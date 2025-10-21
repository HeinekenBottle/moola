# MOOLA Makefile - Stones Guide 80/20 Production Workflow
# Core principle: Focus on high-impact ML operations with minimal complexity

.PHONY: help train eval calibrate report ablate test lint clean doctor install format
.PHONY: train-cpu train-gpu eval-bootstrap eval-calibration ablate-loss ablate-dropout
.PHONY: report-full report-quick check-data check-env

# Default Python and paths
PY := python3
PIP := python3 -m pip
SRC := src
DATA_DIR := data/processed/labeled
ARTIFACTS := artifacts
REPORTS_DIR := $(ARTIFACTS)/reports
LOGS_DIR := logs
CONFIG_DIR := configs

# External directories (sibling to moola/)
DOCS_DIR ?= ../moola_docs_archive
DATABENTO_DIR ?= ../databento
LEGACY_DIR ?= ../$(shell cat ../moola_legacy_latest 2>/dev/null || echo "moola_legacy_20251021_193518")

# Default training data and splits
TRAIN_DATA := $(DATA_DIR)/train_latest.parquet
TRAIN_11D := $(DATA_DIR)/train_latest_11d.parquet
TEMPORAL_SPLIT := $(ARTIFACTS)/splits/v1/fold_0_temporal.json
SPLITS_DIR := $(ARTIFACTS)/splits/v1

# Model configurations
MODEL_PROD := enhanced_simple_lstm
MODEL_BASELINE := simple_lstm
PRETRAINED_11D := $(ARTIFACTS)/encoders/pretrained/bilstm_mae_11d_v1.pt
PRETRAINED_4D := $(ARTIFACTS)/encoders/pretrained/bilstm_mae_4d_v1.pt

# Training hyperparameters (production-optimized)
EPOCHS := 60
BATCH_SIZE := 32
SEED := 1337
DEVICE := cpu

# Help target
help: ## Show this help message
	@echo "MOOLA Makefile - Stones Guide 80/20 Production Workflow"
	@echo ""
	@echo "CORE WORKFLOWS:"
	@echo "  make train         - Train production model with uncertainty weighting"
	@echo "  make stones        - Train all Stones models (Jade, Sapphire, Opal)"
	@echo "  make eval          - Evaluate with bootstrap CI + calibration"
	@echo "  make calibrate     - Temperature scaling + ECE analysis"
	@echo "  make report        - Generate figures + markdown summary"
	@echo "  make ablate        - Loss balancing + dropout sweeps"
	@echo ""
	@echo "STONES COLLECTION:"
	@echo "  make train-jade    - Train Jade (moola-lstm-m-v1.0)"
	@echo "  make train-sapphire - Train Sapphire (moola-preenc-fr-s-v1.0)"
	@echo "  make train-opal    - Train Opal (moola-preenc-ad-m-v1.0)"
	@echo ""
	@echo "TRAINING OPTIONS:"
	@echo "  make train-cpu     - CPU training (for development)"
	@echo "  make train-gpu     - GPU training with pretrained encoder"
	@echo "  make train-11d     - Train with 11D RelativeTransform features"
	@echo ""
	@echo "EVALUATION OPTIONS:"
	@echo "  make eval-bootstrap - Bootstrap confidence intervals"
	@echo "  make eval-calibration - Calibration metrics + plots"
	@echo ""
	@echo "ABLATION STUDIES:"
	@echo "  make ablate-loss   - Loss weighting sweep (uncertainty vs manual)"
	@echo "  make ablate-dropout - Dropout rate sweep for uncertainty"
	@echo ""
	@echo "UTILITIES:"
	@echo "  make test          - Run all tests"
	@echo "  make lint          - Code quality checks"
	@echo "  make clean         - Clean artifacts and logs"
	@echo "  make doctor        - Environment validation"
	@echo "  make install       - Install dependencies"
	@echo "  make check-data    - Validate training data"
	@echo "  make check-env     - Check environment setup"
	@echo ""
	@echo "REPORTS:"
	@echo "  make report-full   - Complete analysis with all figures"
	@echo "  make report-quick  - Quick summary metrics only"
	@echo ""
	@echo "EXAMPLES:"
	@echo "  make train DEVICE=cuda EPOCHS=100"
	@echo "  make eval-bootstrap MODEL=enhanced_simple_lstm RESAMPLES=2000"
	@echo "  make ablate-loss SEED=42 DEVICE=cuda"

# =============================================================================
# CORE WORKFLOWS (Stones Guide 80/20)
# =============================================================================

train: ## Train production model with uncertainty weighting + logs
	@echo "🚀 Starting production training workflow..."
	@mkdir -p $(LOGS_DIR) $(ARTIFACTS)/models/$(MODEL_PROD)
	@echo "✅ Training $(MODEL_PROD) with uncertainty weighting (REQUIRED)"
	@($(PY) -m moola.cli train \
		--model $(MODEL_PROD) \
		--data $(TRAIN_DATA) \
		--split $(TEMPORAL_SPLIT) \
		--device $(DEVICE) \
		--predict-pointers \
		--use-uncertainty-weighting \
		--bootstrap-ci \
		--compute-calibration \

		--save-run \
		--seed $(SEED) \
		--over epochs=$(EPOCHS) \
		--save-checkpoints \
		--monitor-gradients \
		2>&1 | tee $(LOGS_DIR)/train_$(MODEL_PROD)_$(shell date +%Y%m%d_%H%M%S).log) || (echo "❌ Enhanced model failed, falling back to simple_lstm" && make train-simple)

train-simple: ## Fallback simple LSTM training
	@echo "🔄 Using simple LSTM fallback..."
	@mkdir -p $(LOGS_DIR) $(ARTIFACTS)/models/simple_lstm
	$(PY) -m moola.cli train \
		--model simple_lstm \
		--data $(TRAIN_DATA) \
		--split $(TEMPORAL_SPLIT) \
		--device $(DEVICE) \
		--seed $(SEED) \
		--over epochs=$(EPOCHS) \
		--save-run \
		2>&1 | tee $(LOGS_DIR)/train_simple_lstm_$(shell date +%Y%m%d_%H%M%S).log
	@echo "✅ Simple LSTM training complete - check $(LOGS_DIR) for details"

train-cpu: ## CPU training for development
	$(MAKE) train DEVICE=cpu MODEL=$(MODEL_BASELINE) EPOCHS=30

train-gpu: ## GPU training with pretrained encoder (production)
	@echo "🚀 GPU training with pretrained encoder..."
	@if [ ! -f "$(PRETRAINED_11D)" ]; then \
		echo "❌ Pretrained encoder not found: $(PRETRAINED_11D)"; \
		echo "💡 Run: make pretrain-11d or download pretrained weights"; \
		exit 1; \
	fi
	$(MAKE) train DEVICE=cuda MODEL=$(MODEL_PROD) \
		PRETRAINED_ENCODER=$(PRETRAINED_11D) \
		FREEZE_ENCODER=true EPOCHS=60

train-11d: ## Train with 11D RelativeTransform features
	@echo "🚀 Training with 11D RelativeTransform features..."
	@if [ ! -f "$(TRAIN_11D)" ]; then \
		echo "❌ 11D training data not found: $(TRAIN_11D)"; \
		exit 1; \
	fi
	$(MAKE) train DEVICE=$(DEVICE) MODEL=$(MODEL_PROD) \
		DATA=$(TRAIN_11D) INPUT_DIM=11 \
		PRETRAINED_ENCODER=$(PRETRAINED_11D)

# =============================================================================
# STONES COLLECTION TRAINING (Jade, Sapphire, Opal)
# =============================================================================

train-jade: ## Train Jade (moola-lstm-m-v1.0) - Production BiLSTM
	@echo "💎 Training Jade (moola-lstm-m-v1.0)..."
	@mkdir -p $(LOGS_DIR) $(ARTIFACTS)/models/jade
	$(PY) -m moola.cli train \
		--model jade \
		--data $(TRAIN_DATA) \
		--split $(TEMPORAL_SPLIT) \
		--device $(DEVICE) \
		--predict-pointers \
		--use-uncertainty-weighting \
		--bootstrap-ci \
		--compute-calibration \
		--save-run \
		--seed $(SEED) \
		--n-epochs $(EPOCHS) \
		--save-checkpoints \
		--monitor-gradients \
		2>&1 | tee $(LOGS_DIR)/train_jade_$(shell date +%Y%m%d_%H%M%S).log
	@echo "✅ Jade training complete"

train-sapphire: ## Train Sapphire (moola-preenc-fr-s-v1.0) - Frozen Encoder
	@echo "💎 Training Sapphire (moola-preenc-fr-s-v1.0)..."
	@if [ ! -f "$(PRETRAINED_11D)" ]; then \
		echo "❌ Pretrained encoder not found: $(PRETRAINED_11D)"; \
		echo "💡 Run: make pretrain-11d"; \
		exit 1; \
	fi
	@mkdir -p $(LOGS_DIR) $(ARTIFACTS)/models/sapphire
	$(PY) -m moola.cli train \
		--model sapphire \
		--data $(TRAIN_DATA) \
		--split $(TEMPORAL_SPLIT) \
		--device $(DEVICE) \
		--pretrained-encoder $(PRETRAINED_11D) \
		--freeze-encoder \
		--use-uncertainty-weighting \
		--bootstrap-ci \
		--compute-calibration \
		--save-run \
		--seed $(SEED) \
		--n-epochs $(EPOCHS) \
		2>&1 | tee $(LOGS_DIR)/train_sapphire_$(shell date +%Y%m%d_%H%M%S).log
	@echo "✅ Sapphire training complete"

train-opal: ## Train Opal (moola-preenc-ad-m-v1.0) - Adaptive Fine-tuning
	@echo "💎 Training Opal (moola-preenc-ad-m-v1.0)..."
	@if [ ! -f "$(PRETRAINED_11D)" ]; then \
		echo "❌ Pretrained encoder not found: $(PRETRAINED_11D)"; \
		echo "💡 Run: make pretrain-11d"; \
		exit 1; \
	fi
	@mkdir -p $(LOGS_DIR) $(ARTIFACTS)/models/opal
	$(PY) -m moola.cli train \
		--model opal \
		--data $(TRAIN_DATA) \
		--split $(TEMPORAL_SPLIT) \
		--device $(DEVICE) \
		--pretrained-encoder $(PRETRAINED_11D) \
		--unfreeze-encoder-after 10 \
		--use-uncertainty-weighting \
		--bootstrap-ci \
		--compute-calibration \
		--save-run \
		--seed $(SEED) \
		--n-epochs $(EPOCHS) \
		2>&1 | tee $(LOGS_DIR)/train_opal_$(shell date +%Y%m%d_%H%M%S).log
	@echo "✅ Opal training complete"

stones: ## Train all Stones models (Jade, Sapphire, Opal)
	@echo "💎 Training all Stones models..."
	$(MAKE) train-jade DEVICE=$(DEVICE) EPOCHS=$(EPOCHS)
	$(MAKE) train-sapphire DEVICE=$(DEVICE) EPOCHS=$(EPOCHS)
	$(MAKE) train-opal DEVICE=$(DEVICE) EPOCHS=$(EPOCHS)
	@echo "✅ All Stones models trained"

eval: ## Evaluate with temporal K-fold CV
	@echo "📊 Starting evaluation workflow..."
	@mkdir -p $(REPORTS_DIR)/eval
	$(PY) -m moola.cli evaluate \
		--model $(MODEL_PROD) \
		--split-dir $(SPLITS_DIR) \
		--num-folds 5 \

		2>&1 | tee $(LOGS_DIR)/eval_$(MODEL_PROD)_$(shell date +%Y%m%d_%H%M%S).log
	@echo "✅ Evaluation complete - check artifacts/models/$(MODEL_PROD)/metrics.json"

eval-bootstrap: ## Bootstrap confidence intervals only
	@echo "📈 Computing bootstrap confidence intervals..."
	$(MAKE) eval RESAMPLES=2000 CONFIDENCE=0.95

eval-calibration: ## Calibration metrics and reliability plots
	@echo "🎯 Computing calibration metrics..."
	$(PY) -m moola.cli train \
		--model $(MODEL_PROD) \
		--data $(TRAIN_DATA) \
		--split $(TEMPORAL_SPLIT) \
		--device $(DEVICE) \
		--predict-pointers \
		--compute-calibration \
		--plot-reliability \
		--calibration-bins 15 \

		--save-run \
		--seed $(SEED) \
		--n-epochs 10

calibrate: ## Temperature scaling + ECE analysis
	@echo "🌡️  Running temperature scaling calibration..."
	$(MAKE) eval-calibration

report: ## Generate figures + markdown summary
	@echo "📋 Generating comprehensive report..."
	@mkdir -p $(REPORTS_DIR)
	@python3 scripts/generate_report.py $(MODEL_PROD)
	@echo "✅ Report generated in $(REPORTS_DIR)"

report-full: ## Complete analysis with all figures
	@echo "📊 Generating full analysis report..."
	$(MAKE) eval-bootstrap
	$(MAKE) eval-calibration
	$(MAKE) report
	@echo "✅ Full report complete"

report-quick: ## Quick summary metrics only
	@echo "⚡ Generating quick summary..."
	@if [ -f "$(ARTIFACTS)/models/$(MODEL_PROD)/metrics.json" ]; then \
		cat $(ARTIFACTS)/models/$(MODEL_PROD)/metrics.json | jq '.accuracy, .f1, .precision, .recall'; \
	else \
		echo "❌ No metrics found - run 'make eval' first"; \
	fi

ablate: ## Loss balancing + dropout sweeps
	@echo "🔬 Running ablation studies..."
	$(MAKE) ablate-loss
	$(MAKE) ablate-dropout
	@echo "✅ Ablation studies complete"

ablate-loss: ## Loss weighting sweep (uncertainty vs manual)
	@echo "⚖️  Loss weighting ablation study..."
	@mkdir -p $(REPORTS_DIR)/ablation
	@echo "Testing uncertainty weighting vs manual λ weights..."
	# Uncertainty weighting (optimal)
	$(PY) -m moola.cli train \
		--model $(MODEL_PROD) \
		--data $(TRAIN_DATA) \
		--split $(TEMPORAL_SPLIT) \
		--device $(DEVICE) \
		--predict-pointers \
		--use-uncertainty-weighting \
		--seed $(SEED) \
		--over epochs=20 \
		--save-run
	# Manual weights (suboptimal baseline)
	$(PY) -m moola.cli train \
		--model $(MODEL_PROD) \
		--data $(TRAIN_DATA) \
		--split $(TEMPORAL_SPLIT) \
		--device $(DEVICE) \
		--predict-pointers \
		--no-use-uncertainty-weighting \
		--seed $(SEED) \
		--over epochs=20 \
		--save-run
	@echo "✅ Loss weighting ablation complete"

ablate-dropout: ## Dropout rate sweep for uncertainty estimation
	@echo "🎲 Dropout rate ablation study..."
	@mkdir -p $(REPORTS_DIR)/ablation
	@for rate in 0.1 0.15 0.2 0.25; do \
		echo "Testing dropout rate: $$rate"; \
		$(PY) -m moola.cli train \
			--model $(MODEL_PROD) \
			--data $(TRAIN_DATA) \
			--split $(TEMPORAL_SPLIT) \
			--device $(DEVICE) \
			--predict-pointers \
			--use-uncertainty-weighting \
			--mc-dropout \
			--mc-dropout-rate $$rate \
			--mc-passes 50 \
			--seed $(SEED) \
			--over epochs=20 \
			--save-run; \
	done
	@echo "✅ Dropout ablation complete"

# =============================================================================
# UTILITIES
# =============================================================================

format: ## Format code with black and isort
	@echo "🎨 Formatting code..."
	$(PY) -m black src tests
	$(PY) -m isort src tests
	@echo "✅ Code formatting complete"

test: ## Run all tests
	@echo "🧪 Running test suite..."
	$(PY) -m pytest tests/ -v --tb=short
	@echo "✅ Tests complete"

lint: ## Code quality checks
	@echo "🔍 Running code quality checks..."
	@echo "Running Black formatting check..."
	$(PY) -m black --check src tests
	@echo "Running isort import check..."
	$(PY) -m isort --check-only src tests
	@echo "Running Ruff linter..."
	$(PY) -m ruff check src tests
	@echo "✅ Code quality checks passed"

clean: ## Clean artifacts and logs
	@echo "🧹 Cleaning artifacts and logs..."
	@rm -rf $(LOGS_DIR)/*
	@rm -rf $(REPORTS_DIR)/*
	@find $(ARTIFACTS)/runs -name "*.json" -mtime +7 -delete 2>/dev/null || true
	@find $(ARTIFACTS)/models -name "*.pkl" -mtime +7 -delete 2>/dev/null || true
	@echo "✅ Cleanup complete"

doctor: ## Environment validation
	@echo "🩺 Running environment validation..."
	$(PY) -m moola.cli doctor
	@echo "✅ Environment validation complete"

install: ## Install dependencies
	@echo "📦 Installing dependencies..."
	$(PIP) install -e .
	$(PIP) install -U pre-commit black isort ruff
	$(PY) -m pre_commit install
	@echo "✅ Installation complete"

check-data: ## Validate training data
	@echo "🔍 Validating training data..."
	@if [ ! -f "$(TRAIN_DATA)" ]; then \
		echo "❌ Training data not found: $(TRAIN_DATA)"; \
		echo "💡 Run data ingestion pipeline first"; \
		exit 1; \
	fi
	@echo "import pandas as pd" > /tmp/check_data.py
	@echo "import numpy as np" >> /tmp/check_data.py
	@echo "from pathlib import Path" >> /tmp/check_data.py
	@echo "data_path = Path('$(TRAIN_DATA)')" >> /tmp/check_data.py
	@echo "if data_path.exists():" >> /tmp/check_data.py
	@echo "    df = pd.read_parquet(data_path)" >> /tmp/check_data.py
	@echo "    print(f'✅ Training data found: {len(df)} samples')" >> /tmp/check_data.py
	@echo "    print(f'📊 Shape: {df.shape}')" >> /tmp/check_data.py
	@echo "    print(f'📋 Columns: {list(df.columns)}')" >> /tmp/check_data.py
	@echo "    if 'label' in df.columns:" >> /tmp/check_data.py
	@echo "        print(f'🏷️  Label distribution: {df[\"label\"].value_counts().to_dict()}')" >> /tmp/check_data.py
	@echo "    if 'features' in df.columns:" >> /tmp/check_data.py
	@echo "        sample_features = df['features'].iloc[0]" >> /tmp/check_data.py
	@echo "        if isinstance(sample_features, (list, np.ndarray)):" >> /tmp/check_data.py
	@echo "            print(f'🔢 Feature shape: {np.array(sample_features).shape}')" >> /tmp/check_data.py
	@echo "        else:" >> /tmp/check_data.py
	@echo "            print(f'🔢 Feature type: {type(sample_features)}')" >> /tmp/check_data.py
	@echo "else:" >> /tmp/check_data.py
	@echo "    print('❌ Training data not found')" >> /tmp/check_data.py
	$(PY) /tmp/check_data.py
	@rm -f /tmp/check_data.py

check-env: ## Check environment setup
	@echo "🔍 Checking environment setup..."
	@echo "Python version: $$($(PY) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "PYTHONPATH: $$PYTHONPATH"
	@if command -v nvidia-smi >/dev/null 2>&1; then \
		echo "GPU: Available"; \
		nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1; \
	else \
		echo "GPU: Not available"; \
	fi
	@echo "✅ Environment check complete"

# =============================================================================
# PRETRAINING UTILITIES
# =============================================================================

pretrain-11d: ## Pretrain BiLSTM encoder on 11D data
	@echo "🧠 Pretraining BiLSTM encoder on 11D data..."
	@mkdir -p $(ARTIFACTS)/encoders/pretrained
	$(PY) -m moola.cli pretrain-bilstm \
		--input data/raw/unlabeled_windows.parquet \
		--output $(PRETRAINED_11D) \
		--device cuda \
		--epochs 50 \
		--input-dim 11 \
		--mask-strategy patch \
		--batch-size 256
	@echo "✅ 11D pretrained encoder saved to $(PRETRAINED_11D)"

pretrain-4d: ## Pretrain BiLSTM encoder on 4D OHLC data
	@echo "🧠 Pretraining BiLSTM encoder on 4D OHLC data..."
	@mkdir -p $(ARTIFACTS)/encoders/pretrained
	$(PY) -m moola.cli pretrain-bilstm \
		--input data/raw/unlabeled_windows.parquet \
		--output $(PRETRAINED_4D) \
		--device cuda \
		--epochs 50 \
		--input-dim 4 \
		--mask-strategy patch \
		--batch-size 256
	@echo "✅ 4D pretrained encoder saved to $(PRETRAINED_4D)"

# =============================================================================
# ADVANCED WORKFLOWS
# =============================================================================

ensemble: ## Train ensemble of models
	@echo "👥 Training ensemble models..."
	@for model in logreg rf xgb $(MODEL_BASELINE) $(MODEL_PROD); do \
		echo "Training $$model..."; \
		$(PY) -m moola.cli train \
			--model $$model \
			--data $(TRAIN_DATA) \
			--split $(TEMPORAL_SPLIT) \
			--device $(DEVICE) \
			--seed $(SEED) \
			--n-epochs 30; \
	done
	@echo "✅ Ensemble training complete"

oof-generate: ## Generate out-of-fold predictions for stacking
	@echo "🔄 Generating out-of-fold predictions..."
	@for model in logreg rf xgb $(MODEL_BASELINE) $(MODEL_PROD); do \
		echo "Generating OOF for $$model..."; \
		$(PY) -m moola.cli oof \
			--model $$model \
			--seed $(SEED) \
			--device $(DEVICE); \
	done
	@echo "✅ OOF generation complete"

stack-train: ## Train stacking ensemble
	@echo "📚 Training stacking ensemble..."
	$(PY) -m moola.cli train \
		--model stack \
		--data $(TRAIN_DATA) \
		--split $(TEMPORAL_SPLIT) \
		--device $(DEVICE) \
		--seed $(SEED)
	@echo "✅ Stacking ensemble trained"

# =============================================================================
# MONITORING AND DEBUGGING
# =============================================================================

monitor-training: ## Monitor training progress
	@echo "📊 Monitoring training progress..."
	@if [ -f "$(LOGS_DIR)/train_$(MODEL_PROD)_$(shell date +%Y%m%d).log" ]; then \
		tail -f $(LOGS_DIR)/train_$(MODEL_PROD)_$(shell date +%Y%m%d).log; \
	else \
		echo "❌ No training log found for today"; \
		echo "💡 Start training with: make train"; \
	fi

profile-model: ## Profile model performance
	@echo "⚡ Profiling model performance..."
	@echo "import time" > /tmp/profile_model.py
	@echo "import numpy as np" >> /tmp/profile_model.py
	@echo "from pathlib import Path" >> /tmp/profile_model.py
	@echo "import torch" >> /tmp/profile_model.py
	@echo "model_path = Path('$(ARTIFACTS)/models/$(MODEL_PROD)/model.pkl')" >> /tmp/profile_model.py
	@echo "if model_path.exists():" >> /tmp/profile_model.py
	@echo "    import pickle" >> /tmp/profile_model.py
	@echo "    with open(model_path, 'rb') as f:" >> /tmp/profile_model.py
	@echo "        model = pickle.load(f)" >> /tmp/profile_model.py
	@echo "    if hasattr(model, 'input_dim'):" >> /tmp/profile_model.py
	@echo "        input_dim = model.input_dim" >> /tmp/profile_model.py
	@echo "    else:" >> /tmp/profile_model.py
	@echo "        input_dim = 11" >> /tmp/profile_model.py
	@echo "    X_dummy = np.random.randn(100, 105, input_dim).astype(np.float32)" >> /tmp/profile_model.py
	@echo "    device = getattr(model, 'device', 'cpu')" >> /tmp/profile_model.py
	@echo "    for _ in range(10):" >> /tmp/profile_model.py
	@echo "        _ = model.predict(X_dummy[:10])" >> /tmp/profile_model.py
	@echo "    start_time = time.time()" >> /tmp/profile_model.py
	@echo "    for _ in range(100):" >> /tmp/profile_model.py
	@echo "        _ = model.predict(X_dummy)" >> /tmp/profile_model.py
	@echo "    end_time = time.time()" >> /tmp/profile_model.py
	@echo "    avg_time = (end_time - start_time) / 100" >> /tmp/profile_model.py
	@echo "    throughput = 100 / avg_time" >> /tmp/profile_model.py
	@echo "    print(f'✅ Model profiling complete:')" >> /tmp/profile_model.py
	@echo "    print(f'   Average inference time: {avg_time*1000:.2f} ms per sample')" >> /tmp/profile_model.py
	@echo "    print(f'   Throughput: {throughput:.1f} samples/second')" >> /tmp/profile_model.py
	@echo "    print(f'   Device: {device}')" >> /tmp/profile_model.py
	@echo "    print(f'   Input shape: {X_dummy.shape}')" >> /tmp/profile_model.py
	@echo "else:" >> /tmp/profile_model.py
	@echo "    print('❌ Model not found - run make train first')" >> /tmp/profile_model.py
	$(PY) /tmp/profile_model.py
	@rm -f /tmp/profile_model.py

# =============================================================================
# DEFAULT TARGET
# =============================================================================

.DEFAULT_GOAL := help