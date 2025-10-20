# Moola Architecture Refactor - Visual Guide

## 🎯 Before & After Comparison

### BEFORE: Current Structure (Cluttered)
```
moola/
├── 📄 .mcp.json                       ❌ Empty, duplicates global
├── 📄 .env                            ❌ Unused GLM_API_KEY
├── 📄 claude_code_zai_env.sh          ❌ Duplicate of ~/dotfiles
├── 📄 *_agent.md (4 files)            ❌ OpenCode agents (global)
├── 📄 *_command.md (4 files)          ❌ OpenCode commands (global)
├── 📄 model_174_baseline.pkl          ❌ Scattered artifact
├── 📄 model_174_pretrained.pkl        ❌ Scattered artifact
├── 📄 feature_metadata_174.json       ❌ Scattered metadata
├── 📄 test_oof.npy                    ❌ Scattered OOF
├── 📁 test_splits/                    ❌ Duplicate of data/splits/
├── 📁 runpod_bundle_*/                ❌ Build artifacts at root
├── 📁 runpod_results/                 ❌ Duplicate of artifacts/
├── 📄 CLEANUP_SUMMARY_*.md            ❌ Temporary docs
├── 📄 WELCOME_BACK.md                 ❌ Temporary docs
├── 📁 data/
│   ├── processed/
│   │   ├── train_latest.parquet       ⚠️  Unclear: 4D or 11D?
│   │   ├── train_clean.parquet        ⚠️  Unclear: 98 or 174 samples?
│   │   ├── train_combined_174.parquet ⚠️  Unclear: Why 3 versions?
│   │   ├── train_combined_175.parquet ⚠️  Unclear: Why 3 versions?
│   │   ├── train_combined_178.parquet ⚠️  Unclear: Why 3 versions?
│   │   └── train_smote_300.parquet    ⚠️  Unclear: Synthetic?
│   ├── pretraining/
│   │   ├── unlabeled_features.npy     ⚠️  Unclear: 4D or 11D?
│   │   └── unlabeled_ohlc.npy         ⚠️  Unclear: Same as features?
│   └── oof/
│       ├── simple_lstm_clean.npy      ⚠️  Unclear: Supervised or pretrained?
│       └── simple_lstm_augmented.npy  ⚠️  Unclear: What augmentation?
└── 📁 models/
    └── pretrained/
        └── bilstm_encoder.pt          ⚠️  Unclear: Encoder or model?
```

### AFTER: Proposed Structure (Clean)
```
moola/
├── 📄 .env.example                    ✅ Example only (no secrets)
├── 📄 .gitignore                      ✅ Essential
├── 📄 pyproject.toml                  ✅ Essential
├── 📄 Makefile                        ✅ Essential
├── 📄 README.md                       ✅ Essential
├── 📄 CLAUDE.md                       ✅ Essential (Claude Code context)
├── 📄 RUNPOD_QUICK_START.md           ✅ Essential (workflow guide)
│
├── 📁 data/
│   ├── raw/
│   │   ├── unlabeled/
│   │   │   └── unlabeled_windows.parquet  # 2.2M samples, 4D OHLC
│   │   └── labeled/
│   │       └── (future: raw labeled data)
│   │
│   ├── processed/
│   │   ├── unlabeled/
│   │   │   ├── unlabeled_4d_ohlc.npy      # 2.2M × (105, 4)
│   │   │   └── unlabeled_11d_relative.npy # 2.2M × (105, 11)
│   │   │
│   │   ├── labeled/
│   │   │   ├── train_latest.parquet       # 174 samples (current)
│   │   │   ├── train_latest_4d.npy        # 174 × (105, 4)
│   │   │   ├── train_latest_11d.npy       # 174 × (105, 11)
│   │   │   └── metadata/
│   │   │       ├── feature_metadata_174.json
│   │   │       └── dataset_manifest.json
│   │   │
│   │   └── archived/
│   │       ├── train_clean.parquet        # 98 samples (before batch 200)
│   │       ├── train_combined_174.parquet # Historical
│   │       ├── train_smote_300.parquet    # Synthetic augmentation
│   │       └── README.md                  # Explains each dataset
│   │
│   ├── splits/                            # Train/val/test splits
│   ├── batches/                           # Annotation batches
│   └── corrections/                       # Human annotations
│
└── 📁 artifacts/
    ├── encoders/
    │   ├── pretrained/
    │   │   ├── bilstm_mae_4d_v1.pt        # BiLSTM MAE (4D OHLC)
    │   │   ├── bilstm_mae_11d_v1.pt       # BiLSTM MAE (11D Relative)
    │   │   ├── ts2vec_encoder_v1.pt       # TS2Vec contrastive
    │   │   └── tstcc_encoder_v1.pt        # TSTCC contrastive
    │   └── supervised/
    │       └── (future: encoder blocks from supervised training)
    │
    ├── models/
    │   ├── supervised/                    # No pretraining
    │   │   ├── simple_lstm_baseline_174.pkl
    │   │   ├── enhanced_simple_lstm_174.pkl
    │   │   ├── cnn_transformer_174.pkl
    │   │   ├── rwkv_ts_174.pkl
    │   │   ├── logreg_174.pkl
    │   │   ├── rf_174.pkl
    │   │   └── xgb_174.pkl
    │   │
    │   ├── pretrained/                    # Fine-tuned from pretrained encoders
    │   │   ├── simple_lstm_bilstm_mae_4d_174.pkl
    │   │   ├── simple_lstm_bilstm_mae_11d_174.pkl
    │   │   └── enhanced_simple_lstm_ts2vec_174.pkl
    │   │
    │   └── ensemble/                      # Stacking ensemble
    │       ├── stack_rf_meta_174.pkl
    │       └── stack_logreg_meta_174.pkl
    │
    ├── oof/                               # Out-of-fold predictions
    │   ├── supervised/
    │   │   ├── simple_lstm_174.npy
    │   │   ├── enhanced_simple_lstm_174.npy
    │   │   ├── cnn_transformer_174.npy
    │   │   ├── logreg_174.npy
    │   │   ├── rf_174.npy
    │   │   └── xgb_174.npy
    │   └── pretrained/
    │       ├── simple_lstm_bilstm_mae_4d_174.npy
    │       └── simple_lstm_bilstm_mae_11d_174.npy
    │
    ├── metadata/
    │   ├── feature_metadata_174.json
    │   ├── dataset_manifest.json
    │   └── experiment_registry.json
    │
    ├── runpod_bundles/
    │   ├── runpod_bundle_20251020_013740.tar.gz
    │   └── runpod_bundle_build/
    │
    └── runpod_results/
        ├── phase2_results.csv
        └── oof/
```

---

## 🔄 Data Flow: Unlabeled → Labeled → Models

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA (Never Modified)                    │
├─────────────────────────────────────────────────────────────────┤
│ data/raw/unlabeled/unlabeled_windows.parquet                    │
│   • 2.2M samples                                                │
│   • 105 bars × 4 channels (OHLC)                                │
│   • Source: NQ futures 1-min data                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              PROCESSED UNLABELED (For Pretraining)              │
├─────────────────────────────────────────────────────────────────┤
│ data/processed/unlabeled/                                       │
│   ├── unlabeled_4d_ohlc.npy       # (2.2M, 105, 4)             │
│   └── unlabeled_11d_relative.npy  # (2.2M, 105, 11)            │
│                                                                 │
│ Feature Transform:                                              │
│   4D OHLC → 11D RelativeTransform                               │
│   [O, H, L, C] → [O_rel, H_rel, L_rel, C_rel, range, body,     │
│                   upper_wick, lower_wick, HL_mid, OC_mid, vol] │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  PRETRAINED ENCODERS                            │
├─────────────────────────────────────────────────────────────────┤
│ artifacts/encoders/pretrained/                                  │
│   ├── bilstm_mae_4d_v1.pt         # Masked Autoencoder (4D)    │
│   ├── bilstm_mae_11d_v1.pt        # Masked Autoencoder (11D)   │
│   ├── ts2vec_encoder_v1.pt        # TS2Vec contrastive         │
│   └── tstcc_encoder_v1.pt         # TSTCC contrastive          │
│                                                                 │
│ Self-Supervised Learning:                                       │
│   • Masked Autoencoder: Mask 15% of timesteps, reconstruct     │
│   • TS2Vec: Hierarchical contrastive learning                  │
│   • TSTCC: Temporal contrastive coding                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              PROCESSED LABELED (For Supervised Training)        │
├─────────────────────────────────────────────────────────────────┤
│ data/processed/labeled/                                         │
│   ├── train_latest.parquet        # 174 samples (current)      │
│   ├── train_latest_4d.npy         # (174, 105, 4)              │
│   ├── train_latest_11d.npy        # (174, 105, 11)             │
│   └── metadata/                                                 │
│       ├── feature_metadata_174.json                             │
│       └── dataset_manifest.json                                 │
│                                                                 │
│ Source: Human annotations via Candlesticks project              │
│   • Batch 200: 33 keepers (16.6% keeper rate)                  │
│   • Previous batches: 141 samples                               │
│   • Total: 174 labeled samples                                  │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SUPERVISED MODELS                            │
├─────────────────────────────────────────────────────────────────┤
│ artifacts/models/supervised/  (No pretraining)                  │
│   ├── simple_lstm_baseline_174.pkl                              │
│   ├── enhanced_simple_lstm_174.pkl                              │
│   ├── cnn_transformer_174.pkl                                   │
│   ├── rwkv_ts_174.pkl                                           │
│   ├── logreg_174.pkl                                            │
│   ├── rf_174.pkl                                                │
│   └── xgb_174.pkl                                               │
│                                                                 │
│ artifacts/models/pretrained/  (Fine-tuned from pretrained)      │
│   ├── simple_lstm_bilstm_mae_4d_174.pkl                         │
│   ├── simple_lstm_bilstm_mae_11d_174.pkl                        │
│   └── enhanced_simple_lstm_ts2vec_174.pkl                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                OUT-OF-FOLD PREDICTIONS                          │
├─────────────────────────────────────────────────────────────────┤
│ artifacts/oof/supervised/                                       │
│   ├── simple_lstm_174.npy         # (174, 2) probabilities     │
│   ├── enhanced_simple_lstm_174.npy                              │
│   ├── cnn_transformer_174.npy                                   │
│   ├── logreg_174.npy                                            │
│   ├── rf_174.npy                                                │
│   └── xgb_174.npy                                               │
│                                                                 │
│ artifacts/oof/pretrained/                                       │
│   ├── simple_lstm_bilstm_mae_4d_174.npy                         │
│   └── simple_lstm_bilstm_mae_11d_174.npy                        │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ENSEMBLE MODELS                              │
├─────────────────────────────────────────────────────────────────┤
│ artifacts/models/ensemble/                                      │
│   ├── stack_rf_meta_174.pkl       # Random Forest meta-learner │
│   └── stack_logreg_meta_174.pkl   # Logistic Regression meta   │
│                                                                 │
│ Stacking Strategy:                                              │
│   • Input: OOF predictions from all base models                 │
│   • Meta-learner: Random Forest or Logistic Regression          │
│   • Output: Final ensemble prediction                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🏗️ Encoder vs Model vs Weights Taxonomy

### Terminology Clarification

```
┌──────────────────────────────────────────────────────────────────┐
│                         ENCODER                                  │
├──────────────────────────────────────────────────────────────────┤
│ Definition: Reusable feature extraction block                    │
│                                                                  │
│ Examples:                                                        │
│   • BiLSTM encoder (128 hidden units)                            │
│   • RWKV-TS time-mixing block                                    │
│   • CNN-Transformer local convolution block                      │
│                                                                  │
│ Characteristics:                                                 │
│   • Input: (batch, seq_len, features)                            │
│   • Output: (batch, seq_len, hidden_dim) or (batch, hidden_dim) │
│   • No classification head                                       │
│   • Can be pretrained or trained from scratch                    │
│   • Can be frozen or fine-tuned                                  │
│                                                                  │
│ Storage:                                                         │
│   artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt              │
│   artifacts/encoders/supervised/rwkv_ts_encoder_174.pt           │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                          MODEL                                   │
├──────────────────────────────────────────────────────────────────┤
│ Definition: Complete architecture for training/inference         │
│                                                                  │
│ Examples:                                                        │
│   • EnhancedSimpleLSTM = BiLSTM encoder + attention + classifier │
│   • CNN-Transformer = CNN encoder + Transformer + classifier     │
│   • RWKV-TS = RWKV encoder + classifier                          │
│                                                                  │
│ Characteristics:                                                 │
│   • Input: (batch, seq_len, features)                            │
│   • Output: (batch, num_classes) logits or probabilities         │
│   • Includes classification head                                 │
│   • Can load pretrained encoder weights                          │
│   • Trained end-to-end on labeled data                           │
│                                                                  │
│ Storage:                                                         │
│   artifacts/models/supervised/enhanced_simple_lstm_174.pkl       │
│   artifacts/models/pretrained/simple_lstm_bilstm_mae_4d_174.pkl  │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                         WEIGHTS                                  │
├──────────────────────────────────────────────────────────────────┤
│ Definition: Saved checkpoint (encoder or model)                  │
│                                                                  │
│ Types:                                                           │
│   1. Pretrained encoder weights (.pt)                            │
│      • From self-supervised learning                             │
│      • Can be loaded into multiple models                        │
│      • Example: bilstm_mae_4d_v1.pt                              │
│                                                                  │
│   2. Fine-tuned model weights (.pkl)                             │
│      • From supervised training                                  │
│      • Includes encoder + classifier                             │
│      • Example: simple_lstm_bilstm_mae_4d_174.pkl                │
│                                                                  │
│   3. Ensemble meta-learner weights (.pkl)                        │
│      • From stacking ensemble training                           │
│      • Example: stack_rf_meta_174.pkl                            │
│                                                                  │
│ File Extensions:                                                 │
│   • .pt  = PyTorch state_dict (encoder only)                     │
│   • .pkl = Pickled model (encoder + classifier)                  │
└──────────────────────────────────────────────────────────────────┘
```

### Example: SimpleLSTM with Pretrained BiLSTM Encoder

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRETRAINING PHASE                            │
├─────────────────────────────────────────────────────────────────┤
│ Input: 2.2M unlabeled windows (4D OHLC)                         │
│   ↓                                                             │
│ BiLSTM Masked Autoencoder                                       │
│   • Encoder: BiLSTM (128 hidden)                                │
│   • Decoder: MLP (reconstruct masked values)                    │
│   • Loss: MSE on masked timesteps                               │
│   ↓                                                             │
│ Save encoder weights only:                                      │
│   artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt             │
│   • Contains: BiLSTM state_dict                                 │
│   • Size: ~500KB                                                │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   FINE-TUNING PHASE                             │
├─────────────────────────────────────────────────────────────────┤
│ Input: 174 labeled windows (4D OHLC)                            │
│   ↓                                                             │
│ SimpleLSTM Model                                                │
│   • Load pretrained encoder:                                    │
│     bilstm_mae_4d_v1.pt → BiLSTM (128 hidden)                   │
│   • Freeze encoder (first 10 epochs)                            │
│   • Add classifier: Linear(128 → 2)                             │
│   • Train on labeled data                                       │
│   • Unfreeze encoder (last 10 epochs)                           │
│   ↓                                                             │
│ Save complete model:                                            │
│   artifacts/models/pretrained/simple_lstm_bilstm_mae_4d_174.pkl │
│   • Contains: BiLSTM + classifier state_dict                    │
│   • Size: ~600KB                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Naming Convention Examples

### Encoders
```
Format: {architecture}_{pretraining_method}_{features}_v{version}.pt

bilstm_mae_4d_v1.pt              # BiLSTM, Masked Autoencoder, 4D OHLC, v1
bilstm_mae_11d_v1.pt             # BiLSTM, Masked Autoencoder, 11D Relative, v1
ts2vec_encoder_v1.pt             # TS2Vec, contrastive, v1
tstcc_encoder_v1.pt              # TSTCC, contrastive, v1
rwkv_ts_encoder_v1.pt            # RWKV-TS, state-space, v1
```

### Models (Supervised)
```
Format: {architecture}_{dataset_size}.pkl

simple_lstm_baseline_174.pkl     # SimpleLSTM, no pretraining, 174 samples
enhanced_simple_lstm_174.pkl     # EnhancedSimpleLSTM, no pretraining, 174 samples
cnn_transformer_174.pkl          # CNN-Transformer, no pretraining, 174 samples
rwkv_ts_174.pkl                  # RWKV-TS, no pretraining, 174 samples
logreg_174.pkl                   # Logistic Regression, 174 samples
rf_174.pkl                       # Random Forest, 174 samples
xgb_174.pkl                      # XGBoost, 174 samples
```

### Models (Pretrained)
```
Format: {architecture}_{encoder}_{features}_{dataset_size}.pkl

simple_lstm_bilstm_mae_4d_174.pkl       # SimpleLSTM + BiLSTM MAE (4D), 174 samples
simple_lstm_bilstm_mae_11d_174.pkl      # SimpleLSTM + BiLSTM MAE (11D), 174 samples
enhanced_simple_lstm_ts2vec_174.pkl     # EnhancedSimpleLSTM + TS2Vec, 174 samples
cnn_transformer_tstcc_174.pkl           # CNN-Transformer + TSTCC, 174 samples
```

### OOF Predictions
```
Format: {architecture}_{encoder}_{features}_{dataset_size}.npy

# Supervised (no pretraining)
simple_lstm_174.npy              # SimpleLSTM, 174 samples
enhanced_simple_lstm_174.npy     # EnhancedSimpleLSTM, 174 samples

# Pretrained
simple_lstm_bilstm_mae_4d_174.npy       # SimpleLSTM + BiLSTM MAE (4D), 174 samples
simple_lstm_bilstm_mae_11d_174.npy      # SimpleLSTM + BiLSTM MAE (11D), 174 samples
```

---

## 🔍 Quick Reference: Where to Find Things

### "Where is the current training dataset?"
```
data/processed/labeled/train_latest.parquet  # 174 samples (current)
```

### "Where are the pretrained encoder weights?"
```
artifacts/encoders/pretrained/bilstm_mae_4d_v1.pt  # BiLSTM MAE (4D OHLC)
artifacts/encoders/pretrained/bilstm_mae_11d_v1.pt # BiLSTM MAE (11D Relative)
```

### "Where are the trained models?"
```
artifacts/models/supervised/enhanced_simple_lstm_174.pkl  # No pretraining
artifacts/models/pretrained/simple_lstm_bilstm_mae_4d_174.pkl  # With pretraining
```

### "Where are the OOF predictions for stacking?"
```
artifacts/oof/supervised/simple_lstm_174.npy
artifacts/oof/supervised/enhanced_simple_lstm_174.npy
artifacts/oof/supervised/cnn_transformer_174.npy
artifacts/oof/supervised/logreg_174.npy
artifacts/oof/supervised/rf_174.npy
artifacts/oof/supervised/xgb_174.npy
```

### "Where is the unlabeled data for pretraining?"
```
data/raw/unlabeled/unlabeled_windows.parquet  # 2.2M samples (raw)
data/processed/unlabeled/unlabeled_4d_ohlc.npy  # 2.2M samples (processed, 4D)
data/processed/unlabeled/unlabeled_11d_relative.npy  # 2.2M samples (processed, 11D)
```

### "Where are the annotation batches?"
```
data/batches/batch_200.parquet  # Latest batch
data/batches/batch_200_clean_keepers.parquet  # 33 keepers
```

### "Where are the historical datasets?"
```
data/processed/archived/train_clean.parquet  # 98 samples (before batch 200)
data/processed/archived/train_combined_174.parquet  # Historical
data/processed/archived/train_smote_300.parquet  # Synthetic augmentation
```

---

## ✅ Checklist: Post-Refactor Verification

### Root Directory
- [ ] Only essential files at root (≤15 files)
- [ ] No AI agent configs (.mcp.json, .env, *_agent.md, *_command.md)
- [ ] No scattered artifacts (model_*.pkl, feature_*.json, test_*.npy)
- [ ] No duplicate directories (test_splits/, runpod_results/)
- [ ] .env.example exists (no secrets)

### Data Directory
- [ ] Clear separation: raw/ vs processed/
- [ ] Clear separation: unlabeled/ vs labeled/
- [ ] Clear naming: 4D vs 11D features
- [ ] Historical datasets archived with README
- [ ] OOF predictions organized by source (supervised/pretrained)

### Artifacts Directory
- [ ] Encoders separated from models
- [ ] Pretrained encoders clearly named
- [ ] Models organized by training type (supervised/pretrained/ensemble)
- [ ] Consistent naming convention
- [ ] Metadata directory exists

### Code
- [ ] All path references updated
- [ ] Tests pass
- [ ] CLI commands work
- [ ] RunPod bundle creation works

### Documentation
- [ ] CLAUDE.md updated
- [ ] README.md updated
- [ ] docs/ARCHITECTURE.md updated
- [ ] Migration guide created

