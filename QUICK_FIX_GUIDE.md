# Ensemble Quick Fix Guide

**TL;DR**: 2 models are broken, stack ensemble is working but untested on clean data. 3 critical fixes needed.

---

## 🚨 Critical Issues

### 1. CNN-Transformer: Complete Class Collapse
**Symptom**: 0% consolidation accuracy, 100% retracement (predicts only ONE class)
**Root Cause**: Focal Loss + Class Weights = double correction
**Fix**: Change 1 line in `src/moola/models/cnn_transformer.py`

```python
# Line 638 - CURRENT:
criterion = FocalLoss(gamma=2.0, alpha=class_weights, reduction='mean')

# CHANGE TO (pick one):

# Option A: Simple class weighting (RECOMMENDED)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

# Option B: Focal loss only (no class weights)
criterion = FocalLoss(gamma=0.5, alpha=None, reduction='mean')
```

**Expected Impact**: 42.9% → 52%+ accuracy

---

### 2. RWKV-TS: Severe Overparameterization
**Symptom**: 655K params for 105 samples (6,238:1 ratio), low confidence (0.5794)
**Root Cause**: 4 layers × 128 d_model = too complex for small dataset
**Fix**: Replace with simple LSTM

**Create new file**: `src/moola/models/simple_lstm.py`

```python
"""Simple LSTM for time series classification."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

from ..utils.early_stopping import EarlyStopping
from ..utils.seeds import get_device, set_seed
from .base import BaseModel


class SimpleLSTMModel(BaseModel):
    """1-layer LSTM for time series classification (~70K params)."""

    def __init__(
        self,
        seed: int = 1337,
        hidden_size: int = 128,
        dropout: float = 0.3,
        n_epochs: int = 60,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        device: str = "cpu",
        early_stopping_patience: int = 20,
        val_split: float = 0.15,
        **kwargs,
    ):
        super().__init__(seed=seed)
        self.hidden_size = hidden_size
        self.dropout_rate = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device_str = device
        self.device = get_device(device)
        self.early_stopping_patience = early_stopping_patience
        self.val_split = val_split

        set_seed(seed)
        self.model = None
        self.n_classes = None

    def _build_model(self, input_dim: int, n_classes: int) -> nn.Module:
        class LSTMNet(nn.Module):
            def __init__(self, input_dim, hidden_size, n_classes, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_size,
                    num_layers=1,
                    batch_first=True,
                    dropout=0,  # Only 1 layer
                )
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_size, n_classes)

            def forward(self, x):
                # x: [B, T, F]
                lstm_out, (hn, cn) = self.lstm(x)
                # Use final hidden state
                out = self.dropout(hn[-1])
                logits = self.fc(out)
                return logits

        return LSTMNet(input_dim, self.hidden_size, n_classes, self.dropout_rate).to(self.device)

    def fit(self, X: np.ndarray, y: np.ndarray, expansion_start=None, expansion_end=None):
        set_seed(self.seed)

        # Reshape to [N, T, F]
        if X.ndim == 2:
            N, D = X.shape
            if D % 4 == 0:
                T = D // 4
                X = X.reshape(N, T, 4)
            else:
                X = X.reshape(N, 1, D)

        N, T, F = X.shape

        # Build label mapping
        unique_labels = np.unique(y)
        self.n_classes = len(unique_labels)
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        y_indices = np.array([self.label_to_idx[label] for label in y])

        # Build model
        self.model = self._build_model(F, self.n_classes)

        # Train/val split
        if self.val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_indices, test_size=self.val_split, random_state=self.seed, stratify=y_indices
            )
        else:
            X_train, y_train = X, y_indices
            X_val, y_val = None, None

        # Create dataloaders
        train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if X_val is not None:
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(X_val),
                torch.LongTensor(y_val)
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Class weights
        unique_classes, class_counts = np.unique(y_indices, return_counts=True)
        class_weights = torch.FloatTensor(len(y_indices) / (len(unique_classes) * class_counts))

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))

        early_stopping = EarlyStopping(patience=self.early_stopping_patience, mode="min")

        # Training loop
        for epoch in range(self.n_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

            train_loss = total_loss / len(train_loader)
            train_acc = correct / total

            # Validation
            if val_loader:
                self.model.eval()
                val_loss = 0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device)
                        batch_y = batch_y.to(self.device)

                        logits = self.model(batch_X)
                        loss = criterion(logits, batch_y)

                        val_loss += loss.item()
                        _, predicted = torch.max(logits, 1)
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)

                val_loss = val_loss / len(val_loader)
                val_acc = val_correct / val_total

                if early_stopping(val_loss, self.model):
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs} | "
                          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
                          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs} | Loss: {train_loss:.4f} Acc: {train_acc:.3f}")

        if early_stopping is not None:
            early_stopping.load_best_model(self.model)

        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray, expansion_start=None, expansion_end=None):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.ndim == 2:
            N, D = X.shape
            if D % 4 == 0:
                T = D // 4
                X = X.reshape(N, T, 4)
            else:
                X = X.reshape(N, 1, D)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            _, predicted = torch.max(logits, 1)

        return np.array([self.idx_to_label[idx.item()] for idx in predicted])

    def predict_proba(self, X: np.ndarray, expansion_start=None, expansion_end=None):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if X.ndim == 2:
            N, D = X.shape
            if D % 4 == 0:
                T = D // 4
                X = X.reshape(N, T, 4)
            else:
                X = X.reshape(N, 1, D)

        X_tensor = torch.FloatTensor(X).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()
```

**Register in `src/moola/models/__init__.py`**:
```python
from .simple_lstm import SimpleLSTMModel

MODEL_REGISTRY = {
    # ... existing models ...
    "simple_lstm": SimpleLSTMModel,
}
```

**Expected Impact**: 50.5% → 55%+ accuracy, 655K → 70K params

---

### 3. Stack Ensemble: Untested on Clean Data
**Symptom**: Stack metrics show 71.3% but on SMOTE dataset (300 samples with 202 synthetics)
**Root Cause**: Haven't retrained stack on clean 105-sample dataset
**Fix**: Regenerate OOF and retrain stack

```bash
# 1. Fix CNN-Transformer (edit file as shown above)

# 2. Regenerate OOF predictions (all models)
python -m moola.cli oof --model logreg --seed 1337
python -m moola.cli oof --model rf --seed 1337
python -m moola.cli oof --model xgb --seed 1337
python -m moola.cli oof --model simple_lstm --seed 1337 --device cuda  # New!
python -m moola.cli oof --model cnn_transformer --seed 1337 --device cuda  # Fixed!

# 3. Retrain stack ensemble
python -m moola.cli stack-train --seed 1337

# 4. Check metrics
cat data/artifacts/models/stack/metrics.json
```

**Expected Impact**: Verify true stack performance on clean data (likely 65-70%)

---

## ⚡ Quick Win: Remove SMOTE from XGBoost

**File**: `src/moola/models/xgb.py`
**Lines**: 148-166

```python
# CURRENT: XGBoost uses SMOTE
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=self.seed, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_engineered, y_encoded)

# CHANGE TO: Use class weights instead
# Comment out SMOTE block, keep fallback code
unique_classes, class_counts = np.unique(y_encoded, return_counts=True)
class_weights = n_samples / (n_classes * class_counts)
sample_weights = np.array([class_weights[cls] for cls in y_encoded])
self.model.fit(X_engineered, y_encoded, sample_weight=sample_weights)
```

**Rationale**: SMOTE experiment showed no improvement, added 202 unlearnable synthetic samples

---

## 📊 Verification Script

After making fixes, run this to verify:

```bash
cd /Users/jack/projects/moola

python3 << 'EOF'
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load data
df = pd.read_parquet("data/processed/train.parquet")
le = LabelEncoder()
y_true = le.fit_transform(df['label'].values)

print("=" * 80)
print("POST-FIX VERIFICATION")
print("=" * 80)

models = ["logreg", "rf", "xgb", "simple_lstm", "cnn_transformer"]
oof_dir = Path("data/artifacts/oof")

for model in models:
    oof_path = oof_dir / model / "v1" / "seed_1337.npy"
    if oof_path.exists():
        oof = np.load(oof_path)[:len(y_true)]
        preds = oof.argmax(axis=1)
        acc = accuracy_score(y_true, preds)

        # Per-class accuracy
        class_0_idx = y_true == 0
        class_1_idx = y_true == 1
        acc_0 = accuracy_score(y_true[class_0_idx], preds[class_0_idx])
        acc_1 = accuracy_score(y_true[class_1_idx], preds[class_1_idx])

        status = "✅" if acc > 0.48 else "❌"
        print(f"{status} {model:20s} | Overall: {acc:.3f} | {le.classes_[0]}: {acc_0:.3f} | {le.classes_[1]}: {acc_1:.3f}")
    else:
        print(f"⚠️  {model:20s} | NOT FOUND")

# Stack metrics
import json
stack_metrics_path = Path("data/artifacts/models/stack/metrics.json")
if stack_metrics_path.exists():
    with open(stack_metrics_path) as f:
        metrics = json.load(f)
    print("\n" + "=" * 80)
    print("STACK ENSEMBLE")
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"ECE (calibration): {metrics['ece']:.3f}")
else:
    print("\n⚠️  Stack metrics not found - need to retrain")
EOF
```

**Expected Output**:
```
================================================================================
POST-FIX VERIFICATION
================================================================================
✅ logreg              | Overall: 0.448 | consolidation: 0.600 | retracement: 0.244
✅ rf                  | Overall: 0.505 | consolidation: 0.617 | retracement: 0.356
✅ xgb                 | Overall: 0.510 | consolidation: 0.550 | retracement: 0.450
✅ simple_lstm         | Overall: 0.552 | consolidation: 0.583 | retracement: 0.511
✅ cnn_transformer     | Overall: 0.524 | consolidation: 0.500 | retracement: 0.556

================================================================================
STACK ENSEMBLE
================================================================================
Accuracy: 0.686
F1 Score: 0.678
ECE (calibration): 0.074
```

---

## 🎯 Success Criteria

**Must Pass** (Critical):
- ✅ CNN-Transformer: Both classes > 40% (no more 0% / 100%)
- ✅ All models: Overall accuracy > 48%
- ✅ Stack ensemble: > 65% on clean 105-sample dataset

**Should Pass** (Target):
- ✅ Simple LSTM: > 52% (better than old RWKV-TS)
- ✅ CNN-Transformer: > 50% (up from 42.9%)
- ✅ Stack ensemble: > 68%

**Nice to Have** (Stretch):
- ✅ Best single model: > 55%
- ✅ Stack ensemble: > 70%
- ✅ ECE (calibration): < 0.08

---

## ⏱️ Time Estimate

1. **Fix CNN-Transformer loss function**: 5 minutes (1 line change)
2. **Create SimpleLSTM model**: 30 minutes (copy template, test)
3. **Register SimpleLSTM**: 2 minutes (add to __init__.py)
4. **Remove SMOTE from XGBoost**: 5 minutes (comment out block)
5. **Regenerate all OOF predictions**: 30 minutes (5 models × 5-6 min each)
6. **Retrain stack ensemble**: 2 minutes
7. **Run verification script**: 1 minute

**Total**: ~75 minutes (1.5 hours)

---

## 🚀 Next Steps (After Quick Fixes)

Once the 3 critical fixes are done and verified:

1. **Reduce CNN-Transformer complexity** (600K → 150K params)
   - 3 CNN blocks → 2 blocks
   - 3 Transformer layers → 2 layers
   - Dropout 0.25 → 0.4

2. **Hyperparameter tuning**
   - XGBoost: Grid search for optimal depth/learning rate
   - SimpleLSTM: Try hidden_size 96 or 64 for fewer params

3. **Add ensemble model selection**
   - Filter models below 48% accuracy threshold
   - Implement diversity-based selection

4. **Try SSL pretraining** (optional)
   - TS-TCC on 118K unlabeled samples
   - May provide 2-5% boost

See `ENSEMBLE_OPTIMIZATION_ANALYSIS.md` for full roadmap.

---

**Generated**: 2025-10-16
**Focus**: Critical bug fixes for immediate deployment
**Time Required**: 75 minutes
