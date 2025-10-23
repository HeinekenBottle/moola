# MOOLA Model Registry

Centralized model management with codename system following Stones specifications.

## Registry Format

```
moola-{family}-{size}{variant}-v{semver} // codename: {Stone}
```

### Components:
- **family**: Model family (lstm, preenc, etc.)
- **size**: Model size (s=small, m=medium, l=large, xl=extra-large)
- **variant**: Variant identifier (fr=frozen, ad=adaptive, etc.)
- **version**: Semantic version (v1.0, v1.1, etc.)
- **codename**: Stone name (Jade, Sapphire, Opal, etc.)

## Stones Collection

### JadeCompact - Minimal Viable BiLSTM
```
moola-lstm-s-v1.1 // codename: Jade-Compact
```

**Description**: Compact BiLSTM for small datasets with uncertainty-weighted multi-task learning

**Architecture**:
- BiLSTM(10→96×2, 1 layer) → projection (64) → dual heads
- Pointer head: center(sigmoid), length(sigmoid)
- Type head: 3-way logits
- Gradient clip 1.5–2.0
- ReduceLROnPlateau scheduler
- Early stop patience 20

**Stones Compliance**:
- ✅ Pointer = Center+Length with Huber δ≈0.08
- ✅ Loss = Uncertainty-weighted (Kendall) - NO manual λ
- ✅ Dropout: recurrent 0.7, dense 0.6, input 0.3
- ✅ Augment: jitter σ=0.03 + magnitude-warp σ=0.2, ×3 on-the-fly
- ✅ Uncertainty: MC Dropout 50–100 passes + Temperature Scaling

**Usage**:
```python
from moola.models import get_model

# Standard usage
model = get_model("jade", predict_pointers=True)
model.fit(X, y, expansion_start=starts, expansion_end=ends)

# Single-task mode
model = get_model("jade", predict_pointers=False)
model.fit(X, y)
```

## Model Registry API

### Basic Usage

```python
from moola.models.registry import registry, get_model

# Get model by ID
model = get_model("moola-lstm-m-v1.0", predict_pointers=True)

# Get model by codename
model = get_model("Jade", predict_pointers=True)

# List all models
models = registry.list_models()

# Get Stones collection
stones_models = registry.get_stones_models()
```

### Search Models

```python
# Search by family
lstm_models = registry.search(family="lstm")

# Search by size
medium_models = registry.search(size="m")

# Search Stones-compliant models
stones_models = registry.search(stones_compliant=True)
```

### Model Information

```python
# Get model info
info = registry.get("Jade")
print(f"Model ID: {info.model_id}")
print(f"Description: {info.description}")
print(f"Default params: {info.default_params}")
```

## Legacy Models

The registry also maintains backward compatibility with legacy models:

```python
from moola.models import get_model

# Legacy access (still supported)
model = get_model("jade", predict_pointers=True)
model = get_model("enhanced_simple_lstm", device="cuda")
model = get_model("simple_lstm")  # Baseline
```

## Adding New Models

To add a new model to the registry:

```python
from moola.models.registry import registry

# Register new model
registry.register(
    model_id="moola-transformer-l-v1.0",
    codename="Emerald",
    family="transformer",
    size="l",
    variant="",
    version="v1.0",
    description="Large Transformer architecture",
    model_class=EmeraldModel,
    default_params={"hidden_size": 256, "num_layers": 6},
    stones_compliant=True
)
```

## Model Validation

The registry validates model IDs and ensures Stones compliance:

```python
from moola.models.registry import validate_model_id

# Validate model ID format
is_valid = validate_model_id("moola-lstm-m-v1.0")  # True
is_valid = validate_model_id("invalid-format")     # False
```

## Stones Compliance Checklist

All Stones models must meet these requirements:

- [ ] Pointer encoding: Center+Length (not start/end)
- [ ] Loss function: Uncertainty-weighted (Kendall et al.)
- [ ] Huber loss δ≈0.08 for pointer regression
- [ ] Dropout configuration: recurrent 0.6–0.7, dense 0.4–0.5, input 0.2–0.3
- [ ] Data augmentation: jitter σ=0.03 + magnitude-warp σ=0.2, ×3 on-the-fly
- [ ] Uncertainty quantification: MC Dropout 50–100 passes + Temperature Scaling
- [ ] Gradient clipping: 1.5–2.0
- [ ] Scheduler: ReduceLROnPlateau
- [ ] Early stopping: patience 20

## Version Management

Model versions follow semantic versioning:
- **v1.0**: Initial release
- **v1.1**: Minor improvements (backward compatible)
- **v2.0**: Major changes (breaking changes)

## Deployment

For production deployment, use Jade:
```python
from moola.models import get_jade

model = get_jade(
    predict_pointers=True,
    device="cuda",
    use_amp=True,
    save_checkpoints=True
)
```

For research and experimentation, consider Sapphire or Opal for transfer learning benefits.