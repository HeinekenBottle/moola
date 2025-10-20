# PyTorch 2.x & Python 3.11+ Migration Patterns

## Quick Reference Card

### PyTorch 2.x AMP API

#### Pattern 1: GradScaler Initialization
```python
# BEFORE (Deprecated)
scaler = torch.cuda.amp.GradScaler()

# AFTER (PyTorch 2.x)
scaler = torch.amp.GradScaler('cuda')
```

#### Pattern 2: Autocast Context Manager
```python
# BEFORE (Deprecated)
with torch.cuda.amp.autocast():
    output = model(input)

# AFTER (PyTorch 2.x)
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
```

#### Pattern 3: Complete Training Loop
```python
# BEFORE (Deprecated)
scaler = torch.cuda.amp.GradScaler() if use_amp else None

if use_amp:
    with torch.cuda.amp.autocast():
        loss = model(x)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# AFTER (PyTorch 2.x)
scaler = torch.amp.GradScaler('cuda') if use_amp else None

if use_amp:
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        loss = model(x)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Python 3.11+ Type Hints (PEP 604)

#### Pattern 1: Optional Types
```python
# BEFORE (Python 3.9)
from typing import Optional

def foo(x: Optional[int]) -> Optional[str]:
    pass

# AFTER (Python 3.11+)
def foo(x: int | None) -> str | None:
    pass
```

#### Pattern 2: Union Types
```python
# BEFORE (Python 3.9)
from typing import Union

def foo(x: Union[str, int]) -> Union[list, dict]:
    pass

# AFTER (Python 3.11+)
def foo(x: str | int) -> list | dict:
    pass
```

#### Pattern 3: Generic Collections
```python
# BEFORE (Python 3.9)
from typing import Dict, List, Tuple

def foo() -> Dict[str, List[Tuple[int, str]]]:
    pass

# AFTER (Python 3.11+)
def foo() -> dict[str, list[tuple[int, str]]]:
    pass
```

#### Pattern 4: Combined
```python
# BEFORE (Python 3.9)
from typing import Optional, Union, Dict, List

def foo(x: Optional[int], y: Union[str, int]) -> Dict[str, List[Any]]:
    pass

# AFTER (Python 3.11+)
from typing import Any  # Only import what's still needed

def foo(x: int | None, y: str | int) -> dict[str, list[Any]]:
    pass
```

## File-by-File Changes

### 1. src/moola/models/simple_lstm.py

**Line 339:**
```python
# OLD
scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

# NEW
scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
```

**Line 389:**
```python
# OLD
with torch.cuda.amp.autocast():
    logits = self.model(batch_X_aug)
    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    logits = self.model(batch_X_aug)
    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
```

**Line 424:**
```python
# OLD
with torch.cuda.amp.autocast():
    logits = self.model(batch_X)
    loss = criterion(logits, batch_y)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    logits = self.model(batch_X)
    loss = criterion(logits, batch_y)
```

### 2. src/moola/models/cnn_transformer.py

**Line 34:**
```python
# OLD
from typing import Optional, Union

# NEW
from typing import Literal
```

**Line 388:**
```python
# OLD
def forward(self, x: torch.Tensor) -> Union[torch.Tensor, dict]:

# NEW
def forward(self, x: torch.Tensor) -> torch.Tensor | dict:
```

**Lines 492-494:**
```python
# OLD
expansion_start: Optional[np.ndarray] = None,
expansion_end: Optional[np.ndarray] = None,

# NEW
expansion_start: np.ndarray | None = None,
expansion_end: np.ndarray | None = None,
```

**Line 678:**
```python
# OLD
scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

# NEW
scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
```

**Line 758:**
```python
# OLD
with torch.cuda.amp.autocast():
    outputs = self.model(batch_X_aug)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    outputs = self.model(batch_X_aug)
```

**Line 846:**
```python
# OLD
with torch.cuda.amp.autocast():
    outputs = self.model(batch_X)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    outputs = self.model(batch_X)
```

### 3. src/moola/pretraining/masked_lstm_pretrain.py

**Line 27:**
```python
# OLD
from typing import Dict, Literal, Optional

# NEW
from typing import Literal
```

**Lines 120-122:**
```python
# OLD
save_path: Optional[Path] = None,
verbose: bool = True
) -> Dict[str, list]:

# NEW
save_path: Path | None = None,
verbose: bool = True
) -> dict[str, list]:
```

**Line 222:**
```python
# OLD
scaler = torch.cuda.amp.GradScaler()

# NEW
scaler = torch.amp.GradScaler('cuda')
```

**Line 258:**
```python
# OLD
with torch.cuda.amp.autocast():
    reconstruction = self.model(x_masked)
    loss, loss_dict = self.model.compute_loss(reconstruction, batch_X, mask)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    reconstruction = self.model(x_masked)
    loss, loss_dict = self.model.compute_loss(reconstruction, batch_X, mask)
```

**Line 313:**
```python
# OLD
with torch.cuda.amp.autocast():
    reconstruction = self.model(x_masked)
    loss, loss_dict = self.model.compute_loss(reconstruction, batch_X, mask)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    reconstruction = self.model(x_masked)
    loss, loss_dict = self.model.compute_loss(reconstruction, batch_X, mask)
```

**Line 431:**
```python
# OLD
) -> Dict[str, np.ndarray]:

# NEW
) -> dict[str, np.ndarray]:
```

### 4. src/moola/config/performance_config.py

**Line 301:**
```python
# OLD (in docstring)
with torch.cuda.amp.autocast():
    loss = model(x)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    loss = model(x)
```

**Lines 308-312:**
```python
# OLD
return torch.cuda.amp.GradScaler(
    growth_factor=AMP_GROWTH_FACTOR,
    backoff_factor=AMP_BACKOFF_FACTOR,
    growth_interval=AMP_GROWTH_INTERVAL,
)

# NEW
return torch.amp.GradScaler(
    'cuda',
    growth_factor=AMP_GROWTH_FACTOR,
    backoff_factor=AMP_BACKOFF_FACTOR,
    growth_interval=AMP_GROWTH_INTERVAL,
)
```

### 5. src/moola/models/rwkv_ts.py

**Line 407:**
```python
# OLD
scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

# NEW
scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
```

**Line 442:**
```python
# OLD
with torch.cuda.amp.autocast():
    logits = self.model(batch_X_aug)
    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    logits = self.model(batch_X_aug)
    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
```

**Line 479:**
```python
# OLD
with torch.cuda.amp.autocast():
    logits = self.model(batch_X)
    loss = criterion(logits, batch_y)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    logits = self.model(batch_X)
    loss = criterion(logits, batch_y)
```

### 6. src/moola/models/ts_tcc.py

**Line 26:**
```python
# OLD
from typing import Optional, Tuple

# NEW
from typing import Literal
```

**Line 450:**
```python
# OLD
scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

# NEW
scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
```

**Line 478:**
```python
# OLD
with torch.cuda.amp.autocast():
    z1 = self.model(x_aug1)
    z2 = self.model(x_aug2)
    loss = info_nce_loss(z1, z2, temperature=self.temperature)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    z1 = self.model(x_aug1)
    z2 = self.model(x_aug2)
    loss = info_nce_loss(z1, z2, temperature=self.temperature)
```

**Line 517:**
```python
# OLD
with torch.cuda.amp.autocast():
    z1 = self.model(x_aug1)
    z2 = self.model(x_aug2)
    loss = info_nce_loss(z1, z2, temperature=self.temperature)

# NEW
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    z1 = self.model(x_aug1)
    z2 = self.model(x_aug2)
    loss = info_nce_loss(z1, z2, temperature=self.temperature)
```

## Global Search & Replace Commands

### Using grep + sed

```bash
# Find all deprecated AMP usages
grep -r "torch\.cuda\.amp\." src/moola --include="*.py"

# Replace GradScaler (dry run)
grep -rl "torch.cuda.amp.GradScaler()" src/moola --include="*.py" | \
  xargs sed -i.bak 's/torch\.cuda\.amp\.GradScaler()/torch.amp.GradScaler('\''cuda'\'')/g'

# Replace autocast (dry run - more complex, manual review recommended)
grep -rl "torch.cuda.amp.autocast()" src/moola --include="*.py" | \
  xargs sed -i.bak 's/torch\.cuda\.amp\.autocast()/torch.amp.autocast(device_type='\''cuda'\'', dtype=torch.float16)/g'

# Verify no deprecated usages remain
grep -r "torch\.cuda\.amp\." src/moola --include="*.py" | wc -l
# Should output: 0
```

### Using find + perl

```bash
# Replace GradScaler
find src/moola -name "*.py" -exec perl -i -pe 's/torch\.cuda\.amp\.GradScaler\(\)/torch.amp.GradScaler('\''cuda'\'')/g' {} \;

# Replace autocast
find src/moola -name "*.py" -exec perl -i -pe 's/torch\.cuda\.amp\.autocast\(\)/torch.amp.autocast(device_type='\''cuda'\'', dtype=torch.float16)/g' {} \;
```

## Validation Commands

```bash
# Count deprecated usages (should be 0)
grep -r "torch\.cuda\.amp\." src/moola --include="*.py" | wc -l

# Count modern usages (should be 17)
grep -r "torch\.amp\." src/moola --include="*.py" | wc -l

# Find files with Optional/Union imports (for manual review)
grep -r "from typing import.*Optional\|from typing import.*Union" src/moola --include="*.py"

# Verify Python syntax
python3 -m py_compile src/moola/models/simple_lstm.py
python3 -m py_compile src/moola/models/cnn_transformer.py
python3 -m py_compile src/moola/pretraining/masked_lstm_pretrain.py

# Run quick import test
python3 -c "
import sys; sys.path.insert(0, 'src')
from moola.models.simple_lstm import SimpleLSTMModel
from moola.models.cnn_transformer import CnnTransformerModel
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
print('✅ All imports successful')
"
```

## Common Pitfalls

### 1. Missing device_type parameter
```python
# WRONG - Will raise TypeError
with torch.amp.autocast():  # Missing device_type!
    output = model(input)

# CORRECT
with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
```

### 2. Incorrect GradScaler argument order
```python
# WRONG - device is first positional arg
scaler = torch.amp.GradScaler(growth_factor=2.0, 'cuda')

# CORRECT - device is first
scaler = torch.amp.GradScaler('cuda', growth_factor=2.0)
```

### 3. Removing needed typing imports
```python
# WRONG - Literal is still needed!
from typing import Any  # Only Any, removed Literal

def foo(mode: Literal['train', 'eval']) -> None:  # Error: Literal not defined
    pass

# CORRECT - Keep Literal
from typing import Literal, Any

def foo(mode: Literal['train', 'eval']) -> None:
    pass
```

## IDE Support

### VS Code

Add to `.vscode/settings.json`:
```json
{
  "python.analysis.typeCheckingMode": "basic",
  "python.analysis.diagnosticMode": "workspace",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true
}
```

### PyCharm

1. Go to Settings → Editor → Inspections
2. Enable "Type Checker" inspection
3. Set Python version to 3.11+

## CI/CD Integration

### GitHub Actions

```yaml
name: Type Check & Modernization

on: [push, pull_request]

jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install torch>=2.0.0 numpy

      - name: Check for deprecated APIs
        run: |
          DEPRECATED=$(grep -r "torch\.cuda\.amp\." src/moola --include="*.py" | wc -l)
          if [ "$DEPRECATED" -gt 0 ]; then
            echo "❌ Found $DEPRECATED deprecated torch.cuda.amp usages"
            grep -r "torch\.cuda\.amp\." src/moola --include="*.py"
            exit 1
          fi
          echo "✅ No deprecated APIs found"

      - name: Verify imports
        run: |
          python3 -c "
          import sys; sys.path.insert(0, 'src')
          from moola.models.simple_lstm import SimpleLSTMModel
          from moola.models.cnn_transformer import CnnTransformerModel
          print('✅ All imports successful')
          "
```

## References

- [PyTorch 2.0 AMP Migration Guide](https://pytorch.org/docs/stable/amp.html)
- [PEP 604: Union Operator](https://peps.python.org/pep-0604/)
- [Python 3.11 Type Hints](https://docs.python.org/3.11/library/typing.html)
