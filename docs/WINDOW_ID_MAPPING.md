# Moola Window ID Mapping & Data Structure Documentation

## Summary

This document explains how window IDs map to actual OHLC data in Moola, how labeled data references windows, and the complete data extraction pipeline.

---

## 1. Window ID Mapping Explained

### 1.1 Labeled Data Format (train_pivot_134.parquet)

**File:** `/Users/jack/projects/moola/data/processed/train_pivot_134.parquet`

**Structure:**
```
Columns: ['window_id', 'label', 'expansion_start', 'expansion_end', 'features']

Sample records:
- window_id: "0_exp_1"         (String format: base_window_index + experiment marker)
- window_id: "0_exp_2"         (Same base window, different expansion analysis)
- window_id: "102_exp_1"       (Base index 102, experiment 1)
- window_id: "238_exp_1"       (Highest base index: 238)

Total rows: 105 labeled samples
Unique base indices: 66 (range: 0-238)
```

### 1.2 Window ID Components

Each `window_id` follows the pattern: `{base_index}_{exp_num}`

- **base_index** (0-238): Sequential index into the raw OHLC data extraction pipeline
  - NOT a direct timestamp or row number in raw data
  - Derived from sliding window extraction with stride
  - Maps to a 105-bar window starting at position `base_index` in the windowed sequence

- **exp_num** (1, 2, ...): Experiment/variation marker
  - Multiple labeled samples can come from the same 105-bar window
  - Each represents a different pattern analysis or labeling iteration
  - Important: Same base index with different exp_num = different labels/expansions

**Example:** 
- `0_exp_1`: First labeled instance from base window 0 (consolidation, expansion 61-65)
- `0_exp_2`: Second labeled instance from base window 0 (retracement, expansion 61-65)
- Both reference the SAME raw OHLC bars but with different pattern interpretations

### 1.3 Base Index to Raw Data Mapping

The base index maps to raw OHLC data through sliding window extraction:

```
Window Extraction Logic (from extract_unlabeled_windows.py):
├─ Raw OHLC data loaded from file
├─ Sliding window applied: size=105, stride=configurable (typically 1 or 10)
├─ Each window assigned: window_id = f"window_{i}"
│  where i = start_index in raw data
└─ Later labeled windows reuse base numbers from this sequence

Example:
- Raw OHLC has 10,000 bars
- Sliding window (stride=10): extract windows at indices 0, 10, 20, 30, ...
- This produces window_0, window_10, window_20, ... up to ~1000 windows
- Labeled data (train_pivot_134) uses subset of these: 0, 10, 102, 103, 104, ...
```

**Important:** The labeled data was CREATED from this pipeline, so base indices are valid references.

---

## 2. OHLC Data Structure in Parquet

### 2.1 Features Column Format

The `features` column in parquet stores OHLC data as:

```python
# Stored as: array of 105 numpy arrays, each [open, high, low, close]
features = [
    np.array([19584.25, 19585.00, 19581.75, 19581.75]),  # Bar 0: OHLC
    np.array([19582.75, 19583.00, 19580.25, 19580.50]),  # Bar 1: OHLC
    # ... 103 more bars ...
]

# Shape when loaded: (105,) with dtype=object
# Type of each element: np.ndarray with dtype=float64

# To convert to standard 3D array:
X = np.array([np.array(bar) for bar in features])  # Shape: (105, 4)
```

### 2.2 Complete Window Structure (105 bars)

```
Breakdown of 105-bar sequence:
├─ Past context: bars 0-29 (30 bars)
│  Purpose: Historical context for the model
│
├─ Prediction window: bars 30-74 (45 bars)
│  Purpose: Where patterns emerge and are labeled
│  Valid expansion indices: [30, 74]
│
└─ Future outcome: bars 75-104 (30 bars)
   Purpose: Target/outcome for pattern validation
```

### 2.3 Expansion Indices

```python
expansion_start: int  # Index into 0-104 range where pattern begins
expansion_end: int    # Index where pattern ends (inclusive)

Valid range: [30, 74] (must be within prediction window)
Typical patterns: 5-23 bars long
Mean expansion: 6.6 bars

Example (Sample 0):
  expansion_start = 61
  expansion_end = 65
  → Pattern is bars 61, 62, 63, 64, 65 (5 bars)
  → Indices are within [30, 74] ✓
```

---

## 3. Data Loading Code Locations

### 3.1 Core Loading Functions

**File:** `/Users/jack/projects/moola/src/moola/data/load.py`
```python
def validate_expansions(df):
    """Remove samples with invalid expansion indices.
    
    Filters for:
    - expansion_start < expansion_end
    - Both in range [30, 74] (prediction window)
    """
```

### 3.2 Dual-Input Pipeline

**File:** `/Users/jack/projects/moola/src/moola/data/dual_input_pipeline.py`
```python
class DualInputDataProcessor:
    def _extract_raw_ohlc(self, df: pd.DataFrame) -> np.ndarray:
        """Extract raw OHLC data from features column.
        
        Returns: [N, 105, 4] array
        
        Code:
        X_ohlc = np.stack([np.stack(f) for f in df["features"]])
        """
    
    def _extract_engineered_features(
        self,
        X_ohlc: np.ndarray,
        expansion_start: Optional[np.ndarray],
        expansion_end: Optional[np.ndarray]
    ):
        """Extract features from expansion region only.
        
        Uses expansion_start/end to focus on pattern region:
        features_region = X_ohlc[i, expansion_start:expansion_end+1, :]
        """
```

### 3.3 Schemas & Validation

**File:** `/Users/jack/projects/moola/src/moola/data_infra/schemas.py`
```python
class TimeSeriesWindow(BaseModel):
    window_id: str
    features: List[List[float]]  # 105 timesteps × 4 OHLC
    
class LabeledWindow(TimeSeriesWindow):
    label: PatternLabel  # consolidation, retracement, expansion
    expansion_start: int  # Must be ≥30, ≤74
    expansion_end: int    # Must be ≥30, ≤74
```

**File:** `/Users/jack/projects/moola/src/moola/config/data_config.py`
```python
EXPECTED_WINDOW_LENGTH = 105
PAST_WINDOW_START = 0
PAST_WINDOW_END = 30
PREDICTION_WINDOW_START = 30
PREDICTION_WINDOW_END = 75  # Exclusive, so [30:75] = 45 bars
FUTURE_WINDOW_START = 75
FUTURE_WINDOW_END = 105

EXPANSION_START_MIN = 30
EXPANSION_START_MAX = 74
EXPANSION_END_MIN = 30
EXPANSION_END_MAX = 74
```

---

## 4. Window Creation Pipeline

### 4.1 Unlabeled Window Extraction

**Script:** `/Users/jack/projects/moola/scripts/archive/extract_unlabeled_windows.py`

```python
def extract_sliding_windows(
    ohlc_df: pd.DataFrame,
    window_size: int = 105,
    stride: int = 1,  # or 10 for faster extraction
    max_samples: int = 120000
) -> list[dict]:
    """Extract sliding windows from raw OHLC.
    
    Process:
    1. Load raw OHLC: [N_bars, 4] array
    2. Create sliding windows:
       - Position 0: bars [0-104]        → window_0
       - Position 1: bars [1-105]        → window_1 (if stride=1)
       - Position 10: bars [10-114]      → window_10 (if stride=10)
    3. Output: [N_windows, 105, 4]
    
    Output schema per window:
    {
        'window_id': f'window_{i}',      # e.g., "window_0"
        'start_idx': i,                  # Start position in raw data
        'end_idx': i + 105 - 1,          # End position
        'features': ohlc_array,          # [105, 4] OHLC
    }
    """
```

Key insight: **window_id numeric portion = direct index into raw data**

### 4.2 Labeled Data Ingestion

**Script:** `/Users/jack/projects/moola/scripts/archive/ingest_pivot_134_clean.py`

```python
def load_pivot_parquets(parquet_dir: Path) -> pd.DataFrame:
    """Load 134 pre-processed parquet files.
    
    For each file:
    1. Extract: window_id, label, start_idx, end_idx, ohlc
    2. Convert OHLC to [105, 4] array
    3. Create DataFrame:
       {
           'window_id': "0_exp_1",
           'label': "consolidation",
           'expansion_start': 61,
           'expansion_end': 65,
           'features': [[o,h,l,c], ...]  # 105 bars
       }
    """
```

---

## 5. Extraction Methods for Candlesticks Integration

### 5.1 Extract by Sequential Range (Recommended)

**Approach:** Extract windows by start index range

```python
def extract_windows_by_range(
    start_idx: int,
    end_idx: int,
    stride: int = 10
) -> List[tuple]:
    """Extract consecutive windows from [start_idx, end_idx].
    
    Example:
    - Extract windows 0, 10, 20, 30 (stride=10)
    - Returns raw OHLC bars [0-104], [10-114], [20-124], [30-134]
    
    Advantage:
    - Simple mapping: window_N → raw bars [N, N+104]
    - No need to store mapping
    - Sequential access in raw data
    """
```

### 5.2 Extract by Window ID List

**Approach:** Extract specific windows by ID

```python
def extract_windows_by_ids(window_ids: List[str]) -> List[tuple]:
    """Extract specific labeled windows.
    
    Example:
    - Input: ["0_exp_1", "102_exp_1", "238_exp_1"]
    - Extract base indices: [0, 102, 238]
    - Load corresponding windows from parquet or raw data
    
    Advantage:
    - Precise control
    - Can target specific labeled samples
    
    Disadvantage:
    - Requires mapping metadata
    """
```

### 5.3 Candlesticks Integration Strategy

**Recommended approach:**

```python
# Method A: Use labeled data directly
def extract_candlesticks_from_labeled(
    parquet_path: str,
    window_range: tuple = None  # (start_idx, end_idx)
) -> List[CandlestickData]:
    """Extract candlesticks from labeled data.
    
    1. Load parquet: df = pd.read_parquet(parquet_path)
    2. For each row (or filtered rows):
       - Get features: [105, 4] OHLC
       - Get expansion_start/end for highlighting
       - Convert to CandlestickData format
    3. Returns: [(window_id, ohlc_bars, pattern_region), ...]
    """

# Method B: Use sequential indices (cleaner)
def extract_candlesticks_from_raw(
    raw_ohlc_path: str,
    window_indices: List[int],  # e.g., [0, 10, 20, 30, ...]
    stride: int = 10
) -> List[CandlestickData]:
    """Extract candlesticks using sequential indices.
    
    1. Load raw OHLC
    2. For each index in window_indices:
       - Extract bars [index, index+104]
       - Assign window_id based on index/stride relationship
    3. Returns: CandlestickData array
    """
```

---

## 6. Complete Reference: How to Map Any Sample

**Given:** window_id = "102_exp_1" from labeled data

**Step 1: Extract base index**
```python
base_index = int("102_exp_1".split("_")[0])  # → 102
```

**Step 2: Load labeled sample**
```python
df = pd.read_parquet('data/processed/train_pivot_134.parquet')
sample = df[df['window_id'] == "102_exp_1"].iloc[0]

# Get the OHLC data
features = np.array([np.array(bar) for bar in sample['features']])
# Shape: (105, 4)
# Bars 0-104 are the full window
```

**Step 3: Extract pattern region**
```python
expansion_start = sample['expansion_start']  # e.g., 40
expansion_end = sample['expansion_end']      # e.g., 42
pattern_ohlc = features[expansion_start:expansion_end+1]  # [3, 4]

# Bars 40, 41, 42 contain the consolidation/retracement pattern
```

**Step 4: Map to candlesticks**
```python
candlestick_data = {
    'window_id': "102_exp_1",
    'bars': features,  # [105, 4]
    'pattern_start': expansion_start,  # 40
    'pattern_end': expansion_end,      # 42
    'label': sample['label'],  # "consolidation"
    'pattern_region_ohlc': pattern_ohlc  # [3, 4] for rendering
}
```

---

## 7. Data Integrity Checks

### 7.1 Verify window structure
```python
assert X_ohlc.shape == (n_samples, 105, 4), "OHLC shape invalid"
assert np.all(X_ohlc[:, :, 1] >= X_ohlc[:, :, 0]), "High < Open somewhere"
assert np.all(X_ohlc[:, :, 1] >= X_ohlc[:, :, 3]), "High < Close somewhere"
```

### 7.2 Verify expansion indices
```python
assert np.all(expansion_start >= 30), "Start < 30"
assert np.all(expansion_end <= 74), "End > 74"
assert np.all(expansion_start <= expansion_end), "Start > End"
```

### 7.3 Verify window_id format
```python
pattern = r'^(\d+)_exp_(\d+)$'
for wid in df['window_id']:
    assert re.match(pattern, wid), f"Invalid window_id format: {wid}"
```

---

## 8. Key Takeaways for Candlesticks Integration

1. **Window IDs are NOT stored as sequential integers in labeled data**
   - Format: `{base_index}_{exp_num}` (e.g., "102_exp_1")
   - Base index maps to position in windowed extraction

2. **Features column has 105 OHLC bars per sample**
   - Stored as array of arrays
   - Convert to [105, 4] numpy array for processing

3. **Expansion indices define the pattern region**
   - Always in range [30, 74]
   - Should be highlighted/emphasized in candlestick charts

4. **Best extraction approach for Candlesticks:**
   - Use labeled parquet directly
   - No need to go back to raw data
   - Have both full window (105 bars) and pattern region readily available

5. **Validation is critical**
   - Check OHLC relationships (High >= Low, High >= Open, High >= Close)
   - Check expansion indices are within valid range
   - Verify window IDs follow correct format

---

## Files to Reference

| File | Purpose |
|------|---------|
| `src/moola/data_infra/schemas.py` | Pydantic schemas for data validation |
| `src/moola/config/data_config.py` | Constants and specifications |
| `src/moola/data/load.py` | Expansion validation logic |
| `src/moola/data/dual_input_pipeline.py` | OHLC extraction code |
| `scripts/archive/extract_unlabeled_windows.py` | Window creation algorithm |
| `scripts/archive/ingest_pivot_134_clean.py` | Labeled data ingestion |
| `data/processed/train_pivot_134.parquet` | Actual labeled data |

