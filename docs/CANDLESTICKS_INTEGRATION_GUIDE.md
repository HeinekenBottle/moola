# Candlesticks Integration Guide - Moola Window Data Extraction

## Quick Reference: Window ID to Candlesticks Mapping

### The Format
```
window_id: "102_exp_1"
           ├── base_index = 102 (position in windowed sequence)
           └── exp_num = 1 (labeling iteration)

OHLC Data:
├── 105 bars total
├── Past: bars 0-29 (context)
├── Prediction: bars 30-74 (where pattern occurs)
└── Future: bars 75-104 (outcome)

Pattern Location (from labeled data):
├── expansion_start = 40 (bar where pattern begins)
├── expansion_end = 42 (bar where pattern ends, inclusive)
└── Pattern bars = [40, 41, 42] (3 bars)
```

---

## Extraction Workflows

### Workflow 1: Extract Labeled Samples for Visualization

**Use Case:** Show candlesticks for specific labeled patterns

```python
import pandas as pd
import numpy as np

def extract_labeled_candlesticks(
    sample_ids: list = None,  # e.g., ["0_exp_1", "102_exp_1"]
    limit: int = 10
):
    """Extract candlesticks from labeled training data."""
    
    # Load labeled data
    df = pd.read_parquet('data/processed/train_pivot_134.parquet')
    
    if sample_ids:
        df = df[df['window_id'].isin(sample_ids)]
    else:
        df = df.head(limit)
    
    candlesticks_list = []
    
    for _, row in df.iterrows():
        # Convert features to proper OHLC format
        ohlc = np.array([np.array(bar) for bar in row['features']])
        
        candlestick = {
            'window_id': row['window_id'],
            'label': row['label'],
            'full_window': ohlc,  # [105, 4] OHLC
            'pattern_start': row['expansion_start'],
            'pattern_end': row['expansion_end'],
            'pattern_bars': ohlc[row['expansion_start']:row['expansion_end']+1],
            'contexts': {
                'past': ohlc[0:30],       # [30, 4]
                'prediction': ohlc[30:75],  # [45, 4]
                'future': ohlc[75:105]    # [30, 4]
            }
        }
        candlesticks_list.append(candlestick)
    
    return candlesticks_list


# Usage
samples = extract_labeled_candlesticks(
    sample_ids=["0_exp_1", "102_exp_1", "238_exp_1"]
)

for cs in samples:
    print(f"Window: {cs['window_id']}")
    print(f"  Label: {cs['label']}")
    print(f"  Pattern: bars {cs['pattern_start']}-{cs['pattern_end']}")
    print(f"  Full OHLC shape: {cs['full_window'].shape}")
    print(f"  Pattern OHLC shape: {cs['pattern_bars'].shape}")
```

---

### Workflow 2: Extract by Sequential Range

**Use Case:** Get a range of consecutive labeled windows

```python
def extract_range_candlesticks(
    start_idx: int = 0,
    end_idx: int = 50,
    stride: int = 10
):
    """Extract windows matching base indices in range."""
    
    df = pd.read_parquet('data/processed/train_pivot_134.parquet')
    
    # Extract base indices from window_id
    df['base_idx'] = df['window_id'].str.split('_').str[0].astype(int)
    
    # Filter by range
    df_filtered = df[(df['base_idx'] >= start_idx) & (df['base_idx'] <= end_idx)]
    
    candlesticks_list = []
    
    for _, row in df_filtered.iterrows():
        ohlc = np.array([np.array(bar) for bar in row['features']])
        
        candlesticks_list.append({
            'window_id': row['window_id'],
            'base_idx': row['base_idx'],
            'label': row['label'],
            'ohlc': ohlc,
            'pattern_range': (row['expansion_start'], row['expansion_end'])
        })
    
    return candlesticks_list


# Usage: Get all labeled samples with base index 0-50
samples = extract_range_candlesticks(start_idx=0, end_idx=50)
print(f"Extracted {len(samples)} samples in range [0, 50]")
```

---

### Workflow 3: Group by Base Index (Multiple Experiments)

**Use Case:** Show how same window has multiple pattern interpretations

```python
def extract_grouped_candlesticks():
    """Extract windows grouped by base index."""
    
    df = pd.read_parquet('data/processed/train_pivot_134.parquet')
    
    # Extract base index
    df['base_idx'] = df['window_id'].str.split('_').str[0].astype(int)
    
    grouped = {}
    
    for base_idx, group in df.groupby('base_idx'):
        if len(group) > 1:  # Only keep windows with multiple experiments
            
            # Get shared OHLC (same for all experiments from same base)
            shared_ohlc = np.array([np.array(bar) for bar in group.iloc[0]['features']])
            
            experiments = []
            for _, row in group.iterrows():
                experiments.append({
                    'window_id': row['window_id'],
                    'exp_num': int(row['window_id'].split('_')[1]),
                    'label': row['label'],
                    'pattern_start': row['expansion_start'],
                    'pattern_end': row['expansion_end'],
                    'pattern_bars': shared_ohlc[row['expansion_start']:row['expansion_end']+1]
                })
            
            grouped[base_idx] = {
                'shared_ohlc': shared_ohlc,  # Same OHLC for all
                'experiments': experiments    # Different labels/patterns
            }
    
    return grouped


# Usage
grouped = extract_grouped_candlesticks()
print(f"Found {len(grouped)} windows with multiple pattern interpretations")

for base_idx, data in list(grouped.items())[:3]:
    print(f"\nBase window {base_idx}:")
    for exp in data['experiments']:
        print(f"  - {exp['window_id']}: {exp['label']} (bars {exp['pattern_start']}-{exp['pattern_end']})")
```

---

## Data Integrity Validation

### Check OHLC Relationships

```python
def validate_ohlc(ohlc_array):
    """Validate OHLC relationships in extracted candlesticks."""
    
    # ohlc_array: shape [N, 4] where columns are [open, high, low, close]
    
    open_prices = ohlc_array[:, 0]
    high_prices = ohlc_array[:, 1]
    low_prices = ohlc_array[:, 2]
    close_prices = ohlc_array[:, 3]
    
    issues = []
    
    # High must be >= all prices
    if not np.all(high_prices >= open_prices):
        issues.append("High < Open in some bars")
    if not np.all(high_prices >= close_prices):
        issues.append("High < Close in some bars")
    if not np.all(high_prices >= low_prices):
        issues.append("High < Low in some bars")
    
    # Low must be <= all prices
    if not np.all(low_prices <= open_prices):
        issues.append("Low > Open in some bars")
    if not np.all(low_prices <= close_prices):
        issues.append("Low > Close in some bars")
    
    return len(issues) == 0, issues


# Usage
df = pd.read_parquet('data/processed/train_pivot_134.parquet')
for idx, row in df.iterrows():
    ohlc = np.array([np.array(bar) for bar in row['features']])
    valid, issues = validate_ohlc(ohlc)
    if not valid:
        print(f"Sample {row['window_id']}: {issues}")
```

---

## Python Code Example: Complete Extraction

```python
"""
Complete example: Extract candlesticks from Moola labeled data
for visualization in Candlesticks component.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class MoolaCandlestickExtractor:
    """Extract candlestick data from Moola training data."""
    
    PARQUET_PATH = 'data/processed/train_pivot_134.parquet'
    WINDOW_SIZE = 105
    PAST_BARS = 30  # bars 0-29
    PREDICTION_BARS = 45  # bars 30-74
    FUTURE_BARS = 30  # bars 75-104
    
    def __init__(self):
        self.df = pd.read_parquet(self.PARQUET_PATH)
        self._extract_base_indices()
    
    def _extract_base_indices(self):
        """Extract numeric base index from window_id."""
        self.df['base_idx'] = self.df['window_id'].str.split('_').str[0].astype(int)
    
    def extract_by_window_id(self, window_ids: List[str]) -> List[Dict]:
        """Extract specific windows by window_id."""
        result = []
        for wid in window_ids:
            row = self.df[self.df['window_id'] == wid]
            if len(row) == 0:
                print(f"Warning: {wid} not found")
                continue
            result.append(self._row_to_candlestick(row.iloc[0]))
        return result
    
    def extract_range(self, start: int, end: int) -> List[Dict]:
        """Extract windows with base_idx in range [start, end]."""
        filtered = self.df[(self.df['base_idx'] >= start) & (self.df['base_idx'] <= end)]
        return [self._row_to_candlestick(row) for _, row in filtered.iterrows()]
    
    def extract_by_label(self, label: str, limit: int = None) -> List[Dict]:
        """Extract windows by label type."""
        filtered = self.df[self.df['label'] == label]
        if limit:
            filtered = filtered.head(limit)
        return [self._row_to_candlestick(row) for _, row in filtered.iterrows()]
    
    def extract_grouped(self) -> Dict[int, Dict]:
        """Extract windows grouped by base_idx, showing multiple experiments."""
        grouped = {}
        for base_idx, group in self.df.groupby('base_idx'):
            if len(group) > 1:
                ohlc = self._convert_features(group.iloc[0]['features'])
                exps = []
                for _, row in group.iterrows():
                    exps.append({
                        'window_id': row['window_id'],
                        'label': row['label'],
                        'pattern_range': (row['expansion_start'], row['expansion_end'])
                    })
                grouped[base_idx] = {
                    'ohlc': ohlc,
                    'experiments': exps
                }
        return grouped
    
    def _row_to_candlestick(self, row) -> Dict:
        """Convert DataFrame row to candlestick dict."""
        ohlc = self._convert_features(row['features'])
        
        start = row['expansion_start']
        end = row['expansion_end']
        
        return {
            'window_id': row['window_id'],
            'label': row['label'],
            'base_idx': row['base_idx'],
            'ohlc': ohlc,  # Full 105-bar window [105, 4]
            'pattern_start': start,
            'pattern_end': end,
            'pattern_bars': ohlc[start:end+1],  # [end-start+1, 4]
            'contexts': {
                'past': ohlc[0:self.PAST_BARS],
                'prediction': ohlc[self.PAST_BARS:self.PAST_BARS+self.PREDICTION_BARS],
                'future': ohlc[self.PAST_BARS+self.PREDICTION_BARS:]
            }
        }
    
    @staticmethod
    def _convert_features(features) -> np.ndarray:
        """Convert stored features to [105, 4] OHLC array."""
        return np.array([np.array(bar) for bar in features], dtype=np.float64)
    
    def validate_sample(self, window_id: str) -> Tuple[bool, List[str]]:
        """Validate a sample's OHLC data."""
        row = self.df[self.df['window_id'] == window_id]
        if len(row) == 0:
            return False, [f"Window {window_id} not found"]
        
        ohlc = self._convert_features(row.iloc[0]['features'])
        issues = []
        
        # Check shapes
        if ohlc.shape != (105, 4):
            issues.append(f"Invalid OHLC shape: {ohlc.shape}")
        
        # Check OHLC relationships
        o, h, l, c = ohlc[:, 0], ohlc[:, 1], ohlc[:, 2], ohlc[:, 3]
        if not np.all(h >= l):
            issues.append("High < Low in some bars")
        if not np.all(h >= np.maximum(o, c)):
            issues.append("High < max(Open, Close) in some bars")
        if not np.all(l <= np.minimum(o, c)):
            issues.append("Low > min(Open, Close) in some bars")
        
        # Check expansion indices
        start = row.iloc[0]['expansion_start']
        end = row.iloc[0]['expansion_end']
        if not (30 <= start <= end <= 74):
            issues.append(f"Invalid expansion indices: [{start}, {end}]")
        
        return len(issues) == 0, issues


# Usage Examples
if __name__ == "__main__":
    extractor = MoolaCandlestickExtractor()
    
    # Example 1: Extract specific windows
    samples = extractor.extract_by_window_id(["0_exp_1", "102_exp_1"])
    print(f"Extracted {len(samples)} samples")
    
    # Example 2: Extract by label
    consolidations = extractor.extract_by_label('consolidation', limit=5)
    print(f"Found {len(consolidations)} consolidation samples")
    
    # Example 3: Extract range
    range_samples = extractor.extract_range(0, 100)
    print(f"Extracted {len(range_samples)} samples in range [0, 100]")
    
    # Example 4: Validate a sample
    valid, issues = extractor.validate_sample("0_exp_1")
    print(f"Sample validation: {'PASS' if valid else 'FAIL'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
```

---

## Integration Points

### For React Candlesticks Component

```python
# API endpoint to fetch candlesticks for a window

def get_candlesticks_for_window(window_id: str):
    """Fetch candlestick data for API response."""
    
    extractor = MoolaCandlestickExtractor()
    samples = extractor.extract_by_window_id([window_id])
    
    if not samples:
        return {'error': f'Window {window_id} not found'}
    
    cs = samples[0]
    
    # Return format suitable for candlestick chart
    return {
        'window_id': cs['window_id'],
        'label': cs['label'],
        'bars': [
            {
                'open': float(bar[0]),
                'high': float(bar[1]),
                'low': float(bar[2]),
                'close': float(bar[3])
            }
            for bar in cs['ohlc']
        ],
        'pattern_region': {
            'start': cs['pattern_start'],
            'end': cs['pattern_end'],
            'highlighted_bars': [
                {
                    'open': float(bar[0]),
                    'high': float(bar[1]),
                    'low': float(bar[2]),
                    'close': float(bar[3])
                }
                for bar in cs['pattern_bars']
            ]
        }
    }
```

---

## File Locations Summary

| Path | Purpose |
|------|---------|
| `data/processed/train_pivot_134.parquet` | Source of all labeled data |
| `src/moola/data/dual_input_pipeline.py` | Reference for OHLC extraction logic |
| `src/moola/config/data_config.py` | Window structure constants |
| `src/moola/data_infra/schemas.py` | Data validation schemas |

