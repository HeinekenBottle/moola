# Fix Plan: Relative Feature Engineering Pipeline

## Problem Analysis
The current relative feature engineering implementation has several issues:
1. **Wrong data source**: Not using the 5-year NQ data file at `data/raw/nq_ohlcv_1min_2020-09_2025-09_fixed.parquet`
2. **Volume dependency**: Current code expects volume column, but volume should be removed
3. **Pipeline complexity**: Current implementation is overly complex and may not work with the actual data structure

## Proposed Solution

### Phase 1: Data Integration
- Update `parquet_loader.py` to use the correct 5-year data file path
- Examine the actual structure of the NQ data file to understand column format
- Remove volume column handling from the data loader

### Phase 2: Feature Engineering Fixes
- Simplify `relativity.py` to work with 4D OHLC data (no volume)
- Fix the `build_features()` function to properly handle the data structure
- Ensure zigzag features work with 4D input
- Update feature output to maintain 10 dimensions as designed

### Phase 3: Configuration Updates
- Update `configs/features/relativity.yaml` to remove volume references
- Update input specifications to reflect 4D OHLC data
- Ensure feature output remains 10-dimensional

### Phase 4: Integration Testing
- Test the complete pipeline with the actual 5-year data file
- Verify feature generation produces correct shape and values
- Ensure causality and invariance properties are maintained

## Key Changes Required
1. **Data Loader**: Point to correct file path, remove volume column
2. **Feature Builder**: Refactor to handle 4D OHLC input cleanly
3. **Configuration**: Update to reflect 4D input, no volume
4. **Testing**: Verify pipeline works with real 5-year data

The goal is to have a clean, working pipeline that processes the 5-year NQ data into 10-dimensional relative features suitable for the Jade model.