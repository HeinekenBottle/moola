# Plan: Download Missing NQ Data Months and Create Continuous 5-Year Dataset

## Understanding the Current Situation

**Desktop Data (Complete):**
- `nq_ohlcv_1min_2020-09_2024-08.dbn.zst` - 4 years continuous data (Sep 2020 → Aug 2024)

**Moola Data (Spotty):**
- Has: Sep-2024, Oct-2024, Jun-2025, Jul-2025  
- Missing: Aug-2024, Nov-2024 → May-2025, Aug-2025+

## Required Downloads

I need to download 3 specific filler ranges:
1. **Aug 2024 → Sep 2024** (1 month gap filler)
2. **Nov 2024 → May 2025** (7 month gap filler) 
3. **Aug 2025 → Oct 2025** (ongoing data, through current month)

## Technical Approach

**Databento API Integration:**
- Use existing script: `/Users/jack/projects/databento/databento_scripts/download_nq_range.py`
- Configuration: NQ.v.0 (continuous front-month), GLBX.MDP3, ohlcv-1m schema
- Output: `.dbn.zst` files compatible with existing Desktop data

**Data Processing:**
- Convert DBN files to Moola's parquet format
- Handle contract rollovers properly (avoid duplication from overlapping contracts)
- Merge with existing data to create continuous timeline

**Integration Strategy:**
1. Download the 3 missing ranges as separate DBN files
2. Convert each to parquet format using existing Moola data infrastructure  
3. Merge with Desktop data and existing Moola data
4. Validate timeline continuity and data integrity

**Key Considerations:**
- Use your Databento API key (environment variable: DATABENTO_API_KEY)
- Handle continuous contracts properly to avoid overlap issues
- Maintain existing data schema and format compatibility
- Ensure proper timezone and data quality validation

This will create a complete 5-year continuous dataset: Desktop (2020-2024) + filler months + Moola (2024-2025).