#!/usr/bin/env python3
"""
Merge NQ data from multiple sources into continuous 5-year dataset.

Sources:
1. Desktop: nq_ohlcv_1min_2020-09_2024-08.dbn.zst (Sep 2020 → Aug 2024)
2. Filler 1: nq_filler_aug_2024.dbn.zst (Aug 2024 → Sep 2024) 
3. Filler 2: nq_filler_nov2024_may2025.dbn.zst (Nov 2024 → May 2025)
4. Filler 3: nq_filler_aug_oct2025.dbn.zst (Aug 2025 → Oct 2025)
5. Existing Moola data (if any): Sep 2024 → Jul 2025

Creates continuous timeline: Sep 2020 → Oct 2025 (5+ years)
"""

import databento as db
import pandas as pd
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def read_dbn_to_dataframe(dbn_path: str) -> pd.DataFrame:
    """Read DBN file and convert to DataFrame with proper datetime index."""
    logger.info(f"Reading {dbn_path}...")
    store = db.DBNStore.from_file(dbn_path)
    df = store.to_df()
    
    # Convert index to timezone-naive for consistency
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")
    logger.info(f"  Columns: {list(df.columns)}")
    
    return df

def merge_all_data() -> pd.DataFrame:
    """Merge all data sources into continuous timeline."""
    
    # Define all data sources in chronological order
    data_sources = [
        {
            'name': 'Desktop (2020-2024)',
            'path': '/Users/jack/Desktop/nq_ohlcv_1min_2020-09_2024-08.dbn.zst',
            'description': 'Sep 2020 → Aug 2024'
        },
        {
            'name': 'Filler Aug 2024',
            'path': '/Users/jack/projects/moola/data/raw/nq_filler_aug_2024.dbn.zst',
            'description': 'Aug 2024 → Sep 2024'
        },
        {
            'name': 'Filler Sep-Oct 2024',
            'path': '/Users/jack/projects/moola/data/raw/nq_filler_sep_oct_2024.dbn.zst',
            'description': 'Sep 2024 → Nov 2024'
        },
        {
            'name': 'Filler Nov 2024-May 2025',
            'path': '/Users/jack/projects/moola/data/raw/nq_filler_nov2024_may2025.dbn.zst',
            'description': 'Nov 2024 → May 2025'
        },
        {
            'name': 'Filler Jun-Jul 2025',
            'path': '/Users/jack/projects/moola/data/raw/nq_filler_jun_jul_2025.dbn.zst',
            'description': 'Jun 2025 → Aug 2025'
        },
        {
            'name': 'Filler Aug-Oct 2025',
            'path': '/Users/jack/projects/moola/data/raw/nq_filler_aug_oct2025.dbn.zst',
            'description': 'Aug 2025 → Oct 2025'
        }
    ]
    
    all_dataframes = []
    
    for source in data_sources:
        path = Path(source['path'])
        if not path.exists():
            logger.warning(f"Source not found: {path}")
            continue
            
        df = read_dbn_to_dataframe(str(path))
        df['source'] = source['name']  # Track data source
        all_dataframes.append(df)
        logger.info(f"  ✅ Loaded {source['description']}")
    
    if not all_dataframes:
        raise ValueError("No data sources found!")
    
    # Merge all dataframes
    logger.info("Merging all data sources...")
    merged_df = pd.concat(all_dataframes, ignore_index=False)
    
    # Sort by timestamp
    merged_df = merged_df.sort_index()
    
    # Remove any potential duplicates (same timestamp)
    before_dedup = len(merged_df)
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    after_dedup = len(merged_df)
    
    if before_dedup != after_dedup:
        logger.info(f"Removed {before_dedup - after_dedup} duplicate timestamps")
    
    # Verify continuity
    time_diffs = merged_df.index.to_series().diff().dropna()
    expected_freq = pd.Timedelta(minutes=1)
    large_gaps = time_diffs[time_diffs > expected_freq * 10]  # Gaps > 10 minutes
    
    if not large_gaps.empty:
        logger.warning(f"Found {len(large_gaps)} gaps > 10 minutes:")
        for idx, gap in large_gaps.head(10).items():
            logger.warning(f"  Gap at {idx}: {gap}")
    else:
        logger.info("✅ No significant gaps found in timeline")
    
    # Final summary
    logger.info("=" * 60)
    logger.info("MERGED DATASET SUMMARY:")
    logger.info(f"Total records: {len(merged_df):,}")
    logger.info(f"Date range: {merged_df.index.min()} to {merged_df.index.max()}")
    logger.info(f"Duration: {(merged_df.index.max() - merged_df.index.min()).days / 365.25:.1f} years")
    logger.info(f"Columns: {list(merged_df.columns)}")
    logger.info(f"Data sources: {merged_df['source'].value_counts().to_dict()}")
    
    return merged_df

def save_merged_data(df: pd.DataFrame, output_path: str):
    """Save merged data to parquet format."""
    logger.info(f"Saving merged data to {output_path}...")
    
    # Remove source column for final output (keep data clean)
    output_df = df.drop('source', axis=1)
    
    # Save to parquet
    output_df.to_parquet(output_path, index=True, engine='pyarrow')
    
    logger.info(f"✅ Saved {len(output_df):,} records to {output_path}")
    
    # Also save a small summary
    summary_path = output_path.replace('.parquet', '_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"NQ OHLCV 1-min Continuous Dataset Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Total records: {len(output_df):,}\n")
        f.write(f"Date range: {output_df.index.min()} to {output_df.index.max()}\n")
        f.write(f"Duration: {(output_df.index.max() - output_df.index.min()).days / 365.25:.1f} years\n")
        f.write(f"Columns: {list(output_df.columns)}\n")
        f.write(f"\nData quality:\n")
        f.write(f"  Missing values: {output_df.isnull().sum().sum()}\n")
        f.write(f"  Zero volume bars: {(output_df['volume'] == 0).sum()}\n")
    
    logger.info(f"✅ Summary saved to {summary_path}")

def main():
    """Main execution function."""
    logger.info("Starting NQ data merge process...")
    
    try:
        # Merge all data sources
        merged_df = merge_all_data()
        
        # Save merged data
        output_path = '/Users/jack/projects/moola/data/raw/nq_ohlcv_1min_2020-09_2025-10_continuous.parquet'
        save_merged_data(merged_df, output_path)
        
        logger.info("=" * 60)
        logger.info("🎉 NQ data merge completed successfully!")
        logger.info(f"Continuous 5+ year dataset: {output_path}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Error during merge process: {e}")
        raise

if __name__ == "__main__":
    main()
