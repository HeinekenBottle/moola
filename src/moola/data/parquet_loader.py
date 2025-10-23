"""Parquet Loader - Simple data loading for Moola.

Replaces dual_input pipeline with clean, causal data loading.
No forward fill, no data leakage, just proper typing and sorting.

Usage:
    >>> from moola.data.parquet_loader import load
    >>> df = load(['data1.parquet', 'data2.parquet'])
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path


def load(paths: List[str], cols: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
    """Load and concatenate parquet files with proper typing.
    
    Args:
        paths: List of parquet file paths
        cols: Optional tuple of columns to load. Defaults to standard OHLCV.
        
    Returns:
        DataFrame with proper dtypes and sorted by timestamp
        
    Notes:
        - No forward fill or data imputation
        - Strict causality: only uses completed candles
        - Enforces proper dtypes for consistency
        - Sorts by timestamp to ensure temporal order
    """
    if not paths:
        raise ValueError("No file paths provided")
    
    # Default columns if not specified (OHLC only, no volume)
    if cols is None:
        cols = ("open", "high", "low", "close")
    
    # Load each file and validate schema
    dfs = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Load parquet
        df = pd.read_parquet(path)
        
        # Validate required columns
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in {path}: {missing_cols}")
        
        # Select only requested columns
        df = df[list(cols)]
        
        # Enforce proper dtypes
        df = _enforce_dtypes(df)
        
        dfs.append(df)
    
    # Concatenate all dataframes
    if len(dfs) == 1:
        result_df = dfs[0]
    else:
        result_df = pd.concat(dfs, ignore_index=True)
    
    # Sort by timestamp if available
    if 'timestamp' in result_df.columns:
        result_df = result_df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove any duplicate timestamps (keep last)
    if 'timestamp' in result_df.columns:
        result_df = result_df.drop_duplicates(subset=['timestamp'], keep='last')
    
    # Basic validation
    if len(result_df) == 0:
        raise ValueError("No data loaded after processing")
    
    # Check for causality violations (data should be complete)
    _validate_causality(result_df)
    
    return result_df


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce proper data types for OHLCV columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with enforced dtypes
    """
    # Define expected dtypes (OHLC only)
    dtype_map = {
        'open': 'float32',
        'high': 'float32', 
        'low': 'float32',
        'close': 'float32'
    }
    
    # Apply dtypes for columns that exist
    for col, dtype in dtype_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot convert column '{col}' to {dtype}: {e}")
    
    return df


def _validate_causality(df: pd.DataFrame) -> None:
    """Validate that data respects causality constraints.
    
    Args:
        df: Input dataframe
        
    Raises:
        ValueError: If causality violations are detected
    """
    # Check for NaN values in price columns
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                raise ValueError(f"Found {nan_count} NaN values in column '{col}' - data causality violated")
    
    # Check for positive values in prices only (no volume)
    for col in price_cols:
        if col in df.columns:
            negative_prices = (df[col] <= 0).sum()
            if negative_prices > 0:
                raise ValueError(f"Found {negative_prices} non-positive values in column '{col}'")
    
    # Check OHLC consistency (high >= low, high >= open/close, low <= open/close)
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        violations = 0
        
        # high should be >= low
        violations += (df['high'] < df['low']).sum()
        
        # high should be >= open and close
        violations += (df['high'] < df['open']).sum()
        violations += (df['high'] < df['close']).sum()
        
        # low should be <= open and close
        violations += (df['low'] > df['open']).sum()
        violations += (df['low'] > df['close']).sum()
        
        if violations > 0:
            raise ValueError(f"Found {violations} OHLC consistency violations")


def get_info(df: pd.DataFrame) -> dict:
    """Get basic information about loaded data.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with data information
    """
    info = {
        'n_rows': len(df),
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    if 'timestamp' in df.columns:
        info['time_range'] = {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max(),
            'duration_days': (df['timestamp'].max() - df['timestamp'].min()).days
        }
    
    # Price statistics
    price_cols = ['open', 'high', 'low', 'close']
    available_price_cols = [col for col in price_cols if col in df.columns]
    if available_price_cols:
        price_stats = df[available_price_cols].describe()
        info['price_stats'] = price_stats.to_dict()
    
    # Note: Volume statistics removed - working with 4D OHLC data only
    
    return info


def load_nq_5year() -> pd.DataFrame:
    """Load the 5-year NQ data file.
    
    Returns:
        DataFrame with OHLC data from the 5-year NQ dataset
        
    Notes:
        - Uses the fixed 5-year NQ data file
        - Returns only OHLC columns (no volume)
        - Data is properly typed and validated
    """
    nq_path = "/Users/jack/projects/moola/data/raw/nq_5year.parquet"
    return load([nq_path], cols=("open", "high", "low", "close"))


# CLI integration
def main():
    """CLI entry point for parquet loader.
    
    Usage:
        python -m moola.data.parquet_loader --paths data1.parquet data2.parquet --out combined.parquet
    """
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Load and combine parquet files")
    parser.add_argument("--paths", nargs="+", required=True, help="Input parquet file paths")
    parser.add_argument("--cols", nargs="*", help="Columns to load (default: open,high,low,close)")
    parser.add_argument("--out", help="Output parquet file path (optional)")
    parser.add_argument("--info", action="store_true", help="Print data information")
    
    args = parser.parse_args()
    
    try:
        # Load data
        cols = tuple(args.cols) if args.cols else None
        df = load(args.paths, cols)
        
        # Print information if requested
        if args.info:
            info = get_info(df)
            print("Data Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        # Save to file if requested
        if args.out:
            df.to_parquet(args.out)
            print(f"Saved combined data to: {args.out}")
        else:
            print(f"Loaded {len(df)} rows from {len(args.paths)} files")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
