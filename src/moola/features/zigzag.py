"""Causal Zigzag Features - ATR-based swing detection for Moola.

Implements AGENTS.md Section 6 requirements:
- No absolute price leakage (ATR-relative distances)
- Features bounded in [-3, 3] for distances
- Causal: uses only closed candle information
- Hybrid confirmation rules for robust swing detection

Usage:
    >>> from moola.features.zigzag import CausalZigZag
    >>> zigzag = CausalZigZag(atr_period=10, k=1.2, hybrid_lb=5, hybrid_min_atr=0.5)
    >>> prev_SH, prev_SL = zigzag.update(o, h, l, c)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator


class CausalZigZag:
    """Causal zigzag detector with ATR-based confirmation.
    
    Implements hybrid confirmation rules:
    1) Primary: opposite move ≥ k*ATR
    2) Hybrid: extreme occurred within last hybrid_lb bars AND retrace ≥ hybrid_min_atr*ATR
    
    All features are ATR-relative to ensure scale invariance.
    """
    
    def __init__(self, atr_period: int = 10, k: float = 1.2, 
                 hybrid_lb: int = 5, hybrid_min_atr: float = 0.5):
        self.atr_period = atr_period
        self.k = k  # Scalar multiplier for ATR threshold
        self.hybrid_lb = hybrid_lb  # Lookback for hybrid confirmation
        self.hybrid_min_atr = hybrid_min_atr  # Minimum retrace for hybrid
        
        # State tracking for causal detection
        self.atr_history = []
        self.tr_history = []  # True Range history
        self.current_trend = None  # 'up' or 'down'
        self.current_extreme_price = None
        self.current_extreme_idx = None
        self.prev_SH_idx = None  # Previous swing high index
        self.prev_SL_idx = None  # Previous swing low index
        self.prev_SH_price = None
        self.prev_SL_price = None
        self.extreme_candidate_price = None
        self.extreme_candidate_idx = None
        
        # For warming up ATR calculation
        self.bar_count = 0
    
    def _update_atr(self, high: float, low: float, close: float, prev_close: float) -> float:
        """Update ATR calculation using True Range.
        
        TR_t = max(H-L, |H-C_prev|, |L-C_prev|)
        ATR = rolling mean of TR over atr_period
        """
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        self.tr_history.append(tr)
        
        # Keep only recent history for rolling ATR
        if len(self.tr_history) > self.atr_period:
            self.tr_history.pop(0)
        
        # Calculate rolling ATR
        if len(self.tr_history) >= self.atr_period:
            atr = np.mean(self.tr_history)
        else:
            # Use expanding mean during warmup
            atr = np.mean(self.tr_history) if self.tr_history else 0.0
        
        return atr
    
    def _extend_current_leg(self, high: float, low: float, close: float, idx: int):
        """Extend current leg with new extreme if appropriate."""
        if self.current_trend == 'up':
            # Looking for new high
            if high > self.current_extreme_price:
                self.current_extreme_price = high
                self.current_extreme_idx = idx
        elif self.current_trend == 'down':
            # Looking for new low
            if low < self.current_extreme_price:
                self.current_extreme_price = low
                self.current_extreme_idx = idx
    
    def _check_primary_confirmation(self, price: float, atr: float) -> bool:
        """Check primary confirmation: opposite move ≥ k*ATR."""
        if self.current_trend is None or atr == 0:
            return False
        
        if self.current_trend == 'up':
            # Need down move
            required_move = self.k * atr
            actual_move = self.current_extreme_price - price
            return actual_move >= required_move
        else:  # current_trend == 'down'
            # Need up move
            required_move = self.k * atr
            actual_move = price - self.current_extreme_price
            return actual_move >= required_move
    
    def _check_hybrid_confirmation(self, high: float, low: float, close: float, 
                                  atr: float, idx: int) -> bool:
        """Check hybrid confirmation rule."""
        if (self.extreme_candidate_idx is None or 
            atr == 0 or 
            idx - self.extreme_candidate_idx > self.hybrid_lb):
            return False
        
        # Check if we have sufficient retrace from extreme candidate
        if self.current_trend == 'up':
            # Had high candidate, need down retrace
            retrace = self.extreme_candidate_price - low
        else:  # current_trend == 'down'
            # Had low candidate, need up retrace
            retrace = high - self.extreme_candidate_price
        
        required_retrace = self.hybrid_min_atr * atr
        return retrace >= required_retrace
    
    def update(self, open_price: float, high: float, low: float, close: float) -> Tuple[Optional[Tuple[int, float]], Optional[Tuple[int, float]]]:
        """Update zigzag with new closed candle data.
        
        Args:
            open_price, high, low, close: OHLC data for completed bar
            
        Returns:
            Tuple of (prev_SH, prev_SL) where each is (index, price) or None
        """
        self.bar_count += 1
        idx = self.bar_count - 1  # 0-based index
        
        # Need previous close for ATR calculation
        if self.bar_count == 1:
            self.prev_close = close
            return None, None
        
        prev_close = self.prev_close
        self.prev_close = close
        
        # Update ATR
        atr = self._update_atr(high, low, close, prev_close)
        
        # Initialize trend if not set
        if self.current_trend is None:
            # Start trend based on first meaningful move
            if atr > 0:
                # Use close price as initial extreme
                self.current_extreme_price = close
                self.current_extreme_idx = idx
                # Determine initial trend direction (arbitrary, will correct)
                self.current_trend = 'up'
            return None, None
        
        # Extend current leg if appropriate
        self._extend_current_leg(high, low, close, idx)
        
        # Track extreme candidate for hybrid confirmation
        if self.current_trend == 'up' and high > (self.extreme_candidate_price or 0):
            self.extreme_candidate_price = high
            self.extreme_candidate_idx = idx
        elif self.current_trend == 'down' and (low < (self.extreme_candidate_price or float('inf'))):
            self.extreme_candidate_price = low
            self.extreme_candidate_idx = idx
        
        # Check for swing confirmation
        swing_confirmed = False
        
        # Primary confirmation
        if self._check_primary_confirmation(close, atr):
            swing_confirmed = True
        # Hybrid confirmation (if primary fails)
        elif self._check_hybrid_confirmation(high, low, close, atr, idx):
            swing_confirmed = True
        
        if swing_confirmed:
            # Confirm the swing and reverse trend
            if self.current_trend == 'up':
                # Confirm swing high
                self.prev_SH_idx = self.current_extreme_idx
                self.prev_SH_price = self.current_extreme_price
                new_trend = 'down'
                new_extreme = low
                new_extreme_idx = idx
            else:  # current_trend == 'down'
                # Confirm swing low
                self.prev_SL_idx = self.current_extreme_idx
                self.prev_SL_price = self.current_extreme_price
                new_trend = 'up'
                new_extreme = high
                new_extreme_idx = idx
            
            # Reset for new trend
            self.current_trend = new_trend
            self.current_extreme_price = new_extreme
            self.current_extreme_idx = new_extreme_idx
            self.extreme_candidate_price = new_extreme
            self.extreme_candidate_idx = new_extreme_idx
        
        # Return confirmed swings
        prev_SH = (self.prev_SH_idx, self.prev_SH_price) if self.prev_SH_idx is not None else None
        prev_SL = (self.prev_SL_idx, self.prev_SL_price) if self.prev_SL_idx is not None else None
        
        return prev_SH, prev_SL
    
    def get_state(self) -> Dict:
        """Get current zigzag state for debugging."""
        return {
            'current_trend': self.current_trend,
            'current_extreme_price': self.current_extreme_price,
            'current_extreme_idx': self.current_extreme_idx,
            'prev_SH': (self.prev_SH_idx, self.prev_SH_price) if self.prev_SH_idx is not None else None,
            'prev_SL': (self.prev_SL_idx, self.prev_SL_price) if self.prev_SL_idx is not None else None,
            'bar_count': self.bar_count,
            'current_atr': np.mean(self.tr_history) if self.tr_history else 0.0
        }


def swing_relative(close: float, prev_SH_price: Optional[float], prev_SL_price: Optional[float], 
                  atr: float, bars_since_SH: int, bars_since_SL: int, K: float) -> Tuple[float, float, float, float]:
    """Calculate swing-relative features.
    
    Args:
        close: Current close price
        prev_SH_price: Previous swing high price (ATR-relative)
        prev_SL_price: Previous swing low price (ATR-relative) 
        atr: Current ATR
        bars_since_SH: Bars since previous swing high
        bars_since_SL: Bars since previous swing low
        K: Normalization constant for time features
        
    Returns:
        Tuple of (dist_to_prev_SH, dist_to_prev_SL, bars_since_SH_norm, bars_since_SL_norm)
    """
    if atr == 0:
        atr = 1e-8  # Prevent division by zero
    
    # Distance features (ATR-relative, scale-invariant)
    if prev_SH_price is not None:
        dist_to_prev_SH = (prev_SH_price - close) / atr  # Positive when below SH
    else:
        dist_to_prev_SH = 0.0
    
    if prev_SL_price is not None:
        dist_to_prev_SL = (close - prev_SL_price) / atr  # Positive when above SL
    else:
        dist_to_prev_SL = 0.0
    
    # Time features (normalized by K)
    bars_since_SH_norm = bars_since_SH / K
    bars_since_SL_norm = bars_since_SL / K
    
    # Clip to reasonable bounds for stability
    dist_to_prev_SH = np.clip(dist_to_prev_SH, -3, 3)
    dist_to_prev_SL = np.clip(dist_to_prev_SL, -3, 3)
    bars_since_SH_norm = np.clip(bars_since_SH_norm, 0, 3)
    bars_since_SL_norm = np.clip(bars_since_SL_norm, 0, 3)
    
    return dist_to_prev_SH, dist_to_prev_SL, bars_since_SH_norm, bars_since_SL_norm


# AGENTS.md CLI integration
def main():
    """CLI entry point for zigzag features.
    
    Usage: moola features --config configs/features/zigzag.yaml --in data.parquet --out features.parquet
    """
    import argparse
    import sys
    import time
    import yaml
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Build zigzag features from OHLCV data")
    parser.add_argument("--config", required=True, help="Path to zigzag config")
    parser.add_argument("--in", required=True, dest="input", help="Input parquet file")
    parser.add_argument("--out", required=True, dest="output", help="Output parquet file")
    parser.add_argument("--verbose", action="store_true", help="Print detailed information")
    
    args = parser.parse_args()
    
    try:
        start_time = time.time()
        
        # Load data
        data = pd.read_parquet(args.input)
        if args.verbose:
            print(f"Loaded data: {args.input}")
            print(f"Shape: {data.shape}")
            print(f"Columns: {list(data.columns)}")
        
        # Load config
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Build features
        X, mask, meta = build_zigzag_features(data, config.get('zigzag', {}))
        
        # Save results
        feature_names = meta['feature_names']
        n_samples, n_features = X.shape
        
        # Create DataFrame for saving
        df_data = {}
        for i, name in enumerate(feature_names):
            df_data[name] = X[:, i]
        
        df_data['window_id'] = np.arange(n_samples)
        out_df = pd.DataFrame(df_data)
        out_df.to_parquet(args.output)
        
        # Print summary (AGENTS.md Section 15)
        elapsed = time.time() - start_time
        print(f"Input path: {args.input}")
        print(f"Output path: {args.output}")
        print(f"Rows processed: {len(data)}")
        print(f"Windows created: {n_samples}")
        print(f"Features per window: {n_features}")
        print(f"Window size: {meta['window_size']}")
        print(f"Wall time: {elapsed:.2f}s")
        
        if args.verbose:
            print(f"Feature info: {meta}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
