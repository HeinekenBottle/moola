"""Price action feature engineering for financial time series.

ICT-aligned feature extraction focusing on:
- Market structure
- Liquidity zones
- Fair value gaps
- Order blocks
- Imbalance ratios
- Geometry features
- Distance measures
- Candle patterns
- Pattern morphology features

Features are extracted primarily from the inner prediction window [30:75]
with context summaries from left and right buffers.

NEW: Multi-scale feature engineering that separates:
- Pattern-level features (computed on expansion region)
- Context features (computed on fixed [30:75] window)
- Relative position features (how pattern relates to context)

NEW: HopSketch 15-feature extraction from FULL 105-bar window
"""

from typing import Tuple

import numpy as np
from scipy.stats import linregress


def extract_hopsketch_features(X: np.ndarray) -> np.ndarray:
    """Extract 15 geometric features from FULL 105-bar window.

    NO smoothing, NO expansion region filtering. Features are extracted per-bar
    from the entire window. This approach preserves all context information.

    Based on HopSketch architecture from pivot/archive/hopsketch_spec1/.

    Args:
        X: [N, 105, 4] OHLC array (or [N, 420] flattened)

    Returns:
        features: [N, 105*15] = [N, 1575] flattened per-bar features

    Features per bar (15 total):
        - 4 OHLC normalized (by window statistics)
        - 7 geometry (body, range, body_frac, wicks, positions)
        - 4 context (direction, gap_prev, pct_up, run_length)
    """
    # Reshape if needed
    if X.ndim == 2 and X.shape[1] == 420:
        X = X.reshape(-1, 105, 4)

    N = X.shape[0]
    features_list = []

    for sample_idx in range(N):
        sample = X[sample_idx]  # [105, 4]
        o, h, l, c = sample[:, 0], sample[:, 1], sample[:, 2], sample[:, 3]

        # Window normalization (remove price-level dependence)
        window_high = h.max()
        window_low = l.min()
        range_val = window_high - window_low + 1e-8

        # OHLC normalized [4 features per bar]
        o_norm = (o - window_low) / range_val
        h_norm = (h - window_low) / range_val
        l_norm = (l - window_low) / range_val
        c_norm = (c - window_low) / range_val

        # Geometry per bar [7 features]
        body = np.abs(c - o)
        bar_range = h - l + 1e-8
        body_frac = body / bar_range
        upper_wick = (h - np.maximum(o, c)) / bar_range
        lower_wick = (np.minimum(o, c) - l) / bar_range
        close_pos = (c - l) / bar_range
        open_pos = (o - l) / bar_range

        # Context per bar [4 features]
        direction = np.sign(c - o)
        gap_prev = np.concatenate([[0], o[1:] - c[:-1]]) / range_val
        pct_up = (c > o).astype(float)

        # Run length (consecutive same direction)
        run_length = np.zeros(105)
        current_run = 1
        for i in range(1, 105):
            if direction[i] == direction[i-1]:
                current_run += 1
            else:
                current_run = 1
            run_length[i] = current_run

        # Stack all 15 features per bar: [105, 15]
        sample_features = np.column_stack([
            o_norm, h_norm, l_norm, c_norm,  # 4
            body, bar_range, body_frac, upper_wick, lower_wick, close_pos, open_pos,  # 7
            direction, gap_prev, pct_up, run_length  # 4
        ])  # [105, 15]

        # Flatten to [1575] for XGBoost
        features_list.append(sample_features.flatten())

    # Stack all samples: [N, 1575]
    feature_matrix = np.array(features_list, dtype=np.float32)

    # Handle any NaN or inf values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

    return feature_matrix


def engineer_multiscale_features(
    X: np.ndarray,
    expansion_start: np.ndarray = None,
    expansion_end: np.ndarray = None
) -> np.ndarray:
    """Multi-scale feature extraction for financial pattern classification.

    Eliminates signal dilution by separating pattern-level, context, and relative features.
    All features extracted from inner window [30:75] only (no buffer contamination).

    Args:
        X: Raw OHLC data of shape [N, 105, 4] or [N, 420]
            4 features are: Open, High, Low, Close
        expansion_start: Start indices for expansion regions [N] (required)
        expansion_end: End indices for expansion regions [N] (required)

    Returns:
        Engineered features of shape [N, 20] containing:
        - Scale 1 (Pattern): 5 features on expansion region
        - Scale 2 (Context): 10 features on fixed [30:75] window
        - Scale 3 (Relative): 5 features (pattern vs context)
        - Scale 1+2 interaction: 0 features removed (was Williams %R)
    """
    # Reshape if needed
    if X.ndim == 2 and X.shape[1] == 420:
        X = X.reshape(-1, 105, 4)

    N = X.shape[0]

    # Validate expansion indices
    if expansion_start is None or expansion_end is None:
        raise ValueError("Multi-scale features require expansion_start and expansion_end indices")

    # Ensure indices are within [30:75] constraint
    expansion_start = np.clip(expansion_start, 30, 74)
    expansion_end = np.clip(expansion_end, 30, 74)

    # Extract fixed context window [30:75] for all samples
    context_window = X[:, 30:75, :]  # [N, 45, 4]

    all_features = []

    for i in range(N):
        start = int(expansion_start[i])
        end = int(expansion_end[i])

        # Extract pattern region (within [30:75])
        pattern = X[i, start:end+1, :]  # [T_pattern, 4]
        context = context_window[i]  # [45, 4]

        # OHLC extraction
        p_o, p_h, p_l, p_c = pattern[:, 0], pattern[:, 1], pattern[:, 2], pattern[:, 3]
        c_o, c_h, c_l, c_c = context[:, 0], context[:, 1], context[:, 2], context[:, 3]

        features = []

        # ===== SCALE 1: PATTERN-LEVEL FEATURES (5) =====
        # Simple, noise-resistant features computed on expansion region

        # 1. Price change
        if len(p_c) > 0:
            price_change = (p_c[-1] - p_c[0]) / (p_c[0] + 1e-10)
        else:
            price_change = 0.0
        features.append(price_change)

        # 2. Direction
        direction = 1.0 if price_change > 0 else -1.0
        features.append(direction)

        # 3. Range ratio
        if len(p_h) > 0 and len(p_l) > 0 and len(p_c) > 0:
            range_ratio = (p_h.max() - p_l.min()) / (p_c[0] + 1e-10)
        else:
            range_ratio = 0.0
        features.append(range_ratio)

        # 4. Body dominance
        if len(p_o) > 0:
            body = np.abs(p_c - p_o)
            total_range = p_h - p_l + 1e-10
            body_dominance = (body / total_range).mean()
        else:
            body_dominance = 0.5
        features.append(body_dominance)

        # 5. Wick balance (ratio bounded to [0, 1])
        if len(p_o) > 0:
            upper_wick = p_h - np.maximum(p_o, p_c)
            lower_wick = np.minimum(p_o, p_c) - p_l
            total_wick = upper_wick + lower_wick + 1e-10
            wick_balance = (upper_wick / total_wick).mean()  # Now [0, 1]
        else:
            wick_balance = 0.5  # Neutral
        features.append(wick_balance)

        # ===== SCALE 2: CONTEXT FEATURES (10) =====
        # Stable 45-bar indicators on fixed window

        # 6. Volatility (20-bar)
        if len(c_c) >= 20:
            returns = np.diff(c_c) / (c_c[:-1] + 1e-10)
            volatility_20 = returns[-20:].std()
        else:
            volatility_20 = 0.0
        features.append(volatility_20)

        # 7. Volatility (45-bar)
        if len(c_c) > 1:
            returns = np.diff(c_c) / (c_c[:-1] + 1e-10)
            volatility_45 = returns.std()
        else:
            volatility_45 = 0.0
        features.append(volatility_45)

        # 9. Trend strength (slope * r_squared)
        if len(c_c) > 1:
            x = np.arange(len(c_c))
            slope, _, r_value, _, _ = linregress(x, c_c)
            trend_strength = slope * (r_value ** 2)
        else:
            trend_strength = 0.0
        features.append(trend_strength)

        # 10. Support distance
        support = c_l.min()
        support_dist = (c_c[-1] - support) / (c_h.max() - support + 1e-10)
        features.append(support_dist)

        # 11. Resistance distance
        resistance = c_h.max()
        resistance_dist = (resistance - c_c[-1]) / (resistance - c_l.min() + 1e-10)
        features.append(resistance_dist)

        # 12. Average body ratio
        body = np.abs(c_c - c_o)
        total_range = c_h - c_l + 1e-10
        avg_body_ratio = (body / total_range).mean()
        features.append(avg_body_ratio)

        # 13. Average upper wick
        upper_wick = c_h - np.maximum(c_o, c_c)
        avg_upper_wick = (upper_wick / total_range).mean()
        features.append(avg_upper_wick)

        # 14. Average lower wick
        lower_wick = np.minimum(c_o, c_c) - c_l
        avg_lower_wick = (lower_wick / total_range).mean()
        features.append(avg_lower_wick)

        # 15. Position in range
        position_in_range = (c_c[-1] - c_l.min()) / (c_h.max() - c_l.min() + 1e-10)
        features.append(position_in_range)

        # ===== SCALE 3: RELATIVE POSITION FEATURES (5) =====
        # How pattern relates to surrounding context

        # 16. Pattern position (where in window)
        pattern_position = (start - 30) / 45.0
        features.append(pattern_position)

        # 17. Pattern coverage (what % of window)
        pattern_coverage = (end - start) / 45.0
        features.append(pattern_coverage)

        # 18. Pre-pattern momentum
        if start > 30:
            pre_momentum = (c_c[start - 30] - c_c[0]) / (c_c[0] + 1e-10)
        else:
            pre_momentum = 0.0
        features.append(pre_momentum)

        # 19. Post-pattern momentum
        if end < 74:
            post_momentum = (c_c[-1] - c_c[end - 30]) / (c_c[end - 30] + 1e-10)
        else:
            post_momentum = 0.0
        features.append(post_momentum)

        # 20. Pattern vs window range
        pattern_range = p_h.max() - p_l.min() if len(p_h) > 0 else 0.0
        window_range = c_h.max() - c_l.min()
        pattern_vs_window = pattern_range / (window_range + 1e-10)
        features.append(pattern_vs_window)

        # ===== SCALE 1+2 INTERACTION (1) =====
        # Pattern vs context geometric relationships

        # 21. Pattern curvature (geometric invariant)
        if len(p_c) > 2:
            # Second derivative approximation
            first_deriv = np.diff(p_c)
            second_deriv = np.diff(first_deriv)
            curvature = np.mean(np.abs(second_deriv))
            features.append(curvature)
        else:
            features.append(0.0)

        all_features.append(features)

    # Convert to array
    feature_matrix = np.array(all_features, dtype=np.float32)

    # Handle any NaN or inf values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

    return feature_matrix




def engineer_classical_features(
    X: np.ndarray,
    expansion_start: np.ndarray = None,
    expansion_end: np.ndarray = None
) -> np.ndarray:
    """Transform raw OHLC time series to engineered price action features.

    This function extracts ICT-aligned features that capture market microstructure,
    liquidity dynamics, and geometric patterns. Features are extracted from the
    expansion region specified by expansion_start/end indices, or from the default
    inner prediction window [30:75] if indices not provided.

    Args:
        X: Raw OHLC data of shape [N, 105, 4] or [N, 420]
            4 features are: Open, High, Low, Close
        expansion_start: Start indices for expansion regions [N] (optional)
        expansion_end: End indices for expansion regions [N] (optional)

    Returns:
        Engineered features of shape [N, ~35] containing:
        - Market structure (HH/LL, swings)
        - Liquidity zones (equal highs/lows)
        - Fair value gaps
        - Order blocks
        - Imbalance ratios
        - Geometry features (slopes, curvature)
        - Distance measures
        - Candle patterns
        - Market regime features (trend, volatility, range)
        - Buffer context summaries
        - Removed: Williams %R, EMAs, SMAs, RSI, MACD, Bollinger Bands, ATR, OBV
    """
    # Reshape if needed
    if X.ndim == 2 and X.shape[1] == 420:
        X = X.reshape(-1, 105, 4)

    N = X.shape[0]

    # Use expansion indices if provided, otherwise use default windowing
    if expansion_start is not None and expansion_end is not None:
        # Extract expansion regions using provided indices
        # We'll process each sample's expansion region individually
        expansion_regions = []
        for i in range(N):
            start = int(expansion_start[i])
            end = int(expansion_end[i]) + 1  # +1 for inclusive end
            expansion_regions.append(X[i, start:end, :])

        # For uniform processing, we need to handle variable-length regions
        # Extract OHLC from expansion regions
        o_list, h_list, l_list, c_list = [], [], [], []
        for region in expansion_regions:
            o_list.append(region[:, 0])
            h_list.append(region[:, 1])
            l_list.append(region[:, 2])
            c_list.append(region[:, 3])

        # For buffer context, use bars before and after expansion
        o_left_list, h_left_list, l_left_list, c_left_list = [], [], [], []
        c_right_list = []

        for i in range(N):
            start = int(expansion_start[i])
            end = int(expansion_end[i])

            # Left context: 10 bars before expansion (or less if at start)
            left_start = max(0, start - 10)
            left_region = X[i, left_start:start, :]

            # Right context: 10 bars after expansion (or less if at end)
            right_end = min(105, end + 11)
            right_region = X[i, end+1:right_end, :]

            o_left_list.append(left_region[:, 0] if len(left_region) > 0 else np.array([X[i, start, 0]]))
            h_left_list.append(left_region[:, 1] if len(left_region) > 0 else np.array([X[i, start, 1]]))
            l_left_list.append(left_region[:, 2] if len(left_region) > 0 else np.array([X[i, start, 2]]))
            c_left_list.append(left_region[:, 3] if len(left_region) > 0 else np.array([X[i, start, 3]]))
            c_right_list.append(right_region[:, 3] if len(right_region) > 0 else np.array([X[i, end, 3]]))

    else:
        # Fall back to default windowing (legacy behavior)
        from ..utils.windowing import get_window_regions
        left_buffer, inner_window, right_buffer = get_window_regions(X)

        # Extract OHLC from inner window
        o_list = [inner_window[i, :, 0] for i in range(N)]
        h_list = [inner_window[i, :, 1] for i in range(N)]
        l_list = [inner_window[i, :, 2] for i in range(N)]
        c_list = [inner_window[i, :, 3] for i in range(N)]

        o_left_list = [left_buffer[i, :, 0] for i in range(N)]
        h_left_list = [left_buffer[i, :, 1] for i in range(N)]
        l_left_list = [left_buffer[i, :, 2] for i in range(N)]
        c_left_list = [left_buffer[i, :, 3] for i in range(N)]
        c_right_list = [right_buffer[i, :, 3] for i in range(N)]

    # Collect all features (process variable-length regions per-sample)
    all_sample_features = []

    for i in range(N):
        # Get this sample's expansion region
        o_sample = o_list[i]
        h_sample = h_list[i]
        l_sample = l_list[i]
        c_sample = c_list[i]

        # Reshape to [1, T] for feature extraction functions
        o_arr = o_sample.reshape(1, -1)
        h_arr = h_sample.reshape(1, -1)
        l_arr = l_sample.reshape(1, -1)
        c_arr = c_sample.reshape(1, -1)

        sample_features = []

        # ===== 1. MARKET STRUCTURE =====
        sample_features.extend(_extract_market_structure(h_arr, l_arr, c_arr))

        # ===== 2. LIQUIDITY ZONES =====
        sample_features.extend(_extract_liquidity_zones(h_arr, l_arr))

        # ===== 3. FAIR VALUE GAPS (FVG) =====
        sample_features.extend(_extract_fair_value_gaps(h_arr, l_arr))

        # ===== 4. ORDER BLOCKS =====
        sample_features.extend(_extract_order_blocks(o_arr, h_arr, l_arr, c_arr))

        # ===== 5. IMBALANCE RATIOS =====
        sample_features.extend(_extract_imbalance_ratios(o_arr, h_arr, l_arr, c_arr))

        # ===== 6. GEOMETRY FEATURES =====
        sample_features.extend(_extract_geometry_features(c_arr))

        # ===== 7. DISTANCE MEASURES =====
        sample_features.extend(_extract_distance_measures(h_arr, l_arr, c_arr))

        # ===== 8. CANDLE PATTERNS =====
        sample_features.extend(_extract_candle_patterns(o_arr, h_arr, l_arr, c_arr))

        # ===== 9. MARKET REGIME FEATURES =====
        sample_features.extend(_extract_market_regime_features(h_arr, l_arr, c_arr))

        # ===== 10. BUFFER CONTEXT =====
        # Reshape buffer context arrays to [1, T_buffer]
        c_left_arr = c_left_list[i].reshape(1, -1)
        h_left_arr = h_left_list[i].reshape(1, -1)
        l_left_arr = l_left_list[i].reshape(1, -1)
        c_right_arr = c_right_list[i].reshape(1, -1)

        sample_features.extend(_extract_buffer_context(
            c_left_arr, h_left_arr, l_left_arr,
            c_arr,  # inner close (expansion region)
            c_right_arr  # right close
        ))

        # Flatten sample features and add to collection
        sample_feature_vector = np.concatenate([
            f.flatten() if isinstance(f, np.ndarray) else np.array([f])
            for f in sample_features
        ])
        all_sample_features.append(sample_feature_vector)

    # Stack all samples
    feature_matrix = np.vstack(all_sample_features)

    # Handle any NaN or inf values
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=1e6, neginf=-1e6)

    return feature_matrix


def _extract_market_structure(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> list:
    """Extract market structure features (HH/LL, swings, trend).

    Args:
        h, l, c: High, Low, Close arrays [N, T]

    Returns:
        List of feature arrays
    """
    N, T = h.shape
    features = []

    # Local peaks and troughs using simple peak detection
    # Peak: h[i] > h[i-1] and h[i] > h[i+1]
    peaks = np.zeros((N, T), dtype=bool)
    peaks[:, 1:-1] = (h[:, 1:-1] > h[:, :-2]) & (h[:, 1:-1] > h[:, 2:])

    troughs = np.zeros((N, T), dtype=bool)
    troughs[:, 1:-1] = (l[:, 1:-1] < l[:, :-2]) & (l[:, 1:-1] < l[:, 2:])

    # Count peaks and troughs
    features.append(peaks.sum(axis=1))  # num_peaks
    features.append(troughs.sum(axis=1))  # num_troughs

    # Higher highs / Lower lows detection
    # Compare current highs with previous highs
    hh_count = np.zeros(N)
    ll_count = np.zeros(N)

    for i in range(N):
        peak_idx = np.where(peaks[i])[0]
        if len(peak_idx) >= 2:
            # Count how many peaks are higher than previous peaks
            hh_count[i] = np.sum(h[i, peak_idx[1:]] > h[i, peak_idx[:-1]])

        trough_idx = np.where(troughs[i])[0]
        if len(trough_idx) >= 2:
            # Count how many troughs are lower than previous troughs
            ll_count[i] = np.sum(l[i, trough_idx[1:]] < l[i, trough_idx[:-1]])

    features.append(hh_count)  # higher_highs
    features.append(ll_count)  # lower_lows

    # Trend direction: slope of close prices
    trends = np.zeros(N)
    for i in range(N):
        if T > 1:
            slope, _, _, _, _ = linregress(np.arange(T), c[i])
            trends[i] = slope

    features.append(trends)  # trend_slope

    return features


def _extract_liquidity_zones(h: np.ndarray, l: np.ndarray) -> list:
    """Extract liquidity zone features (equal highs/lows).

    Args:
        h, l: High, Low arrays [N, T]

    Returns:
        List of feature arrays
    """
    N, T = h.shape
    features = []
    tolerance = 0.001  # 0.1% tolerance for "equal"

    # Equal highs: highs within tolerance
    equal_highs = np.zeros(N)
    for i in range(N):
        h_sorted = np.sort(h[i])[::-1]  # Sort descending
        # Count consecutive pairs within tolerance
        if len(h_sorted) > 1:
            diffs = np.abs(np.diff(h_sorted) / (h_sorted[:-1] + 1e-10))
            equal_highs[i] = np.sum(diffs < tolerance)

    # Equal lows: lows within tolerance
    equal_lows = np.zeros(N)
    for i in range(N):
        l_sorted = np.sort(l[i])  # Sort ascending
        if len(l_sorted) > 1:
            diffs = np.abs(np.diff(l_sorted) / (l_sorted[:-1] + 1e-10))
            equal_lows[i] = np.sum(diffs < tolerance)

    features.append(equal_highs)
    features.append(equal_lows)

    # Liquidity pool strength: ratio of equal highs to equal lows
    pool_ratio = equal_highs / (equal_lows + 1.0)
    features.append(pool_ratio)

    return features


def _extract_fair_value_gaps(h: np.ndarray, l: np.ndarray) -> list:
    """Extract fair value gap (FVG) features.

    FVG: 3-candle imbalance where high[i-1] < low[i+1] (bullish)
    or low[i-1] > high[i+1] (bearish)

    Args:
        h, l: High, Low arrays [N, T]

    Returns:
        List of feature arrays
    """
    N, T = h.shape
    features = []

    # Bullish FVG: h[i-1] < l[i+1]
    bullish_fvg = np.zeros(N)
    bearish_fvg = np.zeros(N)

    if T >= 3:
        for i in range(N):
            # Check all 3-candle windows
            for t in range(1, T - 1):
                if h[i, t - 1] < l[i, t + 1]:
                    bullish_fvg[i] += 1
                if l[i, t - 1] > h[i, t + 1]:
                    bearish_fvg[i] += 1

    features.append(bullish_fvg)
    features.append(bearish_fvg)

    # FVG ratio: bullish vs bearish
    fvg_ratio = bullish_fvg / (bearish_fvg + 1.0)
    features.append(fvg_ratio)

    return features


def _extract_order_blocks(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> list:
    """Extract order block features.

    Order block: last up/down candle before strong move.

    Args:
        o, h, l, c: OHLC arrays [N, T]

    Returns:
        List of feature arrays
    """
    N, T = o.shape
    features = []

    # Identify strong moves (>0.05% for minute-level crypto data)
    # Adjusted from 2% (stock dailies) to 0.05% (crypto 1-min bars)
    returns = (c - o) / (o + 1e-10)
    strong_up = returns > 0.0005
    strong_down = returns < -0.0005

    # Count order blocks (candles before strong moves)
    ob_count = np.zeros(N)
    ob_strength = np.zeros(N)

    for i in range(N):
        for t in range(1, T):
            if strong_up[i, t]:
                # Previous candle is bullish order block
                ob_count[i] += 1
                ob_strength[i] += abs(returns[i, t - 1])
            elif strong_down[i, t]:
                # Previous candle is bearish order block
                ob_count[i] += 1
                ob_strength[i] += abs(returns[i, t - 1])

    features.append(ob_count)
    features.append(ob_strength)

    # Distance to nearest order block (normalized)
    dist_to_ob = np.zeros(N)
    for i in range(N):
        # Find last order block
        ob_indices = []
        for t in range(1, T):
            if strong_up[i, t] or strong_down[i, t]:
                ob_indices.append(t - 1)

        if ob_indices:
            last_ob = ob_indices[-1]
            dist_to_ob[i] = (T - 1 - last_ob) / T  # Normalized distance
        else:
            dist_to_ob[i] = 1.0  # Max distance if no OB found

    features.append(dist_to_ob)

    return features


def _extract_imbalance_ratios(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> list:
    """Extract imbalance ratio features (wick analysis).

    Args:
        o, h, l, c: OHLC arrays [N, T]

    Returns:
        List of feature arrays
    """
    N, T = o.shape
    features = []

    # Body size: |close - open|
    body = np.abs(c - o)

    # Total range: high - low
    total_range = h - l + 1e-10

    # Body ratio: body / total_range
    body_ratio = body / total_range

    # Upper shadow: h - max(o, c)
    upper_shadow = h - np.maximum(o, c)

    # Lower shadow: min(o, c) - l
    lower_shadow = np.minimum(o, c) - l

    # Upper shadow ratio
    upper_shadow_ratio = upper_shadow / total_range

    # Lower shadow ratio
    lower_shadow_ratio = lower_shadow / total_range

    # Aggregate statistics
    features.append(body_ratio.mean(axis=1))  # avg_body_ratio
    features.append(upper_shadow_ratio.mean(axis=1))  # avg_upper_shadow
    features.append(lower_shadow_ratio.mean(axis=1))  # avg_lower_shadow

    # Imbalance: difference between upper and lower shadows
    imbalance = upper_shadow_ratio - lower_shadow_ratio
    features.append(imbalance.mean(axis=1))  # avg_imbalance

    # Wick dominance: max(upper, lower) / total
    wick_dom = np.maximum(upper_shadow_ratio, lower_shadow_ratio)
    features.append(wick_dom.mean(axis=1))  # avg_wick_dominance

    return features


def _extract_geometry_features(c: np.ndarray) -> list:
    """Extract geometric features (slopes, curvature, angles).

    Args:
        c: Close array [N, T]

    Returns:
        List of feature arrays
    """
    N, T = c.shape
    features = []

    # Linear regression slope
    slopes = np.zeros(N)
    r_values = np.zeros(N)

    for i in range(N):
        if T > 1:
            slope, intercept, r_value, _, _ = linregress(np.arange(T), c[i])
            slopes[i] = slope
            r_values[i] = r_value ** 2  # R-squared

    features.append(slopes)  # slope
    features.append(r_values)  # r_squared

    # Curvature (second derivative approximation)
    # Use central differences for first derivative
    first_deriv = np.zeros((N, T))
    first_deriv[:, 1:-1] = (c[:, 2:] - c[:, :-2]) / 2.0

    # Second derivative
    second_deriv = np.zeros((N, T))
    second_deriv[:, 1:-1] = (first_deriv[:, 2:] - first_deriv[:, :-2]) / 2.0

    # Average curvature
    features.append(np.abs(second_deriv).mean(axis=1))  # avg_curvature

    # Angle of price movement (arctangent of slope)
    angles = np.arctan(slopes) * 180 / np.pi  # Convert to degrees
    features.append(angles)  # price_angle

    return features


def _extract_distance_measures(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> list:
    """Extract distance-based features.

    Args:
        h, l, c: High, Low, Close arrays [N, T]

    Returns:
        List of feature arrays
    """
    N, T = h.shape
    features = []

    # Distance to support (lowest low)
    support = l.min(axis=1, keepdims=True)
    dist_to_support = (c[:, -1:] - support) / (support + 1e-10)
    features.append(dist_to_support.flatten())

    # Distance to resistance (highest high)
    resistance = h.max(axis=1, keepdims=True)
    dist_to_resistance = (resistance - c[:, -1:]) / (resistance + 1e-10)
    features.append(dist_to_resistance.flatten())

    # Position in range (normalized)
    range_span = resistance - support + 1e-10
    position_in_range = (c[:, -1:] - support) / range_span
    features.append(position_in_range.flatten())

    return features


def _extract_candle_patterns(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray) -> list:
    """Extract candle pattern features.

    Args:
        o, h, l, c: OHLC arrays [N, T]

    Returns:
        List of feature arrays
    """
    N, T = o.shape
    features = []

    # Body and range
    body = np.abs(c - o)
    total_range = h - l + 1e-10

    # Doji: body < 10% of range
    doji = (body / total_range) < 0.1
    features.append(doji.sum(axis=1))  # num_doji

    # Engulfing patterns
    bullish_engulf = np.zeros(N)
    bearish_engulf = np.zeros(N)

    for i in range(N):
        for t in range(1, T):
            # Bullish engulfing: current candle up, previous down, current body engulfs previous
            if c[i, t] > o[i, t] and c[i, t - 1] < o[i, t - 1]:
                if o[i, t] < c[i, t - 1] and c[i, t] > o[i, t - 1]:
                    bullish_engulf[i] += 1

            # Bearish engulfing
            if c[i, t] < o[i, t] and c[i, t - 1] > o[i, t - 1]:
                if o[i, t] > c[i, t - 1] and c[i, t] < o[i, t - 1]:
                    bearish_engulf[i] += 1

    features.append(bullish_engulf)
    features.append(bearish_engulf)

    # Hammer/Shooting star
    upper_shadow = h - np.maximum(o, c)
    lower_shadow = np.minimum(o, c) - l

    # Hammer: lower shadow > 2x body, small upper shadow
    hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
    features.append(hammer.sum(axis=1))  # num_hammer

    # Shooting star: upper shadow > 2x body, small lower shadow
    shooting_star = (upper_shadow > 2 * body) & (lower_shadow < body)
    features.append(shooting_star.sum(axis=1))  # num_shooting_star

    return features


def _extract_market_regime_features(h: np.ndarray, l: np.ndarray, c: np.ndarray) -> list:
    """Extract market regime features (pattern-based regime classification).

    Args:
        h, l, c: High, Low, Close arrays [N, T]

    Returns:
        List of feature arrays for market regime classification
    """
    N, T = h.shape
    features = []

    # 1. Trend strength (correlation coefficient)
    trend_strength = np.zeros(N)
    for i in range(N):
        if T > 1:
            x = np.arange(T)
            slope, _, r_value, _, _ = linregress(x, c[i])
            trend_strength[i] = abs(r_value)  # |r| indicates trend strength
    features.append(trend_strength)

    # 2. Volatility regime (std of returns normalized by price)
    vol_regime = np.zeros(N)
    for i in range(N):
        if T > 1:
            returns = np.diff(c[i]) / (c[i][:-1] + 1e-8)
            vol_regime[i] = returns.std() / (np.mean(c[i]) + 1e-8)
    features.append(vol_regime)

    # 3. Range regime (price range relative to average price)
    range_regime = np.zeros(N)
    for i in range(N):
        if T > 0:
            price_range = h[i].max() - l[i].min()
            avg_price = np.mean(c[i])
            if avg_price > 0:
                range_regime[i] = price_range / avg_price
    features.append(range_regime)

    return features


def _extract_buffer_context(c_left: np.ndarray, h_left: np.ndarray, l_left: np.ndarray,
                             c_inner: np.ndarray, c_right: np.ndarray) -> list:
    """Extract context features from buffer regions.

    Args:
        c_left, h_left, l_left: Left buffer OHLC [N, 30]
        c_inner: Inner window close [N, 45]
        c_right: Right buffer close [N, 30]

    Returns:
        List of feature arrays
    """
    N = c_left.shape[0]
    features = []

    # Left buffer momentum (return)
    left_return = (c_left[:, -1] - c_left[:, 0]) / (c_left[:, 0] + 1e-10)
    features.append(left_return)

    # Left buffer volatility (std of returns)
    left_returns = np.diff(c_left, axis=1) / (c_left[:, :-1] + 1e-10)
    left_vol = left_returns.std(axis=1)
    features.append(left_vol)

    # Transition from left to inner (gap)
    left_to_inner_gap = (c_inner[:, 0] - c_left[:, -1]) / (c_left[:, -1] + 1e-10)
    features.append(left_to_inner_gap)

    # Right buffer momentum (forward-looking context)
    right_return = (c_right[:, -1] - c_right[:, 0]) / (c_right[:, 0] + 1e-10)
    features.append(right_return)

    # Transition from inner to right
    inner_to_right_gap = (c_right[:, 0] - c_inner[:, -1]) / (c_inner[:, -1] + 1e-10)
    features.append(inner_to_right_gap)

    return features
