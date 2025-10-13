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
- Williams %R (only approved technical indicator)

Features are extracted primarily from the inner prediction window [30:75]
with context summaries from left and right buffers.
"""

from typing import Tuple

import numpy as np
from scipy.stats import linregress


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
        Engineered features of shape [N, ~40] containing:
        - Market structure (HH/LL, swings)
        - Liquidity zones (equal highs/lows)
        - Fair value gaps
        - Order blocks
        - Imbalance ratios
        - Geometry features (slopes, curvature)
        - Distance measures
        - Candle patterns
        - Williams %R
        - Buffer context summaries
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

        # ===== 9. WILLIAMS %R =====
        sample_features.extend(_extract_williams_r(h_arr, l_arr, c_arr))

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


def _extract_williams_r(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int = 10) -> list:
    """Extract Williams %R indicator.

    Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100

    Args:
        h, l, c: High, Low, Close arrays [N, T]
        period: Lookback period (default 10)

    Returns:
        List of feature arrays
    """
    N, T = h.shape
    features = []

    williams_r = np.zeros(N)

    for i in range(N):
        if T >= period:
            # Calculate for last 'period' bars
            hh = h[i, -period:].max()
            ll = l[i, -period:].min()
            close = c[i, -1]

            if hh > ll:
                williams_r[i] = ((hh - close) / (hh - ll)) * -100
            else:
                williams_r[i] = -50  # Neutral value if no range

    features.append(williams_r)

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
