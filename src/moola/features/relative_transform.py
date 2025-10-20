"""Relative feature transformation pipeline.

Converts absolute OHLC data to scale-invariant relative features for improved
model generalization across different price ranges and market conditions.
"""


import numpy as np
from loguru import logger


class RelativeFeatureTransform:
    """Transform absolute OHLC to scale-invariant relative features.

    Converts [N, 105, 4] OHLC data to [N, 105, 11] relative features:
    - 4 log returns: log(price_t / price_t-1) for O, H, L, C
    - 3 candle ratios: body/range, upper_wick/range, lower_wick/range
    - 4 rolling z-scores: standardized values over 20-bar window for O, H, L, C

    Attributes:
        eps: Small constant to prevent division by zero (default: 1e-8)

    Example:
        >>> transform = RelativeFeatureTransform(eps=1e-8)
        >>> X_ohlc = np.random.randn(100, 105, 4)  # [N, 105, 4]
        >>> X_rel = transform.transform(X_ohlc)    # [N, 105, 11]
        >>> print(X_rel.shape)
        (100, 105, 11)
        >>> print(transform.get_feature_names())
        ['log_return_open', 'log_return_high', ...]
    """

    def __init__(self, eps: float = 1e-8):
        """Initialize the relative feature transformer.

        Args:
            eps: Small constant to prevent division by zero and numerical instability.
                Must be positive and typically between 1e-10 and 1e-6.

        Raises:
            ValueError: If eps is not positive.
        """
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps
        logger.debug(f"Initialized RelativeFeatureTransform with eps={eps}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform [N, 105, 4] OHLC to [N, 105, 11] relative features.

        Args:
            X: OHLC data with shape [N, 105, 4] where:
                - N: number of samples
                - 105: sequence length
                - 4: features [open, high, low, close]

        Returns:
            Transformed features with shape [N, 105, 11]:
                - 4 log returns
                - 3 candle ratios
                - 4 rolling z-scores

        Raises:
            ValueError: If input shape is not [N, 105, 4]
            TypeError: If input is not a numpy array or dtype is not numeric
        """
        # Validate input
        if not isinstance(X, np.ndarray):
            raise TypeError(f"X must be numpy array, got {type(X)}")

        if X.ndim != 3 or X.shape[1] != 105 or X.shape[2] != 4:
            raise ValueError(f"X must have shape [N, 105, 4], got {X.shape}")

        if not np.issubdtype(X.dtype, np.number):
            raise TypeError(f"X must have numeric dtype, got {X.dtype}")

        N, T, F = X.shape
        logger.debug(f"Transforming OHLC data: shape={X.shape}")

        # Extract OHLC components
        open_prices = X[:, :, 0]  # [N, 105]
        high_prices = X[:, :, 1]
        low_prices = X[:, :, 2]
        close_prices = X[:, :, 3]

        # 1. Compute log returns (4 features)
        # For first bar (t=0), use zero as there's no previous price
        log_returns = self._compute_log_returns(
            open_prices, high_prices, low_prices, close_prices
        )  # [N, 105, 4]

        # 2. Compute candle ratios (3 features)
        candle_ratios = self._compute_candle_ratios(
            open_prices, high_prices, low_prices, close_prices
        )  # [N, 105, 3]

        # 3. Compute rolling z-scores (4 features)
        z_scores = self._compute_rolling_zscores(
            open_prices, high_prices, low_prices, close_prices
        )  # [N, 105, 4]

        # Concatenate all features
        X_relative = np.concatenate([log_returns, candle_ratios, z_scores], axis=2)

        # Final validation
        assert X_relative.shape == (N, T, 11), f"Output shape mismatch: {X_relative.shape}"

        # Replace any remaining NaN/inf values with zeros
        X_relative = np.nan_to_num(X_relative, nan=0.0, posinf=0.0, neginf=0.0)

        logger.debug(f"Transformed to relative features: shape={X_relative.shape}")

        return X_relative

    def _compute_log_returns(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
    ) -> np.ndarray:
        """Compute log returns for OHLC.

        Args:
            open_prices: Open prices with shape [N, 105]
            high_prices: High prices with shape [N, 105]
            low_prices: Low prices with shape [N, 105]
            close_prices: Close prices with shape [N, 105]

        Returns:
            Log returns with shape [N, 105, 4]
        """
        N, T = open_prices.shape

        # Initialize with zeros
        log_returns = np.zeros((N, T, 4), dtype=np.float32)

        # Compute log returns for t >= 1
        # log(price_t / price_t-1) = log(price_t) - log(price_t-1)
        for i, price in enumerate([open_prices, high_prices, low_prices, close_prices]):
            # Add eps to prevent log(0)
            price_safe = price + self.eps

            # Compute log difference
            log_price = np.log(price_safe)
            log_returns[:, 1:, i] = log_price[:, 1:] - log_price[:, :-1]

            # First bar (t=0) remains zero

        # Replace any NaN/inf with zeros
        log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)

        return log_returns

    def _compute_candle_ratios(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
    ) -> np.ndarray:
        """Compute candle body and wick ratios.

        Args:
            open_prices: Open prices with shape [N, 105]
            high_prices: High prices with shape [N, 105]
            low_prices: Low prices with shape [N, 105]
            close_prices: Close prices with shape [N, 105]

        Returns:
            Candle ratios with shape [N, 105, 3]:
                - body_ratio: abs(close - open) / range
                - upper_wick_ratio: (high - max(open, close)) / range
                - lower_wick_ratio: (min(open, close) - low) / range
        """
        N, T = open_prices.shape

        # Compute range (high - low)
        range_val = high_prices - low_prices + self.eps  # Add eps to prevent division by zero

        # Compute body ratio: abs(close - open) / range
        body_ratio = np.abs(close_prices - open_prices) / range_val

        # Compute upper wick ratio: (high - max(open, close)) / range
        upper_wick_ratio = (high_prices - np.maximum(open_prices, close_prices)) / range_val

        # Compute lower wick ratio: (min(open, close) - low) / range
        lower_wick_ratio = (np.minimum(open_prices, close_prices) - low_prices) / range_val

        # Stack ratios
        candle_ratios = np.stack([body_ratio, upper_wick_ratio, lower_wick_ratio], axis=2)

        # Replace any NaN/inf with zeros
        candle_ratios = np.nan_to_num(candle_ratios, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip to [0, 1] range (ratios should be between 0 and 1)
        candle_ratios = np.clip(candle_ratios, 0.0, 1.0)

        return candle_ratios

    def _compute_rolling_zscores(
        self,
        open_prices: np.ndarray,
        high_prices: np.ndarray,
        low_prices: np.ndarray,
        close_prices: np.ndarray,
    ) -> np.ndarray:
        """Compute rolling z-scores over 20-bar window.

        Args:
            open_prices: Open prices with shape [N, 105]
            high_prices: High prices with shape [N, 105]
            low_prices: Low prices with shape [N, 105]
            close_prices: Close prices with shape [N, 105]

        Returns:
            Z-scores with shape [N, 105, 4], clipped to [-10, 10]
        """
        N, T = open_prices.shape
        window = 20

        # Initialize z-scores
        z_scores = np.zeros((N, T, 4), dtype=np.float32)

        # Compute z-scores for each OHLC component
        for i, price in enumerate([open_prices, high_prices, low_prices, close_prices]):
            # For each time step t >= window-1, compute z-score over [t-window+1, t]
            for t in range(window - 1, T):
                # Extract window
                window_data = price[:, max(0, t - window + 1) : t + 1]  # [N, window]

                # Compute mean and std
                mean = np.mean(window_data, axis=1, keepdims=True)  # [N, 1]
                std = np.std(window_data, axis=1, keepdims=True) + self.eps  # [N, 1]

                # Compute z-score for current time step
                z_scores[:, t, i] = ((price[:, t : t + 1] - mean) / std).squeeze()

            # For t < window-1, use available data
            for t in range(window - 1):
                if t == 0:
                    # First bar: no history, use zero
                    continue

                # Use all available data from [0, t]
                window_data = price[:, : t + 1]  # [N, t+1]

                mean = np.mean(window_data, axis=1, keepdims=True)
                std = np.std(window_data, axis=1, keepdims=True) + self.eps

                z_scores[:, t, i] = ((price[:, t : t + 1] - mean) / std).squeeze()

        # Replace any NaN/inf with zeros
        z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip extreme values to [-10, 10]
        z_scores = np.clip(z_scores, -10.0, 10.0)

        return z_scores

    def get_feature_names(self) -> list[str]:
        """Return list of 11 feature names.

        Returns:
            List of feature names in order:
                [log_return_open, log_return_high, log_return_low, log_return_close,
                 body_ratio, upper_wick_ratio, lower_wick_ratio,
                 zscore_open, zscore_high, zscore_low, zscore_close]
        """
        return [
            # Log returns (4)
            "log_return_open",
            "log_return_high",
            "log_return_low",
            "log_return_close",
            # Candle ratios (3)
            "body_ratio",
            "upper_wick_ratio",
            "lower_wick_ratio",
            # Rolling z-scores (4)
            "zscore_open",
            "zscore_high",
            "zscore_low",
            "zscore_close",
        ]
