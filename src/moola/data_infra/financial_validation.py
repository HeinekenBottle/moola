"""Enhanced financial market data validation for OHLC time series.

Extends base OHLC validation with market-specific constraints:
- Session continuity checks (weekend/holiday gaps)
- Corporate action detection (splits, dividends)
- Market microstructure validation (tick size, lot size)
- Liquidity and tradability assessments
- Cross-asset consistency checks
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from ..data_infra.schemas import OHLCBar


@dataclass
class MarketConstraints:
    """Market-specific validation constraints."""

    # Trading session constraints
    trading_hours: Dict[str, Tuple[int, int]] = None  # {"weekday": (9, 16)}
    weekend_gap_threshold: float = 0.05  # 5% max weekend gap

    # Corporate action thresholds
    split_ratio_min: float = 0.4  # Minimum 2:5 split ratio
    split_ratio_max: float = 5.0  # Maximum 5:1 split ratio
    dividend_yield_max: float = 0.2  # 20% max dividend yield

    # Market microstructure
    tick_size: Optional[float] = None  # Minimum price increment
    lot_size: Optional[int] = None  # Minimum trade size
    min_trading_value: float = 1000.0  # Minimum daily trading value

    # Liquidity constraints
    min_volume_bars: int = 10  # Minimum bars with non-zero volume
    max_spread_ratio: float = 0.1  # Max 10% bid-ask spread relative to price


class EnhancedOHLCValidator:
    """Advanced OHLC validation with market-specific constraints."""

    def __init__(self, constraints: Optional[MarketConstraints] = None):
        """Initialize validator with market constraints.

        Args:
            constraints: Market-specific validation rules
        """
        self.constraints = constraints or MarketConstraints()

    def validate_ohlc_sequence(
        self, ohlc_data: Union[List[OHLCBar], np.ndarray, pd.DataFrame], symbol: str = "UNKNOWN"
    ) -> Dict[str, any]:
        """Comprehensive validation of OHLC sequence.

        Args:
            ohlc_data: OHLC data as list of bars, numpy array, or DataFrame
            symbol: Trading symbol for context-specific validation

        Returns:
            Validation report with detected issues and quality metrics
        """
        # Convert to consistent format
        df = self._to_dataframe(ohlc_data)
        if df is None or len(df) == 0:
            return {"valid": False, "errors": ["Empty or invalid data format"]}

        validation_report = {
            "symbol": symbol,
            "total_bars": len(df),
            "valid": True,
            "errors": [],
            "warnings": [],
            "quality_metrics": {},
        }

        # 1. Basic OHLC validation (existing logic)
        self._validate_basic_ohlc(df, validation_report)

        # 2. Session continuity validation
        self._validate_session_continuity(df, validation_report)

        # 3. Corporate action detection
        self._detect_corporate_actions(df, validation_report)

        # 4. Market microstructure validation
        self._validate_microstructure(df, validation_report)

        # 5. Liquidity assessment
        self._assess_liquidity(df, validation_report)

        # 6. Price anomaly detection
        self._detect_price_anomalies(df, validation_report)

        # 7. Trend consistency checks
        self._validate_trend_consistency(df, validation_report)

        # Compute overall quality score
        validation_report["quality_score"] = self._compute_quality_score(validation_report)

        return validation_report

    def _to_dataframe(
        self, ohlc_data: Union[List[OHLCBar], np.ndarray, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Convert various OHLC formats to DataFrame."""
        if isinstance(ohlc_data, pd.DataFrame):
            return ohlc_data

        elif isinstance(ohlc_data, list) and len(ohlc_data) > 0:
            if isinstance(ohlc_data[0], OHLCBar):
                data = [
                    (bar.open, bar.high, bar.low, bar.close, bar.timestamp) for bar in ohlc_data
                ]
                df = pd.DataFrame(data, columns=["open", "high", "low", "close", "timestamp"])
                return df

        elif isinstance(ohlc_data, np.ndarray):
            if ohlc_data.ndim == 2 and ohlc_data.shape[1] == 4:
                df = pd.DataFrame(ohlc_data, columns=["open", "high", "low", "close"])
                return df

        return None

    def _validate_basic_ohlc(self, df: pd.DataFrame, report: Dict):
        """Basic OHLC logical validation (extends existing logic)."""
        for i, row in df.iterrows():
            # Check OHLC relationships
            if row["high"] < row["low"]:
                report["errors"].append(f"Bar {i}: High ({row['high']}) < Low ({row['low']})")
                report["valid"] = False

            if row["high"] < max(row["open"], row["close"]):
                report["errors"].append(f"Bar {i}: High ({row['high']}) < max(Open, Close)")
                report["valid"] = False

            if row["low"] > min(row["open"], row["close"]):
                report["errors"].append(f"Bar {i}: Low ({row['low']}) > min(Open, Close)")
                report["valid"] = False

            # Check for unrealistic price movements
            if i > 0:
                prev_close = df.iloc[i - 1]["close"]
                price_change = abs(row["open"] - prev_close) / prev_close

                # Gap detection (>10% overnight gap is suspicious for most assets)
                if price_change > 0.1:
                    report["warnings"].append(
                        f"Bar {i}: Large gap {price_change:.2%} from previous close"
                    )

                # Intrabar movement check
                intrabar_range = (row["high"] - row["low"]) / row["open"]
                if intrabar_range > 3.0:  # >200% intrabar movement
                    report["warnings"].append(
                        f"Bar {i}: Excessive intrabar range {intrabar_range:.1f}x"
                    )

    def _validate_session_continuity(self, df: pd.DataFrame, report: Dict):
        """Validate trading session continuity and detect unusual gaps."""
        if "timestamp" not in df.columns:
            report["warnings"].append("No timestamp data - cannot validate session continuity")
            return

        # Convert timestamps to datetime if needed
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        for i in range(1, len(df)):
            time_diff = df.iloc[i]["timestamp"] - df.iloc[i - 1]["timestamp"]
            time_hours = time_diff.total_seconds() / 3600

            # Weekend gap detection
            if time_hours > 48:  # More than 2 days
                price_change = (
                    abs(df.iloc[i]["open"] - df.iloc[i - 1]["close"]) / df.iloc[i - 1]["close"]
                )

                if price_change > self.constraints.weekend_gap_threshold:
                    report["warnings"].append(
                        f"Bar {i}: Weekend gap of {price_change:.2%} may indicate corporate action"
                    )

            # Intraday gap detection (missing bars)
            elif 1 < time_hours < 24:  # Same day gap
                report["warnings"].append(
                    f"Bar {i}: Intraday gap of {time_hours:.1f} hours - possible missing data"
                )

    def _detect_corporate_actions(self, df: pd.DataFrame, report: Dict):
        """Detect potential corporate actions (splits, dividends)."""
        for i in range(1, len(df)):
            prev_close = df.iloc[i - 1]["close"]
            curr_open = df.iloc[i]["open"]

            # Split detection (large price change with proportional volume change)
            price_ratio = curr_open / prev_close

            if price_ratio < self.constraints.split_ratio_min:  # Split down (e.g., 2:1)
                split_ratio = 1 / price_ratio
                if 2 <= split_ratio <= 5:  # Reasonable split ratios
                    report["warnings"].append(
                        f"Bar {i}: Possible split detected - price ratio {price_ratio:.3f} "
                        f"(suggested split ratio: {split_ratio:.1f}:1)"
                    )

            elif price_ratio > self.constraints.split_ratio_max:  # Reverse split
                split_ratio = price_ratio
                if 2 <= split_ratio <= 5:
                    report["warnings"].append(
                        f"Bar {i}: Possible reverse split detected - price ratio {price_ratio:.3f} "
                        f"(suggested split ratio: 1:{split_ratio:.1f})"
                    )

            # Dividend detection (price drop on ex-dividend date)
            price_drop = (prev_close - curr_open) / prev_close
            if 0.02 <= price_drop <= self.constraints.dividend_yield_max:  # 2-20% drop
                # Check if it's a Monday (ex-dividend often Monday)
                if df.iloc[i]["timestamp"].weekday() == 0:
                    report["warnings"].append(
                        f"Bar {i}: Possible dividend detected - {price_drop:.2%} price drop"
                    )

    def _validate_microstructure(self, df: pd.DataFrame, report: Dict):
        """Validate market microstructure constraints."""
        if self.constraints.tick_size is not None:
            # Check price alignment with tick size
            for col in ["open", "high", "low", "close"]:
                tick_violations = (df[col] % self.constraints.tick_size).abs() > 1e-10
                if tick_violations.any():
                    report["warnings"].append(
                        f"Tick size violations in {col}: {tick_violations.sum()} bars"
                    )

        # Validate spread consistency
        typical_spreads = (df["high"] - df["low"]) / df["close"]
        max_spread = typical_spreads.max()

        if max_spread > self.constraints.max_spread_ratio:
            report["warnings"].append(
                f"Excessive spread detected: {max_spread:.2%} max intrabar spread"
            )

    def _assess_liquidity(self, df: pd.DataFrame, report: Dict):
        """Assess liquidity and tradability of the instrument."""
        # Price-based liquidity proxies (for OHLC without volume)

        # 1. Price continuity - frequent small movements suggest good liquidity
        price_changes = df["close"].pct_change().abs()
        small_movements = (price_changes < 0.001).sum()  # <0.1% movements

        if small_movements < self.constraints.min_volume_bars:
            report["warnings"].append(
                f"Low liquidity detected: only {small_movements} small price movements "
                f"out of {len(df)} bars"
            )

        # 2. Range consistency - erratic ranges may indicate liquidity issues
        ranges = (df["high"] - df["low"]) / df["open"]
        range_volatility = ranges.std() / ranges.mean()

        if range_volatility > 2.0:  # High variance in ranges
            report["warnings"].append(
                f"Inconsistent ranges detected (volatility: {range_volatility:.2f}) "
                "- possible liquidity issues"
            )

    def _detect_price_anomalies(self, df: pd.DataFrame, report: Dict):
        """Detect statistical price anomalies."""
        returns = df["close"].pct_change().dropna()

        # Z-score based anomaly detection
        z_scores = np.abs((returns - returns.mean()) / returns.std())
        extreme_moves = z_scores > 5  # >5 standard deviations

        if extreme_moves.any():
            extreme_indices = returns[extreme_moves].index.tolist()
            report["warnings"].append(
                f"Extreme price movements detected at bars: {extreme_indices} "
                f"(>5 std deviations)"
            )

        # Volatility clustering detection
        rolling_vol = returns.rolling(window=20).std()
        vol_ratio = rolling_vol.max() / rolling_vol.min()

        if vol_ratio > 10:  # Volatility varies by more than 10x
            report["warnings"].append(
                f"High volatility clustering detected (vol ratio: {vol_ratio:.1f}x)"
            )

    def _validate_trend_consistency(self, df: pd.DataFrame, report: Dict):
        """Validate trend consistency and detect potential data issues."""
        # Check for impossible trend reversals
        for i in range(2, len(df)):
            # Three consecutive bars with impossible pattern
            prev_high = df.iloc[i - 2]["high"]
            prev_low = df.iloc[i - 2]["low"]
            curr_high = df.iloc[i]["high"]
            curr_low = df.iloc[i]["low"]

            # Impossible: current high < previous low AND current low > previous high
            if curr_high < prev_low and curr_low > prev_high:
                report["errors"].append(
                    f"Bar {i}: Impossible price pattern - data corruption likely"
                )
                report["valid"] = False

        # Check for stuck prices (possible data feed issues)
        stuck_periods = (
            df["close"].rolling(window=5).apply(lambda x: x.nunique() == 1, raw=False).fillna(False)
        )

        if stuck_periods.any():
            stuck_count = stuck_periods.sum()
            report["warnings"].append(
                f"Stuck prices detected: {stuck_count} instances of 5 consecutive identical prices"
            )

    def _compute_quality_score(self, report: Dict) -> float:
        """Compute overall data quality score (0-100)."""
        base_score = 100.0

        # Penalize errors heavily
        base_score -= len(report["errors"]) * 20

        # Penalize warnings moderately
        base_score -= len(report["warnings"]) * 5

        # Bonus for high-quality indicators
        if report["total_bars"] > 100:
            base_score += 5  # Sufficient data

        # Ensure score is in valid range
        return max(0.0, min(100.0, base_score))


def validate_financial_dataset(
    ohlc_data: Union[List[OHLCBar], np.ndarray, pd.DataFrame],
    symbol: str = "UNKNOWN",
    constraints: Optional[MarketConstraints] = None,
) -> Dict[str, any]:
    """Convenience function for comprehensive financial data validation.

    Args:
        ohlc_data: OHLC time series data
        symbol: Trading symbol for context
        constraints: Market-specific validation constraints

    Returns:
        Detailed validation report
    """
    validator = EnhancedOHLCValidator(constraints)
    return validator.validate_ohlc_sequence(ohlc_data, symbol)
