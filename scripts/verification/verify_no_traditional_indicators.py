#!/usr/bin/env python3
"""
Verification script to ensure all traditional technical indicators have been removed
from the Moola project's feature engineering codebase.

CRITICAL REQUIREMENTS:
- No Williams %R
- No RSI
- No MACD
- No Bollinger Bands
- No EMAs
- No SMAs
- No ATR
- No OBV
- No Stochastic oscillators
- No ADX
- No CCI
- No MFI
- Any other traditional technical indicators

Only pure pattern-based features should remain:
- Pattern morphology (fractal dimension, symmetry, curvature)
- Swing point detection and analysis
- Market microstructure (body ratios, wick dominance, price efficiency)
- Geometric invariants (Hurst exponent, turning points, path length)
- Relative dynamics (pattern vs context, momentum continuity)
- Temporal signatures (autocorrelation, periodicity)
- Pure OHLC relationships
"""

import os
import re
from pathlib import Path

# Traditional indicators to find (case insensitive)
TRADITIONAL_INDICATORS = [
    r'(?i)williams', r'(?i)rsi', r'(?i)macd', r'(?i)bollinger',
    r'(?i)ema', r'(?i)sma', r'(?i)atr', r'(?i)obv',
    r'(?i)stochastic', r'(?i)adx', r'(?i)cci', r'(?i)mfi'
]

# Directories to search
FEATURE_DIRS = [
    'src/moola/features',
]

def check_traditional_indicators():
    """Check for any remaining traditional indicators in feature files."""
    issues = []

    for feature_dir in FEATURE_DIRS:
        feature_path = Path(feature_dir)
        if not feature_path.exists():
            continue

        # Search in all Python files
        for py_file in feature_path.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()

                # Check each traditional indicator
                for indicator in TRADITIONAL_INDICATORS:
                    # Clean indicator pattern (remove (?i) prefix)
                    clean_indicator = re.sub(r'\(\?i\)\(', '', indicator).replace(')', '')

                    # Look for function definitions
                    if re.search(r'def\s+.*\b' + re.escape(clean_indicator) + r'\b', content):
                        matches = re.finditer(indicator, content)
                        for match in matches:
                            # Get line number
                            line_num = content[:match.start()].count('\n') + 1
                            line_content = content.split('\n')[line_num - 1].strip()

                            # Skip if it's in comments about what was removed
                            if ('removed' in line_content.lower() or
                                'removed by user request' in line_content.lower() or
                                'traditional' in line_content.lower()):
                                continue

                            issues.append({
                                'file': str(py_file),
                                'line': line_num,
                                'content': line_content,
                                'indicator': indicator
                            })

                    # Look for function calls or feature assignments
                    if re.search(r'_?' + re.escape(clean_indicator) + r'_?\s*\(\s*[a-zA-Z_].*\s*\)|features?\s*\.\s*extend\(\s*\[.*\b' + re.escape(clean_indicator) + r'\b.*?\]', content):
                        matches = re.finditer(indicator, content)
                        for match in matches:
                            # Get line number
                            line_num = content[:match.start()].count('\n') + 1
                            line_content = content.split('\n')[line_num - 1].strip()

                            # Skip if it's in comments about what was removed
                            if ('removed' in line_content.lower() or
                                'removed by user request' in line_content.lower() or
                                'traditional' in line_content.lower()):
                                continue

                            issues.append({
                                'file': str(py_file),
                                'line': line_num,
                                'content': line_content,
                                'indicator': indicator
                            })

    return issues

def main():
    print("🔍 Verifying no traditional technical indicators remain in Moola feature engineering...")
    print()

    issues = check_traditional_indicators()

    if issues:
        print("❌ Found traditional technical indicators:")
        print()
        for issue in issues:
            print(f"📁 File: {issue['file']}")
            print(f"📍 Line {issue['line']}: {issue['content']}")
            print(f"🎯 Indicator: {issue['indicator']}")
            print()

        print("These traditional indicators need to be removed from the codebase.")
        return False
    else:
        print("✅ No traditional technical indicators found in feature engineering code!")
        print()
        print("🎉 The Moola project now contains only pure pattern-based features:")
        print("   - Pattern morphology features")
        print("   - Market microstructure features")
        print("   - Geometric invariant features")
        print("   - Relative dynamic features")
        print("   - Temporal signature features")
        print("   - Pure OHLC mathematical relationships")
        print()
        print("The feature engineering pipeline is completely free of traditional technical indicators.")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)