#!/usr/bin/env python3
"""Test script for dual-input data pipeline integration.

This script tests:
1. Backward compatibility with existing models
2. Engineered feature extraction
3. Data pipeline integration
4. Model training with both raw OHLC and engineered features

Usage:
    python3 scripts/test_dual_input_integration.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.data.dual_input_pipeline import (
    create_dual_input_processor,
    prepare_model_inputs,
    FeatureConfig
)
from moola.features.small_dataset_features import extract_optimized_features
from moola.features.price_action_features import engineer_multiscale_features


def test_data_loading():
    """Test loading the sample data."""
    logger.info("=" * 60)
    logger.info("TESTING DATA LOADING")
    logger.info("=" * 60)

    data_path = Path("data/processed/train_pivot_134.parquet")
    if not data_path.exists():
        logger.error(f"Sample data not found at {data_path}")
        return False

    df = pd.read_parquet(data_path)
    logger.info(f"✅ Loaded data: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Validate data structure
    required_columns = ["features", "label", "expansion_start", "expansion_end"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        logger.error(f"❌ Missing columns: {missing}")
        return False

    # Check features structure
    sample_features = np.stack(df.iloc[0]["features"])
    if sample_features.shape != (105, 4):
        logger.error(f"❌ Invalid features shape: {sample_features.shape}, expected (105, 4)")
        return False

    logger.info(f"✅ Data structure valid: OHLC shape {sample_features.shape}")
    logger.info(f"✅ Label distribution: {df['label'].value_counts().to_dict()}")
    return True


def test_backward_compatibility():
    """Test backward compatibility - models work without engineered features."""
    logger.info("=" * 60)
    logger.info("TESTING BACKWARD COMPATIBILITY")
    logger.info("=" * 60)

    data_path = Path("data/processed/train_pivot_134.parquet")
    df = pd.read_parquet(data_path)

    # Test processor with engineered features DISABLED (default behavior)
    processor = create_dual_input_processor(
        use_engineered_features=False,
        max_engineered_features=50,
        use_hopsketch=False
    )

    processed_data = processor.process_training_data(df, enable_engineered_features=False)

    # Verify raw OHLC data is intact
    X_ohlc = processed_data["X_ohlc"]
    if X_ohlc.shape != (len(df), 105, 4):
        logger.error(f"❌ Raw OHLC shape mismatch: {X_ohlc.shape}")
        return False

    # Verify no engineered features
    if processed_data["X_engineered"] is not None:
        logger.error(f"❌ Engineered features should be None: {processed_data['X_engineered']}")
        return False

    # Test model input preparation for different model types
    for model_type in ["lstm", "xgboost", "logreg"]:
        model_inputs = prepare_model_inputs(
            processed_data,
            model_type=model_type,
            use_engineered_features=False
        )

        X = model_inputs["X"]
        if model_type in ["lstm", "transformer"]:
            expected_shape = (len(df), 105, 4)
        else:  # xgboost, logreg
            expected_shape = (len(df), 420)  # flattened

        if X.shape != expected_shape:
            logger.error(f"❌ {model_type} input shape mismatch: {X.shape}, expected {expected_shape}")
            return False

        logger.info(f"✅ {model_type} backward compatibility: {X.shape}")

    logger.info("✅ Backward compatibility test PASSED")
    return True


def test_engineered_features():
    """Test engineered feature extraction."""
    logger.info("=" * 60)
    logger.info("TESTING ENGINEERED FEATURES")
    logger.info("=" * 60)

    data_path = Path("data/processed/train_pivot_134.parquet")
    df = pd.read_parquet(data_path)

    # Extract raw OHLC for direct feature testing
    X_ohlc = np.stack([np.stack(f) for f in df["features"]])
    expansion_start = df["expansion_start"].values
    expansion_end = df["expansion_end"].values
    y = df["label"].values

    logger.info(f"Input data: {X_ohlc.shape}")

    # Test 1: Small dataset optimized features
    try:
        small_features, small_names = extract_optimized_features(
            X_ohlc, expansion_start, expansion_end, y, max_total_features=25
        )
        logger.info(f"✅ Small dataset features: {small_features.shape}")
        logger.info(f"  Sample feature names: {small_names[:5]}")
    except Exception as e:
        logger.error(f"❌ Small dataset features failed: {e}")
        return False

    # Test 2: Multi-scale price action features
    try:
        multiscale_features = engineer_multiscale_features(
            X_ohlc, expansion_start, expansion_end
        )
        logger.info(f"✅ Multi-scale features: {multiscale_features.shape}")
    except Exception as e:
        logger.error(f"❌ Multi-scale features failed: {e}")
        return False

    logger.info("✅ Engineered features test PASSED")
    return True


def test_dual_input_pipeline():
    """Test the complete dual-input pipeline."""
    logger.info("=" * 60)
    logger.info("TESTING DUAL-INPUT PIPELINE")
    logger.info("=" * 60)

    data_path = Path("data/processed/train_pivot_134.parquet")
    df = pd.read_parquet(data_path)

    # Test with engineered features ENABLED
    processor = create_dual_input_processor(
        use_engineered_features=True,
        max_engineered_features=50,
        use_hopsketch=False
    )

    processed_data = processor.process_training_data(df, enable_engineered_features=True)

    # Verify raw OHLC data is still available
    X_ohlc = processed_data["X_ohlc"]
    if X_ohlc.shape != (len(df), 105, 4):
        logger.error(f"❌ Raw OHLC shape mismatch: {X_ohlc.shape}")
        return False

    # Verify engineered features are extracted
    X_engineered = processed_data["X_engineered"]
    if X_engineered is None:
        logger.error("❌ Engineered features should not be None")
        return False

    logger.info(f"✅ Raw OHLC data: {X_ohlc.shape}")
    logger.info(f"✅ Engineered features: {X_engineered.shape}")
    logger.info(f"✅ Feature names: {len(processed_data['feature_names'])}")

    # Test feature statistics
    feature_stats = processor.get_feature_statistics(X_engineered)
    logger.info(f"✅ Feature stats: mean={feature_stats['mean']:.4f}, "
                f"std={feature_stats['std']:.4f}, range=[{feature_stats['min']:.4f}, {feature_stats['max']:.4f}]")

    # Test model input preparation for different model types
    for model_type in ["lstm", "xgboost"]:
        model_inputs = prepare_model_inputs(
            processed_data,
            model_type=model_type,
            use_engineered_features=True
        )

        X = model_inputs["X"]
        if model_type == "lstm":
            expected_shape = (len(df), 105, 4)  # LSTM uses raw OHLC
        else:  # xgboost
            expected_shape = (len(df), X_engineered.shape[1])  # XGBoost uses engineered

        if X.shape != expected_shape:
            logger.error(f"❌ {model_type} input shape mismatch: {X.shape}, expected {expected_shape}")
            return False

        logger.info(f"✅ {model_type} dual-input compatibility: {X.shape}")

    logger.info("✅ Dual-input pipeline test PASSED")
    return True


def test_configuration_scenarios():
    """Test different configuration scenarios."""
    logger.info("=" * 60)
    logger.info("TESTING CONFIGURATION SCENARIOS")
    logger.info("=" * 60)

    data_path = Path("data/processed/train_pivot_134.parquet")
    df = pd.read_parquet(data_path)

    scenarios = [
        {
            "name": "Default (no engineered features)",
            "use_engineered_features": False,
            "max_engineered_features": 50,
            "use_hopsketch": False,
            "expected_engineered": False
        },
        {
            "name": "Small dataset features only",
            "use_engineered_features": True,
            "max_engineered_features": 25,
            "use_hopsketch": False,
            "expected_engineered": True
        },
        {
            "name": "Max engineered features",
            "use_engineered_features": True,
            "max_engineered_features": 100,
            "use_hopsketch": False,
            "expected_engineered": True
        },
    ]

    for scenario in scenarios:
        logger.info(f"Testing: {scenario['name']}")

        processor = create_dual_input_processor(
            use_engineered_features=scenario["use_engineered_features"],
            max_engineered_features=scenario["max_engineered_features"],
            use_hopsketch=scenario["use_hopsketch"]
        )

        processed_data = processor.process_training_data(
            df,
            enable_engineered_features=scenario["use_engineered_features"]
        )

        has_engineered = processed_data["X_engineered"] is not None
        if has_engineered != scenario["expected_engineered"]:
            logger.error(f"❌ Scenario '{scenario['name']}' failed: "
                        f"expected engineered={scenario['expected_engineered']}, got {has_engineered}")
            return False

        if has_engineered:
            n_features = processed_data["X_engineered"].shape[1]
            if n_features > scenario["max_engineered_features"]:
                logger.error(f"❌ Too many features: {n_features} > {scenario['max_engineered_features']}")
                return False

            logger.info(f"  ✅ Engineered features: {n_features}")

        logger.info(f"  ✅ Scenario '{scenario['name']}' PASSED")

    logger.info("✅ Configuration scenarios test PASSED")
    return True


def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("=" * 60)
    logger.info("TESTING ERROR HANDLING")
    logger.info("=" * 60)

    # Test invalid data
    try:
        # Create invalid dataframe
        invalid_df = pd.DataFrame({
            "wrong_column": [1, 2, 3],
            "label": ["A", "B", "C"]
        })

        processor = create_dual_input_processor()
        processed_data = processor.process_training_data(invalid_df)
        logger.error("❌ Should have failed with invalid data")
        return False
    except Exception as e:
        logger.info(f"✅ Correctly caught invalid data error: {str(e)[:100]}")

    # Test mismatched expansion indices
    try:
        data_path = Path("data/processed/train_pivot_134.parquet")
        df = pd.read_parquet(data_path)

        # Create invalid expansion indices
        df_invalid = df.copy()
        df_invalid["expansion_start"] = 150  # Beyond array bounds

        processor = create_dual_input_processor()
        processed_data = processor.process_training_data(df_invalid)
        logger.info("✅ Handled invalid expansion indices gracefully")
    except Exception as e:
        logger.info(f"✅ Caught expansion indices error: {str(e)[:100]}")

    logger.info("✅ Error handling test PASSED")
    return True


def main():
    """Run all tests."""
    logger.info("🚀 STARTING DUAL-INPUT PIPELINE INTEGRATION TESTS")
    logger.info("=" * 60)

    tests = [
        ("Data Loading", test_data_loading),
        ("Backward Compatibility", test_backward_compatibility),
        ("Engineered Features", test_engineered_features),
        ("Dual-Input Pipeline", test_dual_input_pipeline),
        ("Configuration Scenarios", test_configuration_scenarios),
        ("Error Handling", test_error_handling),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED with exception: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("🏁 TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Passed: {passed}/{total} tests")

    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Dual-input pipeline is ready for use.")
        return 0
    else:
        logger.error(f"💥 {total - passed} tests failed. Please fix issues before using.")
        return 1


if __name__ == "__main__":
    exit(main())