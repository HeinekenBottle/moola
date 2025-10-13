#!/usr/bin/env python3
"""
Quickstart examples for the Moola SDK Agent

Demonstrates common use cases based on moola ensemble patterns.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path to import main module
sys.path.append(str(Path(__file__).parent.parent))

from main import MoolaAgent


async def ensemble_strategy_example():
    """Example: Get ensemble strategy recommendations"""

    agent = MoolaAgent()

    response = await agent.analyze_ensemble_strategy(
        problem_description="""
        I'm building a trading signal classifier for cryptocurrency markets.
        I want to classify price movements into 3 categories:
        - Consolidation (sideways movement)
        - Uptrend (bullish movement)
        - Downtrend (bearish movement)

        My data consists of 1-hour OHLCV candles for Bitcoin.
        """,
        data_characteristics="""
        Dataset size: ~500 samples
        Timeframe: 1-hour candles
        Features: OHLCV (Open, High, Low, Close, Volume)
        Window size: 24 timesteps (24 hours)
        Target: Predict next 4-hour trend direction
        Classes are somewhat imbalanced (more consolidation than trends)
        """
    )

    print("📈 ENSEMBLE STRATEGY RECOMMENDATIONS")
    print("=" * 50)
    print(response)
    print()


async def feature_engineering_example():
    """Example: Get feature engineering advice"""

    agent = MoolaAgent()

    response = await agent.recommend_feature_engineering(
        data_type="OHLCV time series with 24-timestep windows",
        domain="Cryptocurrency trading signal generation",
        constraints="Limited data (500 samples), need robust features that generalize well, avoid overfitting"
    )

    print("🔧 FEATURE ENGINEERING RECOMMENDATIONS")
    print("=" * 50)
    print(response)
    print()


async def architecture_optimization_example():
    """Example: Get architecture optimization advice"""

    agent = MoolaAgent()

    response = await agent.optimize_model_architecture(
        model_type="CNN-Transformer hybrid for time series classification",
        data_shape=(500, 24, 5),  # 500 samples, 24 timesteps, 5 OHLCV features
        requirements="""
        - Handle class imbalance through Focal Loss or weighted loss
        - Include early stopping to prevent overfitting on small dataset
        - Optimize for GPU training (RTX 4090)
        - Provide uncertainty estimates for predictions
        - Support multi-task learning (trend direction + strength)
        """
    )

    print("🏗️ ARCHITECTURE OPTIMIZATION RECOMMENDATIONS")
    print("=" * 55)
    print(response)
    print()


async def performance_debugging_example():
    """Example: Debug performance issues"""

    agent = MoolaAgent()

    response = await agent.debug_performance_issues(
        problem_description="""
        My ensemble model is achieving 95% training accuracy but only 65% validation accuracy.
        The model appears to be overfitting heavily on the training data.

        I'm using a 3-level stacking ensemble:
        Level 1: XGBoost, Random Forest, Neural Network
        Level 2: Logistic Regression meta-learner

        Training loss decreases smoothly but validation loss starts increasing after epoch 20.
        """,
        metrics={
            "train_accuracy": 0.95,
            "val_accuracy": 0.65,
            "train_loss": 0.12,
            "val_loss": 0.89,
            "f1_score_macro": 0.62,
            "class_imbalance": {
                "consolidation": 0.45,
                "uptrend": 0.30,
                "downtrend": 0.25
            }
        },
        setup_details="""
        Dataset: 500 samples, 24-timestep OHLCV windows
        Split: 80% train, 20% validation (time-based split)
        Features: 15 engineered technical indicators
        Model: Stacking ensemble with 3 base models
        Training: 100 epochs, early stopping patience=15
        Regularization: L2 regularization, dropout=0.3
        """
    )

    print("🐛 PERFORMANCE DEBUGGING RECOMMENDATIONS")
    print("=" * 50)
    print(response)
    print()


async def moola_specific_example():
    """Example: Specific to moola's current ensemble challenges"""

    agent = MoolaAgent()

    response = await agent.debug_performance_issues(
        problem_description="""
        Working on financial pattern classification system with major challenges:

        1. Severe class imbalance: Only 19 reversal samples vs 115+ for other classes
        2. F1 score dropped 10.6% while accuracy increased 2.3% after recent changes
        3. Deep learning models (CNN-Transformer, RWKV) struggling with small dataset
        4. Reversal class showing 0% recall after SMOTE oversampling
        5. Need to migrate from 3-class to 2-class classification (remove reversals)
        """,
        metrics={
            "current_3class": {
                "accuracy": 0.68,
                "f1_macro": 0.45,
                "reversal_recall": 0.00,
                "reversal_precision": 0.00,
                "consolidation_recall": 0.85,
                "retracement_recall": 0.62
            },
            "proposed_2class": {
                "consolidation_samples": 65,
                "retracement_samples": 50,
                "total_samples": 115
            }
        },
        setup_details="""
        Base models: XGBoost, Random Forest, CNN-Transformer, RWKV, Logistic Regression
        Ensemble: Stacking meta-learner
        Data: OHLC time series, 105-timestep windows
        Features: Price action features, technical indicators
        Challenge: Reversal class has insufficient samples for reliable training
        Goal: Robust 2-class classification (consolidation vs retracement)
        """
    )

    print("💰 MOOLA-SPECIFIC OPTIMIZATION RECOMMENDATIONS")
    print("=" * 55)
    print(response)
    print()


async def main():
    """Run all quickstart examples"""

    print("🚀 Moola SDK Quickstart Examples")
    print("Based on moola ensemble system patterns")
    print("=" * 50)
    print()

    # Note: These examples require a valid ANTHROPIC_API_KEY
    print("⚠️  Make sure to set your ANTHROPIC_API_KEY in .env file")
    print("Get your key from: https://console.anthropic.com/")
    print()

    try:
        # Run examples
        await ensemble_strategy_example()
        await feature_engineering_example()
        await architecture_optimization_example()
        await performance_debugging_example()
        await moola_specific_example()

    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print()
        print("💡 Make sure you have:")
        print("1. Set ANTHROPIC_API_KEY in your .env file")
        print("2. Activated the virtual environment: source venv/bin/activate")
        print("3. Installed dependencies: pip install -r requirements.txt")


if __name__ == "__main__":
    asyncio.run(main())