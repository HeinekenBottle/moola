#!/usr/bin/env python3
"""
Moola SDK - ML Orchestrator & Financial Data Specialist Agent

An expert AI agent specialized in:
- Machine Learning ensemble orchestration
- OHLC/financial time series analysis
- Feature engineering and data optimization
- Model architecture recommendations
- Performance tuning and optimization

Based on the moola ML ensemble system patterns.
"""

import asyncio
import os
from pathlib import Path
from typing import Optional

from claude_agent_sdk import ClaudeSDKClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MoolaAgent:
    """ML Orchestrator & Financial Data Specialist Agent"""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Moola Agent

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not found. Set it in your environment or .env file.\n"
                "Get your API key from: https://console.anthropic.com/"
            )

        self.client = ClaudeSDKClient(api_key=self.api_key)

    async def analyze_ensemble_strategy(
        self,
        problem_description: str,
        data_characteristics: Optional[str] = None
    ) -> str:
        """Analyze ML problem and recommend ensemble strategy

        Args:
            problem_description: Description of the ML problem
            data_characteristics: Optional description of data (size, type, etc.)

        Returns:
            Detailed recommendation for ensemble approach
        """
        system_prompt = """You are an expert ML Ensemble Specialist with deep knowledge of:

ENSEMBLE METHODS & STACKING:
- Out-of-Fold (OOF) predictions for meta-learning
- Stacking meta-learners (LogisticRegression, XGBoost, etc.)
- Cross-validation strategies for ensemble training
- Multi-level ensemble architectures

FINANCIAL TIME SERIES EXPERTISE:
- OHLC data patterns and feature engineering
- Windowing strategies for time series (e.g., 105-timestep windows)
- Financial pattern classification (consolidation, retracement, reversal)
- Market regime detection and state modeling

DATA OPTIMIZATION:
- Class imbalance handling (SMOTE, Focal Loss, sample weights)
- Data augmentation (Mixup, CutMix) for small datasets
- Feature engineering for price action patterns
- Dimension reduction and feature selection

MODEL ARCHITECTURES:
- Gradient Boosting (XGBoost) with GPU optimization
- Deep Learning hybrids (CNN-Transformer models)
- Multi-task learning (classification + pointer prediction)
- Early stopping and model persistence

PERFORMANCE OPTIMIZATION:
- GPU acceleration strategies
- Memory-efficient training for small datasets
- Hyperparameter tuning approaches
- Model monitoring and drift detection

Based on the moola ensemble system experience, provide specific, actionable recommendations that balance model complexity with dataset size constraints. Focus on practical implementation details."""

        user_message = f"""I need expert guidance on an ML ensemble strategy for this problem:

PROBLEM DESCRIPTION:
{problem_description}

{'DATA CHARACTERISTICS: ' + data_characteristics if data_characteristics else ''}

Please provide:
1. Recommended ensemble architecture
2. Base model selection rationale
3. Meta-learner strategy
4. Data preprocessing recommendations
5. Class imbalance handling approach
6. Performance optimization tips
7. Potential challenges and solutions

Be specific and actionable, drawing from financial time series ensemble experience."""

        response = await self.client.query(
            message=user_message,
            system_prompt=system_prompt
        )
        return response

    async def recommend_feature_engineering(
        self,
        data_type: str,
        domain: str,
        constraints: Optional[str] = None
    ) -> str:
        """Recommend feature engineering strategies

        Args:
            data_type: Type of data (OHLC, multivariate, etc.)
            domain: Application domain (finance, etc.)
            constraints: Any constraints (small dataset, real-time, etc.)

        Returns:
            Feature engineering recommendations
        """
        system_prompt = """You are an expert Feature Engineering Specialist with deep knowledge of:

OHLC & FINANCIAL FEATURES:
- Price action patterns (support/resistance, trend indicators)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Volume-based features and volatility measures
- Window-based statistical features
- Multi-timeframe feature extraction

TIME SERIES ENGINEERING:
- Lag features and rolling statistics
- Seasonal decomposition and trend components
- Fourier transforms and spectral features
- Autocorrelation and momentum indicators
- Regime-based feature engineering

FEATURE OPTIMIZATION:
- Feature selection techniques (mutual information, recursive elimination)
- Dimension reduction (PCA, t-SNE, UMAP for visualization)
- Feature scaling and normalization strategies
- Handling missing data and outliers
- Feature importance analysis and interpretation

DOMAIN-SPECIALIZED FEATURES:
- Financial market microstructure features
- Order flow and liquidity measures
- Cross-asset correlation features
- Risk-adjusted performance metrics
- Market regime indicators

DATA CONSTRAINTS:
- Small dataset feature engineering (avoid overfitting)
- Real-time feature computation constraints
- Feature stability and robustness
- Computational efficiency for model training

Based on moola's price action feature engineering experience, provide concrete implementation guidance with code examples where helpful."""

        user_message = f"""I need feature engineering recommendations for:

DATA TYPE: {data_type}
DOMAIN: {domain}
{'CONSTRAINTS: ' + constraints if constraints else ''}

Please provide:
1. Core feature categories to create
2. Specific feature formulas/implementations
3. Feature selection strategy
4. Handling of data constraints
5. Validation approach
6. Computational considerations
7. Example code snippets for key features

Focus on practical implementation based on proven financial time series feature engineering patterns."""

        response = await self.client.query(
            message=user_message,
            system_prompt=system_prompt
        )
        return response

    async def optimize_model_architecture(
        self,
        model_type: str,
        data_shape: tuple,
        requirements: str
    ) -> str:
        """Provide model architecture optimization guidance

        Args:
            model_type: Type of model (CNN-Transformer, XGBoost, etc.)
            data_shape: Shape of input data (samples, timesteps, features)
            requirements: Performance and deployment requirements

        Returns:
            Architecture optimization recommendations
        """
        system_prompt = """You are an expert Model Architecture Specialist with deep knowledge of:

DEEP LEARNING ARCHITECTURES:
- CNN-Transformer hybrids for time series
- Multi-scale convolutional feature extraction
- Temporal attention mechanisms and positional encoding
- Multi-task learning architectures
- Causal padding for temporal consistency

GRADIENT BOOSTING OPTIMIZATION:
- XGBoost hyperparameter tuning strategies
- GPU acceleration and histogram-based training
- Regularization techniques (lambda, alpha, subsampling)
- Early stopping and cross-validation strategies
- Feature importance and interpretability

PERFORMANCE OPTIMIZATION:
- GPU memory management and mixed precision training
- Batch size optimization and data loading strategies
- Model compression and quantization
- Inference optimization for deployment
- Distributed training considerations

ARCHITECTURE PATTERNS:
- Residual connections and skip pathways
- Attention mechanisms and self-attention
- Batch normalization and layer normalization
- Dropout and regularization strategies
- Loss function design for imbalanced data

FINANCIAL MODELING SPECIFICS:
- Temporal causality and lookahead bias prevention
- Market regime adaptation and concept drift handling
- Uncertainty quantification and confidence intervals
- Risk-adjusted performance metrics
- Multi-horizon prediction strategies

Based on moola's CNN-Transformer and XGBoost optimization experience, provide specific architectural guidance with implementation details."""

        user_message = f"""I need model architecture optimization for:

MODEL TYPE: {model_type}
DATA SHAPE: {data_shape}
REQUIREMENTS: {requirements}

Please provide:
1. Recommended architecture changes
2. Hyperparameter optimization strategy
3. Training optimization techniques
4. Regularization and overfitting prevention
5. Performance monitoring and debugging
6. Deployment considerations
7. Specific implementation tips

Focus on practical optimizations proven to work with similar financial/time series models."""

        response = await self.client.query(
            message=user_message,
            system_prompt=system_prompt
        )
        return response

    async def debug_performance_issues(
        self,
        problem_description: str,
        metrics: dict,
        setup_details: str
    ) -> str:
        """Debug and troubleshoot model performance issues

        Args:
            problem_description: Description of the performance problem
            metrics: Current performance metrics
            setup_details: Details about model setup and training

        Returns:
            Debugging recommendations and solutions
        """
        system_prompt = """You are an expert ML Performance Debugger with extensive experience in:

PERFORMANCE ISSUE DIAGNOSIS:
- Overfitting vs underfitting identification
- Data leakage and lookahead bias detection
- Class imbalance and distribution shift problems
- Learning rate and optimization issues
- Architecture capacity problems

FINANCIAL MODELING CHALLENGES:
- Temporal validation and proper backtesting
- Market regime change and concept drift
- Stationarity issues in time series
- Signal-to-noise ratio problems
- Survivorship bias and data quality issues

TRAINING OPTIMIZATION:
- Learning rate scheduling and warmup strategies
- Batch size effects on convergence
- Regularization parameter tuning
- Early stopping patience and monitoring
- Loss function selection and weighting

DATA QUALITY ISSUES:
- Missing data and outlier handling
- Feature scaling and normalization problems
- Data preprocessing pipeline bugs
- Label noise and annotation errors
- Feature engineering mistakes

SYSTEMATIC DEBUGGING APPROACH:
- Performance baseline establishment
- Controlled experimentation methodology
- Statistical significance testing
- Error analysis and pattern identification
- Incremental improvement strategies

Based on moola's extensive ensemble debugging experience, provide systematic debugging guidance with specific action items."""

        user_message = f"""I need help debugging model performance issues:

PROBLEM: {problem_description}

CURRENT METRICS:
{metrics}

SETUP DETAILS:
{setup_details}

Please provide:
1. Root cause analysis framework
2. Specific debugging steps to take
3. Data quality checks to perform
4. Model architecture adjustments
5. Training process improvements
6. Validation methodology fixes
7. Performance improvement roadmap

Focus on systematic debugging based on common financial ML ensemble pitfalls and proven solutions."""

        response = await self.client.query(
            message=user_message,
            system_prompt=system_prompt
        )
        return response


async def main():
    """Example usage of the Moola Agent"""

    # Initialize agent
    agent = MoolaAgent()

    print("🤖 Moola SDK - ML Orchestrator & Financial Data Specialist")
    print("=" * 60)
    print()

    # Example 1: Ensemble strategy analysis
    print("📊 Example 1: Ensemble Strategy Analysis")
    print("-" * 40)

    ensemble_response = await agent.analyze_ensemble_strategy(
        problem_description="Financial time series classification of market patterns (consolidation vs retracement) using OHLC data",
        data_characteristics="Small dataset: ~115 samples, 105 timesteps, 4 OHLC features per timestep. Highly imbalanced classes."
    )
    print(ensemble_response)
    print()

    # Example 2: Feature engineering recommendations
    print("🔧 Example 2: Feature Engineering for OHLC Data")
    print("-" * 45)

    feature_response = await agent.recommend_feature_engineering(
        data_type="OHLC time series (105 timesteps, 4 features)",
        domain="Financial pattern classification",
        constraints="Small dataset (~115 samples), need to avoid overfitting, focus on robust features"
    )
    print(feature_response)
    print()

    # Example 3: Model architecture optimization
    print("🏗️ Example 3: CNN-Transformer Architecture Optimization")
    print("-" * 52)

    architecture_response = await agent.optimize_model_architecture(
        model_type="CNN-Transformer hybrid for time series classification",
        data_shape=(115, 105, 4),
        requirements="Optimize for small dataset, include early stopping, handle class imbalance, GPU acceleration"
    )
    print(architecture_response)
    print()


if __name__ == "__main__":
    asyncio.run(main())