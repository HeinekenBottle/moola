# Moola SDK - ML Orchestrator & Financial Data Specialist Agent

An expert AI agent specialized in machine learning ensemble orchestration, financial time series analysis, and data optimization. Built with the Claude Agent SDK and inspired by the moola ML ensemble system.

## 🎯 Purpose

The Moola Agent is designed to be your ML/Finance expert, providing guidance on:

- **Ensemble Methods**: Stacking, meta-learning, OOF predictions, cross-validation strategies
- **OHLC/Financial Data**: Time series feature engineering, pattern recognition, market regime detection
- **Model Architectures**: CNN-Transformer hybrids, XGBoost optimization, multi-task learning
- **Data Optimization**: Class imbalance handling, augmentation, feature selection
- **Performance Tuning**: GPU acceleration, early stopping, hyperparameter optimization

## 🚀 Quick Start

### 1. Installation

```bash
# Clone and setup
git clone <repository> moola-sdk
cd moola-sdk

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
ANTHROPIC_API_KEY=your_api_key_here
```

Get your API key from: https://console.anthropic.com/

### 3. Basic Usage

```python
import asyncio
from main import MoolaAgent

async def main():
    # Initialize agent
    agent = MoolaAgent()

    # Get ensemble strategy recommendations
    response = await agent.analyze_ensemble_strategy(
        problem_description="Financial time series classification using OHLC data",
        data_characteristics="Small dataset: ~115 samples, 105 timesteps, 4 features"
    )

    print(response)

asyncio.run(main())
```

## 📖 Examples

The `examples/` directory contains comprehensive use cases:

```bash
# Run quickstart examples
python examples/quickstart.py
```

### Example Scenarios

1. **Ensemble Strategy Analysis**: Get recommendations for stacking ensembles
2. **Feature Engineering**: OHLC feature engineering for financial data
3. **Architecture Optimization**: CNN-Transformer model tuning
4. **Performance Debugging**: Diagnose overfitting and underfitting
5. **Moola-Specific Issues**: Real challenges from the moola ensemble system

## 🏗️ Architecture

The agent is built on the **Claude Agent SDK (v0.1.3)** and includes expertise in:

### Core Competencies

1. **ML Ensemble Specialist**
   - Stacking meta-learners and OOF predictions
   - Cross-validation strategies for small datasets
   - Multi-level ensemble architectures
   - Model selection and combination strategies

2. **Financial Data Expert**
   - OHLC feature engineering and price action patterns
   - Time series windowing (105-timestep windows)
   - Technical indicators and market regime detection
   - Financial pattern classification

3. **Model Architecture Advisor**
   - CNN-Transformer hybrids for time series
   - XGBoost optimization and GPU acceleration
   - Multi-task learning (classification + pointer prediction)
   - Regularization and overfitting prevention

4. **Data Optimization Specialist**
   - Class imbalance handling (SMOTE, Focal Loss)
   - Data augmentation (Mixup, CutMix)
   - Feature selection and dimension reduction
   - Small dataset optimization strategies

### Inspired by Moola Patterns

This agent is based on the real-world experience from the moola ensemble system:

- **2-class vs 3-class classification** decisions
- **Reversal pattern extraction** (19 samples archived)
- **Stacking meta-learner** implementation
- **CNN-Transformer hybrid** architectures
- **XGBoost with SMOTE** for class imbalance
- **Performance optimization** for RTX 4090
- **Early stopping** and model persistence

## 🛠️ API Reference

### MoolaAgent Class

```python
class MoolaAgent:
    def __init__(self, api_key: Optional[str] = None)

    async def analyze_ensemble_strategy(
        self,
        problem_description: str,
        data_characteristics: Optional[str] = None
    ) -> str

    async def recommend_feature_engineering(
        self,
        data_type: str,
        domain: str,
        constraints: Optional[str] = None
    ) -> str

    async def optimize_model_architecture(
        self,
        model_type: str,
        data_shape: tuple,
        requirements: str
    ) -> str

    async def debug_performance_issues(
        self,
        problem_description: str,
        metrics: dict,
        setup_details: str
    ) -> str
```

## 📊 Use Cases

### 1. Trading Signal Generation

```python
response = await agent.analyze_ensemble_strategy(
    problem_description="Cryptocurrency trading signal classifier",
    data_characteristics="500 samples, 24-hour OHLCV windows, 3 trend classes"
)
```

### 2. Model Architecture Design

```python
response = await agent.optimize_model_architecture(
    model_type="CNN-Transformer for time series",
    data_shape=(500, 24, 5),
    requirements="GPU optimization, class imbalance handling"
)
```

### 3. Performance Debugging

```python
response = await agent.debug_performance_issues(
    problem_description="Overfitting: 95% train vs 65% validation accuracy",
    metrics={"train_acc": 0.95, "val_acc": 0.65},
    setup_details="Stacking ensemble with 3 base models"
)
```

## 🔧 Development

### Dependencies

- `claude-agent-sdk==0.1.3` - Core SDK
- `torch>=2.1.0` - Deep learning support
- `xgboost>=2.0.0` - Gradient boosting
- `numpy>=1.24.0`, `pandas>=2.0.0` - Data processing
- `scikit-learn>=1.3.0` - ML utilities
- `yfinance>=0.2.0` - Financial data
- `imbalanced-learn>=0.11.0` - SMOTE support

### Project Structure

```
moola-sdk/
├── main.py              # Main agent implementation
├── requirements.txt     # Dependencies
├── .env.example        # Environment template
├── README.md           # This file
├── examples/
│   └── quickstart.py   # Usage examples
├── config/             # Configuration files
└── tests/              # Test suite
```

## 🤝 Contributing

This project is inspired by the moola ML ensemble system. When contributing:

1. Follow the existing code style
2. Add comprehensive examples
3. Include documentation for new features
4. Test with real financial ML scenarios

## 📄 License

MIT License - see LICENSE file for details

## 🔗 Related Resources

- **Moola Project**: The original ensemble system that inspired this agent
- **Claude Agent SDK**: https://docs.claude.com/en/api/agent-sdk
- **Financial Pattern Recognition**: Based on real trading system experience

## 💡 Tips

- Start with the `examples/quickstart.py` to understand capabilities
- Provide detailed problem descriptions for better recommendations
- Include data characteristics and constraints when possible
- Use the debugging helper when facing performance issues
- Draw inspiration from the moola ensemble patterns and implementations

---

Built with ❤️ using the Claude Agent SDK, inspired by real-world ML ensemble challenges in financial markets.