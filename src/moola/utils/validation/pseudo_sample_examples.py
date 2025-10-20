"""Example usage and best practices for pseudo-sample generation.

This module provides comprehensive examples demonstrating how to use the
pseudo-sample generation framework effectively for financial time series data.

Examples include:
1. Basic usage with small datasets
2. Integration with existing training pipeline
3. Quality validation and monitoring
4. Performance optimization techniques
5. Self-supervised pseudo-labeling workflows
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

from .pseudo_sample_generation import (
    PseudoSampleGenerationPipeline,
    TemporalAugmentationGenerator,
    PatternBasedSynthesisGenerator,
    StatisticalSimulationGenerator,
    MarketConditionSimulationGenerator,
    SelfSupervisedPseudoLabelingGenerator
)
from .pseudo_sample_validation import FinancialDataValidator, ValidationReport
from .training_pipeline_integration import (
    TrainingPipelineIntegrator,
    AugmentationConfig,
    AugmentedDataset,
    DynamicAugmentedDataset
)


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_financial_data(n_samples: int = 105,
                               seq_length: int = 105,
                               n_features: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Create sample financial data for demonstration.

    Args:
        n_samples: Number of samples to generate
        seq_length: Length of each time series
        n_features: Number of features (OHLC = 4)

    Returns:
        Tuple of (data, labels)
    """
    np.random.seed(1337)

    data = []
    labels = []

    for i in range(n_samples):
        # Generate realistic price path
        initial_price = 100.0 + np.random.normal(0, 10)

        # Generate returns with realistic properties
        returns = np.random.normal(0, 0.02, seq_length)

        # Add some autocorrelation
        for j in range(1, seq_length):
            returns[j] += 0.1 * returns[j-1]

        # Generate prices from returns
        prices = [initial_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        prices = np.array(prices[1:])  # Remove initial price

        # Generate OHLC from prices
        ohlc = np.zeros((seq_length, 4))
        for j in range(seq_length):
            close = prices[j]

            # Realistic intraday variation
            daily_vol = 0.005 * close

            if j == 0:
                open_price = close
            else:
                gap = np.random.normal(0, daily_vol * 0.5)
                open_price = max(1e-6, prices[j-1] + gap)

            # High and low
            if close >= open_price:
                high = close + np.random.uniform(0, daily_vol)
                low = open_price - np.random.uniform(0, daily_vol * 0.5)
            else:
                high = open_price + np.random.uniform(0, daily_vol * 0.5)
                low = close - np.random.uniform(0, daily_vol)

            # Ensure OHLC relationships
            high = max(high, open_price, close)
            low = min(low, open_price, close)

            ohlc[j] = [open_price, high, low, close]

        data.append(ohlc)

        # Generate binary labels (consolidation vs retracement)
        # Consolidation: low volatility, no strong trend
        # Retracement: higher volatility, clearer trend
        volatility = np.std(returns)
        trend = abs(prices[-1] - prices[0]) / prices[0]

        if volatility < 0.01 and trend < 0.05:
            labels.append('consolidation')
        else:
            labels.append('retracement')

    return np.array(data), np.array(labels)


def example_1_basic_generation():
    """Example 1: Basic pseudo-sample generation."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Pseudo-Sample Generation")
    print("=" * 60)

    # Create sample data
    original_data, original_labels = create_sample_financial_data(n_samples=105)
    print(f"Original data shape: {original_data.shape}")
    print(f"Original labels distribution: {np.unique(original_labels, return_counts=True)}")

    # Initialize generation pipeline
    pipeline = PseudoSampleGenerationPipeline(seed=1337)

    # Generate pseudo-samples
    n_pseudo_samples = 200
    generated_data, generated_labels, metadata = pipeline.generate_samples(
        original_data, original_labels, n_pseudo_samples
    )

    print(f"Generated data shape: {generated_data.shape}")
    print(f"Generated labels distribution: {np.unique(generated_labels, return_counts=True)}")
    print(f"Generation metadata: {metadata}")

    # Validate quality
    validator = FinancialDataValidator()
    report = validator.validate_pseudo_samples(original_data, generated_data)

    print(f"\nValidation Results:")
    print(f"Overall Quality Score: {report.overall_quality_score:.3f}")
    print(f"OHLC Integrity: {report.ohlc_integrity:.3f}")

    # Generate human-readable report
    report_str = validator.generate_validation_report(report)
    print("\n" + report_str)

    return original_data, original_labels, generated_data, generated_labels


def example_2_individual_strategies():
    """Example 2: Testing individual generation strategies."""
    print("=" * 60)
    print("EXAMPLE 2: Individual Generation Strategies")
    print("=" * 60)

    # Create sample data
    original_data, original_labels = create_sample_financial_data(n_samples=50)

    # Test each strategy individually
    strategies = {
        'Temporal Augmentation': TemporalAugmentationGenerator(seed=1337),
        'Pattern Synthesis': PatternBasedSynthesisGenerator(seed=1337),
        'Statistical Simulation': StatisticalSimulationGenerator(seed=1337),
        'Market Condition': MarketConditionSimulationGenerator(seed=1337)
    }

    validator = FinancialDataValidator()

    for name, generator in strategies.items():
        print(f"\n--- Testing {name} ---")

        # Generate samples
        gen_data, gen_labels = generator.generate(original_data, original_labels, 50)

        # Validate
        report = validator.validate_pseudo_samples(original_data, gen_data)

        print(f"Generated samples: {len(gen_data)}")
        print(f"Quality score: {report.overall_quality_score:.3f}")
        print(f"OHLC integrity: {report.ohlc_integrity:.3f}")

        # Key metrics
        key_metrics = ['ks_similarity', 'volatility_clustering_similarity', 'autocorr_overall_similarity']
        for metric in key_metrics:
            if metric in report.statistical_similarity or metric in report.temporal_consistency:
                value = (report.statistical_similarity.get(metric) or
                        report.temporal_consistency.get(metric) or 0)
                print(f"  {metric}: {value:.3f}")


def example_3_quality_validation():
    """Example 3: Comprehensive quality validation."""
    print("=" * 60)
    print("EXAMPLE 3: Quality Validation and Visualization")
    print("=" * 60)

    # Create sample data
    original_data, original_labels = create_sample_financial_data(n_samples=100)

    # Generate samples with different quality levels
    pipeline_good = PseudoSampleGenerationPipeline(
        seed=1337, validation_threshold=0.8
    )
    pipeline_poor = PseudoSampleGenerationPipeline(
        seed=999, validation_threshold=0.3
    )

    # Generate good quality samples
    good_data, good_labels, _ = pipeline_good.generate_samples(
        original_data, original_labels, 100
    )

    # Generate poor quality samples (for comparison)
    # Create intentionally distorted samples
    poor_data = original_data[:50].copy()
    poor_data = poor_data + np.random.normal(0, 0.1, poor_data.shape)  # Add noise
    poor_labels = original_labels[:50].copy()

    validator = FinancialDataValidator()

    # Validate both
    print("Good Quality Samples:")
    good_report = validator.validate_pseudo_samples(original_data, good_data)
    print(f"Overall quality: {good_report.overall_quality_score:.3f}")
    print(f"OHLC integrity: {good_report.ohlc_integrity:.3f}")

    print("\nPoor Quality Samples:")
    poor_report = validator.validate_pseudo_samples(original_data, poor_data)
    print(f"Overall quality: {poor_report.overall_quality_score:.3f}")
    print(f"OHLC integrity: {poor_report.ohlc_integrity:.3f}")

    # Generate visualizations
    try:
        from .pseudo_sample_validation import QualityMetricsVisualizer
        visualizer = QualityMetricsVisualizer()

        # Plot validation results
        visualizer.plot_validation_results(good_report, save_path="good_quality_validation.png")
        visualizer.plot_validation_results(poor_report, save_path="poor_quality_validation.png")

        # Plot distribution comparisons
        visualizer.plot_distribution_comparison(
            original_data, good_data, save_path="good_distribution_comparison.png"
        )
        visualizer.plot_distribution_comparison(
            original_data, poor_data, save_path="poor_distribution_comparison.png"
        )

        print("\nVisualization plots saved to current directory")

    except Exception as e:
        print(f"Visualization failed: {e}")


def example_4_training_integration():
    """Example 4: Integration with training pipeline."""
    print("=" * 60)
    print("EXAMPLE 4: Training Pipeline Integration")
    print("=" * 60)

    # Create sample data
    original_data, original_labels = create_sample_financial_data(n_samples=105)

    # Convert labels to numerical format
    label_map = {'consolidation': 0, 'retracement': 1}
    numerical_labels = np.array([label_map[label] for label in original_labels])

    # Configure augmentation
    config = AugmentationConfig(
        enable_augmentation=True,
        augmentation_ratio=2.0,
        quality_threshold=0.7,
        strategy_weights={
            'temporal_augmentation': 0.3,
            'pattern_synthesis': 0.3,
            'statistical_simulation': 0.2,
            'market_condition': 0.2
        }
    )

    # Initialize integrator
    integrator = TrainingPipelineIntegrator(config, logger=logger)

    # Prepare augmented dataloader
    dataloader = integrator.prepare_augmented_dataloader(
        original_data, numerical_labels,
        batch_size=16, shuffle=True, dynamic_generation=True
    )

    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    # Simulate training loop
    print("\nSimulating training loop...")
    for epoch in range(2):
        print(f"\nEpoch {epoch + 1}")

        for batch_idx, (data, labels) in enumerate(dataloader):
            if batch_idx >= 5:  # Only process first 5 batches for demo
                break

            print(f"  Batch {batch_idx + 1}: data shape {data.shape}, labels shape {labels.shape}")

            # Simulate some training work
            # In real training, you would:
            # 1. Forward pass through model
            # 2. Calculate loss
            # 3. Backward pass and optimize

        # Get statistics for dynamic dataset
        if hasattr(dataloader.dataset, 'get_statistics'):
            stats = dataloader.dataset.get_statistics()
            print(f"  Generation stats: {stats}")

    # Get training report
    report = integrator.get_training_report()
    print(f"\nTraining Report:")
    print(f"Total samples generated: {report['training_state']['total_samples_generated']}")
    print(f"Memory usage: {report['memory_usage_gb']:.2f} GB")


def example_5_self_supervised_learning():
    """Example 5: Self-supervised pseudo-labeling."""
    print("=" * 60)
    print("EXAMPLE 5: Self-Supervised Pseudo-Labeling")
    print("=" * 60)

    # Create sample data
    original_data, original_labels = create_sample_financial_data(n_samples=100)

    # Create a simple encoder model for demonstration
    class SimpleEncoder(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
            self.classifier = nn.Linear(output_size, 2)  # Binary classification

        def forward(self, x):
            # x shape: [batch, seq_len, features]
            batch_size, seq_len, features = x.shape

            # Flatten sequence for simple encoder
            x_flat = x.view(batch_size, -1)
            encoded = self.encoder(x_flat)
            logits = self.classifier(encoded)

            return logits

        def predict_proba(self, x):
            """Return class probabilities."""
            with torch.no_grad():
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=-1)
                return probs.cpu().numpy()

    # Initialize model
    model = SimpleEncoder(input_size=105*4, hidden_size=64, output_size=32)
    model.eval()

    # Initialize pipeline with self-supervised learning
    pipeline = PseudoSampleGenerationPipeline(seed=1337)
    pipeline.set_encoder_model(model)
    pipeline.enable_self_supervised(confidence_threshold=0.9)

    print("Self-supervised pseudo-labeling enabled")
    print("Note: In practice, you would first train the encoder on the original data")

    # Generate pseudo-labeled samples
    pseudo_data, pseudo_labels, metadata = pipeline.generate_samples(
        original_data, original_labels, 50
    )

    print(f"Generated {len(pseudo_data)} pseudo-labeled samples")
    print(f"Labels distribution: {np.unique(pseudo_labels, return_counts=True)}")

    # Validate quality
    validator = FinancialDataValidator()
    report = validator.validate_pseudo_samples(original_data, pseudo_data)

    print(f"Quality score: {report.overall_quality_score:.3f}")
    print(f"Self-supervised metrics: {report.statistical_similarity.get('avg_confidence_score', 'N/A')}")


def example_6_performance_optimization():
    """Example 6: Performance optimization and memory management."""
    print("=" * 60)
    print("EXAMPLE 6: Performance Optimization")
    print("=" * 60)

    # Create larger dataset for testing
    original_data, original_labels = create_sample_financial_data(n_samples=200)

    # Test different generation strategies for performance
    strategies = ['temporal_augmentation', 'pattern_synthesis', 'statistical_simulation']

    config = AugmentationConfig(
        max_memory_usage_gb=2.0,  # Conservative memory limit
        validation_threshold=0.7
    )

    integrator = TrainingPipelineIntegrator(config, logger=logger)

    print("Performance comparison:")
    print("-" * 40)

    for strategy in strategies:
        print(f"\nStrategy: {strategy}")

        # Configure pipeline for single strategy
        strategy_weights = {s: 0.0 for s in strategies}
        strategy_weights[strategy] = 1.0

        integrator.generator.strategy_weights = strategy_weights

        # Measure performance
        import time
        start_time = time.time()

        generated_data, generated_labels, metadata = integrator.generate_with_constraints(
            original_data, original_labels, 100
        )

        end_time = time.time()

        print(f"  Generation time: {end_time - start_time:.2f}s")
        print(f"  Samples generated: {len(generated_data)}")
        print(f"  Samples/second: {metadata.get('samples_per_second', 0):.1f}")
        print(f"  Memory usage: {metadata.get('memory_usage_gb', 0):.2f} GB")

        # Quality check
        validator = FinancialDataValidator()
        report = validator.validate_pseudo_samples(original_data, generated_data)
        print(f"  Quality score: {report.overall_quality_score:.3f}")

    print("\nMemory optimization tips:")
    print("1. Use dynamic generation for large datasets")
    print("2. Adjust batch sizes based on available memory")
    print("3. Set appropriate quality thresholds to filter poor samples")
    print("4. Monitor memory usage during training")


def run_all_examples():
    """Run all examples sequentially."""
    print("RUNNING ALL PSEUDO-SAMPLE GENERATION EXAMPLES")
    print("=" * 80)

    examples = [
        example_1_basic_generation,
        example_2_individual_strategies,
        example_3_quality_validation,
        example_4_training_integration,
        example_5_self_supervised_learning,
        example_6_performance_optimization
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            print(f"\n{'='*20} Example {i} {'='*20}")
            example_func()
            print(f"\nExample {i} completed successfully!")
        except Exception as e:
            print(f"\nExample {i} failed with error: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*80)


if __name__ == "__main__":
    print("Financial Pseudo-Sample Generation Examples")
    print("This module demonstrates various usage patterns and best practices")
    print("for generating high-quality financial time series pseudo-samples.")
    print("\nAvailable examples:")
    print("1. Basic generation")
    print("2. Individual strategy testing")
    print("3. Quality validation")
    print("4. Training pipeline integration")
    print("5. Self-supervised learning")
    print("6. Performance optimization")
    print("\nRun individual examples or use run_all_examples() to execute all.")

    # Uncomment to run a specific example:
    # example_1_basic_generation()

    # Uncomment to run all examples:
    # run_all_examples()