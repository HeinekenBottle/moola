#!/usr/bin/env python3
"""Test transfer learning performance with current models.

Compare baseline vs. pretrained encoder performance to understand
the impact of architecture mismatch and transfer learning.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.models.enhanced_simple_lstm import EnhancedSimpleLSTMModel


def load_174_sample_data():
    """Load the 174-sample dataset used in previous experiments."""
    try:
        # Try to load from processed data
        data_path = Path("data/processed/train_clean.parquet")
        if data_path.exists():
            import pandas as pd
            df = pd.read_parquet(data_path)
            
            # Extract OHLC data (assuming it's stored in nested format)
            if 'ohlc' in df.columns:
                X = np.stack(df['ohlc'].values)
            else:
                # Try to extract from flat columns
                ohlc_cols = [col for col in df.columns if col in ['open', 'high', 'low', 'close']]
                if len(ohlc_cols) == 4:
                    # Reshape from flat to sequence format
                    X = df[ohlc_cols].values.reshape(-1, 105, 4)
                else:
                    raise ValueError("Could not find OHLC data")
            
            y = df['label'].values if 'label' in df.columns else df['target'].values
            logger.info(f"Loaded {len(X)} samples from processed data")
            return X, y
            
    except Exception as e:
        logger.warning(f"Could not load processed data: {e}")
    
    # Create synthetic data that matches the 174-sample experiment
    logger.info("Creating synthetic 174-sample data for testing...")
    np.random.seed(1337)
    X = np.random.randn(174, 105, 4).astype(np.float32)
    
    # Create imbalanced binary labels (similar to real data)
    y = np.random.choice([0, 1], size=174, p=[0.7, 0.3])
    
    return X, y


def test_baseline_model(X, y):
    """Train baseline Enhanced SimpleLSTM without pretrained encoder."""
    logger.info("=== Testing Baseline Enhanced SimpleLSTM ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1337, stratify=y
    )
    
    # Create and train model
    model = EnhancedSimpleLSTMModel(
        seed=1337,
        hidden_size=128,
        num_layers=2,  # 2 layers to match architecture
        device="cpu",  # Use CPU for testing
        n_epochs=10,   # Quick training for testing
        batch_size=32,
    )
    
    logger.info(f"Training baseline model on {len(X_train)} samples...")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Baseline accuracy: {accuracy:.4f}")
    logger.info(f"Baseline classification report:\n{classification_report(y_test, y_pred)}")
    
    return accuracy, model


def test_pretrained_model(X, y, encoder_path):
    """Train Enhanced SimpleLSTM with pretrained encoder."""
    logger.info("=== Testing Pretrained Enhanced SimpleLSTM ===")
    
    if not encoder_path.exists():
        logger.error(f"Encoder not found at {encoder_path}")
        return None, None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1337, stratify=y
    )
    
    # Create and train model
    model = EnhancedSimpleLSTMModel(
        seed=1337,
        hidden_size=128,
        num_layers=2,  # 2 layers to match architecture
        device="cpu",  # Use CPU for testing
        n_epochs=10,   # Quick training for testing
        batch_size=32,
    )
    
    logger.info(f"Training pretrained model on {len(X_train)} samples...")
    
    # First build the model
    model.fit(X_train, y_train)
    
    # Then load pretrained encoder
    try:
        model.load_pretrained_encoder(
            encoder_path=encoder_path,
            freeze_encoder=True
        )
        logger.info(f"Loaded pretrained encoder: {model.pretrained_stats}")
        
        # Continue training with pretrained weights
        model.fit(X_train, y_train)
        
    except Exception as e:
        logger.error(f"Failed to load pretrained encoder: {e}")
        return None, None
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    logger.info(f"Pretrained accuracy: {accuracy:.4f}")
    logger.info(f"Pretrained classification report:\n{classification_report(y_test, y_pred)}")
    
    return accuracy, model


def analyze_encoder_compatibility(encoder_path):
    """Analyze encoder architecture compatibility."""
    logger.info("=== Analyzing Encoder Compatibility ===")
    
    if not encoder_path.exists():
        logger.error(f"Encoder not found at {encoder_path}")
        return
    
    try:
        encoder_data = torch.load(encoder_path, map_location='cpu')
        state_dict = encoder_data.get('encoder_state_dict', {})
        hyperparams = encoder_data.get('hyperparams', {})
        
        logger.info(f"Encoder hyperparams: {hyperparams}")
        logger.info(f"Encoder tensors: {len(state_dict)}")
        
        # Expected tensors for 2-layer bidirectional LSTM
        expected_layers = hyperparams.get('num_layers', 1)
        expected_tensors = expected_layers * 4 * 2  # 4 tensors per layer * 2 directions
        
        logger.info(f"Expected tensors ({expected_layers} layers): {expected_tensors}")
        logger.info(f"Actual tensors: {len(state_dict)}")
        
        # List tensor names
        for i, (name, tensor) in enumerate(state_dict.items()):
            logger.info(f"  {i+1:2d}. {name}: {tensor.shape}")
        
        # Compatibility analysis
        match_ratio = len(state_dict) / expected_tensors if expected_tensors > 0 else 0
        logger.info(f"Match ratio: {match_ratio:.1%}")
        
        if match_ratio >= 0.8:
            logger.success("✓ Good architecture compatibility")
        elif match_ratio >= 0.5:
            logger.warning("⚠ Partial architecture compatibility")
        else:
            logger.error("✗ Poor architecture compatibility")
            
    except Exception as e:
        logger.error(f"Failed to analyze encoder: {e}")


def main():
    """Main test function."""
    logger.info("=== Transfer Learning Performance Analysis ===")
    
    # Load data
    X, y = load_174_sample_data()
    logger.info(f"Dataset: {X.shape}, Class distribution: {np.bincount(y)}")
    
    # Test baseline
    baseline_acc, baseline_model = test_baseline_model(X, y)
    
    # Test with new 2-layer pretrained encoder
    encoder_path = Path("data/artifacts/pretrained/bilstm_encoder_2layer.pt")
    analyze_encoder_compatibility(encoder_path)
    
    pretrained_acc, pretrained_model = test_pretrained_model(X, y, encoder_path)
    
    # Compare results
    logger.info("=== Performance Comparison ===")
    logger.info(f"Baseline accuracy:    {baseline_acc:.4f}")
    if pretrained_acc is not None:
        logger.info(f"Pretrained accuracy:  {pretrained_acc:.4f}")
        improvement = pretrained_acc - baseline_acc
        logger.info(f"Improvement:          {improvement:+.4f} ({improvement/baseline_acc:+.1%})")
        
        if improvement > 0.01:
            logger.success("✓ Pretrained encoder improves performance")
        elif improvement > -0.01:
            logger.info("≈ Pretrained encoder has neutral impact")
        else:
            logger.warning("✗ Pretrained encoder hurts performance")
    else:
        logger.error("Could not test pretrained model")
    
    logger.info("=== Analysis Complete ===")


if __name__ == "__main__":
    main()