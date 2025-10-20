#!/usr/bin/env python3
"""Test transfer learning performance with real training data."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.models.enhanced_simple_lstm import EnhancedSimpleLSTMModel


def load_real_data():
    """Load real training data from processed datasets."""
    logger.info("Loading real training data...")
    
    # Load main training data
    train_path = Path("data/processed/train_clean.parquet")
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}")
    
    df = pd.read_parquet(train_path)
    logger.info(f"Loaded {len(df)} samples from main training data")
    
    # Extract features and labels
    if 'features' in df.columns:
        # Features are stored as array of OHLC arrays - need to reshape properly
        feature_list = df['features'].values
        # Convert list of OHLC arrays to proper (samples, timesteps, features) format
        X = np.array([np.stack(bar) for bar in feature_list])
        logger.info(f"Extracted features from 'features' column: {X.shape}")
    else:
        raise ValueError("Could not find 'features' column in training data")
    
    # Extract labels
    if 'label' in df.columns:
        # Convert string labels to binary
        label_map = {'consolidation': 0, 'retracement': 1}
        y = df['label'].map(label_map).values
        logger.info(f"Extracted labels from 'label' column: {np.bincount(y)}")
    else:
        raise ValueError("Could not find 'label' column in training data")
    
    return X, y


def test_baseline_vs_pretrained(X, y, encoder_path):
    """Compare baseline vs pretrained performance on real data."""
    logger.info("=== Testing Transfer Learning on Real Data ===")
    
    # Split data with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1337, stratify=y
    )
    
    logger.info(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")
    logger.info(f"Train class distribution: {np.bincount(y_train)}")
    logger.info(f"Test class distribution: {np.bincount(y_test)}")
    
    # Test 1: Baseline model
    logger.info("--- Baseline Model (No Pretraining) ---")
    baseline_model = EnhancedSimpleLSTMModel(
        seed=1337,
        hidden_size=128,
        num_layers=2,  # Match 2-layer encoder
        device="cpu",
        n_epochs=20,   # More epochs for real data
        batch_size=16,  # Smaller batch for small dataset
        early_stopping_patience=8,
    )
    
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    
    logger.info(f"Baseline accuracy: {baseline_acc:.4f}")
    logger.info(f"Baseline classification report:\n{classification_report(y_test, baseline_pred, zero_division=0)}")
    
    # Test 2: Pretrained model
    logger.info("--- Pretrained Model (2-Layer Encoder) ---")
    pretrained_model = EnhancedSimpleLSTMModel(
        seed=1337,
        hidden_size=128,
        num_layers=2,  # Match 2-layer encoder
        device="cpu",
        n_epochs=20,   # Same epochs for fair comparison
        batch_size=16,
        early_stopping_patience=8,
    )
    
    # Build model first
    pretrained_model.fit(X_train, y_train)
    
    # Load pretrained encoder
    try:
        pretrained_model.load_pretrained_encoder(
            encoder_path=encoder_path,
            freeze_encoder=True
        )
        logger.info(f"✓ Loaded pretrained encoder: {pretrained_model.pretrained_stats}")
        
        # Continue training with pretrained weights
        pretrained_model.fit(X_train, y_train)
        
        pretrained_pred = pretrained_model.predict(X_test)
        pretrained_acc = accuracy_score(y_test, pretrained_pred)
        
        logger.info(f"Pretrained accuracy: {pretrained_acc:.4f}")
        logger.info(f"Pretrained classification report:\n{classification_report(y_test, pretrained_pred, zero_division=0)}")
        
        # Compare results
        improvement = pretrained_acc - baseline_acc
        logger.info(f"Improvement: {improvement:+.4f} ({improvement/baseline_acc:+.1%})")
        
        # Detailed analysis
        logger.info("=== Detailed Analysis ===")
        
        # Class-wise performance
        baseline_report = classification_report(y_test, baseline_pred, zero_division=0, output_dict=True)
        pretrained_report = classification_report(y_test, pretrained_pred, zero_division=0, output_dict=True)
        
        for class_name in ['0', '1']:
            if class_name in baseline_report and class_name in pretrained_report:
                baseline_f1 = baseline_report[class_name]['f1-score']
                pretrained_f1 = pretrained_report[class_name]['f1-score']
                f1_improvement = pretrained_f1 - baseline_f1
                logger.info(f"Class {class_name} F1 improvement: {f1_improvement:+.4f}")
        
        # Overall assessment
        if improvement > 0.05:
            logger.success("✓ SIGNIFICANT IMPROVEMENT: Transfer learning is working well!")
        elif improvement > 0.02:
            logger.info("✓ MODERATE IMPROVEMENT: Transfer learning shows promise")
        elif improvement > 0:
            logger.info("≈ MINIMAL IMPROVEMENT: Transfer learning has small effect")
        else:
            logger.warning("✗ NO IMPROVEMENT: Transfer learning not helping with current setup")
        
        return baseline_acc, pretrained_acc
        
    except Exception as e:
        logger.error(f"✗ Failed to load pretrained encoder: {e}")
        return baseline_acc, None


def main():
    """Main test function with real data."""
    logger.info("=== Real Data Transfer Learning Test ===")
    
    # Load real training data
    try:
        X, y = load_real_data()
        logger.info(f"Dataset shape: {X.shape}, Class distribution: {np.bincount(y)}")
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")
        logger.info("Falling back to synthetic data for demonstration...")
        
        # Fallback to synthetic data
        np.random.seed(1337)
        X = np.random.randn(98, 105, 4).astype(np.float32)
        y = np.random.choice([0, 1], size=98, p=[0.57, 0.43])  # Match real distribution
    
    # Path to 2-layer encoder
    encoder_path = Path("data/artifacts/pretrained/bilstm_encoder_2layer.pt")
    
    if not encoder_path.exists():
        logger.error(f"2-layer encoder not found at {encoder_path}")
        return
    
    # Run comparison test
    baseline_acc, pretrained_acc = test_baseline_vs_pretrained(X, y, encoder_path)
    
    # Final summary
    logger.info("=== Final Summary ===")
    logger.info(f"Baseline accuracy:    {baseline_acc:.4f}")
    if pretrained_acc is not None:
        logger.info(f"Pretrained accuracy:  {pretrained_acc:.4f}")
        improvement = pretrained_acc - baseline_acc
        logger.info(f"Improvement:          {improvement:+.4f} ({improvement/baseline_acc:+.1%})")
        
        # Expected vs actual
        expected_improvement = 0.05  # 5% expected from transfer learning
        if improvement >= expected_improvement:
            logger.success(f"✓ Met or exceeded expected improvement (≥{expected_improvement:.0%})")
        else:
            gap = expected_improvement - improvement
            logger.info(f"⚠ Below expected improvement by {gap:.1%}")
    
    logger.info("=== Test Complete ===")


if __name__ == "__main__":
    main()