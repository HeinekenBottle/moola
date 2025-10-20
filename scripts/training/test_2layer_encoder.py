#!/usr/bin/env python3
"""Test the new 2-layer BiLSTM encoder performance."""

import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.models.enhanced_simple_lstm import EnhancedSimpleLSTMModel


def analyze_encoder(encoder_path):
    """Analyze the 2-layer encoder architecture."""
    logger.info("=== Analyzing 2-Layer Encoder ===")
    
    if not encoder_path.exists():
        logger.error(f"Encoder not found at {encoder_path}")
        return None
    
    try:
        encoder_data = torch.load(encoder_path, map_location='cpu')
        state_dict = encoder_data.get('encoder_state_dict', {})
        hyperparams = encoder_data.get('hyperparams', {})
        
        logger.info(f"Encoder hyperparams: {hyperparams}")
        logger.info(f"Total tensors: {len(state_dict)}")
        
        # Expected for 2-layer bidirectional LSTM
        expected_tensors = 2 * 4 * 2  # 2 layers * 4 tensors * 2 directions = 16
        
        logger.info(f"Expected tensors (2 layers): {expected_tensors}")
        logger.info(f"Actual tensors: {len(state_dict)}")
        
        # List key tensors
        layer_tensors = {}
        for name, tensor in state_dict.items():
            if 'l0' in name:
                layer_tensors.setdefault('layer_0', []).append((name, tensor.shape))
            elif 'l1' in name:
                layer_tensors.setdefault('layer_1', []).append((name, tensor.shape))
        
        for layer, tensors in layer_tensors.items():
            logger.info(f"{layer}: {len(tensors)} tensors")
            for name, shape in tensors:
                logger.info(f"  {name}: {shape}")
        
        match_ratio = len(state_dict) / expected_tensors
        logger.info(f"Match ratio: {match_ratio:.1%}")
        
        if match_ratio >= 0.9:
            logger.success("✓ Excellent architecture compatibility")
            return True
        else:
            logger.warning(f"⚠ Partial compatibility: {match_ratio:.1%}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to analyze encoder: {e}")
        return False


def test_transfer_learning(encoder_path):
    """Test transfer learning with the 2-layer encoder."""
    logger.info("=== Testing Transfer Learning ===")
    
    # Create synthetic data
    np.random.seed(1337)
    X = np.random.randn(100, 105, 4).astype(np.float32)
    y = np.random.choice([0, 1], size=100, p=[0.7, 0.3])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1337, stratify=y
    )
    
    logger.info(f"Training: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Test 1: Baseline (no pretrained weights)
    logger.info("--- Baseline Model ---")
    baseline_model = EnhancedSimpleLSTMModel(
        seed=1337,
        hidden_size=128,
        num_layers=2,  # 2 layers to match encoder
        device="cpu",
        n_epochs=5,    # Quick training
        batch_size=32,
    )
    
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    logger.info(f"Baseline accuracy: {baseline_acc:.4f}")
    
    # Test 2: With pretrained encoder
    logger.info("--- Pretrained Model ---")
    pretrained_model = EnhancedSimpleLSTMModel(
        seed=1337,
        hidden_size=128,
        num_layers=2,  # 2 layers to match encoder
        device="cpu",
        n_epochs=5,    # Quick training
        batch_size=32,
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
        
        # Compare
        improvement = pretrained_acc - baseline_acc
        logger.info(f"Improvement: {improvement:+.4f} ({improvement/baseline_acc:+.1%})")
        
        if improvement > 0.02:
            logger.success("✓ Significant improvement with pretrained encoder")
        elif improvement > 0:
            logger.info("✓ Modest improvement with pretrained encoder")
        else:
            logger.warning("✗ No improvement with pretrained encoder")
            
        return baseline_acc, pretrained_acc
        
    except Exception as e:
        logger.error(f"✗ Failed to load pretrained encoder: {e}")
        return baseline_acc, None


def main():
    """Main test function."""
    logger.info("=== 2-Layer Encoder Performance Test ===")
    
    # Path to the new 2-layer encoder
    encoder_path = Path("data/artifacts/pretrained/bilstm_encoder_2layer.pt")
    
    # Analyze encoder architecture
    is_compatible = analyze_encoder(encoder_path)
    
    if not is_compatible:
        logger.error("Encoder architecture is not compatible, skipping performance test")
        return
    
    # Test transfer learning performance
    baseline_acc, pretrained_acc = test_transfer_learning(encoder_path)
    
    # Summary
    logger.info("=== Summary ===")
    logger.info(f"Baseline accuracy:     {baseline_acc:.4f}")
    if pretrained_acc is not None:
        logger.info(f"Pretrained accuracy:   {pretrained_acc:.4f}")
        improvement = pretrained_acc - baseline_acc
        logger.info(f"Improvement:           {improvement:+.4f} ({improvement/baseline_acc:+.1%})")
    
    logger.info("=== Test Complete ===")


if __name__ == "__main__":
    main()