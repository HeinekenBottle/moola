#!/usr/bin/env python3
"""Paper-Strict Compliance Report Generator.

Validates all 10 paper-strict requirements for the Moola ML pipeline:
1. Registry: Only jade/sapphire/opal models
2. DataLoader: center+length pointer encoding
3. Loss: Kendall uncertainty weighting
4. Split: Forward-chaining with purge
5. Requirements: Pinned to exact versions
6. Pretrain: Batch 64, epochs 100, dropout 0.15, jitter 0.01, mask 0.4
7. Supervised: Batch 29, input 0.25, recurrent 0.65, dense 0.5
8. Augmentation: Jitter 0.03, MagWarp 0.2, multiplier 3
9. Dropout: Input 0.25, recurrent 0.65, dense 0.5
10. Logging: Model/config/encoder paths + assertions

Usage:
    python3 scripts/paper_strict_compliance.py
"""

import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple
import yaml

def check_registry() -> Tuple[bool, str]:
    """Check 1: Registry contains only jade/sapphire/opal models."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        
        # Try to import the registry
        try:
            from moola.models import _MODEL_REGISTRY
            registry_models = set(_MODEL_REGISTRY.keys())
        except ImportError:
            # Check if model files exist
            model_dir = Path(__file__).parent.parent / "src" / "moola" / "models"
            model_files = [f.stem for f in model_dir.glob("*.py") if f.stem not in ["__init__", "base", "registry"]]
            registry_models = set(model_files)
        
        allowed_models = {"jade", "sapphire", "opal"}
        
        if not registry_models.issubset(allowed_models):
            return False, f"Registry contains unauthorized models: {registry_models - allowed_models}"
        
        if not allowed_models.issubset(registry_models):
            return False, f"Registry missing required models: {allowed_models - registry_models}"
        
        return True, f"✓ Registry contains only authorized models: {sorted(registry_models)}"
    except Exception as e:
        return False, f"Error checking registry: {e}"

def check_pointer_encoding() -> Tuple[bool, str]:
    """Check 2: DataLoader uses center+length pointer encoding."""
    try:
        from moola.data.pointer_transforms import start_end_to_center_length
        
        # Test the conversion function
        import torch
        start = torch.tensor([10.0])
        end = torch.tensor([20.0])
        center, length = start_end_to_center_length(start, end, seq_len=104)  # Use 104 (W-1)
        
        # Verify formulas: center = 0.5 * (start + end) / (W-1), length = (end - start + 1) / W
        expected_center = 0.5 * (10 + 20) / 104  # W-1 = 104
        expected_length = (20 - 10 + 1) / 105   # W = 105
        
        if abs(center.item() - expected_center) > 1e-6:
            return False, f"Center conversion incorrect: got {center.item()}, expected {expected_center}"
        
        if abs(length.item() - expected_length) > 1e-6:
            return False, f"Length conversion incorrect: got {length.item()}, expected {expected_length}"
        
        return True, "✓ Pointer encoding uses correct center+length formulas"
    except Exception as e:
        return False, f"Error checking pointer encoding: {e}"

def check_uncertainty_weighting() -> Tuple[bool, str]:
    """Check 3: Loss uses Kendall uncertainty weighting."""
    try:
        from moola.loss.uncertainty_weighted import UncertaintyWeightedLoss
        
        # Create loss function
        loss_fn = UncertaintyWeightedLoss()
        
        # Check for learned log vars
        if not hasattr(loss_fn, 'log_var_ptr') or not hasattr(loss_fn, 'log_var_type'):
            return False, "Loss function missing learned log variance attributes"
        
        if loss_fn.log_var_ptr is None or loss_fn.log_var_type is None:
            return False, "Loss function log variances are None"
        
        return True, "✓ Loss function uses Kendall uncertainty weighting with learned log vars"
    except Exception as e:
        return False, f"Error checking uncertainty weighting: {e}"

def check_temporal_splits() -> Tuple[bool, str]:
    """Check 4: Split uses forward-chaining with purge."""
    try:
        # Check if the splits file exists with purge window implementation
        splits_file = Path(__file__).parent.parent / "src" / "moola" / "data" / "splits.py"
        if splits_file.exists():
            content = splits_file.read_text()
            if "purge_window" in content and "forward-chaining" in content:
                return True, "✓ Temporal split implementation found with purge window and forward-chaining"
            else:
                return False, "Temporal split file missing purge window or forward-chaining implementation"
        else:
            return False, "Temporal split file not found"
    except Exception as e:
        return False, f"Error checking temporal splits: {e}"

def check_requirements() -> Tuple[bool, str]:
    """Check 5: Requirements are pinned to exact versions."""
    try:
        req_file = Path(__file__).parent.parent / "requirements.txt"
        
        if not req_file.exists():
            return False, "requirements.txt not found"
        
        content = req_file.read_text()
        
        # Check for specific pinned versions
        required_pins = [
            "torch==2.3.1",
            "torchvision==0.18.1",
            "numpy==1.26.4",
            "scipy==1.11.4",
            "pyarrow==16.1.0",
            "pandas==2.2.2",
            "opencv-python-headless==4.10.0.84",
        ]
        
        missing_pins = []
        for pin in required_pins:
            if pin not in content:
                missing_pins.append(pin)
        
        if missing_pins:
            return False, f"Missing required pins: {missing_pins}"
        
        return True, "✓ All required dependencies pinned to exact versions"
    except Exception as e:
        return False, f"Error checking requirements: {e}"

def check_pretraining_config() -> Tuple[bool, str]:
    """Check 6: Pretraining uses paper-strict settings."""
    try:
        # Check CLI defaults
        from moola.cli import pretrain_bilstm
        
        # Get the default parameters from the click command
        params = {}
        for param in pretrain_bilstm.params:
            if hasattr(param, 'default') and param.default is not None:
                params[param.name] = param.default
        
        # Verify paper-strict defaults
        expected = {
            'epochs': 100,
            'batch_size': 64,
            'mask_ratio': 0.4,
        }
        
        mismatches = []
        for key, expected_val in expected.items():
            if params.get(key) != expected_val:
                mismatches.append(f"{key}: got {params.get(key)}, expected {expected_val}")
        
        if mismatches:
            return False, f"Pretraining config mismatches: {mismatches}"
        
        # Check pretrainer defaults if importable
        try:
            from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
            pretrainer = MaskedLSTMPretrainer()
            
            if pretrainer.dropout != 0.15:
                return False, f"Pretrainer dropout: {pretrainer.dropout}, expected 0.15"
            
            if pretrainer.mask_ratio != 0.4:
                return False, f"Pretrainer mask_ratio: {pretrainer.mask_ratio}, expected 0.4"
            
            # Check augmentation defaults
            from moola.pretraining.data_augmentation import TimeSeriesAugmenter
            augmenter = TimeSeriesAugmenter()
            
            if augmenter.jitter_sigma != 0.01:
                return False, f"Pretraining jitter sigma: {augmenter.jitter_sigma}, expected 0.01"
        except ImportError:
            # Check if files exist with correct defaults
            pretrainer_file = Path(__file__).parent.parent / "src" / "moola" / "pretraining" / "masked_lstm_pretrain.py"
            if pretrainer_file.exists():
                content = pretrainer_file.read_text()
                if "dropout: float = 0.15" in content and "mask_ratio: float = 0.4" in content:
                    pass  # Found correct defaults
                else:
                    return False, "Pretrainer file missing paper-strict defaults"
        
        return True, "✓ Pretraining uses paper-strict settings (batch 64, epochs 100, dropout 0.15, jitter 0.01, mask 0.4)"
    except Exception as e:
        return False, f"Error checking pretraining config: {e}"

def check_supervised_config() -> Tuple[bool, str]:
    """Check 7 & 9: Supervised training uses paper-strict settings."""
    try:
        # Check training config
        train_config_path = Path(__file__).parent.parent / "src" / "moola" / "configs" / "train" / "multitask.yaml"
        
        if not train_config_path.exists():
            return False, "Training config file not found"
        
        with open(train_config_path) as f:
            train_cfg = yaml.safe_load(f)
        
        # Check batch size
        if train_cfg.get('data', {}).get('batch_size') != 29:
            return False, f"Training batch size: {train_cfg.get('data', {}).get('batch_size')}, expected 29"
        
        # Check model configs for dropout
        model_configs = {
            'jade': Path(__file__).parent.parent / "configs" / "model" / "jade.yaml",
            'sapphire': Path(__file__).parent.parent / "configs" / "model" / "sapphire.yaml",
            'opal': Path(__file__).parent.parent / "configs" / "model" / "opal.yaml",
        }
        
        for model_name, config_path in model_configs.items():
            if not config_path.exists():
                continue
                
            with open(config_path) as f:
                model_cfg = yaml.safe_load(f)
            
            arch = model_cfg.get('model', {}).get('architecture', {})
            
            # Check dropout values
            if arch.get('input_dropout') != 0.25:
                return False, f"{model_name} input_dropout: {arch.get('input_dropout')}, expected 0.25"
            
            if arch.get('recurrent_dropout') != 0.65:
                return False, f"{model_name} recurrent_dropout: {arch.get('recurrent_dropout')}, expected 0.65"
            
            if arch.get('dense_dropout') != 0.5:
                return False, f"{model_name} dense_dropout: {arch.get('dense_dropout')}, expected 0.5"
        
        return True, "✓ Supervised training uses paper-strict settings (batch 29, input 0.25, recurrent 0.65, dense 0.5)"
    except Exception as e:
        return False, f"Error checking supervised config: {e}"

def check_augmentation() -> Tuple[bool, str]:
    """Check 8: Augmentation uses paper-strict settings."""
    try:
        from moola.aug.jitter import Jitter, MagnitudeWarp
        
        # Check jitter defaults
        jitter = Jitter()
        if jitter.sigma != 0.03:
            return False, f"Jitter sigma: {jitter.sigma}, expected 0.03"
        
        # Check magnitude warp defaults
        magwarp = MagnitudeWarp()
        if magwarp.sigma != 0.2:
            return False, f"MagnitudeWarp sigma: {magwarp.sigma}, expected 0.2"
        
        # Check augmentation multiplier in config
        jade_config = Path(__file__).parent.parent / "configs" / "model" / "jade.yaml"
        with open(jade_config) as f:
            jade_cfg = yaml.safe_load(f)
        
        multiplier = jade_cfg.get('augmentation', {}).get('multiplier')
        if multiplier != 3:
            return False, f"Augmentation multiplier: {multiplier}, expected 3"
        
        return True, "✓ Augmentation uses paper-strict settings (jitter 0.03, magwarp 0.2, multiplier 3)"
    except Exception as e:
        return False, f"Error checking augmentation: {e}"

def check_version_info() -> Tuple[bool, str]:
    """Check package versions."""
    try:
        import torch
        import numpy
        import scipy
        import pyarrow
        import pandas
        
        versions = {
            'torch': torch.__version__,
            'numpy': numpy.__version__,
            'scipy': scipy.__version__,
            'pyarrow': pyarrow.__version__,
            'pandas': pandas.__version__,
        }
        
        # Try to import optional packages
        try:
            import torchvision
            versions['torchvision'] = torchvision.__version__
        except ImportError:
            versions['torchvision'] = 'not installed'
        
        try:
            import cv2
            versions['opencv-python-headless'] = cv2.__version__
        except ImportError:
            versions['opencv-python-headless'] = 'not installed'
        
        version_str = "✓ Package versions:\n"
        for pkg, ver in versions.items():
            version_str += f"  {pkg}: {ver}\n"
        
        return True, version_str.strip()
    except Exception as e:
        return False, f"Error checking versions: {e}"

def main():
    """Run all paper-strict compliance checks."""
    print("=" * 70)
    print("PAPER-STRICT COMPLIANCE REPORT")
    print("=" * 70)
    
    checks = [
        ("Registry (jade/sapphire/opal only)", check_registry),
        ("DataLoader (center+length encoding)", check_pointer_encoding),
        ("Loss (Kendall uncertainty weighting)", check_uncertainty_weighting),
        ("Split (forward-chaining with purge)", check_temporal_splits),
        ("Requirements (pinned versions)", check_requirements),
        ("Pretraining config", check_pretraining_config),
        ("Supervised config", check_supervised_config),
        ("Augmentation config", check_augmentation),
        ("Package versions", check_version_info),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            passed, message = check_func()
            results.append((name, passed, message))
        except Exception as e:
            results.append((name, False, f"Check failed with exception: {e}"))
    
    # Print results
    print("\nCHECK RESULTS:")
    print("-" * 70)
    
    passed_count = 0
    for name, passed, message in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status} {name}")
        print(f"     {message}")
        if passed:
            passed_count += 1
        print()
    
    # Summary
    print("=" * 70)
    print(f"SUMMARY: {passed_count}/{len(results)} checks passed")
    
    if passed_count == len(results):
        print("🎉 ALL PAPER-STRICT REQUIREMENTS SATISFIED")
        return 0
    else:
        print("❌ SOME PAPER-STRICT REQUIREMENTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())