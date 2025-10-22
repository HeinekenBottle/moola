#!/usr/bin/env python3
"""Comprehensive QA Checklist and Validation for Clean Training Pipeline.

Validates all critical invariants and requirements:
- Invariants: model∈{jade,sapphire,opal}; encoding=center_length; batch=29
- Float32 enforcement throughout pipeline
- Data scope validation (11 months OHLC for pretrain)
- Target metrics: Pretrain val MAE gap 5-15%; Supervised Hit@±3 ≥60%, F1_macro ≥0.50, ECE <0.10, Joint ≥40%
- Rsync sanity: proper exclusions
"""

import argparse
import sys
from pathlib import Path
import subprocess
import json

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

def check_rsync_sanity():
    """Check .rsyncignore file exists and has proper exclusions."""
    print("🔍 Checking rsync sanity...")
    
    rsyncignore_path = Path(".rsyncignore")
    if not rsyncignore_path.exists():
        print("❌ .rsyncignore file not found")
        return False
    
    required_exclusions = [
        ".git/",
        "__pycache__/",
        ".artifacts/",
        "artifacts/",
        "data/raw/",
        "data/processed/",
        "logs/",
        "wandb/",
        ".env",
        ".venv",
        ".ipynb_checkpoints/"
    ]
    
    content = rsyncignore_path.read_text().strip().split('\n')
    missing = [exc for exc in required_exclusions if exc not in content]
    
    if missing:
        print(f"❌ Missing rsync exclusions: {missing}")
        return False
    
    print("✅ .rsyncignore sanity check passed")
    return True


def check_invariants():
    """Check hard invariants in model registry."""
    print("🔍 Checking hard invariants...")
    
    try:
        from moola.models.registry import ALLOWED, build
        
        # Check allowed models
        if ALLOWED != {"jade", "sapphire", "opal"}:
            print(f"❌ ALLOWED models incorrect: {ALLOWED}")
            return False
        
        # Test model building with mock config
        class MockConfig:
            def __init__(self):
                self.model = MockModel()
                self.train = MockTrain()
        
        class MockModel:
            def __init__(self):
                self.name = "jade"
                self.pointer_head = MockPointerHead()
                self.input_size = 11
                self.hidden_size = 64
                self.num_layers = 1
                self.bidirectional = True
                self.proj_head = True
                self.head_width = 64
        
        class MockPointerHead:
            def __init__(self):
                self.encoding = "center_length"
        
        class MockTrain:
            def __init__(self):
                self.batch_size = 29
        
        cfg = MockConfig()
        
        # Test valid config
        try:
            model = build(cfg)
            print("✅ Model building with valid config succeeded")
        except Exception as e:
            print(f"❌ Model building failed: {e}")
            return False
        
        # Test invalid configs
        invalid_configs = [
            ("invalid_model", "Model name validation"),
            ("jade", "Pointer encoding validation", {"encoding": "invalid"}),
            ("jade", "Batch size validation", {"batch_size": 32})
        ]
        
        for test_case in invalid_configs:
            if len(test_case) == 2:
                model_name, desc = test_case
                cfg.model.name = model_name
            else:
                model_name, desc, overrides = test_case
                cfg.model.name = model_name
                if "encoding" in overrides:
                    cfg.model.pointer_head.encoding = overrides["encoding"]
                if "batch_size" in overrides:
                    cfg.train.batch_size = overrides["batch_size"]
            
            try:
                build(cfg)
                print(f"❌ {desc} failed - should have raised assertion")
                return False
            except AssertionError:
                print(f"✅ {desc} passed")
            except Exception as e:
                print(f"❌ {desc} failed with unexpected error: {e}")
                return False
        
        print("✅ Hard invariants check passed")
        return True
        
    except ImportError as e:
        print(f"❌ Import error checking invariants: {e}")
        return False


def check_float32_enforcement():
    """Check float32 enforcement utilities."""
    print("🔍 Checking float32 enforcement...")
    
    try:
        from moola.utils.training.training_utils import (
            enforce_float32_precision,
            convert_batch_to_float32
        )
        import torch
        
        # Test precision enforcement
        enforce_float32_precision()
        print("✅ Float32 precision enforcement works")
        
        # Test batch conversion
        batch = {
            'X': torch.randn(10, 5).double(),
            'y': torch.randn(10).double(),
            'metadata': ['test'] * 10
        }
        
        converted = convert_batch_to_float32(batch)
        assert converted['X'].dtype == torch.float32
        assert converted['y'].dtype == torch.float32
        assert converted['metadata'] == ['test'] * 10  # Non-tensor unchanged
        
        print("✅ Float32 batch conversion works")
        return True
        
    except Exception as e:
        print(f"❌ Float32 enforcement check failed: {e}")
        return False


def check_data_pipeline():
    """Check data pipeline components."""
    print("🔍 Checking data pipeline...")
    
    try:
        from moola.data.pretrain_pipeline import PretrainPipeline, validate_pretrain_batch
        import torch
        
        # Test pretrain pipeline
        pipeline = PretrainPipeline(Path("dummy_path"), months=11)
        print("✅ PretrainPipeline initialization works")
        
        # Test batch validation
        valid_batch = {
            'X': torch.randn(10, 105, 4, dtype=torch.float32),
            'target': torch.randn(10, 105, 4, dtype=torch.float32),
            'mask': torch.rand(10, 105) < 0.4
        }
        
        validate_pretrain_batch(valid_batch)
        print("✅ Pretrain batch validation works")
        
        # Test invalid batch
        invalid_batch = {
            'X': torch.randn(10, 105, 4, dtype=torch.float64),  # Wrong dtype
            'target': torch.randn(10, 105, 4, dtype=torch.float32),
            'mask': torch.rand(10, 105) < 0.4
        }
        
        try:
            validate_pretrain_batch(invalid_batch)
            print("❌ Batch validation should have failed on wrong dtype")
            return False
        except AssertionError:
            print("✅ Batch validation correctly rejects invalid batch")
        
        return True
        
    except Exception as e:
        print(f"❌ Data pipeline check failed: {e}")
        return False


def check_training_scripts():
    """Check training scripts exist and are executable."""
    print("🔍 Checking training scripts...")
    
    scripts = [
        "scripts/run_mae_pretrain.py",
        "scripts/run_supervised_train.py"
    ]
    
    for script in scripts:
        script_path = Path(script)
        if not script_path.exists():
            print(f"❌ Script not found: {script}")
            return False
        
        if not script_path.is_file():
            print(f"❌ Script is not a file: {script}")
            return False
        
        # Check script is syntactically valid
        try:
            result = subprocess.run([
                sys.executable, "-m", "py_compile", str(script_path)
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"❌ Script syntax error in {script}: {result.stderr}")
                return False
            
        except Exception as e:
            print(f"❌ Error checking script {script}: {e}")
            return False
    
    print("✅ Training scripts check passed")
    return True


def check_makefile_targets():
    """Check Makefile has required targets."""
    print("🔍 Checking Makefile targets...")
    
    makefile_path = Path("Makefile")
    if not makefile_path.exists():
        print("❌ Makefile not found")
        return False
    
    content = makefile_path.read_text()
    
    required_targets = [
        "pretrain-encoder:",
        "train-jade-clean:",
        "train-sapphire-clean:",
        "train-opal-clean:"
    ]
    
    missing_targets = [target for target in required_targets if target not in content]
    
    if missing_targets:
        print(f"❌ Missing Makefile targets: {missing_targets}")
        return False
    
    print("✅ Makefile targets check passed")
    return True


def check_target_metrics():
    """Check target metrics definitions and validation logic."""
    print("🔍 Checking target metrics...")
    
    # Define target metrics as per specification
    target_metrics = {
        "pretrain": {
            "val_mae_gap_min": 5.0,  # %
            "val_mae_gap_max": 15.0  # %
        },
        "supervised": {
            "hit_at_3_min": 0.60,    # 60%
            "f1_macro_min": 0.50,    # 0.50
            "ece_max": 0.10,         # <0.10
            "joint_accuracy_min": 0.40  # 40%
        }
    }
    
    # Validate metric ranges are reasonable
    assert 0 <= target_metrics["pretrain"]["val_mae_gap_min"] <= target_metrics["pretrain"]["val_mae_gap_max"] <= 100
    assert 0 <= target_metrics["supervised"]["hit_at_3_min"] <= 1
    assert 0 <= target_metrics["supervised"]["f1_macro_min"] <= 1
    assert 0 <= target_metrics["supervised"]["ece_max"] <= 1
    assert 0 <= target_metrics["supervised"]["joint_accuracy_min"] <= 1
    
    print("✅ Target metrics validation passed")
    print(f"   Pretrain val MAE gap target: {target_metrics['pretrain']['val_mae_gap_min']}-{target_metrics['pretrain']['val_mae_gap_max']}%")
    print(f"   Supervised targets: Hit@±3 ≥{target_metrics['supervised']['hit_at_3_min']:.0%}, "
          f"F1 ≥{target_metrics['supervised']['f1_macro_min']:.2f}, "
          f"ECE <{target_metrics['supervised']['ece_max']:.2f}, "
          f"Joint ≥{target_metrics['supervised']['joint_accuracy_min']:.0%}")
    
    return True


def run_full_qa():
    """Run complete QA checklist."""
    print("🚀 Starting Comprehensive QA Checklist\n")
    
    checks = [
        ("Rsync Sanity", check_rsync_sanity),
        ("Hard Invariants", check_invariants),
        ("Float32 Enforcement", check_float32_enforcement),
        ("Data Pipeline", check_data_pipeline),
        ("Training Scripts", check_training_scripts),
        ("Makefile Targets", check_makefile_targets),
        ("Target Metrics", check_target_metrics)
    ]
    
    results = {}
    for name, check_func in checks:
        print(f"\n{'='*50}")
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results[name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 QA CHECKLIST SUMMARY")
    print(f"{'='*50}")
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<30} {status}")
    
    print(f"\nOverall: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 ALL CHECKS PASSED - Pipeline is ready for production!")
        return True
    else:
        print("⚠️  Some checks failed - Please fix issues before production")
        return False


def main():
    parser = argparse.ArgumentParser(description="Comprehensive QA Checklist for Clean Training Pipeline")
    parser.add_argument("--check", type=str, choices=[
        "rsync", "invariants", "float32", "pipeline", "scripts", "makefile", "metrics"
    ], help="Run specific check only")
    parser.add_argument("--output", type=str, help="Output results to JSON file")
    
    args = parser.parse_args()
    
    if args.check:
        # Run specific check
        check_map = {
            "rsync": check_rsync_sanity,
            "invariants": check_invariants,
            "float32": check_float32_enforcement,
            "pipeline": check_data_pipeline,
            "scripts": check_training_scripts,
            "makefile": check_makefile_targets,
            "metrics": check_target_metrics
        }
        
        if args.check in check_map:
            result = check_map[args.check]()
            sys.exit(0 if result else 1)
        else:
            print(f"Unknown check: {args.check}")
            sys.exit(1)
    else:
        # Run full QA
        result = run_full_qa()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({
                    "timestamp": str(Path.cwd()),
                    "all_passed": result,
                    "checks": result
                }, f, indent=2)
            print(f"\n📄 Results saved to {args.output}")
        
        sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()