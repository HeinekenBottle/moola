#!/usr/bin/env python3
"""Single LSTM Experiment Runner with MLflow Tracking.

Executes one complete experiment:
1. Pre-training: Bidirectional masked LSTM autoencoder
2. Fine-tuning: SimpleLSTM with pre-trained encoder
3. Evaluation: Test set metrics with per-class accuracy

Usage:
    python run_lstm_experiment.py --experiment_id exp_phase1_timewarp_0.12
    python run_lstm_experiment.py --config experiment_configs.py --experiment_id exp_phase2_arch_128_8
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# MLflow tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("[WARNING] MLflow not installed. Install with: pip install mlflow")

# Import moola modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from moola.models.simple_lstm import SimpleLSTMModel
from moola.pretraining.masked_lstm_pretrain import MaskedLSTMPretrainer
from moola.utils.seeds import set_seed
from experiment_configs import ExperimentConfig


class ExperimentRunner:
    """Runs a single LSTM optimization experiment with full tracking."""

    def __init__(self, config: ExperimentConfig, data_dir: Path):
        """Initialize experiment runner.

        Args:
            config: Experiment configuration
            data_dir: Root data directory containing train/test splits
        """
        self.config = config
        self.data_dir = data_dir
        self.results = {}
        self.start_time = None

        # Set reproducibility
        set_seed(1337)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_data(self) -> tuple:
        """Load augmented training data and test data.

        Returns:
            (X_train_aug, y_train_aug, X_test, y_test)
        """
        print(f"\n{'='*70}")
        print(f"LOADING DATA")
        print(f"{'='*70}")

        # Load augmented unlabeled data (for pre-training)
        unlabeled_path = self.data_dir / "unlabeled_augmented.npz"
        if not unlabeled_path.exists():
            raise FileNotFoundError(
                f"Augmented unlabeled data not found: {unlabeled_path}\n"
                f"Run data augmentation pipeline first!"
            )

        unlabeled_data = np.load(unlabeled_path)
        X_unlabeled = unlabeled_data['X']
        print(f"  Unlabeled data: {X_unlabeled.shape} (for pre-training)")

        # Load labeled training data (for fine-tuning)
        train_path = self.data_dir / "train.npz"
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        train_data = np.load(train_path)
        X_train = train_data['X']
        y_train = train_data['y']
        print(f"  Training data: {X_train.shape}, labels: {y_train.shape}")

        # Load test data (for evaluation)
        test_path = self.data_dir / "test.npz"
        if not test_path.exists():
            raise FileNotFoundError(f"Test data not found: {test_path}")

        test_data = np.load(test_path)
        X_test = test_data['X']
        y_test = test_data['y']
        print(f"  Test data: {X_test.shape}, labels: {y_test.shape}")

        print(f"{'='*70}\n")

        return X_unlabeled, X_train, y_train, X_test, y_test

    def pretrain_encoder(self, X_unlabeled: np.ndarray) -> Path:
        """Pre-train bidirectional masked LSTM encoder.

        Args:
            X_unlabeled: Unlabeled OHLC sequences [N, 105, 4]

        Returns:
            Path to saved encoder checkpoint
        """
        print(f"\n{'='*70}")
        print(f"PRE-TRAINING PHASE")
        print(f"{'='*70}")

        encoder_path = Path(self.config.get_encoder_path())
        encoder_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize pre-trainer
        pretrainer = MaskedLSTMPretrainer(
            input_dim=4,
            hidden_dim=self.config.hidden_size,  # Match SimpleLSTM hidden size
            num_layers=2,
            dropout=0.2,
            mask_ratio=0.15,
            mask_strategy="patch",
            patch_size=7,
            learning_rate=1e-3,
            batch_size=self.config.batch_size,
            device=self.config.device,
            seed=1337
        )

        print(f"  Config:")
        print(f"    - hidden_dim: {self.config.hidden_size}")
        print(f"    - mask_ratio: 0.15")
        print(f"    - mask_strategy: patch")
        print(f"    - epochs: {self.config.pretrain_epochs}")
        print(f"    - device: {self.config.device}")
        print()

        # Pre-train
        pretrain_start = time.time()
        history = pretrainer.pretrain(
            X_unlabeled,
            n_epochs=self.config.pretrain_epochs,
            val_split=0.1,
            patience=10,
            save_path=encoder_path,
            verbose=True
        )
        pretrain_time = time.time() - pretrain_start

        # Log metrics
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1]
        best_val_loss = min(history['val_loss'])

        print(f"  Pre-training complete in {pretrain_time/60:.1f} minutes")
        print(f"  Final train loss: {final_train_loss:.4f}")
        print(f"  Final val loss: {final_val_loss:.4f}")
        print(f"  Best val loss: {best_val_loss:.4f}")
        print(f"  Encoder saved: {encoder_path}")

        # Store results
        self.results['pretrain_time_sec'] = pretrain_time
        self.results['pretrain_final_train_loss'] = final_train_loss
        self.results['pretrain_final_val_loss'] = final_val_loss
        self.results['pretrain_best_val_loss'] = best_val_loss

        if MLFLOW_AVAILABLE:
            mlflow.log_metric("pretrain_time_min", pretrain_time / 60)
            mlflow.log_metric("pretrain_final_train_loss", final_train_loss)
            mlflow.log_metric("pretrain_final_val_loss", final_val_loss)
            mlflow.log_metric("pretrain_best_val_loss", best_val_loss)

        return encoder_path

    def finetune_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        encoder_path: Path
    ) -> SimpleLSTMModel:
        """Fine-tune SimpleLSTM with pre-trained encoder.

        Args:
            X_train: Training OHLC sequences [N, 105, 4]
            y_train: Training labels [N]
            encoder_path: Path to pre-trained encoder

        Returns:
            Trained SimpleLSTMModel
        """
        print(f"\n{'='*70}")
        print(f"FINE-TUNING PHASE")
        print(f"{'='*70}")

        # Initialize SimpleLSTM
        model = SimpleLSTMModel(
            seed=1337,
            hidden_size=self.config.hidden_size,
            num_layers=1,
            num_heads=self.config.num_heads,
            dropout=0.4,
            n_epochs=self.config.finetune_epochs,
            batch_size=self.config.batch_size,
            learning_rate=self.config.finetune_lr,
            device=self.config.device,
            use_amp=True,
            num_workers=16,
            early_stopping_patience=20,
            val_split=0.15,
            mixup_alpha=0.4,
            cutmix_prob=0.5,
            use_temporal_aug=True,
            jitter_prob=self.config.jitter_prob,
            scaling_prob=self.config.scaling_prob,
            time_warp_prob=self.config.time_warp_prob,
        )

        # Update time_warp_sigma (experiment variable)
        if hasattr(model, 'temporal_aug'):
            model.temporal_aug.time_warp_sigma = self.config.time_warp_sigma
            print(f"  Set time_warp_sigma = {self.config.time_warp_sigma}")

        print(f"  Config:")
        print(f"    - hidden_size: {self.config.hidden_size}")
        print(f"    - num_heads: {self.config.num_heads}")
        print(f"    - time_warp_sigma: {self.config.time_warp_sigma}")
        print(f"    - freeze_epochs: {self.config.finetune_freeze_epochs}")
        print(f"    - total_epochs: {self.config.finetune_epochs}")
        print()

        # Build model first (required before loading encoder)
        print(f"  Building model...")
        model.fit(X_train[:1], y_train[:1])  # Build architecture with 1 sample
        model.model.train()

        # Load pre-trained encoder
        print(f"  Loading pre-trained encoder from: {encoder_path}")
        model.load_pretrained_encoder(encoder_path, freeze_encoder=True)

        # Full training with encoder unfreezing
        print(f"  Training with frozen encoder for {self.config.finetune_freeze_epochs} epochs...")
        finetune_start = time.time()
        model.fit(
            X_train,
            y_train,
            unfreeze_encoder_after=self.config.finetune_freeze_epochs
        )
        finetune_time = time.time() - finetune_start

        print(f"  Fine-tuning complete in {finetune_time/60:.1f} minutes")

        # Store results
        self.results['finetune_time_sec'] = finetune_time

        if MLFLOW_AVAILABLE:
            mlflow.log_metric("finetune_time_min", finetune_time / 60)

        return model

    def evaluate_model(
        self,
        model: SimpleLSTMModel,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """Evaluate model on test set.

        Args:
            model: Trained SimpleLSTMModel
            X_test: Test OHLC sequences [N, 105, 4]
            y_test: Test labels [N]

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*70}")
        print(f"EVALUATION PHASE")
        print(f"{'='*70}")

        # Predict
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)

        # Per-class accuracy from confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        class_accuracies = cm.diagonal() / cm.sum(axis=1)

        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)

        print(f"  Overall Accuracy: {accuracy:.4f}")
        print(f"  Class 0 Accuracy: {class_accuracies[0]:.4f}")
        print(f"  Class 1 Accuracy: {class_accuracies[1]:.4f}")
        print()
        print("  Classification Report:")
        print(classification_report(y_test, y_pred))

        # Store results
        metrics = {
            'accuracy': accuracy,
            'class_0_accuracy': class_accuracies[0],
            'class_1_accuracy': class_accuracies[1],
            'class_0_precision': report['0']['precision'],
            'class_0_recall': report['0']['recall'],
            'class_0_f1': report['0']['f1-score'],
            'class_1_precision': report['1']['precision'],
            'class_1_recall': report['1']['recall'],
            'class_1_f1': report['1']['f1-score'],
        }

        # Check if within expected range
        in_expected_range = (
            self.config.expected_accuracy_min <= accuracy <= self.config.expected_accuracy_max
            and self.config.expected_class1_min <= class_accuracies[1] <= self.config.expected_class1_max
        )
        metrics['in_expected_range'] = in_expected_range

        if in_expected_range:
            print(f"  ✓ Results within expected range")
        else:
            print(f"  ⚠ Results outside expected range:")
            print(f"    Expected accuracy: {self.config.expected_accuracy_min:.2f}-{self.config.expected_accuracy_max:.2f}")
            print(f"    Expected class 1: {self.config.expected_class1_min:.2f}-{self.config.expected_class1_max:.2f}")

        self.results.update(metrics)

        # Log to MLflow
        if MLFLOW_AVAILABLE:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value)

        return metrics

    def run(self) -> Dict:
        """Execute complete experiment pipeline.

        Returns:
            Dictionary of all results
        """
        self.start_time = time.time()

        print(f"\n{'#'*70}")
        print(f"# EXPERIMENT: {self.config.experiment_id}")
        print(f"# {self.config.description}")
        print(f"{'#'*70}")

        # Initialize MLflow run
        if MLFLOW_AVAILABLE:
            mlflow.start_run(run_name=self.config.experiment_id)
            mlflow.log_params(self.config.to_dict())

        try:
            # 1. Load data
            X_unlabeled, X_train, y_train, X_test, y_test = self.load_data()

            # 2. Pre-train encoder
            encoder_path = self.pretrain_encoder(X_unlabeled)

            # 3. Fine-tune model
            model = self.finetune_model(X_train, y_train, encoder_path)

            # 4. Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)

            # Total time
            total_time = time.time() - self.start_time
            self.results['total_time_sec'] = total_time

            print(f"\n{'='*70}")
            print(f"EXPERIMENT COMPLETE")
            print(f"{'='*70}")
            print(f"  Total time: {total_time/60:.1f} minutes")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  Class 1 accuracy: {metrics['class_1_accuracy']:.4f}")
            print(f"{'='*70}\n")

            if MLFLOW_AVAILABLE:
                mlflow.log_metric("total_time_min", total_time / 60)

            # Save results
            results_path = Path(self.config.get_save_path()) / "results.json"
            results_path.parent.mkdir(parents=True, exist_ok=True)
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2)

            if MLFLOW_AVAILABLE:
                mlflow.log_artifact(str(results_path))

        except Exception as e:
            print(f"\n[ERROR] Experiment failed: {e}")
            if MLFLOW_AVAILABLE:
                mlflow.log_param("status", "failed")
                mlflow.log_param("error", str(e))
            raise

        finally:
            if MLFLOW_AVAILABLE:
                mlflow.end_run()

        return self.results


def main():
    parser = argparse.ArgumentParser(
        description="Run single LSTM optimization experiment"
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        required=True,
        help="Experiment ID (e.g., exp_phase1_timewarp_0.12)"
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=Path("data/processed"),
        help="Data directory containing train/test splits"
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default="LSTM_Optimization_Phase_IV",
        help="MLflow experiment name"
    )

    args = parser.parse_args()

    # Setup MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        print(f"[MLflow] Tracking URI: {args.mlflow_tracking_uri}")
        print(f"[MLflow] Experiment: {args.mlflow_experiment_name}")

    # Load experiment config
    from experiment_configs import PHASE_1_EXPERIMENTS, get_phase2_experiments, get_phase3_experiments

    # Find config by ID
    all_exps = (
        PHASE_1_EXPERIMENTS +
        get_phase2_experiments() +
        get_phase3_experiments()
    )
    config = next((e for e in all_exps if e.experiment_id == args.experiment_id), None)

    if config is None:
        raise ValueError(f"Unknown experiment_id: {args.experiment_id}")

    # Run experiment
    runner = ExperimentRunner(config, args.data_dir)
    results = runner.run()

    print(f"\n[SUCCESS] Results saved to: {Path(config.get_save_path()) / 'results.json'}")


if __name__ == "__main__":
    main()
