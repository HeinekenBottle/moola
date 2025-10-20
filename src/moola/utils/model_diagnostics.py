"""Model diagnostics and logging utilities.

Provides standardized logging for:
- Model architecture information
- GPU/CUDA diagnostics
- Parameter counting and ratios
- Memory usage tracking
"""

import torch
import torch.nn as nn


class ModelDiagnostics:
    """Diagnostic utilities for model training."""

    @staticmethod
    def log_model_info(model: nn.Module, n_samples: int) -> dict[str, int | float]:
        """Log model architecture information.

        Args:
            model: PyTorch model
            n_samples: Number of training samples

        Returns:
            Dictionary with parameter counts and ratios
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        print(f"[MODEL] Total parameters: {total_params:,}")
        print(f"[MODEL] Trainable parameters: {trainable_params:,}")
        if frozen_params > 0:
            print(f"[MODEL] Frozen parameters: {frozen_params:,}")
        print(f"[MODEL] Parameter-to-sample ratio: {trainable_params/n_samples:.1f}:1")

        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
            "param_to_sample_ratio": trainable_params / n_samples,
        }

    @staticmethod
    def log_gpu_info(device: torch.device, use_amp: bool) -> dict[str, str | float | bool]:
        """Log GPU/CUDA diagnostic information.

        Args:
            device: Target device
            use_amp: Whether mixed precision is enabled

        Returns:
            Dictionary with GPU information
        """
        if device.type != "cuda":
            print(f"[GPU] Training on CPU")
            return {"device": "cpu", "use_amp": False}

        gpu_name = torch.cuda.get_device_name(0)
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved(0) / 1024**3  # GB

        print(f"[GPU] Training on: {gpu_name}")
        print(f"[GPU] Memory allocated: {memory_allocated:.2f} GB")
        print(f"[GPU] Memory reserved: {memory_reserved:.2f} GB")
        print(f"[GPU] Mixed precision (FP16): {use_amp}")

        return {
            "device": "cuda",
            "gpu_name": gpu_name,
            "memory_allocated_gb": memory_allocated,
            "memory_reserved_gb": memory_reserved,
            "use_amp": use_amp,
        }

    @staticmethod
    def log_gpu_memory(device: torch.device, prefix: str = "GPU") -> None:
        """Log current GPU memory usage.

        Args:
            device: Target device
            prefix: Prefix for log message
        """
        if device.type == "cuda":
            memory_gb = torch.cuda.memory_allocated(0) / 1024**3
            print(f"[{prefix}] Memory: {memory_gb:.2f} GB")

    @staticmethod
    def log_class_distribution(y: torch.Tensor, label_to_idx: dict = None) -> dict[int, int]:
        """Log class distribution in dataset.

        Args:
            y: Label tensor (integer indices)
            label_to_idx: Optional label mapping for pretty printing

        Returns:
            Dictionary mapping class index to count
        """
        unique_classes, class_counts = torch.unique(y, return_counts=True)
        class_dist = dict(zip(unique_classes.tolist(), class_counts.tolist()))

        print(f"[CLASS BALANCE] Class distribution: {class_dist}")

        return class_dist

    @staticmethod
    def count_frozen_parameters(model: nn.Module) -> tuple[int, int]:
        """Count frozen vs trainable parameters.

        Args:
            model: PyTorch model

        Returns:
            (trainable_params, frozen_params)
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params

        return trainable_params, frozen_params
