"""Loss normalization patch for multi-task learning.

Add this class to finetune_jade.py to fix pointer silencing issue.

Usage:
    normalizer = LossNormalizer(momentum=0.95)

    # In training loop:
    loss_norm = normalizer.normalize({
        'type': loss_type,
        'ptr': loss_ptr,
        'binary': loss_binary,
        'countdown': loss_countdown,
    })

    # Apply weights to NORMALIZED losses
    total_loss = (
        0.10 * loss_norm['type'] +
        0.70 * loss_norm['ptr'] +
        0.10 * loss_norm['binary'] +
        0.10 * loss_norm['countdown']
    )
"""

import torch
from typing import Dict


class LossNormalizer:
    """Normalize losses by their running mean to enable fair multi-task weighting.

    Solves the scale mismatch problem:
    - Classification (CE): typically 0.5-2.0
    - Regression (MSE/Huber): typically 0.01-0.05

    Without normalization, regression losses get silenced even with high λ weights.

    Args:
        momentum: EMA momentum for running mean (0.95 = slow adaptation)
        warmup_steps: Use batch mean for first N steps (default: 10)
    """

    def __init__(self, momentum: float = 0.95, warmup_steps: int = 10):
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.running_means = {}

    def normalize(self, losses: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Normalize losses by their running mean.

        Args:
            losses: Dict of {task_name: loss_value}

        Returns:
            Dict of {task_name: normalized_loss}
        """
        normalized = {}
        self.step_count += 1

        for name, loss in losses.items():
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss

            # Initialize running mean on first encounter
            if name not in self.running_means:
                self.running_means[name] = loss_value

            # Update running mean (EMA)
            if self.step_count > self.warmup_steps:
                self.running_means[name] = (
                    self.momentum * self.running_means[name] +
                    (1 - self.momentum) * loss_value
                )
            else:
                # During warmup, use simple moving average
                self.running_means[name] = (
                    (self.running_means[name] * (self.step_count - 1) + loss_value) /
                    self.step_count
                )

            # Normalize: loss / running_mean
            # This makes each task contribute ~1.0 on average, then weights control proportions
            mean = self.running_means[name]
            if mean > 1e-8:  # Avoid division by zero
                normalized[name] = loss / mean
            else:
                normalized[name] = loss

        return normalized

    def get_stats(self) -> Dict[str, float]:
        """Get current running means for logging."""
        return self.running_means.copy()


# Example integration into existing training loop:
def example_training_loop():
    """
    # Initialize normalizer ONCE before training
    loss_normalizer = LossNormalizer(momentum=0.95, warmup_steps=10)

    for epoch in range(num_epochs):
        for batch in dataloader:
            # Forward pass
            output = model(features)

            # Compute raw losses
            loss_type = F.cross_entropy(output['logits'], labels)
            loss_ptr = F.huber_loss(output['pointers'], pointer_targets, delta=0.08)
            loss_binary = F.binary_cross_entropy_with_logits(
                output['expansion_binary_logits'],
                binary_targets
            )
            loss_countdown = F.huber_loss(
                output['expansion_countdown'],
                countdown_targets,
                delta=1.0
            )

            # NORMALIZE before weighting
            loss_norm = loss_normalizer.normalize({
                'type': loss_type,
                'ptr': loss_ptr,
                'binary': loss_binary,
                'countdown': loss_countdown,
            })

            # Apply TARGET weights to NORMALIZED losses
            total_loss = (
                0.10 * loss_norm['type'] +      # 10% on classification
                0.70 * loss_norm['ptr'] +        # 70% on pointers (PRIMARY)
                0.10 * loss_norm['binary'] +     # 10% on binary detection
                0.10 * loss_norm['countdown']    # 10% on countdown
            )

            # Backward + optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Log actual contributions (for debugging)
            if step % 100 == 0:
                contrib = {
                    'type': (0.10 * loss_norm['type'] / total_loss * 100).item(),
                    'ptr': (0.70 * loss_norm['ptr'] / total_loss * 100).item(),
                    'binary': (0.10 * loss_norm['binary'] / total_loss * 100).item(),
                    'countdown': (0.10 * loss_norm['countdown'] / total_loss * 100).item(),
                }
                print(f"Loss contributions: {contrib}")
                # Should see: type~10%, ptr~70%, binary~10%, countdown~10%
    """
    pass


if __name__ == "__main__":
    # Quick test
    print("Testing LossNormalizer...")

    normalizer = LossNormalizer(momentum=0.9, warmup_steps=5)

    # Simulate 20 training steps with realistic loss scales
    import random
    for step in range(20):
        # Classification: 0.5-2.0
        loss_type = torch.tensor(random.uniform(0.8, 1.5))
        # Pointers: 0.01-0.05 (80x smaller!)
        loss_ptr = torch.tensor(random.uniform(0.01, 0.03))
        # Binary: 0.3-0.9
        loss_binary = torch.tensor(random.uniform(0.4, 0.8))
        # Countdown: 0.4-1.0
        loss_countdown = torch.tensor(random.uniform(0.5, 0.9))

        # Normalize
        norm = normalizer.normalize({
            'type': loss_type,
            'ptr': loss_ptr,
            'binary': loss_binary,
            'countdown': loss_countdown,
        })

        # Apply target weights (10/70/10/10)
        total = 0.1 * norm['type'] + 0.7 * norm['ptr'] + 0.1 * norm['binary'] + 0.1 * norm['countdown']

        # Calculate actual contributions
        contrib = {
            'type': (0.1 * norm['type'] / total * 100).item(),
            'ptr': (0.7 * norm['ptr'] / total * 100).item(),
            'binary': (0.1 * norm['binary'] / total * 100).item(),
            'countdown': (0.1 * norm['countdown'] / total * 100).item(),
        }

        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Raw losses: type={loss_type.item():.4f}, ptr={loss_ptr.item():.4f}, "
                  f"binary={loss_binary.item():.4f}, countdown={loss_countdown.item():.4f}")
            print(f"  Running means: {normalizer.get_stats()}")
            print(f"  Contributions: type={contrib['type']:.1f}%, ptr={contrib['ptr']:.1f}%, "
                  f"binary={contrib['binary']:.1f}%, countdown={contrib['countdown']:.1f}%")

    print("\n✓ Test complete - contributions should stabilize around target (10/70/10/10)")
