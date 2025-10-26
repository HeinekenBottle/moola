"""Quick local test of expansion-focused training with loss normalization.

Test on small dataset before full RunPod deployment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from moola.models.jade_core import JadeCompact
from moola.features.relativity import build_relativity_features, RelativityConfig


class LossNormalizer:
    """Normalize losses by running mean for fair multi-task weighting."""
    def __init__(self, momentum=0.95, warmup_steps=10):
        self.momentum = momentum
        self.warmup_steps = warmup_steps
        self.step_count = 0
        self.running_means = {}

    def normalize(self, losses):
        normalized = {}
        self.step_count += 1

        for name, loss in losses.items():
            loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss

            if name not in self.running_means:
                self.running_means[name] = loss_value

            if self.step_count > self.warmup_steps:
                self.running_means[name] = (
                    self.momentum * self.running_means[name] +
                    (1 - self.momentum) * loss_value
                )
            else:
                self.running_means[name] = (
                    (self.running_means[name] * (self.step_count - 1) + loss_value) /
                    self.step_count
                )

            mean = self.running_means[name]
            normalized[name] = loss / mean if mean > 1e-8 else loss

        return normalized


def create_expansion_labels(expansion_start, expansion_end, window_length=105):
    """Create binary mask and countdown from pointers."""
    binary_mask = np.zeros(window_length, dtype=np.float32)
    binary_mask[expansion_start:expansion_end+1] = 1.0

    countdown = np.arange(window_length, dtype=np.float32) - expansion_start
    countdown = -countdown
    countdown = np.clip(countdown, -20, 20)  # Clip to prevent MSE explosion

    return binary_mask, countdown


class ExpansionDataset(Dataset):
    """Dataset with expansion labels."""
    def __init__(self, data_path, max_samples=None):
        self.df = pd.read_parquet(data_path)
        if max_samples:
            self.df = self.df.head(max_samples)

        self.label_map = {'consolidation': 0, 'retracement': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Build 12D features from raw OHLC
        ohlc_arrays = [arr for arr in row['features']]
        ohlc_df = pd.DataFrame(ohlc_arrays, columns=['open', 'high', 'low', 'close'])

        cfg = RelativityConfig()
        X_12d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())

        # Labels
        label = self.label_map.get(row['label'], 0)

        # Pointers (normalized to [0, 1])
        center = (row['expansion_start'] + row['expansion_end']) / 2.0 / 105.0
        length = (row['expansion_end'] - row['expansion_start']) / 105.0
        pointers = np.array([center, length], dtype=np.float32)

        # Expansion labels
        binary, countdown = create_expansion_labels(
            row['expansion_start'],
            row['expansion_end']
        )

        return {
            'features': torch.from_numpy(X_12d[0]).float(),
            'label': torch.tensor(label, dtype=torch.long),
            'pointers': torch.from_numpy(pointers).float(),
            'binary': torch.from_numpy(binary).float(),
            'countdown': torch.from_numpy(countdown).float(),
        }


def train_epoch(model, loader, optimizer, normalizer, device):
    """Train one epoch with normalized multi-task loss."""
    model.train()
    total_loss = 0

    for batch in loader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        pointers = batch['pointers'].to(device)
        binary = batch['binary'].to(device)
        countdown = batch['countdown'].to(device)

        # Forward
        output = model(features)

        # Compute raw losses
        loss_type = F.cross_entropy(output['logits'], labels)
        loss_ptr = F.huber_loss(output['pointers'], pointers, delta=0.08)
        loss_binary = F.binary_cross_entropy_with_logits(
            output['expansion_binary_logits'], binary
        )
        loss_countdown = F.huber_loss(output['expansion_countdown'], countdown, delta=1.0)

        # Normalize
        loss_norm = normalizer.normalize({
            'type': loss_type,
            'ptr': loss_ptr,
            'binary': loss_binary,
            'countdown': loss_countdown,
        })

        # Apply weights: 10/70/10/10
        loss = (
            0.10 * loss_norm['type'] +
            0.70 * loss_norm['ptr'] +
            0.10 * loss_norm['binary'] +
            0.10 * loss_norm['countdown']
        )

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    print("=" * 80)
    print("EXPANSION-FOCUSED TRAINING - LOCAL TEST")
    print("=" * 80)

    # Config
    device = 'cpu'
    epochs = 5
    batch_size = 8
    lr = 1e-3

    # Load data
    print("\nLoading data...")
    dataset = ExpansionDataset('data/processed/labeled/train_latest_overlaps_v2.parquet', max_samples=50)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_idx)
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_idx)
    )

    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

    # Model
    print("\nCreating model...")
    model = JadeCompact(
        input_size=12,
        predict_pointers=True,
        predict_expansion_sequence=True,  # Enable expansion heads
    ).to(device)

    params = model.get_num_parameters()
    print(f"Parameters: {params['total']:,} total, {params['trainable']:,} trainable")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Loss normalizer
    normalizer = LossNormalizer(momentum=0.95, warmup_steps=5)

    # Training
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, normalizer, device)

        # Validation (simple)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                labels = batch['label'].to(device)
                pointers = batch['pointers'].to(device)
                binary = batch['binary'].to(device)
                countdown = batch['countdown'].to(device)

                output = model(features)

                loss_type = F.cross_entropy(output['logits'], labels)
                loss_ptr = F.huber_loss(output['pointers'], pointers, delta=0.08)
                loss_binary = F.binary_cross_entropy_with_logits(
                    output['expansion_binary_logits'], binary
                )
                loss_countdown = F.huber_loss(output['expansion_countdown'], countdown, delta=1.0)

                val_loss += (loss_type + loss_ptr + loss_binary + loss_countdown).item()

        val_loss /= len(val_loader)

        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Show loss normalizer stats
        if epoch == epochs - 1:
            print(f"\nFinal loss normalizer stats:")
            for name, mean in normalizer.running_means.items():
                print(f"  {name}: {mean:.6f}")

    print("\n" + "=" * 80)
    print("✓ LOCAL TEST COMPLETE")
    print("=" * 80)
    print("\nNext: Deploy to RunPod for full training on 174 samples")


if __name__ == "__main__":
    main()
