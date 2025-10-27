#!/usr/bin/env python3
"""Production Inference Script for Baseline 100-Epoch Model.

Loads best_model.pt and generates span predictions on unseen data with IoU metrics.

Features:
- Loads JadeCompact model from checkpoint
- Preprocesses OHLC data to 12D relativity features
- Generates binary span predictions with configurable threshold
- Computes IoU (Intersection over Union) for span overlap
- Reports per-window and aggregate metrics
- Saves results to JSON for analysis

Usage:
    # Basic inference on 20 validation windows
    python3 scripts/infer_baseline.py \
        --model artifacts/baseline_100ep_weighted/best_model.pt \
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \
        --output inference_results.json

    # Custom threshold and more windows
    python3 scripts/infer_baseline.py \
        --model artifacts/baseline_100ep_weighted/best_model.pt \
        --data data/processed/labeled/train_latest_overlaps_v2.parquet \
        --output inference_results.json \
        --num-windows 50 \
        --threshold 0.6

Model Architecture:
- Input: 105 bars × 12 features (relativity pipeline)
- Encoder: 1-layer BiLSTM (96 hidden × 2 directions)
- Tasks: Classification (3 classes) + Pointers + Span Detection
- Output: Per-timestep probabilities [0-1] for expansion spans

Performance Baseline (baseline_100ep_weighted):
- Val Span F1: 0.187 (epoch 67)
- Expected IoU: 0.05-0.15 (model hasn't learned task well yet)
- Target IoU: >0.70 (requires better model or features)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path.cwd() / "src"))

from moola.features.relativity import RelativityConfig, build_relativity_features
from moola.models.jade_core import JadeCompact


def load_model(checkpoint_path: str, device: str = "cpu") -> JadeCompact:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model with same architecture as training
    model = JadeCompact(
        input_size=12,
        hidden_size=96,
        num_layers=1,
        predict_pointers=True,
        predict_expansion_sequence=True,
        use_crf=False,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    params = model.get_num_parameters()
    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Val loss: {checkpoint.get('val_loss', 'N/A')}")
    print(f"  - Parameters: {params['total']:,}")

    return model


def preprocess_window(ohlc_arrays):
    """Convert raw OHLC to 12D features."""
    # Each element in ohlc_arrays is a 4-element array [O, H, L, C]
    ohlc_list = [arr for arr in ohlc_arrays]
    ohlc_df = pd.DataFrame(ohlc_list, columns=["open", "high", "low", "close"])
    cfg = RelativityConfig()
    X_13d, _, _ = build_relativity_features(ohlc_df, cfg.model_dump())
    # Take first 12 features (13th is position encoding, not used in training)
    return torch.from_numpy(X_13d[0][:, :12]).float()


def predict_spans(model, features, threshold=0.5, device="cpu"):
    """Generate span predictions from model output."""
    features = features.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(features)

    # Extract predictions
    probs = torch.sigmoid(output["expansion_binary"].squeeze(0)).cpu().numpy()

    # Check for NaN
    if np.isnan(probs).any():
        print("⚠️  Warning: NaN detected in probabilities")
        probs = np.nan_to_num(probs, nan=0.0)

    binary_pred = (probs >= threshold).astype(float)

    # Find contiguous spans
    spans = []
    in_span = False
    start_idx = None

    for i, val in enumerate(binary_pred):
        if val == 1 and not in_span:
            in_span = True
            start_idx = i
        elif val == 0 and in_span:
            in_span = False
            spans.append((start_idx, i - 1))

    if in_span:  # Handle span extending to end
        spans.append((start_idx, len(binary_pred) - 1))

    # Confidence score: mean probability within predicted spans
    if spans:
        span_probs = [probs[s : e + 1].mean() for s, e in spans]
        confidence = float(np.mean(span_probs))
    else:
        confidence = 0.0

    return {
        "spans": spans,
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "pointers": output["pointers"].squeeze(0).cpu().numpy().tolist(),
    }


def compute_overlap_iou(pred_spans, true_start, true_end):
    """Compute IoU between predicted and true spans."""
    if not pred_spans:
        return 0.0

    # Create binary masks
    pred_mask = np.zeros(105, dtype=float)
    true_mask = np.zeros(105, dtype=float)

    for start, end in pred_spans:
        pred_mask[start : end + 1] = 1
    true_mask[true_start : true_end + 1] = 1

    # Compute IoU
    intersection = (pred_mask * true_mask).sum()
    union = ((pred_mask + true_mask) > 0).sum()

    return float(intersection / union) if union > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(description="Production inference for baseline model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, required=True, help="Path to labeled data")
    parser.add_argument(
        "--output", type=str, default="inference_results.json", help="Output JSON file"
    )
    parser.add_argument("--num-windows", type=int, default=20, help="Number of windows to test")
    parser.add_argument("--threshold", type=float, default=0.3, help="Span detection threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Load model
    print("=" * 80)
    print("BASELINE MODEL INFERENCE")
    print("=" * 80)
    model = load_model(args.model, args.device)

    # Load data and get validation split
    print(f"\nLoading data from {args.data}...")
    df = pd.read_parquet(args.data)

    # Use same train/val split as training (80/20)
    train_idx, val_idx = train_test_split(range(len(df)), test_size=0.2, random_state=args.seed)

    # Select windows from validation set
    val_df = df.iloc[val_idx].reset_index(drop=True)
    num_test = min(args.num_windows, len(val_df))
    test_df = val_df.head(num_test)

    print(f"Testing on {num_test} unseen windows from validation set")
    print(f"Threshold: {args.threshold}")
    print()

    # Run inference
    results = []
    overlap_scores = []

    for idx, row in test_df.iterrows():
        # Preprocess
        features = preprocess_window(row["features"])

        # Predict
        prediction = predict_spans(model, features, args.threshold, args.device)

        # Compute overlap metric
        overlap = compute_overlap_iou(
            prediction["spans"], row["expansion_start"], row["expansion_end"]
        )
        overlap_scores.append(overlap)

        # Store result
        result = {
            "window_id": idx,
            "predicted_spans": prediction["spans"],
            "true_span": [int(row["expansion_start"]), int(row["expansion_end"])],
            "confidence": prediction["confidence"],
            "overlap_iou": overlap,
            "label": row.get("label", "unknown"),
            "predicted_center": prediction["pointers"][0],
            "predicted_length": prediction["pointers"][1],
        }
        results.append(result)

        # Print progress
        status = "✓" if overlap >= 0.7 else "✗"
        print(
            f"{status} Window {idx:2d}: IoU={overlap:.3f}, "
            f"Pred={prediction['spans']}, "
            f"True=[{row['expansion_start']}, {row['expansion_end']}]"
        )

    # Compute summary statistics
    mean_overlap = np.mean(overlap_scores)
    passing_rate = np.mean([s >= 0.7 for s in overlap_scores])

    print()
    print("=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"Mean IoU:        {mean_overlap:.3f}")
    print(f"Passing rate:    {passing_rate:.1%} (IoU >= 0.70)")
    print(f"Best IoU:        {max(overlap_scores):.3f}")
    print(f"Worst IoU:       {min(overlap_scores):.3f}")

    if mean_overlap >= 0.7:
        print("\n✅ TARGET MET: Mean IoU >= 0.70")
    else:
        print(f"\n⚠️  Target not met: {mean_overlap:.3f} < 0.70")

    # Save results
    output = {
        "model": str(args.model),
        "data": str(args.data),
        "threshold": args.threshold,
        "num_windows": num_test,
        "summary": {
            "mean_iou": float(mean_overlap),
            "passing_rate": float(passing_rate),
            "max_iou": float(max(overlap_scores)),
            "min_iou": float(min(overlap_scores)),
        },
        "predictions": results,
    }

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
