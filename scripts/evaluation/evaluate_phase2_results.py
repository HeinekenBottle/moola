#!/usr/bin/env python3
"""Evaluate Phase 2 OOF predictions and compare clean vs augmented performance."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def load_oof(oof_path):
    """Load OOF predictions (N, 2) and convert to probabilities."""
    preds = np.load(oof_path)
    return preds

def evaluate_model(y_true, y_pred_probs, model_name, mode):
    """Evaluate a single model's OOF predictions."""
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_pred_probs[:, 1])
    except:
        auc = 0.5

    print(f"\n{'='*60}")
    print(f"{model_name.upper()} ({mode})")
    print(f"{'='*60}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC-ROC:  {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['consolidation', 'retracement']))

    return {'model': model_name, 'mode': mode, 'accuracy': acc, 'auc': auc}

def main():
    # Load labels
    train_df = pd.read_parquet('data/processed/train_clean.parquet')

    # Convert labels to integers if they're strings
    label_map = {'consolidation': 0, 'retracement': 1}
    if train_df['label'].dtype == object:
        y_true = train_df['label'].map(label_map).values
    else:
        y_true = train_df['label'].values

    print(f"\n{'='*60}")
    print(f"PHASE 2 RESULTS EVALUATION")
    print(f"{'='*60}")
    print(f"Dataset: data/processed/train_clean.parquet")
    print(f"Samples: {len(y_true)}")
    print(f"Class distribution: {dict(zip(*np.unique(y_true, return_counts=True)))}")

    # Models to evaluate
    models = ['logreg', 'rf', 'xgb', 'simple_lstm', 'cnn_transformer']
    modes = ['clean', 'augmented']

    results = []

    for model in models:
        for mode in modes:
            oof_path = Path(f'data/oof/{model}_{mode}.npy')

            if not oof_path.exists():
                print(f"\n⚠️  Missing: {oof_path}")
                continue

            y_pred_probs = load_oof(oof_path)

            if len(y_pred_probs) != len(y_true):
                print(f"\n❌ Length mismatch for {model}_{mode}: {len(y_pred_probs)} vs {len(y_true)}")
                continue

            result = evaluate_model(y_true, y_pred_probs, model, mode)
            results.append(result)

    # Summary table
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")

    results_df = pd.DataFrame(results)

    # Pivot table
    pivot_acc = results_df.pivot(index='model', columns='mode', values='accuracy')
    pivot_auc = results_df.pivot(index='model', columns='mode', values='auc')

    print("\nAccuracy Comparison:")
    print(pivot_acc.to_string(float_format=lambda x: f'{x:.4f}'))

    print("\n\nAUC-ROC Comparison:")
    print(pivot_auc.to_string(float_format=lambda x: f'{x:.4f}'))

    # Best models
    print(f"\n{'='*60}")
    print(f"BEST MODELS")
    print(f"{'='*60}")

    best_clean = results_df[results_df['mode'] == 'clean'].nlargest(3, 'accuracy')
    best_aug = results_df[results_df['mode'] == 'augmented'].nlargest(3, 'accuracy')

    print("\nTop 3 Clean Models:")
    for i, row in enumerate(best_clean.itertuples(), 1):
        print(f"  {i}. {row.model}: {row.accuracy:.4f} (AUC: {row.auc:.4f})")

    print("\nTop 3 Augmented Models:")
    for i, row in enumerate(best_aug.itertuples(), 1):
        print(f"  {i}. {row.model}: {row.accuracy:.4f} (AUC: {row.auc:.4f})")

    # Save results
    results_df.to_csv('reports/phase2_results.csv', index=False)
    print(f"\n✅ Results saved to: reports/phase2_results.csv")

if __name__ == '__main__':
    main()
