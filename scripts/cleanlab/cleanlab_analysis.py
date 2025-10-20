#!/usr/bin/env python3
"""Use CleanLab to find label errors in moola training data.

CleanLab identifies mislabeled samples by analyzing out-of-sample predictions
from a trained model. This helps prioritize which samples to review manually.

Workflow:
1. Load full 115 sample dataset (no cleaning)
2. Generate OOF predictions if not available
3. Use CleanLab to identify label errors
4. Export results for manual review

Requirements:
- pip install cleanlab
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from cleanlab.filter import find_label_issues

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def load_oof_predictions(model_name="cnn_transformer"):
    """Load out-of-sample predictions for CleanLab analysis.

    Args:
        model_name: Model to use (cnn_transformer or xgb)

    Returns:
        pred_probs: [N, 2] array of predicted probabilities
        labels: [N] array of true labels (0=consolidation, 1=retracement)
        window_ids: [N] array of window identifiers
    """
    # Load full dataset (no cleaning)
    train_path = Path("data/processed/train.parquet")
    df = pd.read_parquet(train_path)

    print(f"Loaded {len(df)} samples from {train_path}")

    # Load OOF predictions
    oof_path = Path(f"data/artifacts/oof/{model_name}/v1/seed_1337.npy")

    if not oof_path.exists():
        print(f"\n⚠️  OOF predictions not found: {oof_path}")
        print(f"Need to generate OOF predictions for all {len(df)} samples first.")
        print(f"\nRun: python3 -m moola.cli oof --model {model_name} --no-clean")
        return None, None, None

    pred_probs = np.load(oof_path)
    print(f"Loaded OOF predictions: {pred_probs.shape}")

    # Check if predictions match dataset size
    if len(pred_probs) != len(df):
        print(f"\n⚠️  Prediction size mismatch!")
        print(f"Dataset: {len(df)} samples")
        print(f"Predictions: {len(pred_probs)} samples")
        print(f"\nOOF was run on cleaned data. Need to regenerate for full dataset.")
        print(f"Run: python3 -m moola.cli oof --model {model_name} --no-clean")
        return None, None, None

    # Convert labels to integers (0=consolidation, 1=retracement)
    label_map = {'consolidation': 0, 'retracement': 1}
    labels = df['label'].map(label_map).values
    window_ids = df['window_id'].values

    return pred_probs, labels, window_ids


def run_cleanlab_analysis(pred_probs, labels, window_ids):
    """Run CleanLab to find label errors.

    Args:
        pred_probs: [N, 2] predicted probabilities
        labels: [N] true labels
        window_ids: [N] window identifiers

    Returns:
        DataFrame with label issues and quality scores
    """
    print("\n=== Running CleanLab Analysis ===")

    # Get label quality scores (lower = more likely to be wrong)
    from cleanlab.rank import get_label_quality_scores
    quality_scores = get_label_quality_scores(
        labels=labels,
        pred_probs=pred_probs,
    )

    # Find label issues - returns boolean array
    label_issues_mask = find_label_issues(
        labels=labels,
        pred_probs=pred_probs,
        return_indices_ranked_by=None,  # Return boolean array, not indices
    )

    # Create results dataframe
    results_df = pd.DataFrame({
        'window_id': window_ids,
        'given_label': labels,
        'given_label_name': ['consolidation' if l == 0 else 'retracement' for l in labels],
        'pred_label': pred_probs.argmax(axis=1),
        'pred_label_name': ['consolidation' if l == 0 else 'retracement' for l in pred_probs.argmax(axis=1)],
        'prob_consolidation': pred_probs[:, 0],
        'prob_retracement': pred_probs[:, 1],
        'label_quality': quality_scores,
        'is_label_issue': label_issues_mask,
    })

    # Sort by quality score (worst first)
    results_df = results_df.sort_values('label_quality', ascending=True)

    return results_df


def export_cleanlab_results(results_df, output_dir="data/corrections"):
    """Export CleanLab results for manual review.

    Args:
        results_df: DataFrame with CleanLab analysis results
        output_dir: Directory to save results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    full_output = output_dir / "cleanlab_label_quality.csv"
    results_df.to_csv(full_output, index=False)
    print(f"\n✓ Saved full results to {full_output}")

    # Save only issues for review
    issues_df = results_df[results_df['is_label_issue'] == True].copy()
    issues_output = output_dir / "cleanlab_label_issues.csv"
    issues_df.to_csv(issues_output, index=False)
    print(f"✓ Saved {len(issues_df)} label issues to {issues_output}")

    # Print summary
    print(f"\n=== CleanLab Summary ===")
    print(f"Total samples: {len(results_df)}")
    print(f"Label issues found: {len(issues_df)} ({len(issues_df)/len(results_df)*100:.1f}%)")
    print(f"Average label quality: {results_df['label_quality'].mean():.3f}")

    # Show top 10 worst quality samples
    print(f"\n=== Top 10 Most Likely Label Errors ===")
    print(issues_df[['window_id', 'given_label_name', 'pred_label_name',
                     'prob_consolidation', 'prob_retracement', 'label_quality']].head(10).to_string(index=False))

    # Compare with manual flags
    train_df = pd.read_parquet("data/processed/train.parquet")
    manual_flags = set()

    # Flag samples with invalid expansion indices
    invalid_mask = (
        (train_df['expansion_start'] >= train_df['expansion_end']) |
        (train_df['expansion_start'] < 30) | (train_df['expansion_start'] > 74) |
        (train_df['expansion_end'] < 30) | (train_df['expansion_end'] > 74)
    )
    manual_flags = set(train_df[invalid_mask]['window_id'].values)

    print(f"\n=== Comparison with Manual Validation ===")
    print(f"Manually flagged (invalid indices): {len(manual_flags)}")
    print(f"CleanLab flagged (label issues): {len(issues_df)}")

    # Find overlap
    cleanlab_flags = set(issues_df['window_id'].values)
    overlap = manual_flags & cleanlab_flags
    cleanlab_only = cleanlab_flags - manual_flags
    manual_only = manual_flags - cleanlab_flags

    print(f"Overlap: {len(overlap)}")
    print(f"CleanLab only: {len(cleanlab_only)}")
    print(f"Manual only: {len(manual_only)}")

    if len(overlap) > 0:
        print(f"\nOverlapping samples (both flagged):")
        for wid in sorted(overlap):
            print(f"  {wid}")

    return full_output, issues_output


def main():
    """Run CleanLab analysis on moola training data."""
    print("🔍 CleanLab Label Error Detection")
    print("=" * 50)

    # Load OOF predictions
    print("\n1. Loading OOF predictions...")
    pred_probs, labels, window_ids = load_oof_predictions(model_name="cnn_transformer")

    if pred_probs is None:
        print("\n❌ Cannot proceed without OOF predictions")
        print("\nNext steps:")
        print("1. Generate OOF predictions for full 115 samples:")
        print("   python3 scripts/generate_full_oof.py")
        print("2. Re-run this script:")
        print("   python3 scripts/cleanlab_analysis.py")
        return None

    # Run CleanLab analysis
    print("\n2. Running CleanLab analysis...")
    results_df = run_cleanlab_analysis(pred_probs, labels, window_ids)

    # Export results
    print("\n3. Exporting results...")
    full_output, issues_output = export_cleanlab_results(results_df)

    print(f"\n✅ CleanLab analysis complete!")
    print(f"\nNext steps:")
    print(f"1. Review label issues: {issues_output}")
    print(f"2. Manually verify flagged samples")
    print(f"3. Update labels in bespoke annotation tool")
    print(f"4. Import corrections: python3 scripts/import_corrections.py")

    return results_df


if __name__ == "__main__":
    main()
