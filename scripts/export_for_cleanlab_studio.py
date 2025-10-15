#!/usr/bin/env python3
"""Export moola data for CleanLab Studio visual review.

Combines:
1. Manual validation flags (18 samples with invalid expansion indices)
2. CleanLab label issues (10 samples with likely mislabeled data)

Total: 28 samples prioritized for review
"""
import pandas as pd
import numpy as np
from pathlib import Path


def export_for_cleanlab_studio(
    output_dir="data/corrections",
):
    """Export training data with CleanLab quality scores for review.

    Args:
        output_dir: Directory to save CleanLab export
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load original training data
    train_df = pd.read_parquet("data/processed/train.parquet")
    print(f"Loaded {len(train_df)} samples")

    # Load CleanLab quality analysis
    cleanlab_path = output_dir / "cleanlab_label_quality.csv"
    if not cleanlab_path.exists():
        print(f"Error: CleanLab analysis not found: {cleanlab_path}")
        print(f"Run: python3 scripts/cleanlab_analysis.py first")
        return None

    cleanlab_df = pd.read_csv(cleanlab_path)
    print(f"Loaded CleanLab analysis for {len(cleanlab_df)} samples")

    # Merge CleanLab results with training data
    combined_df = train_df.merge(
        cleanlab_df[['window_id', 'label_quality', 'is_label_issue',
                     'prob_consolidation', 'prob_retracement']],
        on='window_id',
        how='left'
    )

    # Flag manual validation issues
    manual_issues = (
        (combined_df['expansion_start'] >= combined_df['expansion_end']) |
        (combined_df['expansion_start'] < 30) | (combined_df['expansion_start'] > 74) |
        (combined_df['expansion_end'] < 30) | (combined_df['expansion_end'] > 74)
    )

    # Create priority score (lower = needs more review)
    combined_df['review_priority'] = 0
    combined_df.loc[manual_issues, 'review_priority'] = 1  # High priority
    combined_df.loc[combined_df['is_label_issue'] == True, 'review_priority'] = 2  # Very high priority
    combined_df.loc[manual_issues & (combined_df['is_label_issue'] == True), 'review_priority'] = 3  # Critical

    # Create CleanLab Studio format
    cleanlab_export = pd.DataFrame({
        # Sample identifier
        'id': combined_df['window_id'],

        # Original label and data
        'label': combined_df['label'],
        'expansion_start': combined_df['expansion_start'],
        'expansion_end': combined_df['expansion_end'],

        # CleanLab scores
        'label_quality_score': combined_df['label_quality'],
        'prob_consolidation': combined_df['prob_consolidation'],
        'prob_retracement': combined_df['prob_retracement'],

        # Flags
        'is_cleanlab_issue': combined_df['is_label_issue'],
        'is_manual_issue': manual_issues,
        'review_priority': combined_df['review_priority'],

        # Notes for review
        'issue_type': combined_df.apply(lambda row: get_issue_type(row, manual_issues), axis=1),
    })

    # Sort by priority (highest first)
    cleanlab_export = cleanlab_export.sort_values('review_priority', ascending=False)

    # Save full export
    full_output = output_dir / "cleanlab_studio_all_samples.csv"
    cleanlab_export.to_csv(full_output, index=False)
    print(f"\n✓ Saved all samples to {full_output}")

    # Save priority samples only (review_priority > 0)
    priority_df = cleanlab_export[cleanlab_export['review_priority'] > 0]
    priority_output = output_dir / "cleanlab_studio_priority_review.csv"
    priority_df.to_csv(priority_output, index=False)
    print(f"✓ Saved {len(priority_df)} priority samples to {priority_output}")

    # Print summary
    print(f"\n=== CleanLab Studio Export Summary ===")
    print(f"Total samples: {len(cleanlab_export)}")
    print(f"Priority samples: {len(priority_df)}")
    print(f"  CleanLab issues only: {(cleanlab_export['is_cleanlab_issue'] & ~manual_issues).sum()}")
    print(f"  Manual issues only: {(manual_issues & ~cleanlab_export['is_cleanlab_issue']).sum()}")
    print(f"  Both flags: {(cleanlab_export['is_cleanlab_issue'] & manual_issues).sum()}")

    # Show top 10 highest priority samples
    print(f"\n=== Top 10 Priority Samples ===")
    print(priority_df[['id', 'label', 'label_quality_score',
                       'is_cleanlab_issue', 'is_manual_issue', 'issue_type']].head(10).to_string(index=False))

    print(f"\n✅ CleanLab Studio export complete!")
    print(f"\nNext steps:")
    print(f"1. Upload {priority_output} to CleanLab Studio")
    print(f"2. Review and correct the {len(priority_df)} flagged samples")
    print(f"3. Download corrected labels from CleanLab Studio")
    print(f"4. Import corrections: python3 scripts/import_corrections.py")

    return priority_output


def get_issue_type(row, manual_issues_mask):
    """Determine issue type for a sample."""
    idx = row.name
    is_cleanlab = row.get('is_cleanlab_issue', False)
    is_manual = manual_issues_mask.iloc[idx] if idx < len(manual_issues_mask) else False

    if is_cleanlab and is_manual:
        return "Both: Label error + Invalid indices"
    elif is_cleanlab:
        return "Label error (CleanLab)"
    elif is_manual:
        return "Invalid expansion indices"
    else:
        return "No issues detected"


def main():
    """Export moola data for CleanLab Studio."""
    print("📤 Exporting for CleanLab Studio")
    print("=" * 50)

    export_for_cleanlab_studio()


if __name__ == "__main__":
    main()
