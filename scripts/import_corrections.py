"""Import bespoke corrections back into moola training data."""
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def import_corrections(
    corrections_path="data/corrections/moola_annotations_corrected.csv",
    original_path="data/processed/train.parquet",
    output_path="data/processed/train_corrected.parquet"
):
    """Apply bespoke corrections to training data (matching HopSketch format).

    Args:
        corrections_path: Path to corrected CSV from bespoke
        original_path: Original train.parquet
        output_path: Output path for corrected data
    """
    # Load corrections
    corrections_path = Path(corrections_path)
    if not corrections_path.exists():
        print(f"Error: Corrections file not found: {corrections_path}")
        print(f"Please annotate in bespoke first and save as {corrections_path.name}")
        return None

    corrections = pd.read_csv(corrections_path)

    # Load original data
    original_df = pd.read_parquet(original_path)

    # Apply corrections
    corrected_df = original_df.copy()
    corrections_applied = 0
    samples_skipped = []

    for idx, corr_row in corrections.iterrows():
        window_id = corr_row['window_id']

        # Find matching row in original data
        mask = corrected_df['window_id'] == window_id

        if not mask.any():
            print(f"Warning: window_id {window_id} not found in original data")
            continue

        # Skip samples marked as 'skip' in correction_type
        if corr_row.get('correction_type') == 'skip':
            print(f"Skipping window {window_id} (marked for removal)")
            corrected_df = corrected_df[~mask]
            samples_skipped.append(window_id)
            continue

        # Apply label correction if provided (not empty string and not NaN)
        if pd.notna(corr_row.get('corrected_label')) and corr_row.get('corrected_label', '') != '':
            corrected_df.loc[mask, 'label'] = corr_row['corrected_label']
            corrections_applied += 1

        # Apply expansion start correction if provided (not NaN)
        if pd.notna(corr_row.get('corrected_expansion_start')):
            corrected_df.loc[mask, 'expansion_start'] = int(corr_row['corrected_expansion_start'])
            corrections_applied += 1

        # Apply expansion end correction if provided (not NaN)
        if pd.notna(corr_row.get('corrected_expansion_end')):
            corrected_df.loc[mask, 'expansion_end'] = int(corr_row['corrected_expansion_end'])
            corrections_applied += 1

    # Validate corrected data
    print("\n=== Validating corrected data ===")
    from moola.data.load import validate_expansions
    corrected_df = validate_expansions(corrected_df)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrected_df.to_parquet(output_path, index=False)

    print(f"\n=== Import Summary ===")
    print(f"Corrections applied: {corrections_applied}")
    print(f"Samples skipped: {len(samples_skipped)}")
    print(f"Original samples: {len(original_df)}")
    print(f"Corrected samples: {len(corrected_df)}")
    print(f"Removed invalid: {len(original_df) - len(corrected_df)}")
    print(f"Output: {output_path}")
    print(f"\nNext steps:")
    print(f"1. Review the validation output above")
    print(f"2. Retrain models: python3 -m moola.cli oof --model xgb")
    print(f"3. Models will automatically use {output_path}")

    return output_path


if __name__ == "__main__":
    import_corrections()
