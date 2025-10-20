"""Apply SMOTE augmentation to balance and expand training dataset.

Loads train_clean.parquet (98 samples: 56 consolidation, 42 retracement)
Applies SMOTE to create ~300 balanced samples
Enforces OHLC constraints on synthetic samples
Saves to train_smote_300.parquet
"""
import numpy as np
import pandas as pd
from pathlib import Path
from imblearn.over_sampling import SMOTE


def fix_ohlc_constraints(features):
    """Ensure OHLC constraints: High >= max(O,C), Low <= min(O,C).

    Args:
        features: Array of shape [N, T, 4] where last dim is [Open, High, Low, Close]

    Returns:
        Fixed features array with valid OHLC relationships
    """
    features = features.copy()

    for i in range(features.shape[0]):
        for t in range(features.shape[1]):
            o, h, l, c = features[i, t]

            # Fix High: must be >= max(Open, Close)
            max_oc = max(o, c)
            if h < max_oc:
                features[i, t, 1] = max_oc

            # Fix Low: must be <= min(Open, Close)
            min_oc = min(o, c)
            if l > min_oc:
                features[i, t, 2] = min_oc

    return features


def main():
    # Paths
    input_path = Path('data/processed/train_clean.parquet')
    output_path = Path('data/processed/train_smote_300.parquet')

    print(f"Loading {input_path}")
    df = pd.read_parquet(input_path)
    print(f"Original dataset: {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")

    # Extract features and labels
    # Features are shape [N, T, F] time series, need to flatten for SMOTE
    X = np.stack([np.stack(f) for f in df['features']])  # [N, T, F]
    y = df['label'].values

    N, T, F = X.shape
    print(f"\nFeature shape: {X.shape} (N={N}, T={T}, F={F})")

    # Flatten time series for SMOTE: [N, T, F] -> [N, T*F]
    X_flat = X.reshape(N, T * F)
    print(f"Flattened for SMOTE: {X_flat.shape}")

    # Apply SMOTE with k=5 neighbors
    # Target ~300 samples total (~150 per class)
    target_count = 150  # per class
    sampling_strategy = {
        'consolidation': target_count,
        'retracement': target_count
    }

    print(f"\nApplying SMOTE with k_neighbors=5, target={target_count} per class")
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=5,
        random_state=1337
    )

    X_resampled_flat, y_resampled = smote.fit_resample(X_flat, y)
    print(f"Resampled shape: {X_resampled_flat.shape}")
    print(f"Resampled distribution:\n{pd.Series(y_resampled).value_counts()}")

    # Reshape back to time series: [N', T*F] -> [N', T, F]
    N_new = X_resampled_flat.shape[0]
    X_resampled = X_resampled_flat.reshape(N_new, T, F)
    print(f"\nReshaped back: {X_resampled.shape}")

    # Fix OHLC constraints for synthetic samples
    print("Enforcing OHLC constraints on synthetic samples...")
    X_resampled = fix_ohlc_constraints(X_resampled)

    # Verify constraints
    n_violations = 0
    for i in range(X_resampled.shape[0]):
        for t in range(X_resampled.shape[1]):
            o, h, l, c = X_resampled[i, t]
            if h < max(o, c) or l > min(o, c):
                n_violations += 1
    print(f"OHLC constraint violations after fix: {n_violations}")

    # Create new DataFrame
    # Convert features back to list of arrays format
    features_list = [X_resampled[i] for i in range(N_new)]

    # Generate synthetic window_ids for new samples
    original_ids = df['window_id'].tolist()
    n_original = len(df)
    n_synthetic = N_new - n_original

    # Original IDs + synthetic IDs
    window_ids = original_ids + [f"synthetic_{i}" for i in range(n_synthetic)]

    # Get expansion indices if they exist
    if 'expansion_start' in df.columns and 'expansion_end' in df.columns:
        # Replicate expansion indices using SMOTE indices
        exp_start = df['expansion_start'].values
        exp_end = df['expansion_end'].values

        # SMOTE generates new samples, we need to map them back
        # For now, use median expansion indices for synthetic samples
        median_start = int(np.median(exp_start))
        median_end = int(np.median(exp_end))

        expansion_start = np.concatenate([
            exp_start,
            np.full(n_synthetic, median_start)
        ])
        expansion_end = np.concatenate([
            exp_end,
            np.full(n_synthetic, median_end)
        ])
    else:
        expansion_start = None
        expansion_end = None

    # Create augmented dataframe
    df_augmented = pd.DataFrame({
        'window_id': window_ids,
        'label': y_resampled,
        'features': features_list
    })

    if expansion_start is not None:
        df_augmented['expansion_start'] = expansion_start
        df_augmented['expansion_end'] = expansion_end

    print(f"\nAugmented dataset shape: {df_augmented.shape}")
    print(f"Final class distribution:\n{df_augmented['label'].value_counts()}")

    # Save augmented dataset
    # Pyarrow has issues with nested arrays, so serialize features column
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert features to serializable format (store as nested lists)
    df_augmented['features'] = df_augmented['features'].apply(lambda x: x.tolist())

    df_augmented.to_parquet(output_path, index=False, engine='pyarrow')
    print(f"\nSaved augmented dataset to {output_path}")
    print(f"Ready for training with {len(df_augmented)} samples (3x expansion)")


if __name__ == '__main__':
    main()
