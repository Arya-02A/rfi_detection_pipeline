import numpy as np
from sklearn.model_selection import train_test_split

def process_ml_dataset(power2d, rfi_mask, patch_size=64, stride=32, threshold=0.2):
    """
    Slices the waterfall into patches and labels them for Binary Classification.
    """
    X, y = [], []
    T, F = power2d.shape

    # 1. Patch Extraction Logic
    for t in range(0, T - patch_size, stride):
        for f in range(0, F - patch_size, stride):
            # Extract local regions
            p_power = power2d[t:t+patch_size, f:f+patch_size]
            p_mask = rfi_mask[t:t+patch_size, f:f+patch_size]

            # 2. Labeling based on RFI density
            rfi_fraction = np.mean(p_mask)
            label = 1 if rfi_fraction > threshold else 0

            X.append(p_power)
            y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    # 3. Z-Score Normalization (Critical for Neural Networks)
    # Formula: (x - mean) / std
    X = (X - np.mean(X)) / (np.std(X) + 1e-8)

    # 4. Reshape for CNN (Samples, Height, Width, Channels)
    X = np.expand_dims(X, axis=-1)

    # 5. Stratified Split (Keeps the RFI/Clean ratio same in both sets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test