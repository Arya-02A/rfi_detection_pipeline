import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def extract_features(X_patches):
    """Vectorized feature extraction for speed."""
    # Reshape X to (N, 4096) for faster stats
    X_flat = X_patches.reshape(X_patches.shape[0], -1)
    
    feats = pd.DataFrame({
        "mean": np.mean(X_flat, axis=1),
        "std": np.std(X_flat, axis=1),
        "max": np.max(X_flat, axis=1),
        "energy": np.sum(X_flat**2, axis=1),
        "skew": skew(X_flat, axis=1),
        "kurtosis": kurtosis(X_flat, axis=1),
        # Variance along axes (Time vs Freq)
        "time_var": np.mean(np.var(X_patches, axis=2), axis=1).flatten(),
        "freq_var": np.mean(np.var(X_patches, axis=1), axis=1).flatten()
    })
    return feats

def train_rfi_model(X, y_binary):
    """Full pipeline: Labeling -> Feature Extraction -> Training."""
    # 1. Create Multiclass Labels
    y_multi = []
    for i in range(len(X)):
        patch = X[i, :, :, 0]
        if y_binary[i] == 0:
            y_multi.append(0) # Clean
        else:
            t_var = np.var(patch, axis=1).mean()
            f_var = np.var(patch, axis=0).mean()
            y_multi.append(1 if f_var > t_var else 2) # 1: Narrow, 2: Broad
    
    y_multi = np.array(y_multi)

    # 2. Extract Features
    features_df = extract_features(X)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        features_df, y_multi, test_size=0.2, stratify=y_multi, random_state=42
    )

    # 4. Pipeline with Scaler + RF
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    accuracy = pipeline.score(X_test, y_test)
    
    return pipeline, accuracy, features_df.columns

def build_overlay_map(predictions, full_shape, patch_t=64, patch_f=64, stride_t=32, stride_f=32):
    T, F = full_shape
    overlay = np.zeros((T, F))
    count_map = np.zeros((T, F)) + 1e-8 # Avoid division by zero
    
    idx = 0
    for t in range(0, T - patch_t, stride_t):
        for f in range(0, F - patch_f, stride_f):
            if idx < len(predictions):
                overlay[t:t+patch_t, f:f+patch_f] += predictions[idx]
                count_map[t:t+patch_t, f:f+patch_f] += 1
                idx += 1
    return overlay / count_map