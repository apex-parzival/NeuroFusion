"""
Extract CSP features from preprocessed BCI2a epochs.

Input:
    processed/bci_iv_2a/bci2a_all_subjects_X.npy
    processed/bci_iv_2a/bci2a_all_subjects_y.npy

Output:
    processed/bci_iv_2a/csp_features_X.npy
    processed/bci_iv_2a/csp_features_y.npy
    models/csp_filters.pkl

Then A4 will train SVM/RF using these CSP features.
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
import joblib


# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------- LOAD PREPROCESSED EEG ----------
X = np.load(DATA_DIR / "bci2a_all_subjects_X.npy")
y = np.load(DATA_DIR / "bci2a_all_subjects_y.npy")

print("Loaded preprocessed EEG:")
print("X shape:", X.shape)  # (n_trials, n_channels, n_times)
print("y shape:", y.shape)  # (n_trials,)


# ---------- TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# ---------- CSP FEATURE EXTRACTION ----------
# Number of CSP components → tune later (2–8 works)
N_COMPONENTS = 8

csp = CSP(
    n_components=N_COMPONENTS,
    reg=None,
    log=True,       # take log-variance = good for MI
    norm_trace=False
)

# Fit CSP on training data
csp.fit(X_train, y_train)

# Transform to CSP feature vectors
X_train_csp = csp.transform(X_train)
X_test_csp = csp.transform(X_test)

print("CSP feature shapes:")
print("Train:", X_train_csp.shape)  # (n_trials, N_COMPONENTS)
print("Test:", X_test_csp.shape)


# ---------- SCALING (Standardization) ----------
scaler = StandardScaler()
X_train_csp = scaler.fit_transform(X_train_csp)
X_test_csp = scaler.transform(X_test_csp)


# ---------- SAVE FEATURES ----------
np.save(DATA_DIR / "csp_features_train_X.npy", X_train_csp)
np.save(DATA_DIR / "csp_features_train_y.npy", y_train)
np.save(DATA_DIR / "csp_features_test_X.npy", X_test_csp)
np.save(DATA_DIR / "csp_features_test_y.npy", y_test)

# Save CSP model + scaler
joblib.dump(csp, MODEL_DIR / "csp_filters.pkl")
joblib.dump(scaler, MODEL_DIR / "csp_scaler.pkl")

print("\n=== CSP FEATURE EXTRACTION DONE ===")
print(f"Saved CSP features and models in:")
print(f"{DATA_DIR} and {MODEL_DIR}")
