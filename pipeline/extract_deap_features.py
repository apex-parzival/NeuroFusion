"""
Extract band-power features from DEAP EEG data and create emotion labels.

Input:
    processed/deap/deap_X.npy          # (N_trials, 32, 8064)
    processed/deap/deap_valence.npy    # (N_trials,)
    processed/deap/deap_arousal.npy    # (N_trials,)

Output:
    processed/deap/deap_features_X.npy         # (N_trials, 32*5)
    processed/deap/deap_labels_valence.npy     # (N_trials,)  (continuous 1–9)
    processed/deap/deap_labels_arousal.npy     # (N_trials,)  (continuous 1–9)
    processed/deap/deap_labels_quadrant.npy    # (N_trials,)  (0–3)

Quadrant mapping (valence, arousal):
    val < 5, aro < 5  -> 0 (low valence, low arousal)  -> "sad/fatigued"
    val < 5, aro >=5  -> 1 (low valence, high arousal) -> "stressed/anxious"
    val >=5, aro < 5  -> 2 (high valence, low arousal) -> "calm/content"
    val >=5, aro >=5  -> 3 (high valence, high arousal)-> "excited/happy"
"""

from pathlib import Path
import numpy as np
from scipy.signal import welch

# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed" / "deap"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- CONSTANTS ----------
FS = 128  # Hz, DEAP preprocessed sampling rate

# Frequency bands (Hz)
BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def bandpower_welch(signal_1d, fs, fmin, fmax):
    """Compute band power using Welch method for one channel."""
    freqs, psd = welch(signal_1d, fs=fs, nperseg=fs * 2)
    # Boolean mask for freq range
    mask = (freqs >= fmin) & (freqs <= fmax)
    return np.trapz(psd[mask], freqs[mask])


def extract_features_for_trial(trial_data):
    """
    trial_data: shape (32, n_times)
    Returns feature vector shape (32 * n_bands,)
    """
    n_channels, n_times = trial_data.shape
    feat_list = []
    for ch in range(n_channels):
        sig = trial_data[ch, :]
        for (fmin, fmax) in BANDS.values():
            bp = bandpower_welch(sig, FS, fmin, fmax)
            feat_list.append(bp)
    return np.array(feat_list, dtype=np.float32)


def create_quadrant_labels(valence, arousal):
    """
    Map continuous valence (1–9) and arousal (1–9) to 4 quadrants (0–3).
    """
    val_high = valence >= 5.0
    aro_high = arousal >= 5.0

    quad = np.zeros_like(valence, dtype=int)

    # low val, high aro -> 1
    quad[(~val_high) & (aro_high)] = 1
    # high val, low aro -> 2
    quad[(val_high) & (~aro_high)] = 2
    # high val, high aro -> 3
    quad[(val_high) & (aro_high)] = 3

    return quad


def main():
    # Load combined DEAP arrays
    X = np.load(DATA_DIR / "deap_X.npy")           # (N_trials, 32, 8064)
    valence = np.load(DATA_DIR / "deap_valence.npy")  # (N_trials,)
    arousal = np.load(DATA_DIR / "deap_arousal.npy")  # (N_trials,)

    print("Loaded DEAP data:")
    print("X shape:", X.shape)
    print("valence shape:", valence.shape)
    print("arousal shape:", arousal.shape)

    n_trials, n_channels, n_times = X.shape
    n_bands = len(BANDS)
    feat_dim = n_channels * n_bands

    print(f"Extracting band-power features: {n_channels} channels * {n_bands} bands = {feat_dim} features/trial")

    X_feats = np.zeros((n_trials, feat_dim), dtype=np.float32)

    for i in range(n_trials):
        if i % 100 == 0:
            print(f"  Processing trial {i+1}/{n_trials}...")
        trial_data = X[i]  # (32, n_times)
        X_feats[i, :] = extract_features_for_trial(trial_data)

    # Quadrant labels
    quad_labels = create_quadrant_labels(valence, arousal)

    print("\n=== FEATURE SHAPES ===")
    print("X_feats:", X_feats.shape)
    print("valence:", valence.shape)
    print("arousal:", arousal.shape)
    print("quadrant:", quad_labels.shape)
    print("Quadrant distribution:", np.bincount(quad_labels))

    # Save to disk
    np.save(DATA_DIR / "deap_features_X.npy", X_feats)
    np.save(DATA_DIR / "deap_labels_valence.npy", valence)
    np.save(DATA_DIR / "deap_labels_arousal.npy", arousal)
    np.save(DATA_DIR / "deap_labels_quadrant.npy", quad_labels)

    print(f"\nSaved features and labels to {DATA_DIR}")


if __name__ == "__main__":
    main()
