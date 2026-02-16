"""
Preprocess DEAP preprocessed Python data for emotion recognition.

- Loads s01.dat ... s32.dat from data/deap/data_preprocessed_python
- Each file contains: 'data' (40 trials, 40 channels, 8064 samples),
  'labels' (40 trials, 4 dims: valence, arousal, dominance, liking)
- Keeps only first 32 channels (EEG), drops peripheral channels
- Stacks all subjects into one big array

Outputs in processed/deap/:
    deap_X.npy          -> shape (N_trials_total, 32, 8064)
    deap_valence.npy    -> shape (N_trials_total,)
    deap_arousal.npy    -> shape (N_trials_total,)
"""

from pathlib import Path
import pickle
import numpy as np

# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEAP_RAW_DIR = PROJECT_ROOT / "data" / "deap" / "data_preprocessed_python"
OUT_DIR = PROJECT_ROOT / "processed" / "deap"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_subject(file_path: Path):
    """Load one DEAP subject .dat file and return (X, valence, arousal)."""
    print(f"Loading {file_path.name} ...")
    with file_path.open("rb") as f:
        # Latin1 encoding is required for DEAP pickles
        subj = pickle.load(f, encoding="latin1")

    # subj["data"]: shape (40 trials, 40 channels, 8064 samples)
    # subj["labels"]: shape (40 trials, 4): [valence, arousal, dominance, liking]
    data = subj["data"]           # (40, 40, 8064)
    labels = subj["labels"]       # (40, 4)

    # Keep only 32 EEG channels (0..31)
    eeg_data = data[:, :32, :]    # (40, 32, 8064)

    valence = labels[:, 0]        # (40,)
    arousal = labels[:, 1]        # (40,)

    print(f"  Trials: {eeg_data.shape[0]}, EEG shape: {eeg_data.shape}")
    return eeg_data, valence, arousal


def main():
    subj_files = sorted(DEAP_RAW_DIR.glob("s*.dat"))
    if not subj_files:
        raise FileNotFoundError(
            f"No DEAP .dat files found in {DEAP_RAW_DIR}. "
            "Make sure s01.dat ... s32.dat are in data/deap/data_preprocessed_python"
        )

    all_X = []
    all_valence = []
    all_arousal = []

    for fpath in subj_files:
        X_subj, v_subj, a_subj = load_subject(fpath)
        all_X.append(X_subj)
        all_valence.append(v_subj)
        all_arousal.append(a_subj)

    # Stack over subjects
    X = np.vstack(all_X)                  # (N_trials_total, 32, 8064)
    valence = np.concatenate(all_valence) # (N_trials_total,)
    arousal = np.concatenate(all_arousal) # (N_trials_total,)

    print("\n=== COMBINED DEAP DATA ===")
    print("X shape:", X.shape)
    print("valence shape:", valence.shape)
    print("arousal shape:", arousal.shape)

    # Save
    np.save(OUT_DIR / "deap_X.npy", X)
    np.save(OUT_DIR / "deap_valence.npy", valence)
    np.save(OUT_DIR / "deap_arousal.npy", arousal)

    print(f"\nSaved combined DEAP arrays in {OUT_DIR}")


if __name__ == "__main__":
    main()
