"""
Preprocess BCI Competition IV 2a (motor imagery) dataset.

- Loads A01T–A09T .gdf files from data/bci_iv_2a
- Denoises with 50 Hz notch filter
- Filters EEG (8–30 Hz)
- Extracts motor imagery epochs (left, right, feet, tongue)
- Applies artifact rejection
- Converts to NumPy arrays
- Saves per-subject X, y and also combined X_all, y_all

Run from project root (NeuroFusion/) with:

    python -m pipeline.preprocess_bci2a
"""

from pathlib import Path
import numpy as np
import mne


# ---------- CONFIG ----------

# Project root = this file's parent parent (NeuroFusion/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Input .gdf files
RAW_DATA_DIR = PROJECT_ROOT / "data" / "bci_iv_2a"

# Output directory for processed arrays
OUT_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Filter settings
LO_FREQ = 8.0    # Hz
HI_FREQ = 30.0   # Hz

# Epoching window (relative to cue/event)
TMIN = 0.0       # seconds after event
TMAX = 4.0       # seconds after event

# In the GDF annotations for BCI IV 2a, motor imagery trials have
# annotation descriptions '769', '770', '771', '772'
# We tell MNE to map those description strings to integer event IDs.
MI_ANNOT_TO_EVENT_ID = {
    "769": 1,  # left hand
    "770": 2,  # right hand
    "771": 3,  # feet
    "772": 4,  # tongue
}

# Then we map those integer event IDs to ML labels 0..3
EVENT_ID_TO_LABEL = {
    1: 0,  # left hand
    2: 1,  # right hand
    3: 2,  # feet
    4: 3,  # tongue
}


# ---------- FUNCTIONS ----------

def preprocess_subject(gdf_path: Path):
    """Load one subject's .gdf, denoise, filter, epoch, and return (X, y)."""
    print(f"\n=== Processing {gdf_path.name} ===")

    # 1) Load raw data
    raw = mne.io.read_raw_gdf(gdf_path, preload=True)
    print(f"Loaded raw: {raw}")

    # 1a) Power-line denoising (India = 50 Hz)
    raw.notch_filter(freqs=[50.0], verbose="ERROR")
    print("Applied 50 Hz notch filter")

    # 2) Keep only EEG channels (ignore EOG for now)
    raw.pick_types(eeg=True, eog=False)
    print(f"After picking EEG: {raw.info['nchan']} channels")

    # 3) Apply band-pass filter (8–30 Hz)
    raw.filter(
        LO_FREQ,
        HI_FREQ,
        fir_design="firwin",
        verbose="ERROR",  # keep logs clean
    )
    print(f"Applied band-pass filter {LO_FREQ}-{HI_FREQ} Hz")

    # 4) Extract ONLY motor imagery events from annotations
    events, event_id_mi = mne.events_from_annotations(
        raw,
        event_id=MI_ANNOT_TO_EVENT_ID,
    )

    print(f"Found {len(events)} motor imagery events")
    print(f"event_id_mi (annotation desc -> event code): {event_id_mi}")

    if len(events) == 0:
        raise RuntimeError(
            "No motor imagery events found. "
            "Check that MI_ANNOT_TO_EVENT_ID matches annotation descriptions."
        )

    # 5) Create epochs around those events WITH artifact rejection
    # ------------------------------------------------------------
    # 2) Artifact rejection threshold
    # 150 µV peak-to-peak is a reasonable starting point
    reject_criteria = dict(eeg=150e-6)

    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_mi,
        tmin=TMIN,
        tmax=TMAX,
        baseline=None,
        preload=True,
        reject=reject_criteria,
        verbose="ERROR",
    )

    data = epochs.get_data()
    print(f"Epochs shape: {data.shape}")  # (n_trials, n_channels, n_times) after rejection

    # 6) Convert to NumPy arrays
    X = data  # shape = (n_trials, n_channels, n_times)

    # events[:, 2] contains the *integer* event ID (1..4)
    y_event_ids = epochs.events[:, 2]

    # Map those event IDs (1..4) to ML labels 0..3
    y = np.vectorize(EVENT_ID_TO_LABEL.get)(y_event_ids)

    print(f"X shape: {X.shape}, y shape: {y.shape}")
    unique, counts = np.unique(y, return_counts=True)
    print("Label distribution (label: count):")
    for lbl, cnt in zip(unique, counts):
        print(f"  {lbl}: {cnt}")

    return X, y


def main():
    # Find all A??T.gdf training files (A01T, A02T, ..., A09T)
    gdf_files = sorted(RAW_DATA_DIR.glob("A??T.gdf"))

    if not gdf_files:
        raise FileNotFoundError(
            f"No .gdf files found in {RAW_DATA_DIR}. "
            f"Make sure A01T.gdf, A02T.gdf, ... are placed there."
        )

    all_X = []
    all_y = []

    for gdf_path in gdf_files:
        subj_id = gdf_path.stem  # e.g. "A01T"
        X, y = preprocess_subject(gdf_path)

        # Save per-subject arrays
        subj_out_X = OUT_DIR / f"{subj_id}_X.npy"
        subj_out_y = OUT_DIR / f"{subj_id}_y.npy"

        np.save(subj_out_X, X)
        np.save(subj_out_y, y)

        print(f"Saved {subj_out_X.name}, {subj_out_y.name} in {OUT_DIR}")

        all_X.append(X)
        all_y.append(y)

    # Concatenate across subjects
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    all_out_X = OUT_DIR / "bci2a_all_subjects_X.npy"
    all_out_y = OUT_DIR / "bci2a_all_subjects_y.npy"

    np.save(all_out_X, X_all)
    np.save(all_out_y, y_all)

    print("\n=== DONE ===")
    print(f"Combined X_all shape: {X_all.shape}")
    print(f"Combined y_all shape: {y_all.shape}")
    print(f"Saved: {all_out_X.name}, {all_out_y.name} in {OUT_DIR}")


if __name__ == "__main__":
    main()
