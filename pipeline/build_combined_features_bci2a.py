"""
Build combined features (temporal + spectral + FBCSP) per subject in one pass.
"""

from pathlib import Path
import numpy as np
from scipy.signal import welch
import mne
from mne.decoding import CSP

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"

FS = 250  # sampling rate for BCI2a (250Hz)

# Temporal windows (0–4s split)
WIN_SEC = [(0,1), (1,2), (2,3), (3,4)]

# Spectral bands
SPECTRAL_BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
}

# FBCSP bands
FBCSP_BANDS = [
    (8,12),
    (12,16),
    (16,22),
    (22,30),
]

CSP_COMPONENTS = 8


# -------------------------------------------------------
# TEMPORAL FEATURES
# -------------------------------------------------------
def temporal_features_from_epoch(epoch, fs=FS):
    feats = []
    for (a_s, b_s) in WIN_SEC:
        a = int(a_s * fs)
        b = int(b_s * fs)
        seg = epoch[:, a:b]
        var = np.var(seg, axis=1)
        logvar = np.log(var + 1e-12)
        feats.extend(logvar.tolist())
    return np.array(feats, dtype=np.float32)


# -------------------------------------------------------
# SPECTRAL FEATURES
# -------------------------------------------------------
def spectral_features_from_epoch(epoch, fs=FS):
    feats = []
    nperseg = min(512, epoch.shape[1])
    for chsig in epoch:
        freqs, psd = welch(chsig, fs=fs, nperseg=nperseg)
        for (fmin, fmax) in SPECTRAL_BANDS.values():
            mask = (freqs >= fmin) & (freqs <= fmax)
            power = np.trapz(psd[mask], freqs[mask])
            feats.append(np.log(power + 1e-12))
    return np.array(feats, dtype=np.float32)


# -------------------------------------------------------
# FBCSP FEATURES
# -------------------------------------------------------
def compute_fbcsp_features(X_subj, y_subj, sfreq=FS):
    n_trials, n_ch, n_t = X_subj.shape
    band_feats = []

    for (l_freq, h_freq) in FBCSP_BANDS:
        X_reshaped = X_subj.reshape(-1, n_t)

        Xf = mne.filter.filter_data(
            X_reshaped,
            sfreq=sfreq,
            l_freq=l_freq,
            h_freq=h_freq,
            fir_design="firwin",
            verbose="ERROR"
        )
        Xf = Xf.reshape(n_trials, n_ch, n_t)

        csp = CSP(n_components=CSP_COMPONENTS, reg=None, log=True, norm_trace=False)
        feats = csp.fit_transform(Xf, y_subj)
        band_feats.append(feats)

    return np.concatenate(band_feats, axis=1).astype(np.float32)


# -------------------------------------------------------
# SUBJECT PROCESSING
# -------------------------------------------------------
def build_for_subject(subj_stem):
    Xp = PROCESSED_DIR / f"{subj_stem}_X.npy"
    yp = PROCESSED_DIR / f"{subj_stem}_y.npy"

    if not Xp.exists():
        print(f"No X file for {subj_stem}, skipping.")
        return None, None

    X = np.load(Xp)
    y = np.load(yp).astype(int)

    n_trials = X.shape[0]
    print(f"\nProcessing {subj_stem} | Trials: {n_trials} | Shape: {X.shape}")

    # temporal
    temporal = np.vstack([temporal_features_from_epoch(X[i]) for i in range(n_trials)])

    # spectral
    spectral = np.vstack([spectral_features_from_epoch(X[i]) for i in range(n_trials)])

    # fbcsp
    fbcsp = compute_fbcsp_features(X, y)

    # combined
    combined = np.concatenate([temporal, spectral, fbcsp], axis=1)

    # save
    np.save(PROCESSED_DIR / f"{subj_stem}_combined_X.npy", combined)
    np.save(PROCESSED_DIR / f"{subj_stem}_combined_y.npy", y)

    print(f"  Saved: {subj_stem}_combined_X.npy | {combined.shape}")
    return combined, y


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
def main():
    subj_files = sorted(PROCESSED_DIR.glob("A??T_X.npy"))
    if not subj_files:
        raise FileNotFoundError("No subject files found. Run preprocess first.")

    all_X = []
    all_y = []

    for p in subj_files:
        subj_stem = p.stem.replace("_X", "")
        Xc, yc = build_for_subject(subj_stem)

        if Xc is not None:
            all_X.append(Xc)
            all_y.append(yc)

    if all_X:
        X_all = np.vstack(all_X)
        y_all = np.concatenate(all_y)
        np.save(PROCESSED_DIR / "bci2a_combined_all_X.npy", X_all)
        np.save(PROCESSED_DIR / "bci2a_combined_all_y.npy", y_all)

        print(f"\nSaved global combined features: {X_all.shape}")


if __name__ == "__main__":
    main()
