"""
UI model loader supporting:
- combined (joblib .pkl pipeline)
- riemann (joblib .pkl pipeline accepting 3D arrays)
- eegnet (.h5 Keras model)
- hybrid (.h5 Keras model) with meta npz saying used_emo flag

Exports:
- load_models() -> mapping
- predict_for_subject(subj, epoch, emo_vector=None) -> (probs, label)
"""
from pathlib import Path
import json, joblib, numpy as np
import warnings

# load tensorflow lazily
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MAPPING = MODELS_DIR / "best_model_per_subject.json"

_models_cache = {}
_mapping_cache = None
_keras_cache = {}  # subj -> keras_model
_hybrid_meta_cache = {}  # subj -> npz meta (used_emo flag)

def _load_mapping():
    global _mapping_cache
    if _mapping_cache is None:
        if not MAPPING.exists():
            raise FileNotFoundError("best_model_per_subject.json not found. Run export_best_all_models.py")
        _mapping_cache = json.loads(MAPPING.read_text())
    return _mapping_cache

def get_available_subjects():
    """Return list of subject IDs available in the mapping."""
    mapping = _load_mapping()
    return sorted(list(mapping.keys()))

def load_specific_model(subj):
    """Load ONLY the model for the given subject."""
    if subj in _models_cache:
        return _models_cache[subj]
    
    mapping = _load_mapping()
    if subj not in mapping:
        raise KeyError(f"Subject {subj} not found in model mapping")
        
    info = mapping[subj]
    model_path = PROJECT_ROOT / info["model_path"]
    suffix = model_path.suffix.lower()
    
    if suffix == ".pkl":
        model = joblib.load(model_path)
        mtype = info["best"]
        entry = {"best": mtype, "model": model, "model_type": mtype, "path": str(model_path)}
    elif suffix == ".h5":
        if load_model is None:
            raise RuntimeError("TensorFlow not available in this environment; cannot load .h5 models")
        # Check cache for keras model to avoid re-loading shared objects if any
        if subj in _keras_cache:
            ker = _keras_cache[subj]
        else:
            ker = load_model(model_path)
            _keras_cache[subj] = ker
            
        mtype = info["best"]
        entry = {"best": mtype, "model": ker, "model_type": mtype, "path": str(model_path)}
        
        # try to load hybrid meta if exists
        hybrid_meta = (PROJECT_ROOT / "models" / "hybrid" / f"{subj}_hybrid_meta.npz")
        if hybrid_meta.exists():
            try:
                _hybrid_meta_cache[subj] = np.load(hybrid_meta)
            except Exception:
                pass
    else:
        raise RuntimeError(f"Unsupported model suffix: {suffix}")
        
    _models_cache[subj] = entry
    return entry

def load_models():
    """Deprecated: Loads ALL models. Use load_specific_model instead."""
    mapping = _load_mapping()
    out = {}
    for subj in mapping:
        out[subj] = load_specific_model(subj)
    return out

# small helpers for combined-feature building if needed (copied from previous)
from scipy.signal import welch
import mne
from mne.decoding import CSP

FS = 250
WIN_SEC = [(0,1),(1,2),(2,3),(3,4)]
SPECTRAL_BANDS = {"theta":(4,8),"alpha":(8,13),"beta":(13,30)}
FBCSP_BANDS = [(8,12),(12,16),(16,22),(22,30)]
CSP_COMPONENTS = 8  # ensure matches training
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"
_csp_cache = {}

def temporal_features_from_epoch(epoch, fs=FS):
    feats=[]
    for (a_s,b_s) in WIN_SEC:
        a=int(a_s*fs); b=int(b_s*fs)
        seg=epoch[:,a:b]; var=np.var(seg,axis=1)
        feats.extend(np.log(var+1e-12).tolist())
    return np.array(feats,dtype=np.float32)

def spectral_features_from_epoch(epoch, fs=FS):
    feats=[]
    nperseg=min(512, epoch.shape[1])
    for chsig in epoch:
        freqs, psd = welch(chsig, fs=fs, nperseg=nperseg)
        for (fmin,fmax) in SPECTRAL_BANDS.values():
            mask=(freqs>=fmin)&(freqs<=fmax)
            power=np.trapz(psd[mask], freqs[mask]) if mask.any() else 0.0
            feats.append(np.log(power+1e-12))
    return np.array(feats,dtype=np.float32)

def _fit_csp_per_subject(subj):
    if subj in _csp_cache and _csp_cache[subj].get("fitted"):
        return
    Xp=PROCESSED_DIR / f"{subj}_X.npy"
    yp=PROCESSED_DIR / f"{subj}_y.npy"
    if not Xp.exists() or not yp.exists():
        raise FileNotFoundError(f"Processed subject data not found for {subj}")
    X_all=np.load(Xp); y_all=np.load(yp).astype(int)
    n_trials,n_ch,n_t = X_all.shape
    csp_per_band=[]
    for (l_freq,h_freq) in FBCSP_BANDS:
        X_reshaped = X_all.reshape(-1, n_t)
        Xf = mne.filter.filter_data(X_reshaped, sfreq=FS, l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")
        Xf = Xf.reshape(n_trials, n_ch, n_t)
        csp = CSP(n_components=CSP_COMPONENTS, reg='ledoit_wolf', log=True, norm_trace=False)
        csp.fit(Xf, y_all)
        csp_per_band.append(csp)
    _csp_cache[subj] = {"csp_per_band": csp_per_band, "fitted": True}

def fbcsp_features_from_epoch_with_fitted_csp(epoch, subj):
    if subj not in _csp_cache or not _csp_cache[subj].get("fitted"):
        _fit_csp_per_subject(subj)
    feats_bands=[]
    n_trials=1
    n_ch,n_t = epoch.shape
    for idx,(l_freq,h_freq) in enumerate(FBCSP_BANDS):
        arr = epoch.reshape(-1,n_t)
        Xf = mne.filter.filter_data(arr, sfreq=FS, l_freq=l_freq, h_freq=h_freq, fir_design="firwin", verbose="ERROR")
        Xf = Xf.reshape(n_trials, n_ch, n_t)
        csp = _csp_cache[subj]["csp_per_band"][idx]
        feats = csp.transform(Xf)
        feats_bands.append(feats)
    return np.concatenate(feats_bands, axis=1).astype(np.float32).reshape(-1)

def compute_combined_features_from_epoch(epoch, subj):
    temp = temporal_features_from_epoch(epoch)
    spec = spectral_features_from_epoch(epoch)
    fbcsp = fbcsp_features_from_epoch_with_fitted_csp(epoch, subj)
    combined = np.concatenate([temp, spec, fbcsp], axis=0)
    return combined

# ---------------- prediction ----------------
def predict_for_subject(subj, epoch=None, emo_vector=None, combined_features=None):
    """
    Returns (probs, label).
    - subj: 'A01T' etc.
    - epoch: 2D array (n_ch, n_t)
    - emo_vector: 1D array of emotion features (if hybrid used_emo True)
    - combined_features: optional 1D array (precomputed)
    """
    """
    predict_for_subject(subj, epoch=None, emo_vector=None, combined_features=None)
    """
    # Load just this subject's model
    entry = load_specific_model(subj)
    model = entry["model"]
    mtype = entry["model_type"]

    if mtype in ("combined", "riemann"):
        # combined expects 2D features; riemann expects 3D epoch arrays
        if mtype == "combined":
            if combined_features is None:
                if epoch is None:
                    raise ValueError("combined model requires combined_features or epoch")
                combined_features = compute_combined_features_from_epoch(epoch, subj)
            X = np.asarray(combined_features).reshape(1, -1)
            probs = model.predict_proba(X)[0]
            label = int(np.argmax(probs))
            return probs, label
        else:  # riemann
            if epoch is None:
                raise ValueError("riemann model requires epoch")
            arr = np.asarray(epoch)
            if arr.ndim == 2:
                arr = arr.reshape(1, arr.shape[0], arr.shape[1])
            probs = model.predict_proba(arr)[0]
            label = int(np.argmax(probs))
            return probs, label

    elif mtype == "eegnet":
        # keras model expects (1, n_ch, n_t, 1)
        if load_model is None:
            raise RuntimeError("Keras not available in this environment")
        if epoch is None:
            raise ValueError("EEGNet model requires epoch input")
        arr = np.asarray(epoch).astype(np.float32)
        if arr.ndim == 2:
            arr = arr.reshape(1, arr.shape[0], arr.shape[1], 1)
        else:
            arr = arr.reshape(1, arr.shape[1], arr.shape[2], 1)
        probs = model.predict(arr, verbose=0)[0]
        label = int(np.argmax(probs))
        return probs, label

    elif mtype == "hybrid":
        if load_model is None:
            raise RuntimeError("Keras not available in this environment")
        # check hybrid meta to see whether model expects emotion vector
        meta = _hybrid_meta_cache.get(subj)
        used_emo = bool(meta.get("used_emo", 0)) if meta is not None else False
        if used_emo:
            if epoch is None or emo_vector is None:
                raise ValueError("Hybrid model requires both epoch and emo_vector (used_emo=True)")
            arr = np.asarray(epoch).astype(np.float32)
            if arr.ndim == 2:
                arr = arr.reshape(1, arr.shape[0], arr.shape[1], 1)
            emo = np.asarray(emo_vector).reshape(1, -1).astype(np.float32)
            probs = model.predict([arr, emo], verbose=0)[0]
            label = int(np.argmax(probs))
            return probs, label
        else:
            # EEG-only hybrid: behave like eegnet
            if epoch is None:
                raise ValueError("Hybrid EEG-only model requires epoch")
            arr = np.asarray(epoch).astype(np.float32)
            if arr.ndim == 2:
                arr = arr.reshape(1, arr.shape[0], arr.shape[1], 1)
            probs = model.predict(arr, verbose=0)[0]
            label = int(np.argmax(probs))
            return probs, label

    else:
        raise RuntimeError(f"Unknown model_type: {mtype}")
