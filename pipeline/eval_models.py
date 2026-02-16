"""
Evaluate available models (combined, riemann, eegnet, hybrid) per subject on a held-out test split.
Writes per-model meta files containing {"acc": float}. For Keras models we save .npz meta files.
"""
from pathlib import Path
import numpy as np
import joblib
import warnings

# keras import inside try (may be heavy)
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import LabelBinarizer
except Exception:
    tf = None
    load_model = None
    LabelBinarizer = None

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_BCI = PROJECT_ROOT / "processed" / "bci_iv_2a"
PROCESSED_EMO = PROJECT_ROOT / "processed" / "deap"
MODELS_DIR = PROJECT_ROOT / "models"

def eval_combined(subj):
    Xp = PROCESSED_BCI / f"{subj}_combined_X.npy"
    yp = PROCESSED_BCI / f"{subj}_combined_y.npy"
    mdlp = MODELS_DIR / f"{subj}_combined_model.pkl"
    meta_out = MODELS_DIR / f"{subj}_combined_meta.pkl"
    if not Xp.exists() or not yp.exists() or not mdlp.exists():
        return None
    X = np.load(Xp); y = np.load(yp).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    mdl = joblib.load(mdlp)
    ypred = mdl.predict(Xte)
    acc = float((ypred == yte).mean())
    joblib.dump({"acc": acc}, meta_out)
    print(f"[EVAL] {subj} combined acc: {acc*100:.2f}%")
    return acc

def eval_riemann(subj):
    Xp = PROCESSED_BCI / f"{subj}_X.npy"
    yp = PROCESSED_BCI / f"{subj}_y.npy"
    mdlp = MODELS_DIR / f"{subj}_riemann_model.pkl"
    meta_out = MODELS_DIR / f"{subj}_riemann_meta.pkl"
    if not Xp.exists() or not yp.exists() or not mdlp.exists():
        return None
    X = np.load(Xp); y = np.load(yp).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    mdl = joblib.load(mdlp)
    ypred = mdl.predict(Xte)
    acc = float((ypred == yte).mean())
    joblib.dump({"acc": acc}, meta_out)
    print(f"[EVAL] {subj} riemann acc: {acc*100:.2f}%")
    return acc

def eval_eegnet(subj):
    # Keras model: models/eegnet/<subj>_eegnet.h5 and meta npz contains classes
    modelp = MODELS_DIR / "eegnet" / f"{subj}_eegnet.h5"
    meta_in = MODELS_DIR / "eegnet" / f"{subj}_eegnet_meta.npz"
    meta_out = MODELS_DIR / f"{subj}_eegnet_meta_eval.npz"
    Xp = PROCESSED_BCI / f"{subj}_X.npy"
    yp = PROCESSED_BCI / f"{subj}_y.npy"
    if not modelp.exists() or not meta_in.exists() or not Xp.exists() or not yp.exists():
        return None
    if load_model is None:
        warnings.warn("TensorFlow not installed; skipping eegnet eval.")
        return None
    X = np.load(Xp); y = np.load(yp).astype(int)
    n_trials, n_ch, n_t = X.shape
    X = X.reshape((n_trials, n_ch, n_t, 1)).astype(np.float32)
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])
    Xtr, Xte, ytr, yte = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    model = load_model(modelp)
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    np.savez(meta_out, acc=acc)
    print(f"[EVAL] {subj} eegnet acc: {acc*100:.2f}%")
    return float(acc)

def eval_hybrid(subj):
    # Hybrid model path / meta: models/hybrid/<subj>_hybrid.h5 ; meta npz has used_emo flag
    modelp = MODELS_DIR / "hybrid" / f"{subj}_hybrid.h5"
    meta_in = MODELS_DIR / "hybrid" / f"{subj}_hybrid_meta.npz"
    meta_out = MODELS_DIR / f"{subj}_hybrid_meta_eval.npz"
    Xp = PROCESSED_BCI / f"{subj}_X.npy"
    yp = PROCESSED_BCI / f"{subj}_y.npy"
    if not modelp.exists() or not meta_in.exists() or not Xp.exists() or not yp.exists():
        return None
    if load_model is None:
        warnings.warn("TensorFlow not installed; skipping hybrid eval.")
        return None
    meta = np.load(meta_in)
    used_emo = bool(meta.get("used_emo", False))
    Xmi = np.load(Xp); y = np.load(yp).astype(int)
    n_trials, n_ch, n_t = Xmi.shape
    Xmi_in = Xmi.reshape((n_trials, n_ch, n_t, 1)).astype(np.float32)
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.hstack([1 - Y, Y])
    if used_emo:
        emo_path = PROCESSED_EMO / f"{subj}_emo_X.npy"
        if not emo_path.exists():
            warnings.warn(f"Hybrid model for {subj} expects emotion features but {emo_path} not found; skipping eval.")
            return None
        Xemo = np.load(emo_path)
        # ensure same trials
        if Xemo.shape[0] != n_trials:
            warnings.warn(f"Emotion features length mismatch for {subj}; skipping hybrid eval.")
            return None
        tr_idx, te_idx = train_test_split(np.arange(n_trials), test_size=0.2, stratify=y, random_state=42)
        Xmi_te = Xmi_in[te_idx]; Xemo_te = Xemo[te_idx]; yte = Y[te_idx]
        model = load_model(modelp)
        loss, acc = model.evaluate([Xmi_te, Xemo_te], yte, verbose=0)
        np.savez(meta_out, acc=acc, used_emo=1)
        print(f"[EVAL] {subj} hybrid (with emo) acc: {acc*100:.2f}%")
        return float(acc)
    else:
        # EEG-only hybrid (saved as hybrid) — evaluate as EEGNet
        tr_idx, te_idx = train_test_split(np.arange(n_trials), test_size=0.2, stratify=y, random_state=42)
        Xmi_te = Xmi_in[te_idx]; yte = Y[te_idx]
        model = load_model(modelp)
        loss, acc = model.evaluate(Xmi_te, yte, verbose=0)
        np.savez(meta_out, acc=acc, used_emo=0)
        print(f"[EVAL] {subj} hybrid (EEG-only) acc: {acc*100:.2f}%")
        return float(acc)

def main():
    subj_files = sorted(PROCESSED_BCI.glob("A??T_X.npy"))
    if not subj_files:
        raise FileNotFoundError("No subject epoch files found.")
    subs = [p.stem.replace("_X","") for p in subj_files]

    results = {}
    for s in subs:
        res = {}
        a = eval_combined(s); res['combined'] = a
        b = eval_riemann(s); res['riemann'] = b
        c = eval_eegnet(s); res['eegnet'] = c
        d = eval_hybrid(s); res['hybrid'] = d
        results[s] = res

    print("\n=== EVAL SUMMARY ===")
    for s, r in results.items():
        print(s, r)

if __name__ == "__main__":
    main()
