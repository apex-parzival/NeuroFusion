"""
Realtime helpers for Motor Imagery (BCI2a).

Functions:
    load_mi_models()
    mi_predict_epoch(epoch, models)
"""

from pathlib import Path
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"
MODEL_DIR = PROJECT_ROOT / "models"


def load_mi_models():
    """
    Load CSP, scaler, and MI classifier.

    Returns:
        dict with keys: 'csp', 'scaler', 'clf'
    """
    csp_path = MODEL_DIR / "csp_filters.pkl"
    scaler_path = MODEL_DIR / "csp_scaler.pkl"
    # use tuned model name if you saved it with a different filename
    clf_path = MODEL_DIR / "mi_svm_classifier_tuned.pkl"
    if not clf_path.exists():
        # fallback to baseline name
        clf_path = MODEL_DIR / "mi_svm_classifier.pkl"

    csp = joblib.load(csp_path)
    scaler = joblib.load(scaler_path)
    clf = joblib.load(clf_path)

    return {"csp": csp, "scaler": scaler, "clf": clf}


def mi_predict_epoch(epoch, models):
    """
    Predict motor imagery class for a single epoch.

    Args:
        epoch: np.ndarray, shape (n_channels, n_times)
        models: dict from load_mi_models()

    Returns:
        pred_label: int in {0,1,2,3}
        proba: np.ndarray, shape (n_classes,) with probabilities
    """
    csp = models["csp"]
    scaler = models["scaler"]
    clf = models["clf"]

    # CSP expects shape (n_trials, n_channels, n_times)
    X = epoch[np.newaxis, :, :]  # (1, C, T)

    X_csp = csp.transform(X)     # (1, n_components)
    X_scaled = scaler.transform(X_csp)

    proba = clf.predict_proba(X_scaled)[0]  # (n_classes,)
    pred = int(np.argmax(proba))
    return pred, proba


def load_mi_demo_epochs(split="test"):
    """
    Load some demo epochs and labels from precomputed CSP split.

    For now, we use the same train/test split as CSP features,
    but we will go back to raw epochs later if needed.

    Returns:
        X_epochs, y_labels
    """
    # We originally saved only CSP features as train/test, but the raw
    # epochs are all in bci2a_all_subjects_X.npy, y.npy.
    # For simple simulation, we just sample epochs from the full set.
    X_all = np.load(PROCESSED_DIR / "bci2a_all_subjects_X.npy")
    y_all = np.load(PROCESSED_DIR / "bci2a_all_subjects_y.npy")
    return X_all, y_all
