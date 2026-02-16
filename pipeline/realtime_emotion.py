"""
Realtime helpers for Emotion recognition (DEAP).

Functions:
    load_emotion_models()
    emo_predict_features(feature_vec, models)
    load_emotion_demo_features()
"""

from pathlib import Path
import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "deap"
MODEL_DIR = PROJECT_ROOT / "models"


def load_emotion_models():
    """
    Load scaler and emotion classifier.

    Returns:
        dict with keys: 'scaler', 'clf'
    """
    scaler_path = MODEL_DIR / "emotion_scaler.pkl"
    clf_path = MODEL_DIR / "emotion_svm_classifier.pkl"

    scaler = joblib.load(scaler_path)
    clf = joblib.load(clf_path)

    return {"scaler": scaler, "clf": clf}


def emo_predict_features(feature_vec, models):
    """
    Predict emotion quadrant for a single feature vector.

    Args:
        feature_vec: np.ndarray, shape (n_features,)
        models: dict from load_emotion_models()

    Returns:
        pred_label: int in {0,1,2,3}
        proba: np.ndarray, shape (n_classes,)
    """
    scaler = models["scaler"]
    clf = models["clf"]

    X = feature_vec[np.newaxis, :]  # (1, n_features)
    X_scaled = scaler.transform(X)
    proba = clf.predict_proba(X_scaled)[0]
    pred = int(np.argmax(proba))
    return pred, proba


def load_emotion_demo_features():
    """
    Load all DEAP features and labels for demo.

    Returns:
        X_feats, y_quadrant
    """
    X_feats = np.load(PROCESSED_DIR / "deap_features_X.npy")
    y_quad = np.load(PROCESSED_DIR / "deap_labels_quadrant.npy")
    return X_feats, y_quad
