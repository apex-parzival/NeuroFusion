"""
Train an emotion classifier on DEAP band-power features.

Inputs:
    processed/deap/deap_features_X.npy        # (N_trials, n_features)
    processed/deap/deap_labels_quadrant.npy   # (N_trials,) 0–3

Outputs:
    models/emotion_svm_classifier.pkl
    Prints accuracy, classification report, confusion matrix
"""

from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed" / "deap"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LOAD FEATURES ----------
X = np.load(DATA_DIR / "deap_features_X.npy")
y = np.load(DATA_DIR / "deap_labels_quadrant.npy")

print("Loaded DEAP features:")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Class distribution:", np.bincount(y))

# ---------- TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# ---------- SCALING ----------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- HYPERPARAMETER TUNING ----------
param_grid = {
    "C": [0.1, 1, 10],
    "gamma": ["scale", 0.1, 0.01],
    "kernel": ["rbf", "linear"],
}

base_svm = SVC(probability=True, random_state=42)

grid = GridSearchCV(
    estimator=base_svm,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy",
    verbose=1,
)

print("\nRunning GridSearchCV for emotion classifier...")
grid.fit(X_train_scaled, y_train)

print("\nBest parameters found:")
print(grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_ * 100:.2f}%")

best_clf = grid.best_estimator_

# ---------- EVALUATE ON TEST SET ----------
y_pred = best_clf.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\n=== EMOTION CLASSIFIER RESULTS ===")
print(f"Test Accuracy: {acc * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

# ---------- SAVE MODEL + SCALER ----------
model_path = MODEL_DIR / "emotion_svm_classifier.pkl"
scaler_path = MODEL_DIR / "emotion_scaler.pkl"

joblib.dump(best_clf, model_path)
joblib.dump(scaler, scaler_path)

print(f"\nSaved emotion classifier to: {model_path}")
print(f"Saved scaler to: {scaler_path}")
print("\nDone.")
