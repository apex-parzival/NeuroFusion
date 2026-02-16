"""
Train a tuned motor imagery classifier on CSP features from BCI Competition IV 2a.

Performs:
- GridSearchCV over SVM hyperparameters
- Train on best params
- Evaluate on held-out test set
- Save best model to models/mi_svm_classifier_tuned.pkl
"""

from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import joblib

# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- LOAD CSP FEATURES ----------
X_train = np.load(DATA_DIR / "csp_features_train_X.npy")
y_train = np.load(DATA_DIR / "csp_features_train_y.npy")
X_test = np.load(DATA_DIR / "csp_features_test_X.npy")
y_test = np.load(DATA_DIR / "csp_features_test_y.npy")

print("Loaded CSP features:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)

# ---------- HYPERPARAMETER GRID ----------
param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.1, 0.01, 0.001],
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

print("\nRunning GridSearchCV over SVM hyperparameters...")
grid.fit(X_train, y_train)

print("\nBest parameters found:")
print(grid.best_params_)
print(f"Best CV accuracy: {grid.best_score_ * 100:.2f}%")

best_clf = grid.best_estimator_

# ---------- EVALUATE ON TEST SET ----------
y_pred = best_clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n=== TUNED MOTOR IMAGERY CLASSIFIER RESULTS ===")
print(f"Test Accuracy: {acc * 100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test, y_pred, digits=4))

print("Confusion matrix (rows = true, cols = predicted):")
print(confusion_matrix(y_test, y_pred))

# ---------- SAVE BEST MODEL ----------
model_path = MODEL_DIR / "mi_svm_classifier_tuned.pkl"
joblib.dump(best_clf, model_path)

print(f"\nSaved tuned classifier to: {model_path}")
print("\nDone.")
