"""
Compare SVM performance on different feature types:

- Temporal features
- Spectral features
- FBCSP (spatio-spectral)
"""

from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PROCESSED_DIR = Path("processed/bci_iv_2a")


def train_and_print(name: str, X_file: str, y_file: str):
    X = np.load(PROCESSED_DIR / X_file)
    y = np.load(PROCESSED_DIR / y_file)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = SVC(kernel="rbf", probability=False, random_state=42)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))

    print(f"{name} accuracy: {acc * 100:.2f}%")


def main():
    train_and_print(
        "TEMPORAL",
        "bci2a_temporal_features.npy",
        "bci2a_temporal_labels.npy",
    )
    train_and_print(
        "SPECTRAL",
        "bci2a_spectral_features.npy",
        "bci2a_spectral_labels.npy",
    )
    train_and_print(
        "FBCSP",
        "bci2a_fbcsp_features.npy",
        "bci2a_fbcsp_labels.npy",
    )


if __name__ == "__main__":
    main()
