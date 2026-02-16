"""
Subject-wise CSP + SVM training and evaluation for BCI Competition IV 2a.

For each subject:
    - Load A0xT_X.npy and A0xT_y.npy from processed/bci_iv_2a
    - Train/test split within that subject
    - Fit CSP on training epochs
    - Extract CSP features
    - Standardize features
    - Train tuned SVM (small grid)
    - Report accuracy per subject

Finally:
    - Print mean and std of accuracies across subjects

Run from project root:

    python -m pipeline.train_bci2a_subjectwise_csp
"""

from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from mne.decoding import CSP


# ---------- PATHS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"


def evaluate_subject(subj_stem: str):
    """
    Evaluate CSP+SVM on a single subject.

    subj_stem: e.g. 'A01T'
    """
    X_path = PROCESSED_DIR / f"{subj_stem}_X.npy"
    y_path = PROCESSED_DIR / f"{subj_stem}_y.npy"

    if not X_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing files for subject {subj_stem}")

    X = np.load(X_path)  # (n_trials, n_channels, n_times)
    y = np.load(y_path).astype(int)

    print(f"\n=== Subject {subj_stem} ===")
    print("X shape:", X.shape, "| y shape:", y.shape)

    # Train/test split WITHIN subject
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # CSP: spatial feature extraction
    N_COMPONENTS = 6
    csp = CSP(
        n_components=N_COMPONENTS,
        reg=None,
        log=True,
        norm_trace=False,
    )

    X_train_csp = csp.fit_transform(X_train, y_train)  # (n_trials, n_components)
    X_test_csp = csp.transform(X_test)

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_csp)
    X_test_scaled = scaler.transform(X_test_csp)

    # SVM with small grid search
    param_grid = {
        "C": [0.1, 1, 10],
        "gamma": ["scale", 0.01, 0.001],
        "kernel": ["rbf"],
    }

    base_svm = SVC(probability=False, random_state=42)

    grid = GridSearchCV(
        estimator=base_svm,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring="accuracy",
        verbose=0,
    )

    grid.fit(X_train_scaled, y_train)
    best_clf = grid.best_estimator_

    y_pred = best_clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print("  Best params:", grid.best_params_)
    print(f"  Subject accuracy: {acc * 100:.2f}%")

    return acc


def main():
    # Find all subject-level files like A01T_X.npy
    subj_files = sorted(PROCESSED_DIR.glob("A??T_X.npy"))
    if not subj_files:
        raise FileNotFoundError(
            f"No subject X files found in {PROCESSED_DIR}. "
            "Expected files like A01T_X.npy, A02T_X.npy, ..."
        )

    subj_stems = [p.stem.replace("_X", "") for p in subj_files]

    all_acc = []

    print("Found subjects:", subj_stems)

    for subj in subj_stems:
        acc = evaluate_subject(subj)
        all_acc.append(acc)

    all_acc = np.array(all_acc)
    mean_acc = all_acc.mean() * 100.0
    std_acc = all_acc.std() * 100.0

    print("\n=== SUMMARY (Subject-wise CSP + SVM) ===")
    for subj, acc in zip(subj_stems, all_acc):
        print(f"{subj}: {acc * 100:.2f}%")
    print(f"\nMean accuracy: {mean_acc:.2f}%")
    print(f"Std deviation: {std_acc:.2f}%")


if __name__ == "__main__":
    main()
