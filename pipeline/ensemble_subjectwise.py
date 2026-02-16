"""
Per-subject ensemble: Combined-feature SVM + Riemannian pipeline (soft voting).
Trains both models per subject (with small grid), builds voting ensemble, evaluates.
Run from project root:

    python -u -m pipeline.ensemble_subjectwise
"""
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# pyriemann imports
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "processed" / "bci_iv_2a"

def train_combined_svm(Xtr, ytr):
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))])
    param_grid = {
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.01],
        "clf__kernel": ["rbf"]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="accuracy")
    grid.fit(Xtr, ytr)
    return grid.best_estimator_, grid.best_params_

def train_riemann_svm(Xtr, ytr):
    # Xtr shape: (n_trials, n_ch, n_t)
    riem_pipe = Pipeline([
        ("cov", Covariances(estimator="scm")),
        ("ts", TangentSpace()),
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ])
    param_grid = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01]}
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(riem_pipe, param_grid, cv=cv, n_jobs=-1, scoring="accuracy")
    # grid.fit expects 2D X; our pipeline handles 3D input (pyriemann estimators accept 3D)
    grid.fit(Xtr, ytr)
    return grid.best_estimator_, grid.best_params_

def evaluate_subject(subj):
    print(f"\n[ENSEMBLE] Evaluating subject {subj}")
    # Combined features files:
    Xc_path = PROCESSED / f"{subj}_combined_X.npy"
    yc_path = PROCESSED / f"{subj}_combined_y.npy"
    # Raw epoch files for Riemannian:
    Xr_path = PROCESSED / f"{subj}_X.npy"
    yr_path = PROCESSED / f"{subj}_y.npy"

    if not Xc_path.exists() or not Xr_path.exists():
        print("  Missing combined or raw epoch files -- skipping")
        return None

    Xc = np.load(Xc_path); yc = np.load(yc_path).astype(int)
    Xr = np.load(Xr_path); yr = np.load(yr_path).astype(int)

    # Split indices consistently for both datasets (same trial order)
    # We'll create an index split so that the same trials are used for train/test in both feature sets
    idx = np.arange(len(yc))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=yc, random_state=42)

    Xc_tr, Xc_te = Xc[train_idx], Xc[test_idx]
    yc_tr, yc_te = yc[train_idx], yc[test_idx]

    Xr_tr, Xr_te = Xr[train_idx], Xr[test_idx]
    yr_tr, yr_te = yr[train_idx], yr[test_idx]
    # sanity
    assert np.array_equal(yc_te, yr_te)

    # Train both models
    svm_comb, p1 = train_combined_svm(Xc_tr, yc_tr)
    svm_riem, p2 = train_riemann_svm(Xr_tr, yr_tr)

    # Build voting ensemble
    # Need to wrap the riemann pipeline to accept combined X for predict_proba? No — we use separate inputs.
    # So we'll use predictions from both directly: predict_proba on their respective test inputs and average probabilities.
    prob1 = svm_comb.predict_proba(Xc_te)  # (n_test, n_classes)
    prob2 = svm_riem.predict_proba(Xr_te)

    # average probabilities (ensure same label order)
    avg_prob = (prob1 + prob2) / 2.0
    ypred = np.argmax(avg_prob, axis=1)
    acc = accuracy_score(yc_te, ypred)

    print(f"  Combined best params: {p1}")
    print(f"  Riemannian best params: {p2}")
    print(f"  Ensemble accuracy: {acc*100:.2f}%")
    return acc

def main():
    subj_files = sorted(PROCESSED.glob("A??T_X.npy"))
    subs = [p.stem.replace("_X","") for p in subj_files]
    accs = []
    for s in subs:
        a = evaluate_subject(s)
        if a is not None:
            accs.append(a)
    accs = np.array(accs)
    print("\n=== ENSEMBLE SUMMARY ===")
    for s,a in zip(subs, accs):
        print(f"{s}: {a*100:.2f}%")
    print(f"Mean accuracy: {accs.mean()*100:.2f}%")
    print(f"Std deviation: {accs.std()*100:.2f}%")

if __name__ == "__main__":
    main()
