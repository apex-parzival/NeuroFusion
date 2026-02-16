"""
Train subject-wise Riemannian pipeline (Cov -> Tangent -> SVM) and save best per subject.
Saves: models/<SUBJ>_riemann_model.pkl
"""
from pathlib import Path
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def evaluate_and_save(subj):
    Xp = PROCESSED_DIR / f"{subj}_X.npy"
    yp = PROCESSED_DIR / f"{subj}_y.npy"
    if not Xp.exists() or not yp.exists():
        print(f"[SKIP] Missing epoch files for {subj}")
        return None
    X = np.load(Xp); y = np.load(yp).astype(int)
    print(f"[TRAIN-R] {subj} | X: {X.shape}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([
        ("cov", Covariances(estimator="scm")),
        ("ts", TangentSpace()),
        ("scaler", StandardScaler()),
        ("clf", SVC(probability=True, random_state=42))
    ])
    param_grid = {"clf__C":[0.1,1,10], "clf__gamma":["scale",0.01]}
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="accuracy", verbose=0)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    ypred = best.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"  {subj} Riemannian best: {grid.best_params_} Acc: {acc*100:.2f}%")
    joblib.dump(best, MODELS_DIR / f"{subj}_riemann_model.pkl")
    joblib.dump({"acc": acc}, MODELS_DIR / f"{subj}_riemann_meta.pkl")
    return acc

def main():
    subj_files = sorted(PROCESSED_DIR.glob("A??T_X.npy"))
    subs = [p.stem.replace("_X","") for p in subj_files]
    accs = []
    for s in subs:
        a = evaluate_and_save(s)
        if a is not None:
            accs.append(a)
    accs = np.array(accs)
    np.save(MODELS_DIR / "riemann_results.npy", accs)
    print("\n=== RIEMANN SUMMARY ===")
    for s,a in zip(subs, accs):
        print(f"{s}: {a*100:.2f}%")
    print(f"Mean accuracy: {accs.mean()*100:.2f}%")
    print(f"Std deviation: {accs.std()*100:.2f}%")

if __name__ == "__main__":
    main()
