"""
Train subject-wise combined-feature SVM, save best model per subject.
Saves models to: models/<SUBJ>_combined_model.pkl
Also writes per-subject accuracy to models/combined_results.npy
"""
from pathlib import Path
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def evaluate_and_save(subj):
    Xp = PROCESSED_DIR / f"{subj}_combined_X.npy"
    yp = PROCESSED_DIR / f"{subj}_combined_y.npy"
    if not Xp.exists() or not yp.exists():
        print(f"[SKIP] Missing combined features for {subj}")
        return None
    X = np.load(Xp); y = np.load(yp).astype(int)
    print(f"[TRAIN-C] {subj} | X: {X.shape}")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=42))])
    param_grid = {"clf__C":[0.01,0.1,1,10,100], "clf__gamma":["scale",0.1,0.01,0.001], "clf__kernel":["rbf"]}
    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=-1, scoring="accuracy", verbose=0)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    ypred = best.predict(Xte)
    acc = accuracy_score(yte, ypred)
    print(f"  {subj} Combined best: {grid.best_params_} Acc: {acc*100:.2f}%")
    # save model
    joblib.dump(best, MODELS_DIR / f"{subj}_combined_model.pkl")
    # save test split indices for reproducible eval (optional)
    joblib.dump({"test_idx": list(range(len(yte))), "acc": acc}, MODELS_DIR / f"{subj}_combined_meta.pkl")
    return acc

def main():
    subj_files = sorted(PROCESSED_DIR.glob("A??T_combined_X.npy"))
    if not subj_files:
        raise FileNotFoundError("No combined features found. Run builder first.")
    subs = [p.stem.replace("_combined_X","") for p in subj_files]
    accs = []
    for s in subs:
        a = evaluate_and_save(s)
        if a is not None:
            accs.append(a)
    accs = np.array(accs)
    np.save(MODELS_DIR / "combined_results.npy", accs)
    print("\n=== COMBINED SUMMARY ===")
    for s,a in zip(subs, accs):
        print(f"{s}: {a*100:.2f}%")
    print(f"Mean accuracy: {accs.mean()*100:.2f}%")
    print(f"Std deviation: {accs.std()*100:.2f}%")

if __name__ == "__main__":
    main()
