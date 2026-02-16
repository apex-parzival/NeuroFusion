"""
Choose best model per subject based on real evaluation meta (acc).
Copies chosen model to models/best_models/<SUBJ>_best.* and writes models/best_model_per_subject.json
"""
from pathlib import Path
import json, shutil
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
BEST_DIR = MODELS_DIR / "best_models"
BEST_DIR.mkdir(parents=True, exist_ok=True)

def load_acc(path):
    # path may be joblib pkl meta (pkl) or npz
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix == ".pkl":
        try:
            import joblib
            meta = joblib.load(p)
            return float(meta.get("acc"))
        except Exception:
            return None
    if p.suffix == ".npz":
        try:
            arr = np.load(p)
            return float(arr.get("acc"))
        except Exception:
            return None
    return None

def candidate_meta(subj):
    # returns dict model_type -> (model_path, meta_path, acc)
    cand = {}
    # combined
    p_comb = MODELS_DIR / f"{subj}_combined_model.pkl"
    m_comb = MODELS_DIR / f"{subj}_combined_meta.pkl"
    if p_comb.exists():
        cand['combined'] = (p_comb, m_comb, load_acc(m_comb))
    # riemann
    p_riem = MODELS_DIR / f"{subj}_riemann_model.pkl"
    m_riem = MODELS_DIR / f"{subj}_riemann_meta.pkl"
    if p_riem.exists():
        cand['riemann'] = (p_riem, m_riem, load_acc(m_riem))
    # eegnet
    p_eeg = MODELS_DIR / "eegnet" / f"{subj}_eegnet.h5"
    m_eeg = MODELS_DIR / "eegnet" / f"{subj}_eegnet_meta_eval.npz"
    if p_eeg.exists():
        cand['eegnet'] = (p_eeg, m_eeg, load_acc(m_eeg))
    # hybrid
    p_hyb = MODELS_DIR / "hybrid" / f"{subj}_hybrid.h5"
    m_hyb = MODELS_DIR / "hybrid" / f"{subj}_hybrid_meta_eval.npz"
    if p_hyb.exists():
        cand['hybrid'] = (p_hyb, m_hyb, load_acc(m_hyb))
    return cand

def choose_best_for_subject(subj):
    cand = candidate_meta(subj)
    if not cand:
        return None
    # pick max acc where acc is not None; if all None, pick a reasonable default order
    best_type = None
    best_acc = -1.0
    for t, (p, m, acc) in cand.items():
        if acc is not None and acc > best_acc:
            best_acc = acc
            best_type = t
    if best_type is None:
        # fallback preference: combined > eegnet > hybrid > riemann
        for pref in ['combined','eegnet','hybrid','riemann']:
            if pref in cand:
                best_type = pref
                break
    chosen_path = cand[best_type][0]
    # copy to best_models
    dst = BEST_DIR / f"{subj}_best{chosen_path.suffix}"
    shutil.copyfile(chosen_path, dst)
    reason = f"picked={best_type},acc={best_acc if best_acc>=0 else 'NA'}"
    return {"best": best_type, "model_path": str(dst.relative_to(PROJECT_ROOT)), "reason": reason}

def main():
    # find subjects by looking at processed BCI files
    processed = PROJECT_ROOT / "processed" / "bci_iv_2a"
    subj_files = sorted(processed.glob("A??T_X.npy"))
    subs = [p.stem.replace("_X","") for p in subj_files]
    mapping = {}
    for s in subs:
        pick = choose_best_for_subject(s)
        if pick:
            mapping[s] = pick
    # write JSON
    (MODELS_DIR / "best_model_per_subject.json").write_text(json.dumps(mapping, indent=2))
    print("Wrote best_model_per_subject.json with mapping:")
    print(json.dumps(mapping, indent=2))

if __name__ == "__main__":
    main()
