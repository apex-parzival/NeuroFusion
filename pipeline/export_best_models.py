"""
Compare saved combined vs riemann model results and pick best per subject.
Writes: models/best_model_per_subject.json
Also copies best model files into models/best_models/<SUBJ>_best.pkl
"""
from pathlib import Path
import json, shutil
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
BEST_DIR = MODELS_DIR / "best_models"
BEST_DIR.mkdir(exist_ok=True)

# list subjects with combined model first
combined = sorted([p.name for p in MODELS_DIR.glob("A??T_combined_model.pkl")])
riemann = sorted([p.name for p in MODELS_DIR.glob("A??T_riemann_model.pkl")])

subjects = sorted({name.split("_")[0] for name in combined+riemann})

mapping = {}
for subj in subjects:
    comb_path = MODELS_DIR / f"{subj}_combined_model.pkl"
    riem_path = MODELS_DIR / f"{subj}_riemann_model.pkl"
    comb_acc = None; riem_acc = None
    # meta files may have acc values
    comb_meta = MODELS_DIR / f"{subj}_combined_meta.pkl"
    riem_meta = MODELS_DIR / f"{subj}_riemann_meta.pkl"
    try:
        import joblib
        if comb_meta.exists():
            comb_acc = joblib.load(comb_meta).get("acc")
        if riem_meta.exists():
            riem_acc = joblib.load(riem_meta).get("acc")
    except Exception:
        pass

    # choose model: prefer higher acc if available, otherwise prefer combined
    chosen = None
    reason = ""
    if comb_path.exists() and riem_path.exists():
        if comb_acc is not None and riem_acc is not None:
            chosen = "combined" if comb_acc >= riem_acc else "riemann"
            reason = f"combined_acc={comb_acc:.3f},riemann_acc={riem_acc:.3f}"
        else:
            # fallback: choose combined by default
            chosen = "combined"
            reason = "no_meta,default_combined"
    elif comb_path.exists():
        chosen = "combined"; reason = "only_combined"
    elif riem_path.exists():
        chosen = "riemann"; reason = "only_riemann"
    else:
        continue

    src = MODELS_DIR / f"{subj}_{chosen}_model.pkl"
    dst = BEST_DIR / f"{subj}_best.pkl"
    shutil.copyfile(src, dst)
    mapping[subj] = {"best": chosen, "model_path": str(dst.relative_to(PROJECT_ROOT)), "reason": reason}

# Save mapping
with open(MODELS_DIR / "best_model_per_subject.json","w") as f:
    json.dump(mapping, f, indent=2)

print("Saved best_model_per_subject.json")
print(json.dumps(mapping, indent=2))
