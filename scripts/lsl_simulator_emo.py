"""
Simulates emotion feature stream via LSL (DEAP dataset).
live_demo  → balanced cycling through all 4 emotion quadrants
scenarios  → forced to the correct quadrant using real DEAP labels
"""
import time
import json
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from pathlib import Path

FS            = 1
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "deap"

SCENARIO_QUAD = {
    "sleep_mode":     0,   # sad/fatigued
    "emergency_test": 1,   # stressed/anxious
    "movie_night":    2,   # calm/content
}

def main():
    print("Loading DEAP Emotion features...")
    feat_path  = PROCESSED_DIR / "deap_features_X.npy"
    label_path = PROCESSED_DIR / "deap_labels_quadrant.npy"

    if not feat_path.exists():
        print(f"Missing: {feat_path}")
        return

    X      = np.load(str(feat_path),  mmap_mode='r')
    labels = np.load(str(label_path), mmap_mode='r').astype(int)

    # Build per-quadrant index pools from real labels
    idx_per_quad = {q: np.where(labels == q)[0] for q in range(4)}
    for q, pool in idx_per_quad.items():
        print(f"  Quadrant {q}: {len(pool)} samples")

    n_trials, n_feats = X.shape
    info   = StreamInfo('NeuroFusion_Emo', 'EmotionFeatures',
                        n_feats, FS, 'float32', 'nf_emo_sim')
    outlet = StreamOutlet(info)
    print(f"✅ LSL stream 'NeuroFusion_Emo' started — {n_feats} features @ {FS} Hz")
    print("Ctrl+C to stop.\n")

    cfg_path     = PROJECT_ROOT / "runtime" / "config.json"
    cfg_tick     = 0
    cur_scenario = "live_demo"

    # live_demo: cycle through quadrants every 8 seconds
    cycle_quad  = 0
    cycle_count = 0
    CYCLE_EVERY = 8   # seconds (= ticks at 1 Hz)

    try:
        while True:
            # Read config every 3 ticks
            if cfg_tick % 3 == 0:
                try:
                    if cfg_path.exists():
                        cur_scenario = json.loads(cfg_path.read_text()).get("scenario", "live_demo")
                except Exception:
                    pass
            cfg_tick += 1

            forced = SCENARIO_QUAD.get(cur_scenario)

            if forced is not None:
                # Scenario active → pick only from correct quadrant
                pool = idx_per_quad.get(forced, [])
                trial_idx = int(np.random.choice(pool)) if len(pool) else np.random.randint(0, n_trials)
            else:
                # live_demo → cycle quadrants so ALL emotions are shown
                cycle_count += 1
                if cycle_count >= CYCLE_EVERY:
                    cycle_count = 0
                    cycle_quad  = (cycle_quad + 1) % 4

                pool = idx_per_quad.get(cycle_quad, [])
                trial_idx = int(np.random.choice(pool)) if len(pool) else np.random.randint(0, n_trials)

            feat = X[trial_idx].tolist()
            outlet.push_sample(feat)
            time.sleep(1.0 / FS)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()
