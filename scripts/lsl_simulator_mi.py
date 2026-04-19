"""
Simulates EEG headset broadcasting Motor Imagery data via LSL.
Broadcasts real BCI Competition IV 2a trials at 250 Hz.
"""
import time
import json
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from pathlib import Path
from collections import deque

FS            = 250
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"

def main():
    print("Loading MI dataset...")
    data_path = PROCESSED_DIR / "bci2a_all_subjects_X.npy"
    if not data_path.exists():
        print(f"Missing: {data_path}\nRun: python scripts/reconstruct_data.py")
        return

    X = np.load(str(data_path), mmap_mode='r')
    n_trials, n_ch, n_t = X.shape
    print(f"  {n_trials} trials, {n_ch} channels, {n_t} samples each")

    info   = StreamInfo('NeuroFusion_MI', 'EEG', n_ch, FS,
                        'float32', 'nf_mi_sim')
    outlet = StreamOutlet(info)
    print(f"✅ LSL stream 'NeuroFusion_MI' started — {n_ch} ch @ {FS} Hz")
    print("Ctrl+C to stop.\n")

    EEG_BUF_FILE = PROJECT_ROOT / "runtime" / "raw_eeg_buffer.json"
    cfg_path     = PROJECT_ROOT / "runtime" / "config.json"
    raw_buffer   = deque(maxlen=500)
    write_ctr    = 0
    noise        = 0.0   # read from config

    try:
        while True:
            # Read noise from config at start of every trial (~4 s)
            try:
                if cfg_path.exists():
                    noise = float(json.loads(cfg_path.read_text()).get("noise_level", 0.0))
            except Exception:
                pass

            trial_idx = np.random.randint(0, n_trials)
            epoch     = np.array(X[trial_idx])   # (n_ch, n_t)

            # Scale noise to data amplitude so it is always visible
            if noise > 0.0:
                data_std = float(np.std(epoch)) or 1.0
                epoch    = epoch + np.random.normal(0, noise * data_std, epoch.shape).astype(np.float32)

            for i in range(n_t):
                sample = epoch[:, i].tolist()
                outlet.push_sample(sample)

                raw_buffer.append(epoch[:8, i].tolist())
                write_ctr += 1
                if write_ctr % 25 == 0:
                    try:
                        EEG_BUF_FILE.parent.mkdir(parents=True, exist_ok=True)
                        EEG_BUF_FILE.write_text(
                            json.dumps({"buffer": list(raw_buffer),
                                        "timestamp": time.time()}),
                            encoding="utf-8"
                        )
                    except Exception:
                        pass

                time.sleep(1.0 / FS)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == '__main__':
    main()
