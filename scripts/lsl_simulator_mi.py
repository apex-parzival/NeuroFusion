"""
Simulates an EEG headset broadcasting Motor Imagery data via Lab Streaming Layer (LSL).
Instead of real hardware, this broadcasts random trials from the BCI Competition IV 2a dataset
over the local network at 250 Hz.
"""

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from pathlib import Path

FS = 250
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "bci_iv_2a"

def main():
    print("Loading MI dataset for simulation...")
    data_path = PROCESSED_DIR / "bci2a_all_subjects_X.npy"
    if not data_path.exists():
        print(f"Error: Could not find {data_path}. Please run scripts/reconstruct_data.py first.")
        return

    # Using mmap_mode 'r' to save memory
    X = np.load(str(data_path), mmap_mode='r')
    n_trials, n_ch, n_t = X.shape
    
    # Create the LSL Stream Information
    info = StreamInfo(
        name='NeuroFusion_MI',
        type='EEG',
        channel_count=n_ch,
        nominal_srate=FS,
        channel_format='float32',
        source_id='neurofusion_mi_sim_12345'
    )
    
    # Create the outlet to broadcast data
    outlet = StreamOutlet(info)
    
    print(f"✅ Started LSL Stream 'NeuroFusion_MI'. Broadcasting {n_ch} channels at {FS} Hz.")
    print("Press Ctrl+C to stop the simulator.")
    
    try:
        while True:
            # Pick a random trial to simulate continuous brain activity
            trial_idx = np.random.randint(0, n_trials)
            epoch = X[trial_idx] # shape (n_ch, n_t)
            
            # Stream it sample by sample (columns)
            for i in range(n_t):
                sample = epoch[:, i].tolist()
                outlet.push_sample(sample)
                # Sleep to match the sampling rate (1/250s = 4ms)
                time.sleep(1.0 / FS)
    except KeyboardInterrupt:
        print("\nStopping MI simulator.")

if __name__ == '__main__':
    main()
