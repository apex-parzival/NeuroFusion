"""
Simulates an EEG headset passing through an emotion-feature extraction pipeline via LSL.
Broadcasts a new emotion feature vector from the DEAP dataset every 1 second.
"""

import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet
from pathlib import Path

# Streaming features, so we don't need 250Hz. Let's broadcast 1 updated state per second.
FS = 1 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "processed" / "deap"

def main():
    print("Loading Emotion Feature dataset for simulation...")
    data_path = PROCESSED_DIR / "deap_features_X.npy"
    
    if not data_path.exists():
        print(f"Error: Could not find {data_path}. Please download DEAP data and run the preprocessing script.")
        return

    X = np.load(str(data_path), mmap_mode='r')
    n_trials, n_feats = X.shape
    
    # Create the LSL Stream Information
    info = StreamInfo(
        name='NeuroFusion_Emo',
        type='EmotionFeatures',
        channel_count=n_feats,
        nominal_srate=FS,
        channel_format='float32',
        source_id='neurofusion_emo_sim_67890'
    )
    
    outlet = StreamOutlet(info)
    
    print(f"✅ Started LSL Stream 'NeuroFusion_Emo'. Broadcasting {n_feats} features at {FS} Hz.")
    print("Press Ctrl+C to stop the simulator.")
    
    try:
        while True:
            # Pick a random feature vector to simulate continuous affective state changes
            trial_idx = np.random.randint(0, n_trials)
            feat = X[trial_idx].tolist()
            
            outlet.push_sample(feat)
            time.sleep(1.0 / FS)
            
    except KeyboardInterrupt:
        print("\nStopping Emotion simulator.")

if __name__ == '__main__':
    main()
