"""
Real-time Inference Engine for NeuroFusion.
Listens to 'NeuroFusion_MI' and 'NeuroFusion_Emo' LSL streams, buffers the incoming chunks,
and feeds them to the subject-specific ML models for continuous command predictions.
"""
import sys
import time
import numpy as np
from pathlib import Path
from pylsl import resolve_stream, StreamInlet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ui_model_loader import predict_for_subject
from pipeline.realtime_emotion import load_emotion_models, emo_predict_features

MI_CLASSES = ["left", "right", "feet", "tongue"]
EMO_CLASSES = ["sad/fatigued", "stressed/anxious", "calm/content", "excited/happy"]

def apply_mi_command(label, state):
    """Update smart home state dictionary locally (for now) based on MI command."""
    if label == 0:
        state["light"] = (state.get("light", 0) + 1) % 4
        return f"💡 Lights -> {state['light']}"
    elif label == 1:
        state["fan"] = (state.get("fan", 0) + 1) % 4
        return f"🌀 Fan -> {state['fan']}"
    elif label == 2:
        state["tv"] = not state.get("tv", False)
        return f"📺 TV -> {'ON' if state['tv'] else 'OFF'}"
    else:
        state["emergency"] = True
        return "🚨 EMERGENCY!"

def apply_emo_mode(label, state):
    """Update smart home ambient state based on emotion quadrant."""
    state["ambient"] = EMO_CLASSES[label]
    return f"🎭 Ambient -> {state['ambient']}"

def main():
    print("🧠 Initializing NeuroFusion Real-time Inference Engine...")
    
    # We load A01T by default. Ideally passed via cmd args for the smart home owner's profile.
    subj_id = "A01T" 
    
    # Internal simulated state (in Phase 3, this will be pushed to an IoT broker)
    state = {"light": 0, "fan": 0, "tv": False, "emergency": False, "ambient": "neutral"}
    
    print("Loading Emotion classifiers...")
    emo_models = load_emotion_models()
    print(f"Loading Base Motor Imagery Models for {subj_id} will happen dynamically on first inference.")

    print("\n📡 Looking for Motor Imagery LSL stream ('NeuroFusion_MI')...")
    mi_streams = resolve_stream('name', 'NeuroFusion_MI')
    mi_inlet = StreamInlet(mi_streams[0])
    
    print("📡 Looking for Emotion LSL stream ('NeuroFusion_Emo')...")
    emo_streams = resolve_stream('name', 'NeuroFusion_Emo')
    emo_inlet = StreamInlet(emo_streams[0])
    
    print("\n✅ Connected to all active Brain Streams! Beginning continuous inference loop.\n")
    print("-" * 50)
    
    fs = int(mi_inlet.info().nominal_srate()) # usually 250 Hz
    epoch_samples = 4 * fs # 1000 samples required for 4-second BCI epoch
    
    # Buffer to hold incoming temporal stream
    mi_buffer = []

    try:
        while True:
            # 1. Pull Motor Imagery Data Chunk
            chunk, timestamps = mi_inlet.pull_chunk(timeout=0.0)
            if chunk:
                mi_buffer.extend(chunk)
                
            # 2. Pull Emotion Data Chunk
            # The emotion stream handles vectors per trial (~1Hz), so we don't need heavy buffering.
            emo_chunk, emo_ts = emo_inlet.pull_chunk(timeout=0.0)
            if emo_chunk:
                # Process the latest available emotion vector
                latest_emo_feat = np.array(emo_chunk[-1])
                emo_pred, emo_prob = emo_predict_features(latest_emo_feat, emo_models)
                
                action = apply_emo_mode(emo_pred, state)
                print(f"[AFFECTIVE] {action} | Raw Pred: {EMO_CLASSES[emo_pred]}")
                
            # 3. Perform MI Inference when Buffer reaches 4 seconds (1000 samples)
            if len(mi_buffer) >= epoch_samples:
                # Transpose chunk from (samples, channels) back to (channels, samples) for ML models
                epoch_data = np.array(mi_buffer[:epoch_samples]).T
                
                try:
                    # Pass the epoch to the subject-specific inference loader
                    mi_prob, mi_label = predict_for_subject(subj_id, epoch=epoch_data)
                    
                    # Normalize predictions output (model variance fix)
                    if not isinstance(mi_label, (int, np.integer)):
                        mi_label, mi_prob = mi_prob, mi_label
                        
                    action = apply_mi_command(int(mi_label), state)
                    print(f"[MOTOR]     {action}  | Raw Pred: {MI_CLASSES[int(mi_label)]}")
                    
                except Exception as e:
                    print(f"[ERROR] MI Prediction Failed: {e}")
                    
                # Sliding Window Strategy: We keep the last 500 samples (2 seconds) 
                # so the BCI evaluates every 2 seconds with a 50% overlap of data!
                slide_offset = int(epoch_samples / 2)
                mi_buffer = mi_buffer[-slide_offset:]

            # Small sleep to prevent pinning CPU at 100%
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping Inference Engine.")

if __name__ == "__main__":
    main()
