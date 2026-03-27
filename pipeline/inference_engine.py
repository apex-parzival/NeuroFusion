"""
Real-time Inference Engine for NeuroFusion.
Listens to 'NeuroFusion_MI' and 'NeuroFusion_Emo' LSL streams, buffers the incoming chunks,
and feeds them to the subject-specific ML models for continuous command predictions.

Communication: Writes state to a local JSON file (runtime/state.json) that the
Streamlit dashboard polls. No external broker dependency.
"""
import sys
import time
import json
import numpy as np
from pathlib import Path
from pylsl import resolve_byprop, StreamInlet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ui_model_loader import predict_for_subject
from pipeline.realtime_emotion import load_emotion_models, emo_predict_features

MI_CLASSES = ["left", "right", "feet", "tongue"]
EMO_CLASSES = ["sad/fatigued", "stressed/anxious", "calm/content", "excited/happy"]

# --- Local File IPC ---
RUNTIME_DIR = PROJECT_ROOT / "runtime"
STATE_FILE = RUNTIME_DIR / "state.json"

def write_state_to_file(state, last_event=None):
    """Write the current smart home state to a local JSON file (atomic)."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "last_event": last_event,
        "timestamp": time.time()
    }
    # Read existing event log
    event_log = []
    try:
        if STATE_FILE.exists():
            old = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            event_log = old.get("event_log", [])
    except Exception:
        pass

    if last_event:
        from datetime import datetime
        event_log.insert(0, {
            "event": last_event,
            "time": datetime.now().strftime("%H:%M:%S"),
        })
        event_log = event_log[:30]  # keep last 30

    payload["event_log"] = event_log

    # Atomic write: write to temp then rename
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)

def apply_mi_command(label, state):
    """Update smart home state dictionary based on MI command."""
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
    
    subj_id = "A01T"
    state = {"light": 0, "fan": 0, "tv": False, "emergency": False, "ambient": "neutral"}
    
    print("Loading Emotion classifiers...")
    emo_models = load_emotion_models()
    print(f"Loading Base Motor Imagery Models for {subj_id} will happen dynamically on first inference.")

    # Write initial state so the dashboard sees something immediately
    write_state_to_file(state, last_event="🧠 Inference Engine started")
    print(f"📁 State file: {STATE_FILE}")

    print("\n📡 Looking for Motor Imagery LSL stream ('NeuroFusion_MI')...")
    mi_streams = resolve_byprop('name', 'NeuroFusion_MI')
    mi_inlet = StreamInlet(mi_streams[0])
    
    print("📡 Looking for Emotion LSL stream ('NeuroFusion_Emo')...")
    emo_streams = resolve_byprop('name', 'NeuroFusion_Emo')
    emo_inlet = StreamInlet(emo_streams[0])
    
    print("\n✅ Connected to all active Brain Streams! Beginning continuous inference loop.\n")
    print("-" * 50)
    
    fs = int(mi_inlet.info().nominal_srate())
    epoch_samples = 4 * fs
    mi_buffer = []

    try:
        while True:
            # 1. Pull Motor Imagery Data Chunk
            chunk, timestamps = mi_inlet.pull_chunk(timeout=0.0)
            if chunk:
                mi_buffer.extend(chunk)
                
            # 2. Pull Emotion Data Chunk
            emo_chunk, emo_ts = emo_inlet.pull_chunk(timeout=0.0)
            if emo_chunk:
                latest_emo_feat = np.array(emo_chunk[-1])
                emo_pred, emo_prob = emo_predict_features(latest_emo_feat, emo_models)
                
                action = apply_emo_mode(emo_pred, state)
                # Store full probability distribution
                if hasattr(emo_prob, 'tolist'):
                    state["emo_probs"] = {EMO_CLASSES[i]: round(float(p), 4) for i, p in enumerate(emo_prob)}
                else:
                    state["emo_probs"] = {EMO_CLASSES[emo_pred]: 1.0}
                log_msg = f"[AFFECTIVE] {action} | Raw Pred: {EMO_CLASSES[emo_pred]}"
                print(log_msg)
                write_state_to_file(state, last_event=log_msg)
                
            # 3. Perform MI Inference when Buffer reaches 4 seconds
            if len(mi_buffer) >= epoch_samples:
                epoch_data = np.array(mi_buffer[:epoch_samples]).T
                
                try:
                    mi_prob, mi_label = predict_for_subject(subj_id, epoch=epoch_data)
                    
                    if not isinstance(mi_label, (int, np.integer)):
                        mi_label, mi_prob = mi_prob, mi_label
                    
                    # Store full probability distribution
                    if hasattr(mi_prob, 'tolist'):
                        state["mi_probs"] = {MI_CLASSES[i]: round(float(p), 4) for i, p in enumerate(mi_prob)}
                    else:
                        state["mi_probs"] = {MI_CLASSES[int(mi_label)]: 1.0}
                        
                    action = apply_mi_command(int(mi_label), state)
                    log_msg = f"[MOTOR]     {action}  | Raw Pred: {MI_CLASSES[int(mi_label)]}"
                    print(log_msg)
                    write_state_to_file(state, last_event=log_msg)
                    
                except Exception as e:
                    print(f"[ERROR] MI Prediction Failed: {e}")
                    
                slide_offset = int(epoch_samples / 2)
                mi_buffer = mi_buffer[-slide_offset:]

            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\nStopping Inference Engine.")

if __name__ == "__main__":
    main()
