"""
Simulated real-time BCI for NeuroFusion:

- Streams random Motor Imagery epochs (BCI2a) and DEAP emotion features
- Uses trained models to predict:
    - MI command (0: left, 1: right, 2: feet, 3: tongue)
    - Emotion quadrant (0..3)
- Updates a virtual smart home state and prints it.

Run from project root:

    python .\pipeline\realtime_simulation.py
"""

import time
import random
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import numpy as np

from pipeline.realtime_mi import load_mi_models, load_mi_demo_epochs, mi_predict_epoch
from pipeline.realtime_emotion import (
    load_emotion_models,
    load_emotion_demo_features,
    emo_predict_features,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ------- SMART HOME STATE -------

def init_smart_home_state():
    return {
        "light_brightness": 0,   # 0, 1, 2, 3
        "fan_speed": 0,          # 0, 1, 2, 3
        "tv_on": False,
        "emergency_alert": False,
        "ambient_mode": "neutral",  # 'sad', 'stressed', 'calm', 'excited'
    }


def apply_mi_command(label, proba, state):
    """
    Update smart home state based on MI label.

    label:
        0 -> left hand
        1 -> right hand
        2 -> feet
        3 -> tongue
    """
    if label == 0:
        # cycle light brightness: 0->1->2->3->0
        state["light_brightness"] = (state["light_brightness"] + 1) % 4
        action = f"Lights brightness set to {state['light_brightness']}"
    elif label == 1:
        # cycle fan speed similarly
        state["fan_speed"] = (state["fan_speed"] + 1) % 4
        action = f"Fan speed set to {state['fan_speed']}"
    elif label == 2:
        state["tv_on"] = not state["tv_on"]
        action = f"TV turned {'ON' if state['tv_on'] else 'OFF'}"
    else:  # 3
        state["emergency_alert"] = True
        action = "EMERGENCY ALERT TRIGGERED!"

    return action


def apply_emotion_quadrant(label, proba, state):
    """
    Update ambient_mode based on emotion quadrant.

    label meaning:
        0 -> sad/fatigued
        1 -> stressed/anxious
        2 -> calm/content
        3 -> excited/happy
    """
    mapping = {
        0: "sad/fatigued",
        1: "stressed/anxious",
        2: "calm/content",
        3: "excited/happy",
    }
    state["ambient_mode"] = mapping.get(label, "neutral")
    return f"Ambient mode set to: {state['ambient_mode']}"


def main():
    # Load models
    print("Loading MI models...")
    mi_models = load_mi_models()
    print("Loading Emotion models...")
    emo_models = load_emotion_models()

    # Load demo data
    print("Loading BCI2a demo epochs...")
    X_mi, y_mi = load_mi_demo_epochs()
    print("Loading DEAP demo features...")
    X_emo, y_emo = load_emotion_demo_features()

    n_mi = X_mi.shape[0]
    n_emo = X_emo.shape[0]

    print(f"MI epochs available: {n_mi}")
    print(f"Emotion feature samples available: {n_emo}")

    state = init_smart_home_state()

    print("\n=== Starting simulated NeuroFusion loop ===")
    print("Press Ctrl+C to stop.\n")

    try:
        for step in range(20):  # simulate 20 cycles; adjust as you like
            print(f"\n--- Step {step+1} ---")

            # pick random MI and Emotion samples
            idx_mi = random.randint(0, n_mi - 1)
            idx_emo = random.randint(0, n_emo - 1)

            epoch = X_mi[idx_mi]        # (C, T)
            true_mi = int(y_mi[idx_mi])

            feat = X_emo[idx_emo]       # (n_features,)
            true_emo = int(y_emo[idx_emo])

            # Predict MI
            mi_label, mi_proba = mi_predict_epoch(epoch, mi_models)
            mi_action = apply_mi_command(mi_label, mi_proba, state)

            # Predict Emotion
            emo_label, emo_proba = emo_predict_features(feat, emo_models)
            emo_action = apply_emotion_quadrant(emo_label, emo_proba, state)

            # Print summary
            mi_classes = ["left", "right", "feet", "tongue"]
            emo_classes = ["sad/fatigued", "stressed/anxious", "calm/content", "excited/happy"]

            print(f"MI TRUE: {true_mi} ({mi_classes[true_mi]}) | PRED: {mi_label} ({mi_classes[mi_label]})")
            print(f"  MI probs: {np.round(mi_proba, 3)}")
            print(f"  -> Action: {mi_action}")

            print(f"Emotion TRUE: {true_emo} ({emo_classes[true_emo]}) | PRED: {emo_label} ({emo_classes[emo_label]})")
            print(f"  Emotion probs: {np.round(emo_proba, 3)}")
            print(f"  -> Ambient: {emo_action}")

            print(f"Current SMART HOME STATE: {state}")

            time.sleep(0.5)  # small delay to feel like "time"

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")


if __name__ == "__main__":
    main()
