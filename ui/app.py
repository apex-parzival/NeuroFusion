import sys
import random
from pathlib import Path

import numpy as np
import streamlit as st

# ---------- PATH & IMPORT FIX ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from pipeline.realtime_mi import (
        load_mi_demo_epochs,
    )
    from pipeline.realtime_emotion import (
        load_emotion_models,
        load_emotion_demo_features,
        emo_predict_features,
    )
    from pipeline.ui_model_loader import (
        get_available_subjects, 
        load_specific_model, 
        predict_for_subject
    )
    IMPORT_OK = True
except Exception as e:
    IMPORT_OK = False
    IMPORT_ERROR = e


MI_CLASSES = ["left", "right", "feet", "tongue"]
EMO_CLASSES = ["sad/fatigued", "stressed/anxious", "calm/content", "excited/happy"]

# ---------- CACHED LOADERS ----------
@st.cache_data
def get_cached_mi_epochs():
    """Cache the heavy MI epochs to avoid reloading on every rerun."""
    if not IMPORT_OK: return None, None
    return load_mi_demo_epochs()

@st.cache_data
def get_cached_emotion_features():
    """Cache the heavy Emotion features."""
    if not IMPORT_OK: return None, None
    return load_emotion_demo_features()

@st.cache_resource
def get_cached_emotion_models():
    """Cache the emotion classifier model."""
    if not IMPORT_OK: return None
    return load_emotion_models()

@st.cache_resource
def get_cached_subject_list():
    """Cache the list of available subjects."""
    if not IMPORT_OK: return []
    try:
        return get_available_subjects()
    except Exception:
        return []

# ---------- STATE HELPERS ----------
def init_state_if_needed():
    """Initialize Streamlit session state variables if not present."""
    if "smart_home" not in st.session_state:
        st.session_state.smart_home = {
            "light_brightness": 0,
            "fan_speed": 0,
            "tv_on": False,
            "emergency_alert": False,
            "ambient_mode": "neutral"
        }
    
    # Load heavy data via cache
    if "X_mi" not in st.session_state:
        X_mi, y_mi = get_cached_mi_epochs()
        if X_mi is not None:
            st.session_state.X_mi = X_mi
            st.session_state.y_mi = y_mi
            
    if "X_emo" not in st.session_state:
        X_emo, y_emo = get_cached_emotion_features()
        if X_emo is not None:
            st.session_state.X_emo = X_emo
            st.session_state.y_emo = y_emo

    if "emo_models" not in st.session_state:
        st.session_state.emo_models = get_cached_emotion_models()

    if "last_step" not in st.session_state:
        st.session_state.last_step = None
    if "event_log" not in st.session_state:
        st.session_state.event_log = []
    if "step_counter" not in st.session_state:
        st.session_state.step_counter = 0


def apply_mi_command(label, state):
    """Apply Motor Imagery command to smart home state."""
    if label == 0:
        state["light_brightness"] = (state["light_brightness"] + 1) % 4
        action = f"💡 Lights brightness → {state['light_brightness']}"
    elif label == 1:
        state["fan_speed"] = (state["fan_speed"] + 1) % 4
        action = f"🌀 Fan speed → {state['fan_speed']}"
    elif label == 2:
        state["tv_on"] = not state["tv_on"]
        action = f"📺 TV turned {'ON' if state['tv_on'] else 'OFF'}"
    else:
        state["emergency_alert"] = True
        action = "🚨 EMERGENCY ALERT TRIGGERED!"
    return action


def apply_emotion_quadrant(label, state):
    """Apply emotion quadrant to ambient mode."""
    mapping = {
        0: "sad/fatigued",
        1: "stressed/anxious",
        2: "calm/content",
        3: "excited/happy",
    }
    state["ambient_mode"] = mapping.get(label, "neutral")
    return f"🎭 Ambient mode → {state['ambient_mode']}"


# ---------- MAIN APP ----------
def main():
    st.set_page_config(
        page_title="NeuroFusion Smart Home",
        page_icon="🧠",
        layout="wide",
    )
    init_state_if_needed()

    # ----- SIDEBAR -----
    with st.sidebar:
        st.markdown("## 🧠 NeuroFusion")
        st.markdown(
            "Hybrid **Motor Imagery + Emotion** BCI controlling a simulated smart home.\n\n"
            "- 🧩 Motor Imagery: BCI Competition IV 2a\n"
            "- 🎧 Emotion: DEAP EEG dataset\n"
        )

        # Show model stats (your actual results)
        st.markdown("### 📊 Model Performance")
        st.markdown("- Motor Imagery (subject-wise combined): **77.59%**")
        st.markdown("- Riemannian (subject-wise): **69.54%**")

        st.markdown("### ℹ️ How to demo")
        st.markdown(
            "1. Select a subject in the dropdown below.\n"
            "2. Click **Next Brain Step**.\n"
            "3. Watch control + mood predictions and Smart Home state updates.\n"
        )

        # Subject selection (list models available)
        subj_options = get_cached_subject_list()
        selected_subj = st.selectbox("Choose subject (used to load best model)", subj_options, index=0 if subj_options else None)
        st.session_state.selected_subj = selected_subj

    st.title("🧠 NeuroFusion Smart Home – BCI + Emotion Control")

    # If imports failed, show error instead of blank page
    if not IMPORT_OK:
        st.error("Failed to import pipeline modules.")
        st.code(repr(IMPORT_ERROR))
        st.stop()

    smart_home = st.session_state.smart_home
    X_mi = st.session_state.X_mi
    y_mi = st.session_state.y_mi
    X_emo = st.session_state.X_emo
    y_emo = st.session_state.y_emo
    subj_id = st.session_state.get("selected_subj")

    col_main, col_state = st.columns([2.2, 1])

    # ----- LEFT: LIVE CONTROL + DETAILS -----
    with col_main:
        st.subheader("🔄 Live Brain Event Simulation")

        st.write(
            "Each step samples a random **motor imagery epoch** "
            "and a random **emotion feature vector**, then updates the smart home "
            "based on both *intention* (MI) and *affective state* (DEAP)."
        )

        if st.button("▶ Next Brain Step"):
            st.session_state.step_counter += 1

            # pick random samples
            idx_mi = random.randint(0, X_mi.shape[0] - 1)
            idx_emo = random.randint(0, X_emo.shape[0] - 1)

            epoch = X_mi[idx_mi]  # shape (n_ch, n_times)
            true_mi = int(y_mi[idx_mi])

            feat = X_emo[idx_emo]
            true_emo = int(y_emo[idx_emo])

            # predictions: MI using our new loader (auto-choose best model for chosen subject)
            try:
                mi_proba, mi_label = predict_for_subject(subj_id, epoch=epoch)
                # predict_for_subject returns (probs, label) — earlier versions used (probs,label). ensure order:
                if isinstance(mi_proba, np.ndarray) and isinstance(mi_label, (int, np.integer)):
                    pass
                else:
                    # some saved models return (label, probs) — normalize
                    mi_label, mi_proba = mi_proba, mi_label
            except Exception as e:
                st.error(f"MI prediction error: {e}")
                mi_label = 0
                mi_proba = np.zeros(len(MI_CLASSES))

            mi_action = apply_mi_command(mi_label, smart_home)

            # emotion predictor remains same helper (uses prebuilt features)
            emo_label, emo_proba = emo_predict_features(feat, st.session_state.emo_models)
            emo_action = apply_emotion_quadrant(emo_label, smart_home)

            # save last step
            step_info = {
                "step": st.session_state.step_counter,
                "true_mi": true_mi,
                "mi_label": int(mi_label),
                "mi_proba": mi_proba.tolist() if hasattr(mi_proba, "tolist") else list(mi_proba),
                "true_emo": true_emo,
                "emo_label": int(emo_label),
                "emo_proba": emo_proba.tolist() if hasattr(emo_proba, "tolist") else list(emo_proba),
                "mi_action": mi_action,
                "emo_action": emo_action,
            }
            st.session_state.last_step = step_info
            st.session_state.event_log.insert(0, step_info)  # newest first
            st.session_state.event_log = st.session_state.event_log[:20]  # keep last 20

        last = st.session_state.last_step
        if last is not None:
            st.markdown("### 🧠 Last Inference")

            c1, c2 = st.columns(2)

            with c1:
                st.markdown("**Motor Imagery (Control Channel)**")
                st.write(
                    f"True: `{last['true_mi']} ({MI_CLASSES[last['true_mi']]})`  \n"
                    f"Predicted: `{last['mi_label']} ({MI_CLASSES[last['mi_label']]})`"
                )
                st.bar_chart(np.array(last["mi_proba"]))
                st.success(last["mi_action"])

            with c2:
                st.markdown("**Emotion (Ambient Channel)**")
                st.write(
                    f"True: `{last['true_emo']} ({EMO_CLASSES[last['true_emo']]})`  \n"
                    f"Predicted: `{last['emo_label']} ({EMO_CLASSES[last['emo_label']]})`"
                )
                st.bar_chart(np.array(last["emo_proba"]))
                st.info(last["emo_action"])

        # Event log
        with st.expander("🧾 Event Log (last 20 brain steps)", expanded=False):
            if not st.session_state.event_log:
                st.caption("No events yet. Click **Next Brain Step** to start.")
            else:
                for ev in st.session_state.event_log:
                    st.markdown(
                        f"**Step {ev['step']}**  \n"
                        f"- MI: `{ev['true_mi']} → {ev['mi_label']} "
                        f"({MI_CLASSES[ev['mi_label']]})`  \n"
                        f"- Emotion: `{ev['true_emo']} → {ev['emo_label']} "
                        f"({EMO_CLASSES[ev['emo_label']]})`  \n"
                        f"- Action: {ev['mi_action']}  \n"
                        f"- Ambient: {ev['emo_action']}"
                    )
                    st.markdown("---")

    # ----- RIGHT: SMART HOME STATE -----
    with col_state:
        st.subheader("🏠 Smart Home State")

        brightness_labels = ["Off", "Low", "Medium", "High"]
        fan_labels = ["Off", "Low", "Medium", "High"]

        st.markdown("### 💡 Lights")
        st.metric("Brightness", brightness_labels[smart_home["light_brightness"]])
        st.progress((smart_home["light_brightness"] + 1) / 4)

        st.markdown("### 🌀 Fan")
        st.metric("Speed", fan_labels[smart_home["fan_speed"]])
        st.progress((smart_home["fan_speed"] + 1) / 4)

        st.markdown("### 📺 TV")
        st.metric("Power", "ON" if smart_home["tv_on"] else "OFF")

        st.markdown("### 🚨 Emergency")
        if smart_home["emergency_alert"]:
            st.error("EMERGENCY ALERT ACTIVE")
        else:
            st.success("No active emergency")

        st.markdown("### 🎭 Ambient Mood")
        st.metric("Mode", smart_home["ambient_mode"])

        if st.button("🔁 Reset Smart Home State"):
            st.session_state.smart_home = {
                "light_brightness": 0,
                "fan_speed": 0,
                "tv_on": False,
                "emergency_alert": False,
                "ambient_mode": "neutral",
            }
            st.session_state.last_step = None
            st.session_state.event_log = []
            st.session_state.step_counter = 0
            st.rerun()



# Streamlit executes top-down, so call main()
main()
