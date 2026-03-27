"""
NeuroFusion Smart Home Dashboard (Local File IPC Architecture)

This UI does NOT run any ML inference itself.
It reads the state from 'runtime/state.json' written by the Inference Engine,
and passively displays the live smart home state.

Run: streamlit run ui/app.py
"""
import json
import time
from pathlib import Path

import streamlit as st

# ---------- CONSTANTS ----------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATE_FILE = PROJECT_ROOT / "runtime" / "state.json"

BRIGHTNESS_LABELS = ["Off", "Low", "Medium", "High"]
FAN_LABELS = ["Off", "Low", "Medium", "High"]


def _read_state_file():
    """Read the latest state from the local JSON file."""
    try:
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            return data
    except Exception:
        pass
    return None


# ---------- MAIN APP ----------
def main():
    st.set_page_config(
        page_title="NeuroFusion Smart Home",
        page_icon="🧠",
        layout="wide",
    )

    # ----- SIDEBAR -----
    with st.sidebar:
        st.markdown("## 🧠 NeuroFusion")
        st.markdown(
            "Hybrid **Motor Imagery + Emotion** BCI controlling a simulated smart home.\n\n"
            "- 🧩 Motor Imagery: BCI Competition IV 2a\n"
            "- 🎧 Emotion: DEAP EEG dataset\n"
        )

        st.markdown("### 📊 Model Performance")
        st.markdown("- Motor Imagery (subject-wise combined): **77.59%**")
        st.markdown("- Riemannian (subject-wise): **69.54%**")

        st.divider()
        st.markdown("### ℹ️ How it works")
        st.markdown(
            "1. The **LSL Simulators** broadcast brain data.\n"
            "2. The **Inference Engine** decodes thoughts & emotions.\n"
            "3. State is written to a **local file** (`runtime/state.json`).\n"
            "4. **This dashboard** polls the file and displays the live state.\n"
        )

        st.divider()
        st.markdown("### 🔗 Connection Status")

    # Read the latest state from file
    data = _read_state_file()
    state = data.get("state") if data else None
    event_log = data.get("event_log", []) if data else []

    # Update sidebar connection status
    with st.sidebar:
        if state is not None:
            st.success("🟢 Connected to Inference Engine")
        else:
            st.warning("🟡 Waiting for backend...")

    # ----- HEADER -----
    st.title("🧠 NeuroFusion Smart Home – Live IoT Dashboard")
    st.caption(
        "This dashboard is a **passive listener**. It does not run any AI models. "
        "It reads the state file written by the backend inference engine."
    )

    if state is None:
        st.info(
            "⏳ **No data received yet.** Make sure these 3 scripts are running:\n\n"
            "1. `python scripts/lsl_simulator_mi.py`\n"
            "2. `python scripts/lsl_simulator_emo.py`\n"
            "3. `python -m pipeline.inference_engine`\n\n"
            "Once the backend starts publishing, this page will update automatically."
        )
        time.sleep(2)
        st.rerun()
        return

    # ----- MAIN LAYOUT -----
    col_home, col_log = st.columns([1.5, 1])

    with col_home:
        st.subheader("🏠 Smart Home State")

        # Device cards in a 2x2 grid
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 💡 Lights")
            light_val = state.get("light", 0)
            st.metric("Brightness", BRIGHTNESS_LABELS[light_val])
            st.progress((light_val + 1) / 4)

        with c2:
            st.markdown("### 🌀 Fan")
            fan_val = state.get("fan", 0)
            st.metric("Speed", FAN_LABELS[fan_val])
            st.progress((fan_val + 1) / 4)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("### 📺 TV")
            tv_on = state.get("tv", False)
            st.metric("Power", "ON" if tv_on else "OFF")

        with c4:
            st.markdown("### 🚨 Emergency")
            if state.get("emergency", False):
                st.error("🔴 EMERGENCY ALERT ACTIVE")
            else:
                st.success("✅ No active emergency")

        st.divider()
        st.markdown("### 🎭 Ambient Mood")
        ambient = state.get("ambient", "neutral")
        mood_emoji = {
            "sad/fatigued": "😔",
            "stressed/anxious": "😰",
            "calm/content": "😌",
            "excited/happy": "🤩",
            "neutral": "😐",
        }
        st.metric("Current Mood", f"{mood_emoji.get(ambient, '😐')} {ambient}")

    # ----- PROBABILITY DISTRIBUTIONS -----
    st.divider()
    prob_col1, prob_col2 = st.columns(2)

    with prob_col1:
        st.subheader("🧩 Motor Imagery Probabilities")
        mi_probs = state.get("mi_probs")
        if mi_probs and isinstance(mi_probs, dict):
            import pandas as pd
            df_mi = pd.DataFrame({
                "Class": list(mi_probs.keys()),
                "Probability": list(mi_probs.values())
            }).set_index("Class")
            st.bar_chart(df_mi, horizontal=True)
        else:
            st.caption("Waiting for first MI prediction...")

    with prob_col2:
        st.subheader("🎭 Emotion Probabilities")
        emo_probs = state.get("emo_probs")
        if emo_probs and isinstance(emo_probs, dict):
            import pandas as pd
            df_emo = pd.DataFrame({
                "Class": list(emo_probs.keys()),
                "Probability": list(emo_probs.values())
            }).set_index("Class")
            st.bar_chart(df_emo, horizontal=True)
        else:
            st.caption("Waiting for first Emotion prediction...")

    # ----- EVENT LOG -----
    st.divider()
    with col_log:
        st.subheader("📜 Live Event Log")
        if not event_log:
            st.caption("No events received yet.")
        else:
            for ev in event_log:
                st.markdown(f"`{ev['time']}` — {ev['event']}")

    # Auto-refresh every 2 seconds to pick up new state
    time.sleep(2)
    st.rerun()


# Streamlit executes top-down
main()
