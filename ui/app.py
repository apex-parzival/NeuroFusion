"""
NeuroFusion Smart Home Dashboard (Phase 4 - MQTT Listener Architecture)

This UI does NOT run any ML inference itself.
It subscribes to the MQTT topic 'neurofusion/smarthome/state' and passively
displays whatever the backend Inference Engine publishes.

Run: streamlit run ui/app.py
"""
import json
import threading
import time
from datetime import datetime

import streamlit as st
import paho.mqtt.client as mqtt

# ---------- CONSTANTS ----------
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_TOPIC = "neurofusion/smarthome/state"

BRIGHTNESS_LABELS = ["Off", "Low", "Medium", "High"]
FAN_LABELS = ["Off", "Low", "Medium", "High"]

# ---------- MQTT BACKGROUND LISTENER ----------
# We use a threading lock to safely share data between the MQTT callback thread
# and the Streamlit main thread.

_mqtt_lock = threading.Lock()
_latest_payload = {"state": None, "event_log": []}


def _on_connect(client, userdata, flags, rc):
    """Called when the MQTT client connects to the broker."""
    client.subscribe(MQTT_TOPIC)


def _on_message(client, userdata, msg):
    """Called when a new message arrives on the subscribed topic."""
    global _latest_payload
    try:
        data = json.loads(msg.payload.decode())
        with _mqtt_lock:
            _latest_payload["state"] = data.get("state")
            event = data.get("last_event")
            ts = data.get("timestamp", time.time())
            if event:
                log_entry = {
                    "event": event,
                    "time": datetime.fromtimestamp(ts).strftime("%H:%M:%S"),
                }
                _latest_payload["event_log"].insert(0, log_entry)
                # Keep only last 30 events
                _latest_payload["event_log"] = _latest_payload["event_log"][:30]
    except Exception:
        pass


@st.cache_resource
def _get_mqtt_client():
    """Start a single persistent MQTT client in the background (cached across reruns)."""
    client = mqtt.Client()
    client.on_connect = _on_connect
    client.on_message = _on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()
    return client


def _get_state():
    """Thread-safe read of the latest smart home state from MQTT."""
    with _mqtt_lock:
        return _latest_payload.get("state")


def _get_event_log():
    """Thread-safe read of the event log."""
    with _mqtt_lock:
        return list(_latest_payload.get("event_log", []))


# ---------- MAIN APP ----------
def main():
    st.set_page_config(
        page_title="NeuroFusion Smart Home",
        page_icon="🧠",
        layout="wide",
    )

    # Start the MQTT listener (runs once, cached)
    _get_mqtt_client()

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
        st.markdown("### ℹ️ How it works now")
        st.markdown(
            "1. The **LSL Simulators** broadcast brain data.\n"
            "2. The **Inference Engine** decodes thoughts & emotions.\n"
            "3. Commands are sent over **MQTT** (IoT protocol).\n"
            "4. **This dashboard** passively listens and displays the live state.\n"
        )

        st.divider()
        st.markdown("### 🔗 Connection Status")
        state = _get_state()
        if state is not None:
            st.success("🟢 Connected to Inference Engine")
        else:
            st.warning("🟡 Waiting for backend... Click refresh below.")

    # ----- HEADER -----
    st.title("🧠 NeuroFusion Smart Home – Live IoT Dashboard")
    st.caption(
        "This dashboard is a **passive listener**. It does not run any AI models. "
        "It subscribes to the MQTT topic and displays whatever the backend inference engine broadcasts."
    )

    # Read the latest state
    state = _get_state()

    if state is None:
        st.info(
            "⏳ **No data received yet.** Make sure these 3 scripts are running:\n\n"
            "1. `python scripts/lsl_simulator_mi.py`\n"
            "2. `python scripts/lsl_simulator_emo.py`\n"
            "3. `python -m pipeline.inference_engine`\n\n"
            "Once the backend starts publishing, this page will update automatically."
        )
        # Auto-refresh every 2 seconds while waiting
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

    with col_log:
        st.subheader("📜 Live Event Log")
        event_log = _get_event_log()
        if not event_log:
            st.caption("No events received yet.")
        else:
            for ev in event_log:
                st.markdown(f"`{ev['time']}` — {ev['event']}")

    # Auto-refresh every 2 seconds to pick up new MQTT messages
    time.sleep(2)
    st.rerun()


# Streamlit executes top-down
main()
