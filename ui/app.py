"""
NeuroFusion Smart Home Dashboard
"""
import json
import time
from pathlib import Path
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from components.theme import inject_theme
from components.eeg_oscilloscope import render_oscilloscope
from components.smart_home_svg import render_smart_home
from components.confidence_gauge import render_confidence_gauge
from components.model_comparison import render_model_comparison

PROJECT_ROOT     = Path(__file__).resolve().parents[1]
RUNTIME_DIR      = PROJECT_ROOT / "runtime"
STATE_FILE       = RUNTIME_DIR / "state.json"
CONFIG_FILE      = RUNTIME_DIR / "config.json"
EEG_BUFFER_FILE  = RUNTIME_DIR / "raw_eeg_buffer.json"
MANUAL_CMD_FILE  = RUNTIME_DIR / "manual_command.json"

SUBJECTS = [f"A{i:02d}T" for i in range(1, 10)]

SCENARIOS = {
    "live_demo":      "Live Demo 🕹️",
    "movie_night":    "Movie Night 🎬",
    "sleep_mode":     "Sleep Mode 💤",
    "emergency_test": "Stress Test 😰",
}
SCENE_KEY = {v: k for k, v in SCENARIOS.items()}
SCENE_EFFECT = {
    "live_demo":      "All 4 emotions cycle randomly",
    "movie_night":    "Locks emotion → Calm / Content 😌",
    "sleep_mode":     "Locks emotion → Sad / Fatigued 😔",
    "emergency_test": "Locks emotion → Stressed / Anxious 😰 (emotion only — not emergency protocol)",
}

FAN_LABELS   = ["Off", "Low", "Medium", "High"]
LIGHT_LABELS = ["OFF", "LOW", "MEDIUM", "HIGH"]

MI_META = {
    "left":   {"icon": "👈", "color": "#00d4ff", "action": "Cycles LIGHT (Off→Low→Med→High)"},
    "right":  {"icon": "👉", "color": "#22c55e", "action": "Cycles FAN speed (Off→Low→Med→High)"},
    "feet":   {"icon": "🦶", "color": "#a78bfa", "action": "Toggles TV On/Off"},
    "tongue": {"icon": "👅", "color": "#ff4d6d", "action": "Triggers EMERGENCY (×2 in a row)"},
    "—":      {"icon": "❓", "color": "#475569", "action": "Waiting..."},
}
EMO_META = {
    "sad/fatigued":    {"icon": "😔", "color": "#60a5fa",  "label": "Sad / Fatigued"},
    "stressed/anxious":{"icon": "😰", "color": "#f87171",  "label": "Stressed / Anxious"},
    "calm/content":    {"icon": "😌", "color": "#34d399",  "label": "Calm / Content"},
    "excited/happy":   {"icon": "🤩", "color": "#fbbf24",  "label": "Excited / Happy"},
    "neutral":         {"icon": "😐", "color": "#94a3b8",  "label": "Neutral"},
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_json(p, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default
    except Exception:
        return default

def send_manual_command(action: str):
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        MANUAL_CMD_FILE.write_text(
            json.dumps({"action": action, "processed": False, "ts": time.time()}),
            encoding="utf-8")
    except Exception as e:
        print(f"Manual command write error: {e}")

def write_state_direct(state_file, runtime_dir, state_data, patched_state):
    import time as _time
    payload = {**state_data, "state": patched_state, "timestamp": _time.time(),
               "last_event": "🛡️ Emergency manually reset via UI"}
    event_log = state_data.get("event_log", [])
    from datetime import datetime
    event_log.insert(0, {"event": "🛡️ Emergency manually reset via UI",
                          "time": datetime.now().strftime("%H:%M:%S")})
    payload["event_log"] = event_log[:30]
    tmp = state_file.with_suffix(".tmp")
    try:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(state_file)
    except Exception as e:
        print(f"State write error: {e}")

def _write_config(d):
    try:
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"Config write error: {e}")

def _card(icon, label, value, color="#00d4ff", sub=None):
    sub_html = f"<div style='color:#64748b;font-size:0.68rem;margin-top:3px;'>{sub}</div>" if sub else ""
    return f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid {color}33;
            border-radius:12px;padding:14px 10px;text-align:center;
            box-shadow:0 0 12px {color}18;height:100%;">
  <div style="font-size:1.5rem;">{icon}</div>
  <div style="color:#64748b;font-size:0.65rem;font-family:Orbitron,sans-serif;
              letter-spacing:1px;margin-top:4px;">{label}</div>
  <div style="color:{color};font-size:1rem;font-family:Orbitron,sans-serif;
              font-weight:700;margin-top:4px;">{value}</div>
  {sub_html}
</div>"""

def _init_ss(config):
    if "nf_init" not in st.session_state:
        st.session_state.nf_init      = True
        st.session_state.nf_subject   = config.get("subject_id", "A01T")
        st.session_state.nf_scenario  = SCENARIOS.get(config.get("scenario","live_demo"), "Live Demo 🕹️")
        st.session_state.nf_noise     = float(config.get("noise_level", 0.0))
        st.session_state.nf_threshold = float(config.get("confidence_threshold", 0.65))

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="NeuroFusion | BCI", page_icon="🧠",
                       layout="wide", initial_sidebar_state="expanded")
    st_autorefresh(interval=800, key="nf_refresh")
    inject_theme()

    config = _read_json(CONFIG_FILE, {"subject_id":"A01T","noise_level":0.0,
                                       "confidence_threshold":0.65,"scenario":"live_demo"})
    _init_ss(config)

    # ── SIDEBAR ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<h1>🧠 NeuroFusion</h1>", unsafe_allow_html=True)
        st.caption("Hybrid Motor Imagery + Emotion BCI")
        st.divider()

        # ── Subject ──────────────────────────────────────────────────────
        st.markdown("#### 🧬 Active Subject")
        st.caption("Switches which person's trained ML model is used for Motor Imagery decoding")
        new_subject = st.selectbox("Subject", SUBJECTS, key="nf_subject", label_visibility="collapsed")

        st.divider()

        # ── Scenario ─────────────────────────────────────────────────────
        st.markdown("#### 🎬 Scenario Preset")
        st.caption("Controls which emotion is shown on the dashboard")
        selected_disp = st.radio("Scenario", list(SCENARIOS.values()),
                                  key="nf_scenario", label_visibility="collapsed")
        new_scenario  = SCENE_KEY[selected_disp]
        st.info(f"Effect: {SCENE_EFFECT[new_scenario]}", icon="ℹ️")

        st.divider()

        # ── Noise ─────────────────────────────────────────────────────────
        st.markdown("#### 📶 Signal Noise Injection")
        st.caption("Adds noise to the EEG stream — visible in the oscilloscope and reduces confidence scores")
        new_noise = st.slider("Noise", 0.0, 1.0, step=0.05,
                               key="nf_noise", label_visibility="collapsed")
        noise_desc = ["Clean signal", "Mild noise", "Moderate noise", "Heavy noise — confidence drops"]
        st.caption(f"→ {noise_desc[min(int(new_noise / 0.26), 3)]}")

        st.divider()

        # ── Threshold ─────────────────────────────────────────────────────
        st.markdown("#### 🎯 Min Confidence Threshold")
        st.caption("Commands only fire when confidence exceeds this value — purple line on gauge")
        new_threshold = st.slider("Threshold", 0.50, 0.99, step=0.01,
                                   key="nf_threshold", label_visibility="collapsed")
        st.caption(f"→ Commands fire only when confidence > {new_threshold:.0%}")

        st.divider()

        # ── Manual MI command override ────────────────────────────────────
        st.markdown("#### 🎮 Manual Command Override")
        st.caption("Force-trigger any MI command — useful for demo/testing")
        mc1, mc2 = st.columns(2)
        mc3, mc4 = st.columns(2)
        if mc1.button("👈 LEFT",   use_container_width=True):
            send_manual_command("left");   st.toast("👈 LEFT sent", icon="💡")
        if mc2.button("👉 RIGHT",  use_container_width=True):
            send_manual_command("right");  st.toast("👉 RIGHT sent", icon="🌀")
        if mc3.button("🦶 FEET",   use_container_width=True):
            send_manual_command("feet");   st.toast("🦶 FEET sent", icon="📺")
        if mc4.button("👅 TONGUE", use_container_width=True):
            send_manual_command("tongue"); st.toast("👅 TONGUE sent", icon="⚡")

        st.divider()

        # ── Emergency info ────────────────────────────────────────────────
        st.markdown("#### 🚨 Emergency Protocol")

        # Tongue streak progress (read from current state)
        _live_state = (_read_json(STATE_FILE) or {}).get("state") or {}
        _streak = int(_live_state.get("tongue_streak", 0))
        streak_color = "#ff4d6d" if _streak >= 1 else "#475569"
        st.markdown(f"""
<div style='background:rgba(255,77,109,0.1);border:1px solid rgba(255,77,109,0.3);
            border-radius:8px;padding:10px;font-size:0.8rem;color:#f87171;'>
  <b>How to activate:</b><br>
  Tongue motor imagery × <b>2 consecutive epochs</b><br>
  above confidence threshold. Same action deactivates.<br><br>
  <b>Tongue streak:</b>
  <span style='color:{streak_color};font-size:1rem;font-weight:bold;'> {_streak}/2</span>
  {"🔴" if _streak >= 1 else "⚪"}
</div>""", unsafe_allow_html=True)

        if st.button("🛡️ Reset Emergency", use_container_width=True, type="secondary"):
            state_data_now = _read_json(STATE_FILE)
            if state_data_now and state_data_now.get("state"):
                patched = state_data_now["state"]
                patched["emergency"] = False
                patched["tongue_streak"] = 0
                write_state_direct(STATE_FILE, RUNTIME_DIR, state_data_now, patched)
                st.toast("✅ Emergency cleared", icon="🛡️")

        st.divider()

        # ── Write config if changed ────────────────────────────────────────
        changed = (
            new_subject  != config.get("subject_id") or
            new_scenario != config.get("scenario") or
            abs(new_noise     - float(config.get("noise_level", 0.0)))           > 0.001 or
            abs(new_threshold - float(config.get("confidence_threshold", 0.65))) > 0.001
        )
        if changed:
            _write_config({"subject_id": new_subject, "noise_level": new_noise,
                           "confidence_threshold": new_threshold, "scenario": new_scenario})
            st.toast(f"✅ Applied: {new_subject} · {selected_disp}", icon="⚙️")

        # ── Status ────────────────────────────────────────────────────────
        st.markdown("#### 🔗 System Status")

    # ── Read live state ───────────────────────────────────────────────────
    state_data = _read_json(STATE_FILE)
    state      = state_data.get("state") if state_data else None
    event_log  = state_data.get("event_log", []) if state_data else []
    eeg_data   = _read_json(EEG_BUFFER_FILE)
    eeg_buffer = eeg_data.get("buffer", []) if eeg_data else []

    with st.sidebar:
        if state is not None:
            active_subj = state.get("active_subject", new_subject)
            st.success(f"🟢 ONLINE — Model: **{active_subj}**")
            ts = state_data.get("timestamp", time.time())
            st.caption(f"Last update: {time.strftime('%H:%M:%S', time.localtime(ts))}")
        else:
            st.warning("🟡 Waiting for Inference Engine...")

    st.title("NeuroFusion — Live IoT Control")

    if state is None:
        st.info("⏳ Start the Inference Engine to see live data.")
        return

    # ── Extract current values ────────────────────────────────────────────
    s_light     = state.get("light", 0)
    s_fan       = state.get("fan", 0)
    s_tv        = state.get("tv", False)
    s_emergency = state.get("emergency", False)
    s_ambient   = state.get("ambient", "neutral")

    # ── SESSION STATS BAR ─────────────────────────────────────────────────
    session_start  = state.get("session_start", time.time())
    elapsed        = int(time.time() - session_start)
    h, m, s_sec   = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    uptime_str     = f"{h:02d}:{m:02d}:{s_sec:02d}"
    commands_count = state.get("commands_count", 0)
    class_counts   = state.get("class_counts", {})
    top_cls        = max(class_counts, key=class_counts.get) if class_counts else "—"
    top_cls_count  = class_counts.get(top_cls, 0) if top_cls != "—" else 0
    top_cls_icon   = MI_META.get(top_cls, MI_META["—"])["icon"] if top_cls != "—" else "❓"

    sb1, sb2, sb3 = st.columns(3)
    with sb1:
        st.markdown(_card("⏱️", "SESSION UPTIME", uptime_str, "#64748b"), unsafe_allow_html=True)
    with sb2:
        st.markdown(_card("⚡", "COMMANDS FIRED", str(commands_count), "#22c55e"), unsafe_allow_html=True)
    with sb3:
        st.markdown(_card(top_cls_icon, "TOP CLASS", f"{top_cls.upper()} ×{top_cls_count}",
                          MI_META.get(top_cls, MI_META["—"])["color"] if top_cls != "—" else "#475569"),
                    unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    mi_probs  = state.get("mi_probs", {})
    mi_class  = max(mi_probs, key=mi_probs.get) if mi_probs else state.get("last_top_class", "—")
    mi_conf   = mi_probs.get(mi_class, state.get("last_top_confidence", 0.0)) if mi_probs else 0.0

    emo_probs = state.get("emo_probs", {})
    emo_class = max(emo_probs, key=emo_probs.get) if emo_probs else s_ambient

    mi_m  = MI_META.get(mi_class,  MI_META["—"])
    emo_m = EMO_META.get(emo_class, {"icon":"❓","color":"#94a3b8","label":emo_class})

    l_colors = ["#475569", "#fbbf24", "#ffd700", "#ffffff"]
    f_colors = ["#475569", "#94a3b8", "#00d4ff", "#ff4d6d"]

    # ── TOP STATUS ROW ────────────────────────────────────────────────────
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    rows = [
        (c1, _card("💡","LIGHT", LIGHT_LABELS[s_light], l_colors[s_light],
                   sub="Left hand MI cycles")),
        (c2, _card("🌀","FAN",   FAN_LABELS[s_fan],     f_colors[s_fan],
                   sub="Right hand MI cycles")),
        (c3, _card("📺","TV",    "ON" if s_tv else "OFF",
                   "#1a6fff" if s_tv else "#475569", sub="Feet MI toggles")),
        (c4, _card(mi_m["icon"], "MOTOR DETECTED", mi_class.upper(), mi_m["color"],
                   sub=f"{mi_conf:.0%} conf")),
        (c5, _card(emo_m["icon"],"EMOTION",        emo_m["label"],   emo_m["color"],
                   sub="Sets ambient mood")),
        (c6, _card("🚨" if s_emergency else "🛡️",
                   "EMERGENCY",
                   "ACTIVE" if s_emergency else "STANDBY",
                   "#ff4d6d" if s_emergency else "#22c55e",
                   sub="Tongue ×2 triggers")),
    ]
    for col, html in rows:
        with col:
            st.markdown(html, unsafe_allow_html=True)

    # ── Motor command guide row ───────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    g1,g2,g3,g4 = st.columns(4)
    for col, cls in [(g1,"left"),(g2,"right"),(g3,"feet"),(g4,"tongue")]:
        m = MI_META[cls]
        border = f"2px solid {m['color']}" if mi_class == cls else "1px solid #1e3a5f"
        with col:
            st.markdown(f"""
<div style="border:{border};border-radius:8px;padding:8px;text-align:center;
            background:{'rgba('+(','.join([str(int(m['color'].lstrip('#')[i:i+2],16)) for i in (0,2,4)]) )+',0.08)' if mi_class==cls else 'transparent'}">
  <span style="font-size:1.3rem;">{m['icon']}</span>
  <span style="color:{m['color']};font-family:Orbitron,sans-serif;font-size:0.75rem;
               font-weight:700;margin-left:6px;">{cls.upper()}</span>
  <div style="color:#64748b;font-size:0.68rem;margin-top:3px;">{m['action']}</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── MAIN LAYOUT ───────────────────────────────────────────────────────
    left_col, right_col = st.columns([1.2, 1.8])

    with left_col:
        st.subheader("🏠 Smart Home")
        render_smart_home(state)

        st.markdown("---")
        d1, d2 = st.columns(2)
        with d1:
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid {mi_m['color']}44;
            border-radius:10px;padding:12px;text-align:center;">
  <div style="color:#64748b;font-size:0.68rem;font-family:Orbitron,sans-serif;">MOTOR INTENT</div>
  <div style="font-size:2rem;margin:6px 0;">{mi_m['icon']}</div>
  <div style="color:{mi_m['color']};font-family:Orbitron,sans-serif;font-weight:700;">
    {mi_class.upper()}</div>
  <div style="color:#64748b;font-size:0.72rem;margin-top:4px;">{mi_m['action']}</div>
  <div style="color:#94a3b8;font-size:0.78rem;margin-top:4px;">{mi_conf:.0%} confidence</div>
</div>""", unsafe_allow_html=True)

        with d2:
            st.markdown(f"""
<div style="background:rgba(255,255,255,0.03);border:1px solid {emo_m['color']}44;
            border-radius:10px;padding:12px;text-align:center;">
  <div style="color:#64748b;font-size:0.68rem;font-family:Orbitron,sans-serif;">AFFECTIVE STATE</div>
  <div style="font-size:2rem;margin:6px 0;">{emo_m['icon']}</div>
  <div style="color:{emo_m['color']};font-family:Orbitron,sans-serif;font-weight:700;">
    {emo_m['label']}</div>
  <div style="color:#64748b;font-size:0.72rem;margin-top:4px;">sets ambient lighting</div>
</div>""", unsafe_allow_html=True)

    with right_col:
        st.subheader("🧠 Motor Cortex EEG Stream")
        if eeg_buffer:
            render_oscilloscope(eeg_buffer)
        else:
            st.markdown("""
<div style="height:350px;display:flex;align-items:center;justify-content:center;
            border:1px dashed #1e3a5f;color:#94a3b8;border-radius:8px;">
  Waiting for raw EEG stream…
</div>""", unsafe_allow_html=True)

        st.markdown("---")
        ga, gb = st.columns(2)
        with ga:
            render_confidence_gauge(mi_conf, f"Motor: {mi_class.upper()}", threshold=new_threshold)
            status = state.get("mi_status", "WAITING")
            if "LOW_CONFIDENCE" in status:
                st.warning(f"⏸ {status}")
            elif "COMMAND_SENT" in status:
                st.success("✅ COMMAND SENT")
        with gb:
            if emo_probs:
                render_confidence_gauge(emo_probs.get(emo_class, 0.0),
                                        f"Emotion: {emo_m['label']}", threshold=0.40)
            else:
                render_confidence_gauge(0.0, "Emotion Confidence", threshold=0.40)

    # ── MODEL COMPARISON ──────────────────────────────────────────────────
    st.markdown("---")
    render_model_comparison(state, subject_id=new_subject)

    # ── EVENT LOG ─────────────────────────────────────────────────────────
    st.markdown("---")
    log_header, log_export = st.columns([4, 1])
    log_header.subheader("📜 Event Log")

    if event_log:
        import io
        csv_lines = ["time,event"]
        for ev in event_log:
            clean = ev["event"].replace('"', "'")
            csv_lines.append(f'{ev["time"]},"{clean}"')
        csv_bytes = "\n".join(csv_lines).encode("utf-8")
        log_export.download_button("⬇️ Export CSV", data=csv_bytes,
                                   file_name="neurofusion_event_log.csv",
                                   mime="text/csv", use_container_width=True)

    if not event_log:
        st.caption("No events yet.")
    else:
        html = "<div style='max-height:380px;overflow-y:auto;padding-right:6px;'>"
        for ev in event_log:
            txt = ev["event"]
            if "[MOTOR]" in txt:
                txt = txt.replace("[MOTOR]",
                    "<span style='background:#00d4ff22;color:#00d4ff;padding:2px 6px;"
                    "border-radius:4px;font-size:0.72rem;'>MOTOR</span>")
            if "[AFFECTIVE]" in txt:
                txt = txt.replace("[AFFECTIVE]",
                    "<span style='background:#7c3aed22;color:#a78bfa;padding:2px 6px;"
                    "border-radius:4px;font-size:0.72rem;'>EMOTION</span>")
            if "[MANUAL]" in txt:
                txt = txt.replace("[MANUAL]",
                    "<span style='background:#f59e0b22;color:#fbbf24;padding:2px 6px;"
                    "border-radius:4px;font-size:0.72rem;'>MANUAL</span>")
            html += (
                f"<div style='background:rgba(255,255,255,0.025);border:1px solid "
                f"rgba(255,255,255,0.06);padding:10px 14px;border-radius:8px;"
                f"margin-bottom:6px;font-size:0.83rem;line-height:1.5;'>"
                f"<span style='color:#475569;font-size:0.72rem;'>{ev['time']}</span>"
                f"&nbsp;&nbsp;{txt}</div>"
            )
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
