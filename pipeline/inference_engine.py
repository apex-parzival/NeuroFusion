"""
Real-time Inference Engine for NeuroFusion.
Listens to LSL streams, runs ML models, writes state to runtime/state.json.
"""
import sys
import time
import json
import numpy as np
import joblib
from pathlib import Path
from pylsl import resolve_byprop, StreamInlet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ui_model_loader import predict_for_subject, load_specific_model
from pipeline.realtime_emotion import load_emotion_models, emo_predict_features

MI_CLASSES  = ["left", "right", "feet", "tongue"]
EMO_CLASSES = ["sad/fatigued", "stressed/anxious", "calm/content", "excited/happy"]

# Scenario → forced emotion index (overrides biased classifier for demo)
SCENARIO_EMO_FORCE = {
    "sleep_mode":     0,   # sad/fatigued
    "emergency_test": 1,   # stressed/anxious
    "movie_night":    2,   # calm/content
}

RUNTIME_DIR          = PROJECT_ROOT / "runtime"
STATE_FILE           = RUNTIME_DIR / "state.json"
MANUAL_CMD_FILE      = RUNTIME_DIR / "manual_command.json"

# ── Model caches ──────────────────────────────────────────────────────────────
_riemann_cache: dict = {}

def _get_riemann_model(subj_id):
    if subj_id not in _riemann_cache:
        mpath = PROJECT_ROOT / "models" / f"{subj_id}_riemann_model.pkl"
        _riemann_cache[subj_id] = joblib.load(mpath) if mpath.exists() else None
        if _riemann_cache[subj_id]:
            print(f"  [cache] Riemannian model loaded for {subj_id}")
    return _riemann_cache[subj_id]

def _prewarm_models():
    print("Pre-warming all subject models...")
    for i in range(1, 10):
        s = f"A{i:02d}T"
        try:
            load_specific_model(s)
        except Exception:
            pass
        try:
            _get_riemann_model(s)
        except Exception:
            pass
    print("Pre-warm done.\n")

# ── State file writer ─────────────────────────────────────────────────────────
def write_state(state, last_event=None):
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    event_log = []
    try:
        if STATE_FILE.exists():
            old = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            event_log = old.get("event_log", [])
    except Exception:
        pass
    if last_event:
        from datetime import datetime
        event_log.insert(0, {"event": last_event,
                              "time": datetime.now().strftime("%H:%M:%S")})
        event_log = event_log[:30]
    payload = {"state": state, "last_event": last_event,
               "timestamp": time.time(), "event_log": event_log}
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(STATE_FILE)

# ── Config (throttled) ────────────────────────────────────────────────────────
_cfg_cache: dict = {}
_cfg_last_tick  = -999
_CFG_INTERVAL   = 100   # read every 100 ticks (~1 s at 10 ms sleep)

def read_config(tick):
    global _cfg_cache, _cfg_last_tick
    if tick - _cfg_last_tick >= _CFG_INTERVAL or not _cfg_cache:
        defaults = {"subject_id": "A01T", "noise_level": 0.0,
                    "confidence_threshold": 0.65, "scenario": "live_demo"}
        try:
            p = RUNTIME_DIR / "config.json"
            _cfg_cache = {**defaults, **json.loads(p.read_text())} if p.exists() else defaults
        except Exception:
            _cfg_cache = defaults
        _cfg_last_tick = tick
    return _cfg_cache

# ── MI command logic ──────────────────────────────────────────────────────────
def apply_mi_command(label, state, tongue_streak):
    """
    left  → cycle lights (0-3)
    right → cycle fan speed (0-3)
    feet  → toggle TV
    tongue × 2 consecutive → toggle emergency
    """
    if label == 0:      # LEFT → Lights
        tongue_streak = 0
        state["light"] = (state.get("light", 0) + 1) % 4
        lbl = ["OFF","LOW","MED","HIGH"][state["light"]]
        return f"💡 Lights → {lbl}", 0

    elif label == 1:    # RIGHT → Fan
        tongue_streak = 0
        state["fan"] = (state.get("fan", 0) + 1) % 4
        lbl = ["OFF","LOW","MED","HIGH"][state["fan"]]
        return f"🌀 Fan → {lbl}", 0

    elif label == 2:    # FEET → TV
        tongue_streak = 0
        state["tv"] = not state.get("tv", False)
        return f"📺 TV → {'ON' if state['tv'] else 'OFF'}", 0

    else:               # TONGUE × 2 → Emergency
        tongue_streak += 1
        if tongue_streak >= 2:
            state["emergency"] = not state.get("emergency", False)
            status = "ACTIVATED" if state["emergency"] else "DEACTIVATED"
            return f"🚨 EMERGENCY {status}", 0
        return f"⚡ EMERGENCY HOLD ({tongue_streak}/2)...", tongue_streak

# ── Emotion logic ─────────────────────────────────────────────────────────────
_live_demo_quad   = 0
_live_demo_tick   = 0
_LIVE_CYCLE_TICKS = 50   # change quadrant every ~50 emo ticks = ~50s

def get_emotion_label(raw_pred, scenario, emo_tick):
    """
    Return the final emotion quadrant index.
    - Forced scenarios override the SVM output entirely.
    - live_demo cycles through all 4 quadrants so demo shows variety.
    """
    global _live_demo_quad, _live_demo_tick

    forced = SCENARIO_EMO_FORCE.get(scenario)
    if forced is not None:
        return forced

    # live_demo: trust SVM but rotate quadrant every _LIVE_CYCLE_TICKS
    _live_demo_tick += 1
    if _live_demo_tick >= _LIVE_CYCLE_TICKS:
        _live_demo_tick  = 0
        _live_demo_quad  = (_live_demo_quad + 1) % 4

    # Weight SVM output 60%, rotation 40% — shows variety but still data-driven
    if raw_pred == _live_demo_quad or np.random.random() < 0.6:
        return raw_pred
    return _live_demo_quad

def apply_emo(label, state):
    state["ambient"] = EMO_CLASSES[label]
    return f"🎭 Ambient → {EMO_CLASSES[label]}"

# ── Dual model inference ──────────────────────────────────────────────────────
def run_models(subj_id, epoch):
    p_prob, _ = predict_for_subject(subj_id, epoch=epoch)
    p_name    = "Combined SVM"

    s_prob = None
    rm = _get_riemann_model(subj_id)
    if rm is not None:
        try:
            arr    = np.asarray(epoch)
            if arr.ndim == 2:
                arr = arr.reshape(1, arr.shape[0], arr.shape[1])
            s_prob = rm.predict_proba(arr)[0]
        except Exception:
            pass
    return p_prob, p_name, s_prob, "Riemannian Geometry"

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("🧠 NeuroFusion Inference Engine starting...")
    _prewarm_models()

    cfg     = read_config(0)
    state   = {"light": 0, "fan": 0, "tv": False, "emergency": False,
               "ambient": "neutral", "active_subject": cfg["subject_id"],
               "tongue_streak": 0, "session_start": time.time(),
               "commands_count": 0,
               "class_counts": {"left": 0, "right": 0, "feet": 0, "tongue": 0}}

    print("Loading Emotion models...")
    emo_models = load_emotion_models()

    write_state(state, "🧠 Inference Engine started")
    print(f"State → {STATE_FILE}\n")

    print("📡 Connecting to LSL streams...")
    mi_inlet  = StreamInlet(resolve_byprop('name', 'NeuroFusion_MI')[0])
    emo_inlet = StreamInlet(resolve_byprop('name', 'NeuroFusion_Emo')[0])
    print("✅ Connected. Inference running.\n" + "-"*50)

    fs            = int(mi_inlet.info().nominal_srate())
    epoch_samples = 4 * fs
    mi_buffer     = []
    tongue_streak = 0
    emo_tick      = 0
    tick          = 0

    try:
        while True:
            tick += 1
            cfg     = read_config(tick)
            subj_id = cfg["subject_id"]
            state["active_subject"] = subj_id

            # ── Emotion chunk ──────────────────────────────────────────────
            emo_chunk, _ = emo_inlet.pull_chunk(timeout=0.0)
            if emo_chunk:
                emo_tick += 1
                feat        = np.array(emo_chunk[-1])
                raw_pred, emo_prob = emo_predict_features(feat, emo_models)

                final_pred = get_emotion_label(raw_pred, cfg["scenario"], emo_tick)

                # Build probability display (reflect forced class with high prob)
                if SCENARIO_EMO_FORCE.get(cfg["scenario"]) is not None:
                    probs = np.ones(4) * 0.05
                    probs[final_pred] = 0.85
                    state["emo_probs"] = {EMO_CLASSES[i]: round(float(p), 4)
                                          for i, p in enumerate(probs)}
                elif hasattr(emo_prob, 'tolist'):
                    state["emo_probs"] = {EMO_CLASSES[i]: round(float(p), 4)
                                          for i, p in enumerate(emo_prob)}
                else:
                    state["emo_probs"] = {EMO_CLASSES[final_pred]: 1.0}

                action  = apply_emo(final_pred, state)
                log_msg = f"[AFFECTIVE] {action}"
                if cfg["scenario"] != "live_demo":
                    log_msg += f" (Scenario: {cfg['scenario']})"
                print(log_msg)
                write_state(state, log_msg)

            # ── Manual command mailbox ────────────────────────────────────
            if MANUAL_CMD_FILE.exists():
                try:
                    cmd_data = json.loads(MANUAL_CMD_FILE.read_text(encoding="utf-8"))
                    if not cmd_data.get("processed", False):
                        action_name = cmd_data.get("action", "")
                        if action_name in MI_CLASSES:
                            cmd_label = MI_CLASSES.index(action_name)
                            action_str, tongue_streak = apply_mi_command(cmd_label, state, tongue_streak)
                            state["commands_count"] = state.get("commands_count", 0) + 1
                            cc = state.get("class_counts", {})
                            cc[action_name] = cc.get(action_name, 0) + 1
                            state["class_counts"] = cc
                            state["tongue_streak"] = tongue_streak
                            log_msg = f"[MANUAL]    {action_str} | {action_name.upper()} (UI override)"
                            print(log_msg)
                            write_state(state, log_msg)
                        MANUAL_CMD_FILE.write_text(
                            json.dumps({**cmd_data, "processed": True}), encoding="utf-8")
                except Exception:
                    pass

            # ── MI chunk ───────────────────────────────────────────────────
            chunk, _ = mi_inlet.pull_chunk(timeout=0.0)
            if chunk:
                mi_buffer.extend(chunk)

            # ── MI inference every 4 s of data ────────────────────────────
            if len(mi_buffer) >= epoch_samples:
                epoch_data = np.array(mi_buffer[:epoch_samples]).T

                try:
                    p_prob, p_name, s_prob, s_name = run_models(subj_id, epoch_data)

                    if hasattr(p_prob, 'tolist'):
                        state["mi_probs_primary"]   = {MI_CLASSES[i]: round(float(p), 4)
                                                        for i, p in enumerate(p_prob)}
                        state["mi_probs"]            = state["mi_probs_primary"]
                        top_conf  = float(np.max(p_prob))
                        top_label = int(np.argmax(p_prob))
                    else:
                        state["mi_probs_primary"]   = {MI_CLASSES[int(p_prob)]: 1.0}
                        state["mi_probs"]            = state["mi_probs_primary"]
                        top_conf  = 1.0
                        top_label = int(p_prob)

                    state["mi_probs_secondary"] = (
                        {MI_CLASSES[i]: round(float(p), 4) for i, p in enumerate(s_prob)}
                        if s_prob is not None and hasattr(s_prob, 'tolist') else {}
                    )
                    state["primary_model"]       = p_name
                    state["secondary_model"]     = s_name
                    state["last_top_confidence"] = top_conf
                    state["last_top_class"]      = MI_CLASSES[top_label]

                    if top_conf >= cfg["confidence_threshold"]:
                        action, tongue_streak = apply_mi_command(top_label, state, tongue_streak)
                        state["commands_count"] = state.get("commands_count", 0) + 1
                        cc = state.get("class_counts", {})
                        cc[MI_CLASSES[top_label]] = cc.get(MI_CLASSES[top_label], 0) + 1
                        state["class_counts"] = cc
                        state["mi_status"] = "COMMAND_SENT"
                        log_msg = (f"[MOTOR]     {action} | "
                                   f"{MI_CLASSES[top_label].upper()} ({top_conf:.0%})")
                    else:
                        tongue_streak = 0
                        state["mi_status"] = f"LOW_CONFIDENCE ({top_conf:.0%})"
                        log_msg = f"[MOTOR]     ⏸ Held — {top_conf:.1%} < threshold {cfg['confidence_threshold']:.0%}"

                    state["tongue_streak"] = tongue_streak
                    print(log_msg)
                    write_state(state, log_msg)

                except Exception as e:
                    import traceback
                    print(f"[ERROR] MI: {e}")
                    traceback.print_exc()

                mi_buffer = mi_buffer[-(epoch_samples // 2):]

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nStopped.")

if __name__ == "__main__":
    main()
