import json
import streamlit as st
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAPPING_FILE = PROJECT_ROOT / "models" / "best_model_per_subject.json"

_mapping_cache = None

def _load_mapping():
    global _mapping_cache
    if _mapping_cache is None:
        try:
            _mapping_cache = json.loads(MAPPING_FILE.read_text())
        except Exception:
            _mapping_cache = {}
    return _mapping_cache

def _parse_accuracy(reason_str: str) -> str:
    """Extract accuracy from reason string like 'picked=combined,acc=0.8275...'"""
    try:
        for part in reason_str.split(","):
            if part.startswith("acc="):
                return f"{float(part.split('=')[1]):.1%}"
    except Exception:
        pass
    return "N/A"

def render_model_comparison(state, subject_id: str = "A01T"):
    st.subheader("🔬 Live Model Comparison")

    mapping          = _load_mapping()
    subj_info        = mapping.get(subject_id, {})
    best_type        = subj_info.get("best", "combined")
    primary_accuracy = _parse_accuracy(subj_info.get("reason", ""))

    primary_model_name   = state.get("primary_model",   "Combined SVM")
    secondary_model_name = state.get("secondary_model", "Riemannian Geometry")
    probs_primary        = state.get("mi_probs_primary",  state.get("mi_probs", {}))
    probs_secondary      = state.get("mi_probs_secondary", {})

    c1, c2 = st.columns(2)

    def _draw_panel(container, title, probs_dict, accuracy_str, is_best):
        with container:
            badge = " 🏅" if is_best else ""
            st.markdown(f"**{title}{badge}**  —  Subject accuracy: `{accuracy_str}`")

            if not probs_dict:
                st.caption("Waiting for inference…")
                return 0.0

            top_class = max(probs_dict, key=probs_dict.get)
            top_prob  = probs_dict[top_class]

            st.markdown(f"### Prediction: **{top_class.upper()}**")
            st.markdown(f"Confidence: **{top_prob:.1%}**")

            sorted_cls = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
            html = "<div style='font-family:monospace; margin-top:16px;'>"
            for cls, prob in sorted_cls:
                filled = int(prob * 10)
                bar    = "█" * filled + "░" * (10 - filled)
                color  = "#00d4ff" if cls == top_class else "#64748b"
                html  += (f"<div style='color:{color}; margin-bottom:5px;'>"
                          f"{cls.ljust(7)} | {bar} | {prob:5.1%}</div>")
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)
            return top_prob

    p1 = _draw_panel(c1, primary_model_name,   probs_primary,   primary_accuracy,  True)
    p2 = _draw_panel(c2, secondary_model_name, probs_secondary, "see README",       False)

    if probs_primary and probs_secondary:
        st.markdown("---")
        winner = primary_model_name if (p1 or 0) >= (p2 or 0) else secondary_model_name
        diff   = abs((p1 or 0) - (p2 or 0))
        st.info(f"🏆 **Winner this epoch:** {winner}  (Δ confidence: +{diff:.1%})")
