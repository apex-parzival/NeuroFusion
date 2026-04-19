import streamlit as st

LIGHT_LABELS = ["OFF", "LOW", "MED", "HIGH"]
FAN_LABELS   = ["OFF", "LOW", "MED", "HIGH"]

def render_smart_home(state):
    if state is None:
        return

    s_light     = state.get('light', 0)
    s_fan       = state.get('fan', 0)
    s_tv        = state.get('tv', False)
    s_emergency = state.get('emergency', False)
    s_ambient   = state.get('ambient', 'neutral')

    # ── visual properties ──────────────────────────────────────────────
    fan_durations = ["0s", "1.8s", "0.8s", "0.3s"]
    fan_speed     = fan_durations[s_fan] if s_fan < 4 else "0s"
    fan_anim      = f"spin {fan_speed} linear infinite" if s_fan > 0 else "none"

    light_opacities = [0.0, 0.35, 0.65, 1.0]
    light_op        = light_opacities[s_light]
    light_fill      = "#ffd700" if s_light > 0 else "#2a2a4a"
    light_radii     = [0, 30, 55, 80]
    light_glow_r    = light_radii[s_light]
    light_glow      = (f"drop-shadow(0 0 {light_glow_r}px rgba(255, 215, 0, {light_op}))"
                       if s_light > 0 else "none")

    tv_fill  = "#1a6fff" if s_tv else "#1a1a2e"
    tv_glow  = "drop-shadow(0 0 18px rgba(26,111,255,0.85))" if s_tv else "none"
    tv_label = "ON" if s_tv else "OFF"
    tv_label_col = "#1a6fff" if s_tv else "#475569"

    mood_colors = {
        "sad/fatigued":    "#0d1b3b",
        "stressed/anxious":"#3b0d0d",
        "calm/content":    "#0d3b2e",
        "excited/happy":   "#3b380d",
        "neutral":         "#121b2d",
    }
    room_bg = mood_colors.get(s_ambient, "#121b2d")

    emergency_display = "block" if s_emergency else "none"
    fan_label_col     = ["#475569", "#94a3b8", "#00d4ff", "#ff4d6d"][s_fan]
    light_label_col   = ["#475569", "#fbbf24", "#ffd700", "#ffffff"][s_light]

    svg_html = f"""
<div style="width:100%; border:1px solid rgba(0,212,255,0.2); border-radius:12px; background:#050d1a; overflow:hidden;">
  <style>
    @keyframes spin {{ from {{ transform:rotate(0deg); }} to {{ transform:rotate(360deg); }} }}
    @keyframes epulse {{ 0%,100% {{ fill:rgba(255,0,0,0.08); }} 50% {{ fill:rgba(255,0,0,0.45); }} }}
    @keyframes tpulse {{ 0%,100% {{ opacity:0.5; }} 50% {{ opacity:1; }} }}
    .fan-blades {{ transform-box:fill-box; transform-origin:center; animation:{fan_anim}; }}
    .light-bulb {{ fill:{light_fill}; opacity:{max(light_op, 0.3)}; filter:{light_glow}; transition:all 0.6s ease; }}
    .tv-screen  {{ fill:{tv_fill}; filter:{tv_glow}; transition:all 0.5s ease; }}
    .room-area  {{ fill:{room_bg}; transition:fill 2s ease; }}
    .e-overlay  {{ animation:epulse 0.9s ease-in-out infinite; display:{emergency_display}; }}
    .e-text     {{ animation:tpulse 0.9s ease-in-out infinite; }}
  </style>

  <svg viewBox="0 0 620 370" width="100%" height="400" xmlns="http://www.w3.org/2000/svg">

    <!-- Room fills (ambient mood) -->
    <rect class="room-area" x="15" y="15" width="355" height="340" rx="5"/>
    <rect class="room-area" x="385" y="15" width="220" height="340" rx="5"/>

    <!-- Walls -->
    <path d="M15 15 L605 15 L605 355 L15 355 Z" fill="none" stroke="#1e3a5f" stroke-width="5"/>
    <line x1="380" y1="15"  x2="380" y2="145" stroke="#1e3a5f" stroke-width="5"/>
    <line x1="380" y1="220" x2="380" y2="355" stroke="#1e3a5f" stroke-width="5"/>

    <!-- Room labels -->
    <text x="30"  y="38" font-family="Orbitron,sans-serif" font-size="11" fill="#2a4a6a" letter-spacing="2">LIVING ROOM</text>
    <text x="395" y="38" font-family="Orbitron,sans-serif" font-size="11" fill="#2a4a6a" letter-spacing="2">BEDROOM</text>

    <!-- ── TV (living room left wall) ── -->
    <rect x="25" y="100" width="22" height="140" rx="3" fill="#1e293b"/>
    <rect class="tv-screen" x="28" y="105" width="16" height="130" rx="2"/>
    <!-- TV label -->
    <text x="36" y="255" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="9"
          fill="{tv_label_col}">{tv_label}</text>
    <!-- TV antenna lines -->
    <line x1="33" y1="100" x2="28" y2="88"  stroke="#334155" stroke-width="1.5"/>
    <line x1="39" y1="100" x2="44" y2="88"  stroke="#334155" stroke-width="1.5"/>

    <!-- ── Sofa ── -->
    <rect x="240" y="95"  width="75"  height="180" rx="10" fill="#1e3a5f"/>
    <rect x="220" y="115" width="32"  height="140" rx="5"  fill="#25456e"/>
    <rect x="310" y="115" width="20"  height="140" rx="5"  fill="#25456e"/>

    <!-- ── Light (living room ceiling) ── -->
    <g transform="translate(190,72)">
      <circle cx="0" cy="0" r="18" fill="#1e293b" stroke="#334155" stroke-width="2"/>
      <circle class="light-bulb" cx="0" cy="0" r="11"/>
      <!-- light state label -->
      <text x="0" y="38" text-anchor="middle" font-family="Orbitron,sans-serif"
            font-size="9" fill="{light_label_col}">{LIGHT_LABELS[s_light]}</text>
    </g>

    <!-- ── Fan (living room) ── -->
    <g transform="translate(190,285)">
      <circle cx="0" cy="0" r="9" fill="#475569"/>
      <g class="fan-blades">
        <path d="M -4,-32 Q 16,-42 4,-6 L 4,6 Q -16,42 -4,6 Z" fill="#94a3b8"/>
        <path d="M 32,-4  Q 42,16  6,4  L -6,4 Q -42,-16 -6,-4 Z" fill="#94a3b8"/>
      </g>
      <!-- fan state label -->
      <text x="0" y="52" text-anchor="middle" font-family="Orbitron,sans-serif"
            font-size="9" fill="{fan_label_col}">{FAN_LABELS[s_fan]}</text>
    </g>

    <!-- ── Bed ── -->
    <rect x="400" y="115" width="185" height="155" rx="6" fill="#1e3a5f"/>
    <rect x="560" y="125" width="22"  height="135" rx="3" fill="#25456e"/>
    <rect x="412" y="135" width="45"  height="55"  rx="9" fill="#94a3b8"/>
    <rect x="412" y="200" width="45"  height="55"  rx="9" fill="#94a3b8"/>

    <!-- ── Bedroom light ── -->
    <g transform="translate(497,72)">
      <circle cx="0" cy="0" r="13" fill="#1e293b" stroke="#334155" stroke-width="2"/>
      <circle class="light-bulb" cx="0" cy="0" r="7"/>
      <text x="0" y="28" text-anchor="middle" font-family="Orbitron,sans-serif"
            font-size="9" fill="{light_label_col}">{LIGHT_LABELS[s_light]}</text>
    </g>

    <!-- ── Device status strip at bottom ── -->
    <rect x="15" y="310" width="590" height="45" rx="0" fill="rgba(0,0,0,0.35)"/>

    <!-- Light badge -->
    <rect x="25" y="318" width="120" height="28" rx="6" fill="rgba(255,215,0,0.08)" stroke="rgba(255,215,0,0.25)" stroke-width="1"/>
    <text x="85" y="328" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="8" fill="#94a3b8">LIGHT</text>
    <text x="85" y="341" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="11" fill="{light_label_col}" font-weight="bold">{LIGHT_LABELS[s_light]}</text>

    <!-- Fan badge -->
    <rect x="158" y="318" width="120" height="28" rx="6" fill="rgba(0,212,255,0.06)" stroke="rgba(0,212,255,0.2)" stroke-width="1"/>
    <text x="218" y="328" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="8" fill="#94a3b8">FAN</text>
    <text x="218" y="341" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="11" fill="{fan_label_col}" font-weight="bold">{FAN_LABELS[s_fan]}</text>

    <!-- TV badge -->
    <rect x="291" y="318" width="90" height="28" rx="6" fill="rgba(26,111,255,0.08)" stroke="rgba(26,111,255,0.25)" stroke-width="1"/>
    <text x="336" y="328" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="8" fill="#94a3b8">TV</text>
    <text x="336" y="341" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="11" fill="{tv_label_col}" font-weight="bold">{tv_label}</text>

    <!-- Ambient badge -->
    <rect x="393" y="318" width="207" height="28" rx="6" fill="rgba(124,58,237,0.08)" stroke="rgba(124,58,237,0.2)" stroke-width="1"/>
    <text x="496" y="328" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="8" fill="#94a3b8">AMBIENT</text>
    <text x="496" y="341" text-anchor="middle" font-family="Orbitron,sans-serif" font-size="10" fill="#a78bfa" font-weight="bold">{s_ambient.upper()}</text>

    <!-- ── Emergency overlay ── -->
    <rect class="e-overlay" x="0" y="0" width="620" height="355"/>
    <text class="e-text" x="310" y="185" text-anchor="middle" display="{emergency_display}"
          font-family="Orbitron,sans-serif" font-weight="bold" font-size="26"
          fill="#ff4d6d">⚠ EMERGENCY PROTOCOL ⚠</text>

  </svg>
</div>
"""
    st.components.v1.html(svg_html, height=405)
