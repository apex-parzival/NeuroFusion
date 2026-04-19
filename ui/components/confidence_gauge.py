import plotly.graph_objects as go
import streamlit as st

def render_confidence_gauge(confidence: float, label: str, threshold: float = 0.70):
    val       = confidence * 100
    thresh_pct = threshold * 100

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={'suffix': "%", 'font': {'color': '#00d4ff', 'size': 34}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#334155', 'tickwidth': 2},
            'bar':  {'color': '#00d4ff', 'thickness': 0.25},
            'bgcolor': '#0a1628',
            'bordercolor': '#1e3a5f',
            'steps': [
                {'range': [0,           thresh_pct * 0.7], 'color': 'rgba(255,77,109,0.18)'},
                {'range': [thresh_pct * 0.7, thresh_pct], 'color': 'rgba(251,191,36,0.18)'},
                {'range': [thresh_pct,       100],         'color': 'rgba(34,197,94,0.18)'},
            ],
            'threshold': {
                'line':      {'color': '#7c3aed', 'width': 4},
                'thickness': 0.75,
                'value':     thresh_pct,
            }
        },
        title={'text': label, 'font': {'color': '#94a3b8', 'size': 13}}
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        height=220,
        margin=dict(l=20, r=20, t=40, b=10)
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
