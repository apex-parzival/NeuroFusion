"""
Theme constraints and custom CSS injection for NeuroFusion Dark Glassmorphism.
"""
import streamlit as st

def inject_theme():
    # Inject Custom CSS
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Orbitron:wght@500;700&display=swap');

        /* Main App Background (Radial Gradient) */
        .stApp {
            background: radial-gradient(circle at 50% -20%, #0a1628, #050d1a);
            font-family: 'Inter', sans-serif;
            color: #f1f5f9;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff;
            text-shadow: 0 0 10px rgba(0, 212, 255, 0.3);
        }
        h1 { margin-bottom: 0.2rem !important; }

        /* Metric Cards - Glassmorphism */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(0, 212, 255, 0.15);
            border-radius: 12px;
            padding: 16px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 0 15px rgba(0, 212, 255, 0.05);
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
        }
        
        div[data-testid="metric-container"]:hover {
            border-color: rgba(0, 212, 255, 0.4);
            box-shadow: 0 0 25px rgba(0, 212, 255, 0.15), inset 0 0 20px rgba(0, 212, 255, 0.1);
            transform: translateY(-2px);
        }

        /* Metric Value Text */
        div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
            color: #f8fafc;
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        /* Metric Label Text */
        div[data-testid="metric-container"] div[data-testid="stMetricLabel"] {
            color: #94a3b8;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0d1f3c;
            border-right: 1px solid rgba(0, 212, 255, 0.2);
            box-shadow: 5px 0 20px rgba(0, 0, 0, 0.5);
        }
        
        /* Sidebar Logo/Title */
        section[data-testid="stSidebar"] h1 {
            font-size: 2rem;
            background: -webkit-linear-gradient(45deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: none;
            margin-top: 1rem;
        }

        /* Buttons with Cyan Glow */
        .stButton button {
            background: rgba(0, 212, 255, 0.1);
            border: 1px solid rgba(0, 212, 255, 0.3);
            color: #00d4ff;
            border-radius: 8px;
            transition: all 0.2s ease;
            font-family: 'Orbitron', sans-serif;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .stButton button:hover {
            background: rgba(0, 212, 255, 0.2);
            border-color: rgba(0, 212, 255, 0.8);
            box-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
            transform: translateY(-2px);
            color: #ffffff;
        }

        /* Dropdowns/Selectboxes */
        div[data-baseweb="select"] > div {
            background-color: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(0, 212, 255, 0.2);
            border-radius: 8px;
            color: white;
        }

        /* Sliders */
        div[role="slider"] {
            background-color: #00d4ff;
        }
        div[data-baseweb="slider"] > div > div {
            background-color: rgba(0, 212, 255, 0.2);
        }

        /* Block Dividers */
        hr {
            border-color: rgba(0, 212, 255, 0.15) !important;
            margin: 1.5em 0 !important;
        }

        /* Expander headers */
        .streamlit-expanderHeader {
            font-family: 'Orbitron', sans-serif;
            color: #00d4ff !important;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            border: 1px solid rgba(0, 212, 255, 0.1);
        }

        /* Progress Bars */
        .stProgress > div > div > div > div {
            background-image: linear-gradient(90deg, #7c3aed, #00d4ff);
        }

        /* Subtle Neural Pulse Background Animation */
        .stApp::before {
            content: '';
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2300d4ff' fill-opacity='0.03'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
            z-index: -1;
            animation: pulse-bg 10s infinite alternate;
        }
        @keyframes pulse-bg {
            0% { opacity: 0.5; }
            100% { opacity: 1; }
        }

        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #050d1a; }
        ::-webkit-scrollbar-thumb { background: rgba(0, 212, 255, 0.3); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: rgba(0, 212, 255, 0.6); }
        
    </style>
    """, unsafe_allow_html=True)

