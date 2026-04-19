import plotly.graph_objects as go
import streamlit as st

def render_oscilloscope(buffer_data):
    """
    Renders scrolling EEG oscilloscope in Streamlit.
    buffer_data is a list of up to 500 samples, each sample is a list of 8 channel values.
    """
    # channel_names based on typical 10-20 layout for BCI Competition IV 2a (C3, Cz, C4, Fz, Pz, etc.)
    # The dataset has 22 EEG channels, we'll plot the first 8 to avoid clutter.
    channel_names = ["Fz", "FC3", "FC1", "FCz", "FC2", "FC4", "C5", "C3"]
    
    # Gradient colors from cyan to violet
    colors = [
        '#00d4ff', '#1cb6fb', '#3a98f7', '#577af2',
        '#755ced', '#923ee9', '#b020fe', '#ca04d8'
    ]
    
    fig = go.Figure()
    
    num_samples = len(buffer_data)
    if num_samples == 0:
        return
        
    num_channels = min(8, len(buffer_data[0]))
    
    # Plot each channel with a vertical offset
    offset_scale = 100.0  # Adjust according to typical μV range in dataset (approx 10-50μV)
    
    for ch in range(num_channels):
        # Extract the ch-th channel across all samples
        raw_y = [sample[ch] for sample in buffer_data]
        v_min, v_max = min(raw_y), max(raw_y)
        v_ptp = v_max - v_min if v_max != v_min else 1.0
        
        # Scale to strictly fit within its own lane (e.g. 50% of the offset scale to leave visual gaps)
        y_scaled = [((v - v_min) / v_ptp) * (offset_scale * 0.7) for v in raw_y]
        y_data = [val + ((7 - ch) * offset_scale) for val in y_scaled]
        x_data = list(range(num_samples))
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=y_data,
            mode='lines',
            line=dict(width=1.2, color=colors[ch % len(colors)]),
            name=channel_names[ch]
        ))
        
        # Add channel label annotation on the left
        fig.add_annotation(
            x=0,
            y=(7 - ch) * offset_scale + (offset_scale * 0.35), # Centered vertically in lane
            xanchor="right",
            text=f"{channel_names[ch]} ",
            font=dict(color="#64748b", size=10),
            showarrow=False
        )

    fig.update_layout(
        title=dict(
            text="🧠 Live EEG Stream — Cortex",
            font=dict(color="#94a3b8", size=14)
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        margin=dict(l=40, r=20, t=40, b=20),
        height=350,
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False, # Hide x axis
            range=[0, 500]
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False, # Hide y axis
        )
    )
    
    st.plotly_chart(fig, config={'displayModeBar': False})
