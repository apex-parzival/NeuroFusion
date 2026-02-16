# NeuroFusion Smart Home BCI

## Overview
NeuroFusion is a hybrid Brain-Computer Interface (BCI) application that combines Motor Imagery (MI) control with Emotion recognition to control a simulated Smart Home environment.

- **Motor Imagery**: Controls specific devices (Lights, Fan, TV) based on EEG patterns (Left Hand, Right Hand, Feet, Tongue).
- **Emotion Recognition**: Adjusts the ambient environment mode based on emotional state (Calm, Excited, Sad, Stressed).

## Features
- Real-time simulation of brain states using the BCI Competition IV 2a and DEAP datasets.
- Interactive Streamlit dashboard for monitoring brain activity and controlling the smart home.
- Support for multiple subject models (CSP+LDA, Riemannian Geometry, EEGNet, Hybrid Fusion).

## Installation

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd NeuroFusion
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # Mac/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the application**:
    ```bash
    streamlit run ui/app.py
    ```

2.  **Navigate**: Open your browser to `http://localhost:8501`.
3.  **Interact**: Select a subject and click "Next Brain Step" to simulate BCI commands.

## Project Structure
- `ui/`: Contains the Streamlit web application.
- `pipeline/`: Core BCI logic, feature extraction, and model handling.
- `models/`: Trained model files.
- `data/` & `processed/`: Dataset storage (git-ignored).
