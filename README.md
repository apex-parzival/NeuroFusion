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

## Data Setup (Important)

This repository excludes large dataset files. You must generate them before running the app.

### 1. Motor Imagery Data (BCI IV 2a)
The individual files are included. Run this script to combine them:
```bash
python scripts/reconstruct_data.py
```

### 2. Emotion Data (DEAP)
1.  Download **"Data_Preprocessed_Python"** from [DEAP Dataset](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).
2.  Place `s01.dat`...`s32.dat` in `data/deap/data_preprocessed_python/`.
3.  Run: `python -m pipeline.preprocess_deap_emotion`.

See [INSTRUCTIONS.md](INSTRUCTIONS.md) for full details.

## Project Structure
- `ui/`: Streamlit application.
- `pipeline/`: Data processing and model training scripts.
- `models/`: Trained models (Git LFS).
- `scripts/`: Helper scripts (e.g., data reconstruction).
- `processed/`: Processed data files.
