# NeuroFusion Project Setup Instructions

This guide provides detailed steps to set up the NeuroFusion project, specifically focusing on reproducing the necessary data files that were excluded from the repository due to size constraints.

## Prerequisites

- Python 3.8 or higher
- Git

## 1. Initial Setup

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/apex-parzival/NeuroFusion.git
    cd NeuroFusion
    ```

2.  **Create and activate a virtual environment**:
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

## 2. Data Setup

The project relies on two datasets: **BCI Competition IV 2a** (Motor Imagery) and **DEAP** (Emotion).

### A. BCI Competition IV 2a (Required for Motor Imagery)

The individual subject files for this dataset are included in the repository, but the large combined file needed for training/simulation was excluded. You can reconstruct it easily.

1.  **Run the reconstruction script**:
    ```bash
    python scripts/reconstruct_data.py
    ```

    *This script will scan `processed/bci_iv_2a/` for individual `A0*T_X.npy` files and create the missing `bci2a_all_subjects_X.npy` and `bci2a_all_subjects_y.npy`.*

### B. DEAP Dataset (Required for Emotion Recognition)

The DEAP dataset files are too large to host on GitHub and must be downloaded manually.

1.  **Download the data**:
    - Go to the [DEAP Dataset Website](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).
    - Sign the EULA and request access.
    - Download the **"Data_Preprocessed_Python.zip"** version.

2.  **Place the files**:
    - Unzip the downloaded file.
    - You should have files named `s01.dat`, `s02.dat`, ..., `s32.dat`.
    - Create the directory: `data/deap/data_preprocessed_python/` inside the project root.
    - Move all `.dat` files into that directory.

    Structure should look like:
    ```
    NeuroFusion/
    └── data/
        └── deap/
            └── data_preprocessed_python/
                ├── s01.dat
                ├── ...
                └── s32.dat
    ```

3.  **Process the data**:
    Run the preprocessing pipeline to generate the `.npy` files:
    ```bash
    python -m pipeline.preprocess_deap_emotion
    ```

## 3. Running the Application

Once the data is set up:

```bash
streamlit run ui/app.py
```

## 4. Troubleshooting

- **Missing Files Error**: If you see errors about missing `.npy` files, ensure you ran the reconstruction script (Step 2A) and/or downloaded the DEAP data (Step 2B).
- **Git LFS**: This repo uses Git LFS for some model files. If models are missing or 0 bytes, make sure you have Git LFS installed (`git lfs install`) and run `git lfs pull`.
