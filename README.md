# 🧠 NeuroFusion: The Mind-Controlled Smart Home

Welcome to **NeuroFusion**! This README is designed to explain everything about this project from the ground up, so even if you know nothing about Artificial Intelligence (AI) or neuroscience, you will understand exactly what this is, why it exists, and how it works.

---

## 📖 The "TL;DR" (What is this project?)

**NeuroFusion** is a software system that allows a person to control a "Smart Home" (lights, fan, TV) using only their **thoughts** and **emotions**. Right now, we are simulating a live environment completely in software—meaning our code behaves *exactly* like it would if someone were wearing a physical mind-reading headset.

---

## ❓ The Problem

Imagine a person who has severe physical disabilities (such as paralysis) and cannot move their arms, legs, or speak. How can they turn on a TV? How can they turn off a blazing fan if they feel cold?
Standard technology requires physical interaction (pushing a button) or voice commands, which they cannot provide. This strips them of independence.

---

## 💡 The Solution

A **Brain-Computer Interface (BCI)**.
A BCI is a system that allows a human brain to talk directly to a computer without involving muscles. We read the brain's electrical signals, use Artificial Intelligence to figure out what the person is thinking, and then send a command to a smart device (like a smart lightbulb or TV) to do the action.

---

## ⚡ What are Brain Signals?

Your brain is made of billions of cells called neurons. Every time you think, feel, or move, these neurons pass tiny electrical charges to each other.
If we put metal sensors on the outside of your head (your scalp), we can measure these tiny electrical changes. This technique is called **EEG (Electroencephalography)**. Think of it like putting microphones outside a football stadium; you can't hear one specific person talking, but you can hear the overall cheer of the crowd when someone scores a goal.

---

## 🎯 What Signals Are We Using?

To control a complex environment, we combine (or "fuse") two different types of brain signals:

1. **Motor Imagery (The "Action" Button)**

   * **What it is:** When you *think* about moving your left hand, your brain creates a specific electrical pattern—even if your hand doesn't actually move.
   * **The Data:** We use a famous public dataset called **BCI Competition IV 2a**. It recorded people thinking about moving their:
     * Left Hand
     * Right Hand
     * Feet
     * Tongue
   * **How we use it:** We map these thoughts to hard commands. For example, thinking about your left hand might translate to "Turn up the lights."
2. **Emotion Recognition (The "Ambient" Controller)**

   * **What it is:** Your brain waves look different when you are stressed versus when you are relaxed.
   * **The Data:** We use a dataset called **DEAP**, which recorded brainwaves of people while they watched emotional music videos.
   * **How we use it:** We constantly monitor the affective state (mood) of the user. If the AI sees the user is getting "Stressed", the Smart Home might automatically dim the lights and change to a "Calming" mode, without the user specifically asking for it!

---

## Real-time Inference Engine (`pipeline/inference_engine.py`)

This is the "brain" of the backend. It runs as a separate background process:

1. Listens to the `NeuroFusion_MI` and `NeuroFusion_Emo` LSL streams.
2. Automatically buffers the incoming stream data into 4-second brainwave epochs.
3. Triggers the Machine Learning models (EEGNet/SVM) to predict the user's intent.
4. Writes every prediction (with full probability distributions) to a **local state file** (`runtime/state.json`), which the dashboard polls every 2 seconds.

---

## 🏗️ The Architecture: How Does the Data Flow?

The system is split into **4 independent processes** that communicate via LSL streams and a local state file:

```
┌──────────────────────┐     ┌──────────────────────┐
│  LSL Simulator (MI)  │     │  LSL Simulator (Emo)  │
│  250 Hz, 25 channels │     │  1 Hz, 160 features   │
└──────────┬───────────┘     └──────────┬────────────┘
           │  LSL Stream                │  LSL Stream
           └──────────┐    ┌────────────┘
                      ▼    ▼
              ┌───────────────────┐
              │  Inference Engine │
              │  (AI Backend)     │
              │  EEGNet / SVM     │
              └────────┬──────────┘
                       │  Writes JSON
                       ▼
              ┌───────────────────┐
              │ runtime/state.json│
              └────────┬──────────┘
                       │  Reads JSON
                       ▼
              ┌───────────────────┐
              │  Streamlit UI     │
              │  (Dashboard)      │
              └───────────────────┘
```

### Step 1: The Headset Simulators (`scripts/lsl_simulator_mi.py` & `emo.py`)

* **What it does:** These scripts load the giant files of pre-recorded brain waves (`.npy` files). They slice the data up and broadcast it over our computer's local network continuously via **LSL (Lab Streaming Layer)**, exactly like a real physical headset transmitting via Bluetooth.
* **Why it flows there:** To create a realistic environment. When we buy a real headset later, we just turn off these scripts and connect the real headset. Our application won't even know the difference!

### Step 2: The Data Buffer/Listener (Internal Pipeline)

* **What it does:** The inference engine "listens" to the LSL streams. Instead of looking at 1 millisecond of brain data (which is too short to understand), it gathers a "chunk" of time—like recording a 4-second video clip of the brain waves. This chunk is called an **epoch**.

### Step 3: Feature Extraction (Cleaning the Noise)

* **What it does:** Raw brain waves look like scribbles. We pass the 4-second "epoch" through filters (specifically, something called CSP - Common Spatial Pattern).
* **Why it flows there:** This cleans off the "noise" and highlights the exact part of the brain wave that proves the person is thinking about their left hand.

### Step 4: The AI Brain (`pipeline/ui_model_loader.py`)

* **What it does:** The cleaned data goes into our Machine Learning models. We have trained complex math models (like EEGNet or Support Vector Machines) on hours of brain data.
* **Why it flows there:** The AI looks at the shape of the cleaned wave and makes a guess: *"I am 95% sure this person is thinking about their feet!"*

### Step 5: State File Output (`runtime/state.json`)

* **What it does:** The prediction result (e.g., "TV ON", "Ambient → stressed") along with **full probability distributions** for all classes is written to a local JSON file.
* **Why it flows there:** This provides a simple, reliable IPC (Inter-Process Communication) mechanism. No external network dependency — the dashboard simply reads this file.

### Step 6: The Smart Home Dashboard (`ui/app.py`)

* **What it does:** The Streamlit UI is a **passive listener**. It reads the `runtime/state.json` file and displays whatever the backend has written. It does **zero** AI work itself.
* **Where it goes after:** The dashboard auto-refreshes every 2 seconds, showing the latest device states, probability bar charts for all prediction classes, and a live event log.

---

## 📦 Installation

### Prerequisites

* Python 3.9+
* Git

### Setup

```bash
# Clone the repository
git clone https://github.com/apex-parzival/NeuroFusion.git
cd NeuroFusion

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run the Project

The system requires **4 terminal windows** running simultaneously. **Start them in order:**

### Terminal 1: Start the Brain Simulator (Motor Imagery)

```bash
python scripts/lsl_simulator_mi.py
```

> Loads pre-recorded BCI IV 2a data and broadcasts at 250 Hz over LSL.

### Terminal 2: Start the Brain Simulator (Emotion)

```bash
python scripts/lsl_simulator_emo.py
```

> Loads pre-extracted DEAP emotion features and broadcasts at 1 Hz over LSL.

### Terminal 3: Start the Inference Engine (AI Backend)

```bash
python -m pipeline.inference_engine
```

> Connects to both LSL streams, runs real-time ML inference, and writes predictions to `runtime/state.json`.

### Terminal 4: Start the Smart Home Dashboard (UI)

```bash
streamlit run ui/app.py
```

> Opens a live dashboard at `http://localhost:8501`.

---

## 🖥️ Dashboard Features

- **Smart Home State** — Live status of Lights, Fan, TV, and Emergency
- **Ambient Mood** — Real-time emotion classification (sad/fatigued, stressed/anxious, calm/content, excited/happy)
- **Probability Bar Charts** — Full probability distributions for both Motor Imagery and Emotion predictions
- **Live Event Log** — Timestamped stream of all predictions and state changes

---

## 📊 Model Performance

| Model                                  | Accuracy         |
| -------------------------------------- | ---------------- |
| Motor Imagery (Combined, subject-wise) | **77.59%** |
| Riemannian (subject-wise)              | **69.54%** |

Open your browser to `http://localhost:8501` and watch as the dashboard **automatically updates** in real-time as the AI decodes simulated brain waves!





The EEG waves you see on your dashboard are the heartbeat of the **NeuroFusion** system. Here is a breakdown of how they are calculated and the scientific meaning behind them:

### 1. How the Waves are Calculated

The "calculation" happens in two stages: simulation and visualization.

* **Real Data Source (Simulation):** While in demo mode, the system isn't just generating random lines. It is loading real brain activity from the  **BCI Competition IV 2a dataset** . Researchers recorded these signals from actual subjects performing motor imagery. The system streams these recordings at **250 samples per second (Hz)** to mimic a live headset.
* **Signal Normalization (Visualization):** Because raw brain signals are incredibly tiny (measured in  **micro-volts, μV** ), they vary in scale. The `eeg_oscilloscope.py` script performs a calculation called  **Min-Max Scaling** . It looks at the strongest and weakest parts of the current signal window and stretches them to fit perfectly within their designated "lane" on your screen so the patterns are visible to the human eye.
* **Noise Injection:** When you move the "Noise Injection" slider in the UI, the system uses a **Gaussian Random Calculation** to add artificial electrical interference to the clean data, testing how well the AI models can "see" through the static.

---

### 2. What Each Wave Signifies

Each line on the graph represents a specific location on the brain, following the  **International 10-20 System** .

#### The Electrode Positions

* **Fz (Frontal Zero):** Located at the top-front of the head. It signifies high-level cognition, focus, and attention.
* **C3, Cz, C4 (Central):** These are the most critical for your dashboard. They sit directly over the  **Motor Cortex** .
  * **C3 (Left side):** Signifies intent to move the  **Right Hand** .
  * **C4 (Right side):** Signifies intent to move the  **Left Hand** .
  * **Cz (Center):** Often signifies intent to move the  **Feet** .

#### The Frequency "Language"

The AI doesn't just look at the shape; it calculates the "power" in different frequency bands:

* **Delta (1–4 Hz):** Signifies deep sleep or unconsciousness.
* **Theta (4–8 Hz):** Signifies deep relaxation or drowsiness.
* **Alpha / Mu (8–13 Hz):** This is the "idle" rhythm of the brain. When you stop moving and relax your hands, your **Mu waves** (a type of Alpha wave over the motor cortex) get very strong. When you *think* about moving, these waves suddenly "collapse"—this is the primary pattern the AI looks for to trigger a smart home action.
* **Beta (13–30 Hz):** Signifies active thinking, focus, and the actual execution of movement.
* **Gamma (30+ Hz):** Signifies high-level multi-sensory processing (like being very excited or highly alert).

---

## 🗂️ Complete File & Module Reference

### Directory Tree

```
NeuroFusion/
├── data/
│   └── bci_iv_2a/          ← Raw .gdf files: A01T.gdf … A09T.gdf
├── processed/
│   ├── bci_iv_2a/          ← Per-subject X.npy / y.npy arrays
│   └── deap/               ← deap_features_X.npy, deap_labels_quadrant.npy, etc.
├── models/
│   ├── best_models/        ← A01T_best.pkl … A09T_best.pkl (auto-selected)
│   ├── eegnet/             ← A0xT_eegnet.h5 + meta .npz
│   ├── hybrid/             ← A0xT_hybrid.h5 + meta .npz
│   ├── best_model_per_subject.json
│   ├── csp_filters.pkl / csp_scaler.pkl
│   ├── emotion_svm_classifier.pkl / emotion_scaler.pkl
│   ├── mi_svm_classifier.pkl / mi_svm_classifier_tuned.pkl
│   └── A0xT_riemann_model.pkl / A0xT_riemann_meta.pkl
├── pipeline/
│   ├── preprocess_bci2a.py
│   ├── preprocess_deap_emotion.py
│   ├── extract_deap_features.py
│   ├── feature_csp_bci2a.py
│   ├── build_combined_features_bci2a.py
│   ├── train_bci2a_subjectwise_csp.py
│   ├── train_bci2a_feature_classifier.py
│   ├── train_eegnet_mi.py
│   ├── train_subjectwise_riemann_save.py
│   ├── train_subjectwise_combined_save.py
│   ├── train_emotion_classifier.py
│   ├── train_hybrid_fusion.py
│   ├── train_mi_classifier.py
│   ├── ensemble_subjectwise.py
│   ├── export_best_models.py
│   ├── export_best_all_models.py
│   ├── eval_models.py
│   ├── realtime_mi.py
│   ├── realtime_emotion.py
│   ├── realtime_simulation.py
│   ├── inference_engine.py      ← Main runtime AI process
│   └── ui_model_loader.py       ← Model registry + prediction API
├── scripts/
│   ├── lsl_simulator_mi.py      ← Motor imagery LSL broadcaster
│   ├── lsl_simulator_emo.py     ← Emotion features LSL broadcaster
│   └── reconstruct_data.py
├── ui/
│   ├── app.py                   ← Streamlit dashboard
│   └── components/
│       ├── confidence_gauge.py  ← Radial confidence meter
│       ├── eeg_oscilloscope.py  ← Live waveform + noise injection
│       ├── model_comparison.py  ← Side-by-side model probability bars
│       ├── smart_home_svg.py    ← Animated SVG floor-plan widget
│       └── theme.py             ← Dark-mode CSS constants
├── runtime/
│   ├── state.json               ← Live IPC state file (written by engine, read by UI)
│   ├── config.json              ← Runtime config (subject_id, noise_level, scenario)
│   └── manual_command.json      ← UI override mailbox
├── requirements.txt
└── start.bat                    ← Windows launcher
```

---

## 🧪 Full ML Pipeline Walk-through

### Phase 0 – Data Acquisition (Offline, Done Once)

| Dataset | Subjects | Trials | Classes | Sampling Rate | Channels |
|---------|----------|--------|---------|--------------|----------|
| BCI Competition IV 2a | 9 | ~288 per subject | Left Hand, Right Hand, Feet, Tongue | 250 Hz | 22 EEG + 3 EOG |
| DEAP | 32 | 40 videos × 32 subjects | Valence/Arousal (continuous 1–9) | 128 Hz | 32 EEG |

### Phase 1 – Preprocessing (`pipeline/preprocess_bci2a.py`)

Raw `.gdf` files are loaded using **MNE-Python**. The steps applied to each subject are:

1. Load raw GDF via `mne.io.read_raw_gdf`
2. **50 Hz Notch filter** — removes Indian power-line interference
3. **Band-pass filter 8–30 Hz** — retains mu and beta rhythms relevant to motor imagery; removes slow drift and muscle noise
4. **Pick EEG channels only** — drops EOG (eye movement) channels to prevent artifacts from contaminating spatial filters
5. **Epoch extraction** — events `769` (left), `770` (right), `771` (feet), `772` (tongue) are read from annotations; a 0–4 second window is cut around each cue
6. **Artifact rejection** — epochs with peak-to-peak amplitude > 100 µV are discarded
7. **Save** — resulting arrays `A0xT_X.npy` (shape: `n_trials × 22 × 1000`) and `A0xT_y.npy` (labels 0–3) are written to `processed/bci_iv_2a/`

### Phase 2 – DEAP Preprocessing (`pipeline/preprocess_deap_emotion.py` + `extract_deap_features.py`)

1. Load per-participant `.dat` files (MATLAB format) containing 32-channel EEG at 128 Hz
2. Extract **band-power features** using Welch's method for five bands:

| Band | Frequency Range | Emotion Relevance |
|------|----------------|-------------------|
| Delta | 1–4 Hz | Sleep/deep relaxation |
| Theta | 4–8 Hz | Drowsiness, meditation |
| Alpha | 8–13 Hz | Relaxed wakefulness |
| Beta | 13–30 Hz | Active cognition, stress |
| Gamma | 30–45 Hz | Excitement, alertness |

3. **Quadrant mapping** — continuous valence (1–9) and arousal (1–9) scores are thresholded at 5.0 to produce 4 discrete emotion classes:
   - `0` → sad/fatigued (low V, low A)
   - `1` → stressed/anxious (low V, high A)
   - `2` → calm/content (high V, low A)
   - `3` → excited/happy (high V, high A)

4. Feature matrix `deap_features_X.npy` has shape `(N_trials, 32×5 = 160)` — 160 features per trial.

### Phase 3 – Feature Engineering for Motor Imagery

Three parallel feature sets are computed:

#### A. CSP (Common Spatial Patterns)
- Trains spatial filters that maximally separate variance between any two MI classes
- Produces **log-variance** features from 6 spatial components
- Simple, interpretable, well-proven for mu/beta ERD (Event-Related Desynchronization)

#### B. FBCSP (Filter Bank CSP) — used in Combined model
Operates on 4 frequency sub-bands: `[8–12, 12–16, 16–22, 22–30 Hz]`
- Each band is bandpass-filtered independently
- CSP with 8 components is applied per band
- Results are concatenated → richer spectral representation

#### C. Combined Features (used in best-performing model)
Three feature types are stacked into one vector per trial:
1. **Temporal** — log-variance in 4 one-second windows × 22 channels = 88 features
2. **Spectral** — log band-power (Theta, Alpha, Beta) × 22 channels = 66 features
3. **FBCSP** — 4 bands × 8 components = 32 features
- **Total: ~186 features per trial**

#### D. Riemannian Geometry
- Computes **sample covariance matrix** of each epoch (22×22 matrix)
- Projects to **Tangent Space** of the Riemannian manifold
- Tangent vectors (vectorized upper triangle) fed to SVM
- Robust to non-stationarity, no spatial filter fitting required

### Phase 4 – Model Training

#### Combined SVM (best model for 7/9 subjects)
```
pipeline/train_subjectwise_combined_save.py
```
- Features: Combined (Temporal + Spectral + FBCSP)
- Classifier: SVC with RBF kernel, probability=True
- Hyperparameter search: GridSearchCV (C ∈ {0.1,1,10}, gamma ∈ {scale, 0.1, 0.01})
- Cross-validation: 5-fold StratifiedKFold

#### Riemannian SVM (best model for 2/9 subjects — A06T, A08T)
```
pipeline/train_subjectwise_riemann_save.py
```
- Pipeline: `Covariances(scm) → TangentSpace → StandardScaler → SVC`
- Uses **pyriemann** library

#### EEGNet (deep learning baseline)
```
pipeline/train_eegnet_mi.py
```
Architecture:
```
Input (22 ch × 1000 t × 1)
  └─ Conv2D(F1=8, 1×64)         ← Temporal filter
  └─ DepthwiseConv2D(22×1, D=2) ← Spatial filter per temporal filter
  └─ BatchNorm → ELU → AvgPool(1×4) → Dropout(0.5)
  └─ SeparableConv2D(16, 1×16)  ← Pointwise + depthwise
  └─ BatchNorm → ELU → AvgPool(1×8) → Dropout(0.5)
  └─ Flatten → Dense(4, softmax)
```
Training: Adam(1e-3), EarlyStopping(patience=12), ReduceLROnPlateau

#### Hybrid Fusion Model
```
pipeline/train_hybrid_fusion.py
```
- EEGNet branch (EEG) + MLP branch (emotion features, 160-dim)
- Branches merged via Concatenate → Dense(128) → Dense(4, softmax)
- Falls back to EEG-only when per-subject DEAP data unavailable

#### Emotion SVM
```
pipeline/train_emotion_classifier.py
```
- Input: 160 band-power features per trial
- Pipeline: StandardScaler → SVC(RBF, probability=True)
- GridSearchCV: C ∈ {0.1,1,10}, gamma ∈ {scale, 0.1, 0.01}

### Phase 5 – Model Selection (`pipeline/export_best_all_models.py`)

For each subject the best model is selected by held-out test accuracy. Results saved to `models/best_model_per_subject.json`:

| Subject | Best Model | Accuracy |
|---------|-----------|----------|
| A01T | Combined SVM | 82.76% |
| A02T | Combined SVM | 72.41% |
| A03T | Combined SVM | 86.21% |
| A04T | Combined SVM | 75.86% |
| A05T | Combined SVM | 74.14% |
| A06T | Riemannian SVM | 56.90% |
| A07T | Combined SVM | **89.66%** |
| A08T | Riemannian SVM | **93.10%** |
| A09T | Combined SVM | 75.86% |
| **Mean (Combined)** | | **77.59%** |
| **Mean (Riemannian)** | | **69.54%** |

---

## ⚙️ Runtime Config Reference

The file `runtime/config.json` is read every ~1 second by the inference engine and can be edited while the system is running:

```json
{
  "subject_id": "A01T",
  "noise_level": 0.0,
  "confidence_threshold": 0.65,
  "scenario": "live_demo"
}
```

| Key | Type | Description |
|-----|------|-------------|
| `subject_id` | string | Which subject model to use (A01T–A09T) |
| `noise_level` | float 0–1 | Gaussian noise injected into raw signal |
| `confidence_threshold` | float 0–1 | Minimum confidence before MI command fires |
| `scenario` | string | `live_demo`, `sleep_mode`, `emergency_test`, `movie_night` |

### Scenarios

| Scenario | Effect |
|----------|--------|
| `live_demo` | SVM output used, emotion cycles through quadrants for demo variety |
| `sleep_mode` | Emotion forced to **sad/fatigued** (index 0) |
| `emergency_test` | Emotion forced to **stressed/anxious** (index 1) |
| `movie_night` | Emotion forced to **calm/content** (index 2) |

---

## 🔁 IPC State File Schema (`runtime/state.json`)

```json
{
  "state": {
    "light": 0,
    "fan": 0,
    "tv": false,
    "emergency": false,
    "ambient": "calm/content",
    "active_subject": "A01T",
    "tongue_streak": 0,
    "session_start": 1700000000.0,
    "commands_count": 12,
    "class_counts": {"left":3,"right":4,"feet":4,"tongue":1},
    "mi_probs": {"left":0.6,"right":0.2,"feet":0.1,"tongue":0.1},
    "mi_probs_primary": {"left":0.6,"right":0.2,"feet":0.1,"tongue":0.1},
    "mi_probs_secondary": {"left":0.5,"right":0.25,"feet":0.15,"tongue":0.1},
    "emo_probs": {"sad/fatigued":0.05,"stressed/anxious":0.05,"calm/content":0.85,"excited/happy":0.05},
    "primary_model": "Combined SVM",
    "secondary_model": "Riemannian Geometry",
    "last_top_confidence": 0.82,
    "last_top_class": "left",
    "mi_status": "COMMAND_SENT"
  },
  "last_event": "[MOTOR] 💡 Lights → LOW | LEFT (82%)",
  "timestamp": 1700000010.0,
  "event_log": [
    {"event": "...", "time": "14:22:01"}
  ]
}
```

---

## 🎮 MI Command Mapping

| Motor Imagery Class | Smart Home Action | Detail |
|--------------------|-------------------|--------|
| Left Hand | Cycle Lights | 4-level cycle: OFF → LOW → MED → HIGH |
| Right Hand | Cycle Fan | 4-level cycle: OFF → LOW → MED → HIGH |
| Feet | Toggle TV | ON/OFF toggle |
| Tongue (×2 consecutive) | Emergency Alert | Must predict Tongue twice in a row to toggle |

The **double-tongue** requirement for Emergency prevents false positives — a single tongue classification is printed as a hold warning but takes no action.

---

## 🌊 LSL Stream Specifications

### Motor Imagery Stream (`NeuroFusion_MI`)
- **Type:** EEG
- **Channels:** 22 (EEG) + 3 (EOG) = 25 total
- **Sample rate:** 250 Hz
- **Data:** Raw µV values replayed from BCI IV 2a `.npy` arrays

### Emotion Stream (`NeuroFusion_Emo`)
- **Type:** Features
- **Channels:** 160 (32 EEG channels × 5 frequency bands)
- **Sample rate:** 1 Hz (one feature vector per second)
- **Data:** Pre-extracted DEAP band-power features

---

## 🖥️ UI Component Details (`ui/components/`)

| Component | File | Description |
|-----------|------|-------------|
| Smart Home SVG | `smart_home_svg.py` | Animated SVG floor-plan with glowing lights, spinning fan, TV screen effect, blinking emergency light |
| EEG Oscilloscope | `eeg_oscilloscope.py` | Live scrolling waveform for Fz, C3, Cz, C4 channels; adds Gaussian noise based on slider |
| Confidence Gauge | `confidence_gauge.py` | Radial dial showing top-class probability 0–100% with color transitions green→orange→red |
| Model Comparison | `model_comparison.py` | Two side-by-side Plotly bar charts comparing Combined SVM vs Riemannian probability distributions |
| Theme | `theme.py` | Dark-mode CSS: background `#0e1117`, accent `#7c3aed` (purple), card `#1a1d2e` |

---

## 🔬 Scientific Background & Literature Context

### Brain-Computer Interface (BCI)

A BCI creates a direct communication pathway between the brain and an external device, bypassing the normal motor output pathways through nerves and muscles. The primary application domain driving BCI research is **assistive technology** for individuals with motor disabilities (ALS, spinal cord injury, locked-in syndrome).

### Motor Imagery (MI-BCI)

Motor imagery exploits the **Sensorimotor Rhythm (SMR)** — specifically the **Event-Related Desynchronization (ERD)** of mu (8–12 Hz) and beta (13–30 Hz) rhythms over the contralateral motor cortex when a subject imagines a movement. The neural basis is the mirror-neuron system: imagining movement activates the same motor cortex areas as actual movement, but without muscular output.

- Left hand imagery → ERD over right hemisphere (C4 electrode)
- Right hand imagery → ERD over left hemisphere (C3 electrode)
- Feet imagery → ERD over central scalp (Cz electrode)

### Valence-Arousal Emotion Model

The DEAP dataset uses the **Russell Circumplex Model of Affect**, where emotions are described in a 2D space:
- **Valence (horizontal axis):** Pleasure vs. displeasure
- **Arousal (vertical axis):** Activation energy (excited vs. calm)

This produces 4 quadrants directly usable as discrete classes for a classifier.

### EEGNet Architecture Justification

EEGNet (Lawhern et al., 2018) was chosen as the deep learning baseline because:
1. It was designed specifically for EEG data with limited training samples
2. The depthwise separable convolution reduces parameter count dramatically vs. standard CNNs
3. The temporal-then-spatial filtering structure mirrors the manual CSP pipeline

### Riemannian Geometry Justification

Traditional CSP/SVM pipelines assume Euclidean distance between covariance matrices, which is not geometrically correct. Riemannian methods (Congedo et al., 2017) treat covariance matrices as points on a symmetric positive-definite manifold and compute distances using the **geodesic** (shortest curved path), leading to better generalization and robustness to non-stationarity.

---

## 🏥 Real-World Deployment Path

The current implementation is a **software-only simulation** using the offline BCI IV 2a and DEAP datasets. The pathway to a real physical deployment would be:

| Step | What Changes | What Stays the Same |
|------|-------------|---------------------|
| Replace simulator | Turn off `lsl_simulator_mi.py`; connect a real EEG headset that streams via LSL | The entire inference engine, models, and UI are unchanged |
| Real-time model adaptation | Add online covariance normalization or Euclidean Alignment per session | The model architecture is unchanged |
| IoT integration | Replace `runtime/state.json` writes with MQTT publish to real smart-home hub (code exists in older branches) | Command logic stays identical |
| Clinical validation | Run on N=20+ subjects, measure Information Transfer Rate (ITR) | Pipeline structure identical |

---

## 🧩 Dependency Summary

| Library | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32 | Dashboard web UI |
| numpy | ≥1.24 | Array math |
| scipy | ≥1.10 | Welch PSD, signal filtering |
| scikit-learn | ≥1.2 | SVM, GridSearchCV, StandardScaler |
| mne | ≥1.6 | EEG preprocessing, CSP, GDF I/O |
| tensorflow | ≥2.15 | EEGNet and Hybrid deep models |
| pylsl | ≥1.16 | Lab Streaming Layer IPC |
| pyriemann | latest | Riemannian covariance geometry |
| plotly | ≥5.18 | Interactive probability charts |
| joblib | ≥1.2 | Model serialization |
| paho-mqtt | ≥1.6 | (legacy) MQTT IoT transport |

---

## 📐 Thesis Chapter Mapping

This project maps directly to a standard 5-chapter thesis structure:

| Chapter | Title | Key Sections from this README |
|---------|-------|-------------------------------|
| 1 | Introduction | Problem, Solution, BCI motivation, project goals |
| 2 | Literature Review | EEGNet reference, Riemannian geometry reference, Valence-Arousal model, BCI competition IV 2a, DEAP dataset |
| 3 | Methodology | Architecture diagram, Phase 0–5 pipeline, preprocessing steps, feature engineering, model architectures, EEGNet equations |
| 4 | Results & Evaluation | Subject-wise accuracy table, mean accuracies, model comparison chart, dual-model runtime, confidence threshold effects |
| 5 | Conclusion & Future Work | Real-world deployment path, limitations, IoT integration, online adaptation |

---

## 🖥️ Dashboard UI — Full Walkthrough (`ui/app.py`)

The Streamlit dashboard auto-refreshes every **800 milliseconds** via `streamlit_autorefresh`. It is purely a **display and control layer** — it does zero machine learning itself. Every number it shows was written to `runtime/state.json` by the inference engine.

### Sidebar Controls

| Control | Type | What it does |
|---------|------|-------------|
| **Active Subject** | Dropdown (A01T–A09T) | Writes `subject_id` to `config.json`; inference engine picks it up within ~1 second and switches to that subject's best ML model |
| **Scenario Preset** | Radio buttons | Changes `scenario` in `config.json`; forces specific emotion quadrant in some modes |
| **Signal Noise Injection** | Slider 0.0–1.0 | Writes `noise_level`; LSL simulator reads this and adds Gaussian noise scaled to the data's own standard deviation, making it visible on the oscilloscope |
| **Min Confidence Threshold** | Slider 0.50–0.99 | Writes `confidence_threshold`; no command fires unless `top_prob ≥ threshold`; shown as a purple vertical line on the gauge |
| **Manual Command Override** | 4 buttons (LEFT/RIGHT/FEET/TONGUE) | Writes `runtime/manual_command.json` with `processed: false`; inference engine reads the mailbox every iteration and executes the command as if the AI predicted it |
| **Emergency Reset** | Button | UI directly patches `state.json` to set `emergency: false` and `tongue_streak: 0` — the only case where the UI writes state rather than config |

### Session Stats Bar

Three cards shown at the top of the main area:
- **SESSION UPTIME** — `hh:mm:ss` computed from `state.session_start` (Unix timestamp set when the engine starts)
- **COMMANDS FIRED** — `state.commands_count` counter incremented every time an MI prediction passes the confidence threshold
- **TOP CLASS** — The MI class with the highest count in `state.class_counts` across the whole session

### Status Row (6 Cards)

| Card | Source Field | Colors |
|------|-------------|--------|
| LIGHT | `state.light` (0–3) | Grey/Amber/Gold/White |
| FAN | `state.fan` (0–3) | Grey/Grey/Cyan/Red |
| TV | `state.tv` (bool) | Grey/Blue |
| MOTOR DETECTED | `state.mi_probs` top key | Per-class color |
| EMOTION | `state.ambient` | Per-quadrant color |
| EMERGENCY | `state.emergency` (bool) | Green/Red |

### Command Guide Row

Four panels highlight **which class the model currently predicts** (border glows with that class's color) and remind the user what action each class triggers.

### Smart Home SVG Panel

The `render_smart_home(state)` function generates inline SVG on every refresh. Key visual effects:

| Device | Visual Representation | Animation |
|--------|-----------------------|-----------|
| Light bulb | Circle fill color + radial glow | `filter: drop-shadow` radius proportional to brightness level (0, 30, 55, 80 px) |
| Fan | 4-blade SVG polygon | CSS `@keyframes spin` — speed changes with fan level (1.8s / 0.8s / 0.3s per revolution) |
| TV | Rectangle | Blue fill + blue drop-shadow glow when ON |
| Room walls | Background rect fill | Transitions over 2s to ambient mood color (`#0d1b3b` for sad, `#3b0d0d` for stressed, `#0d3b2e` for calm, `#3b380d` for excited) |
| Emergency | Red pulsing overlay | `@keyframes epulse` — alpha oscillates 0.08 → 0.45 at 0.9 s period |

### EEG Oscilloscope Panel

`render_oscilloscope(buffer_data)` renders a Plotly figure with 8 channels:
- Channels plotted: `Fz, FC3, FC1, FCz, FC2, FC4, C5, C3`
- Each channel occupies its own **horizontal lane** using a vertical offset of 100 µV
- Signal is **min-max scaled per channel** to fill 70% of its lane height — ensures all channels are visually comparable regardless of absolute amplitude
- Colors gradient from cyan (`#00d4ff`) to violet (`#ca04d8`) across the 8 channels
- X-axis spans 500 samples (2 seconds at 250 Hz) — the buffer is a sliding window `deque(maxlen=500)` maintained by the LSL simulator

### Confidence Gauge

A Plotly Indicator (radial gauge) with 3 color zones:
- **Red zone** (0 → 70% of threshold): confidence too low, no command will fire
- **Yellow zone** (70%–100% of threshold): approaching threshold
- **Green zone** (threshold → 100%): command zone — predictions here trigger smart home actions
- **Purple line**: the threshold value set by the slider

Two gauges shown side-by-side: one for MI (primary model), one for emotion.

### Model Comparison Panel

Shows the **probability distribution of both models** (Combined SVM and Riemannian) in ASCII bar format per epoch:
```
left    | ██████████ |  62.3%
right   | ██░░░░░░░░ |  18.7%
feet    | █░░░░░░░░░ |   9.2%
tongue  | ░░░░░░░░░░ |   9.8%
```
A "winner this epoch" line shows which model was more confident and by how much.

### Event Log

Scrollable log of the last 30 events with timestamp and colour-coded type badges:
- `MOTOR` (cyan) — MI inference result
- `EMOTION` (purple) — affective state update
- `MANUAL` (amber) — UI override command

A **CSV export button** lets users download the full log for analysis.

---

## 🔧 How to Train Models From Scratch

If you have the raw data and want to reproduce the trained models, follow this exact sequence:

### Step 1 — Preprocess BCI IV 2a

```bash
python -m pipeline.preprocess_bci2a
```
- Reads `data/bci_iv_2a/A01T.gdf` … `A09T.gdf`
- Outputs `processed/bci_iv_2a/A0xT_X.npy` (shape: `n_trials × 22 × 1000`) and `A0xT_y.npy`

### Step 2 — Build Combined Features

```bash
python -m pipeline.build_combined_features_bci2a
```
- Reads per-subject `_X.npy` arrays
- Computes Temporal + Spectral + FBCSP features per trial
- Outputs `processed/bci_iv_2a/A0xT_combined_X.npy` (shape: `n_trials × ~186`)

### Step 3 — Preprocess DEAP

```bash
python -m pipeline.preprocess_deap_emotion
python -m pipeline.extract_deap_features
```
- Reads `data/deap/data_preprocessed_python/s01.dat` … `s32.dat`
- Outputs `processed/deap/deap_features_X.npy` (shape: `N × 160`) and `deap_labels_quadrant.npy`

### Step 4 — Train All Motor Imagery Models

```bash
# Combined SVM (best for most subjects)
python -m pipeline.train_subjectwise_combined_save

# Riemannian SVM (best for A06T, A08T)
python -m pipeline.train_subjectwise_riemann_save

# EEGNet (deep learning baseline)
python -m pipeline.train_eegnet_mi

# Hybrid Fusion (EEG + emotion features)
python -m pipeline.train_hybrid_fusion
```

### Step 5 — Train Emotion Classifier

```bash
python -m pipeline.train_emotion_classifier
```
- Outputs `models/emotion_svm_classifier.pkl` and `models/emotion_scaler.pkl`

### Step 6 — Select Best Model Per Subject

```bash
python -m pipeline.export_best_all_models
```
- Compares Combined SVM accuracy vs Riemannian accuracy per subject
- Writes `models/best_model_per_subject.json` and copies winning models to `models/best_models/`

### Step 7 — Reconstruct Simulation Data

```bash
python scripts/reconstruct_data.py
```
- Concatenates all subject arrays into `processed/bci_iv_2a/bci2a_all_subjects_X.npy` for the LSL simulator

---

## 🔬 Mathematical Formulas (Thesis-Ready)

### CSP Objective

CSP finds spatial filters **W** such that:

```
W = argmax  [ var(W^T X_class1) / var(W^T X_class2) ]
```

Solved as a generalized eigenvalue problem:
```
Σ₁ w = λ Σ₂ w
```
where Σ₁, Σ₂ are the covariance matrices of the two classes. The eigenvectors corresponding to the largest and smallest eigenvalues are selected — they capture maximum variance for one class and minimum for the other.

### Band Power via Welch's Method

```
P(f₁, f₂) = ∫[f₁ to f₂] PSD(f) df  ≈  Σ PSD(fₖ) · Δf   for fₖ ∈ [f₁, f₂]
```

PSD is estimated using Welch's periodogram: the signal is divided into overlapping segments, each segment is windowed and FFT-computed, and the results are averaged.

### Riemannian Distance

For two covariance matrices P, Q on the SPD manifold:
```
δR(P, Q) = || log(P^{-½} Q P^{-½}) ||_F
```

The tangent space projection at reference point P_ref:
```
S = P_ref^{½} · log(P_ref^{-½} · Cov · P_ref^{-½}) · P_ref^{½}
```
The upper triangular of the symmetric S (vectorized) forms the feature vector fed to the SVM.

### Confidence Threshold Decision Rule

```
action_fires = True   if   max(softmax_output) ≥ θ_confidence
action_fires = False  otherwise
```
where `θ_confidence` is the user-adjustable threshold (default 0.65).

### Noise Injection Formula

```
epoch_noisy = epoch_clean + N(0, noise_level × σ_epoch)
```
where `σ_epoch = std(epoch_clean)` — this ensures noise scales with signal amplitude, making the visual disruption proportional to the original signal strength.

### Valence-Arousal Quadrant Mapping

```
quadrant(v, a) =
  0   if v < 5 AND a < 5     (sad / fatigued)
  1   if v < 5 AND a ≥ 5     (stressed / anxious)
  2   if v ≥ 5 AND a < 5     (calm / content)
  3   if v ≥ 5 AND a ≥ 5     (excited / happy)
```

---

## 🔄 Live Config Hot-Reload Mechanism

The inference engine never stops to reload — it reads `runtime/config.json` **every 100 ticks** (approximately every 1 second at 10 ms sleep). This allows settings to change live without restarting:

```
Tick 0:         Engine reads config → subj=A01T, threshold=0.65
User changes:   Sidebar → selects A07T, threshold=0.75
Tick ~100:      Engine re-reads config → switches to A07T model (cached from prewarm)
Tick ~100+:     All subsequent MI predictions use A07T's Combined SVM
```

The model is cached after first load (`_models_cache` dict keyed by subject ID) so switching subjects is near-instant after the initial prewarm. The **prewarm** step at startup preloads all 9 subjects' models into memory so there is no latency spike when the user first selects a new subject.

---

## 📬 Manual Command Override System

The manual override is a **JSON mailbox pattern**:

```
UI writes:      runtime/manual_command.json  →  { "action": "left", "processed": false }
Engine reads:   On every tick, checks if file exists AND processed == false
Engine acts:    Runs apply_mi_command() as if the AI predicted "left"
Engine marks:   Overwrites file with { ..., "processed": true }
```

This is why pressing a manual button in the UI doesn't wait for the engine — the UI just drops the command file and the engine picks it up on its next 10 ms tick. The `processed` flag prevents the same command being executed twice.

---

## 🧬 Subject-Wise Modelling — Why Not Cross-Subject?

EEG signals are **highly subject-specific**. The spatial distribution of brain activity, the amplitude of mu-rhythm desynchronisation, and signal-to-noise ratio vary enormously between individuals due to:

1. **Skull thickness** — thicker skulls attenuate signals more
2. **Cortical folding patterns** — different gyral geometry means the same motor action produces different scalp topographies
3. **Task engagement** — different subjects engage differently with motor imagery instructions

A **cross-subject model** trained on all 9 subjects together would underfit each individual. NeuroFusion instead trains **one model per subject**, selects the best architecture for that subject, and stores 9 separate model files. At runtime, the active subject is selected in the sidebar.

---

## 🔀 Ensemble Model (`pipeline/ensemble_subjectwise.py`)

In addition to the per-subject single best models, the codebase includes an ensemble approach using **soft voting**:

1. Train **Combined SVM** on combined features for subject X
2. Train **Riemannian SVM** on raw epochs for subject X  
3. At test time, average the probability vectors from both models:
   ```
   P_ensemble = 0.5 × P_combined + 0.5 × P_riemannian
   label = argmax(P_ensemble)
   ```

The ensemble is evaluated but the **runtime inference engine runs both models in parallel** and displays them separately in the Model Comparison panel — giving the user live insight into agreement or disagreement between the two approaches.

---

## 📊 Per-Subject Accuracy Breakdown (From Actual Training)

```
Subject | Best Model     | Held-Out Test Accuracy
--------|----------------|------------------------
A01T    | Combined SVM   | 82.76%
A02T    | Combined SVM   | 72.41%   ← lowest — challenging subject
A03T    | Combined SVM   | 86.21%
A04T    | Combined SVM   | 75.86%
A05T    | Combined SVM   | 74.14%
A06T    | Riemannian     | 56.90%   ← Riemannian won here
A07T    | Combined SVM   | 89.66%   ← near top
A08T    | Riemannian     | 93.10%   ← best subject overall
A09T    | Combined SVM   | 75.86%
--------|----------------|------------------------
Mean    | Combined SVM   | 77.59%
Mean    | Riemannian     | 69.54%
```

**Why does A08T perform best?** — Subject A08T likely produces strong, consistent motor imagery signals with low noise, making even the Riemannian approach (which is more geometry-preserving but less feature-engineered) highly accurate.

**Why does A06T score lowest?** — Some subjects produce weak or atypical motor imagery EEG patterns. Riemannian still edges out Combined SVM for A06T, suggesting the covariance geometry is more informative than hand-crafted spectral features for this subject.

**Chance level for 4 classes = 25%.** All subjects are well above chance. This is consistent with BCI Competition IV 2a state-of-the-art results reported in literature (typically 70–85% for CSP+SVM pipelines).

---

## 🛠️ Troubleshooting Guide

### Issue: Inference engine starts but no predictions appear

**Cause:** LSL simulators not running, or wrong stream name.

**Fix:**
1. Confirm Terminals 1 and 2 are running (`lsl_simulator_mi.py` and `lsl_simulator_emo.py`)
2. Check the engine prints `✅ Connected` — if it hangs at `📡 Connecting...`, the streams aren't visible
3. On Windows, allow Python through the firewall when prompted — LSL uses local UDP broadcast

### Issue: Dashboard shows "Waiting for Inference Engine"

**Cause:** `runtime/state.json` doesn't exist yet.

**Fix:** Start the inference engine (Terminal 3) first. The dashboard polls every 800 ms and will update automatically once the file appears.

### Issue: All predictions are the same class (e.g., always "left")

**Cause:** Biased SVM model for this subject, or the subject's data has class imbalance.

**Fix:** Switch to a different subject in the sidebar (A07T and A08T tend to have the most balanced predictions).

### Issue: `FileNotFoundError: bci2a_all_subjects_X.npy`

**Fix:**
```bash
python scripts/reconstruct_data.py
```

### Issue: TensorFlow/EEGNet models fail to load

**Cause:** TensorFlow not installed, or the `.h5` files were not generated.

**Fix:** The system automatically falls back to the Combined SVM or Riemannian model (which are pure scikit-learn `.pkl` files and require no TensorFlow). This is why 9 `.pkl` files exist in `models/best_models/`.

### Issue: Model `.pkl` files are 0 bytes (Git LFS issue)

**Fix:**
```bash
git lfs install
git lfs pull
```

### Issue: Dashboard is slow / high CPU

**Cause:** The 800 ms auto-refresh triggers a full Streamlit rerun including JSON file I/O.

**Fix:** Reduce refresh rate by editing `st_autorefresh(interval=800)` in `ui/app.py` to `interval=2000` for slower machines.

### Issue: Emotion always shows same quadrant

**Cause:** Non-`live_demo` scenario is selected, which forces a fixed emotion.

**Fix:** Change Scenario in sidebar to **Live Demo**.

---

## 🌐 LSL (Lab Streaming Layer) — Technical Details

**LSL** is an open-source standard for streaming time-series data over a local network, widely used in neuroscience research. It handles:

- **Time synchronization** — all streams share a common clock via NTP-like correction
- **Metadata** — stream name, type, channel count, sample rate are broadcast via mDNS
- **Pull model** — consumers call `pull_chunk()` which returns all samples buffered since the last call (non-blocking with `timeout=0.0`)

In NeuroFusion:
- The simulators **push** at 250 Hz (MI) and 1 Hz (Emotion)
- The inference engine **pulls** in a 10 ms loop, accumulating chunks
- When 4 seconds of data (1000 samples at 250 Hz) are buffered, inference is triggered
- After inference, the buffer is **shifted by 50%** (500 samples) — a 50% overlapping window for continuous inference without gaps

---

## 🧠 The Inference Engine — Tick-by-Tick Execution

Every 10 ms (`time.sleep(0.01)`), the engine performs:

```
Tick N:
  1. Read config.json if 100 ticks have passed (hot-reload)
  2. Pull emotion chunk from NeuroFusion_Emo LSL (non-blocking)
     → If data available: run emotion SVM → write state
  3. Check manual_command.json mailbox
     → If unprocessed command: execute MI action → mark processed
  4. Pull MI chunk from NeuroFusion_MI LSL (non-blocking)
     → Append to mi_buffer
  5. If mi_buffer >= 1000 samples:
     a. Slice epoch (first 1000 samples)
     b. Run Combined SVM (primary)
     c. Run Riemannian SVM (secondary, if model exists)
     d. Evaluate confidence threshold
     e. If confidence ≥ threshold: apply MI command
     f. Write updated state to state.json (atomic: .tmp → rename)
     g. Trim buffer by 50% (sliding window)
  6. Sleep 10 ms
```

The atomic write (`tmp → rename`) prevents the UI from reading a partially-written JSON file.

---

## 🗃️ Data Formats Reference

### `processed/bci_iv_2a/A0xT_X.npy`
- Shape: `(n_trials, 22, 1000)`
- dtype: `float64`
- Contents: bandpass-filtered (8–30 Hz) EEG epochs, units approximately µV
- n_trials: typically 288 per subject (72 trials × 4 classes)

### `processed/bci_iv_2a/A0xT_y.npy`
- Shape: `(n_trials,)`
- dtype: `int64`
- Values: `{0: left, 1: right, 2: feet, 3: tongue}`

### `processed/bci_iv_2a/A0xT_combined_X.npy`
- Shape: `(n_trials, ~186)`
- dtype: `float32`
- Contents: `[88 temporal features | 66 spectral features | 32 FBCSP features]`

### `processed/deap/deap_features_X.npy`
- Shape: `(N_trials, 160)`
- dtype: `float32`
- Contents: log band-power per channel per band; `32 channels × 5 bands`

### `processed/deap/deap_labels_quadrant.npy`
- Shape: `(N_trials,)`
- dtype: `int32`
- Values: `{0: sad/fatigued, 1: stressed/anxious, 2: calm/content, 3: excited/happy}`

### `runtime/raw_eeg_buffer.json`
- Written by LSL simulator every 25 samples (100 ms)
- Contains `{ "buffer": [[ch0..ch7], ...], "timestamp": unix }` — last 500 samples of first 8 channels for oscilloscope visualization

---

## 🎯 Design Decisions & Rationale

### Why Streamlit and not a web framework (Flask/React)?

Streamlit was chosen for rapid prototyping of a data-science dashboard. A production BCI system would use a more responsive stack (e.g., FastAPI backend + React frontend), but Streamlit's auto-refresh and native Plotly support made it the fastest path to a working demo.

### Why local file IPC instead of MQTT or WebSocket?

Early versions used **MQTT** (paho-mqtt is still in requirements.txt). It was replaced with a local JSON file (`runtime/state.json`) because:
1. No broker setup required — zero external dependencies
2. Atomic rename on write prevents partial reads
3. The entire state is always one JSON read away — no subscription management
4. Simpler debugging (you can just open the file in a text editor)

For a real IoT deployment with physical smart home devices, MQTT or WebSocket would be reintroduced.

### Why 4-second epochs?

4 seconds is the standard epoch length in motor imagery literature (BCI Competition IV 2a was recorded with 4 s trial windows). Shorter epochs reduce latency but reduce the amount of spectral information available. Longer epochs increase accuracy but feel unresponsive (delay between thought and action). 4 s is the accepted balance.

### Why 50% overlap on the inference window?

Full non-overlapping windows (4 s → wait 4 s → next inference) create a jarring rhythm and miss predictions that span window boundaries. 50% overlap means a new prediction fires every 2 seconds, providing a more responsive experience while using 80% of the same data.

### Why subject-wise training instead of transfer learning?

Transfer learning across EEG subjects is an active research area (domain adaptation, alignment techniques). For this project, the simpler subject-wise approach was chosen because:
1. All subjects' data was available offline
2. The per-subject models demonstrate higher absolute accuracy
3. The added complexity of domain adaptation was beyond the project scope

---

## 📦 Key Python Dependencies — Deep Dive

### MNE-Python (`mne`)
The gold-standard EEG/MEG analysis library. Used for:
- `mne.io.read_raw_gdf()` — reads BCI Competition GDF format files
- `raw.notch_filter()`, `raw.filter()` — digital filtering
- `mne.events_from_annotations()` — converts annotation strings to event arrays
- `mne.Epochs()` — segments continuous recording into trials
- `mne.decoding.CSP` — Common Spatial Patterns implementation

### PyRiemann (`pyriemann`)
Implements Riemannian geometry operations on covariance matrices:
- `Covariances(estimator='scm')` — computes sample covariance matrix per epoch
- `TangentSpace()` — projects covariance matrices to their tangent space, producing Euclidean-compatible feature vectors

### PyLSL (`pylsl`)
Python bindings for the Lab Streaming Layer library:
- `StreamInfo` — declares a stream with metadata
- `StreamOutlet` — broadcasts samples over the local network
- `resolve_byprop('name', 'NeuroFusion_MI')` — discovers streams by name
- `StreamInlet` — subscribes to a stream and pulls samples

### TensorFlow/Keras (`tensorflow`)
Used only for EEGNet and Hybrid models:
- Lazy-imported to avoid slowing down startup when only SVM models are needed
- If TensorFlow is unavailable, the system transparently falls back to `.pkl` models

### Joblib (`joblib`)
Serializes and deserializes scikit-learn pipelines:
- `joblib.dump(model, path)` — saves a fitted pipeline
- `joblib.load(path)` — restores it; compressed `.pkl` format
- Used for all SVM models and CSP scaler objects

---

## 📝 Glossary

| Term | Full Form | Plain English Meaning |
|------|-----------|----------------------|
| BCI | Brain-Computer Interface | System that reads brain signals and translates them into computer commands |
| EEG | Electroencephalography | Recording of electrical activity from the scalp using metal electrodes |
| MI | Motor Imagery | Mentally imagining a movement without physically performing it |
| ERD | Event-Related Desynchronization | Decrease in mu/beta power when motor cortex activates |
| CSP | Common Spatial Patterns | Algorithm that finds the optimal electrode combinations to separate two MI classes |
| FBCSP | Filter Bank CSP | CSP applied in multiple frequency sub-bands for richer features |
| SVM | Support Vector Machine | Classification algorithm that finds the optimal boundary between classes |
| RBF | Radial Basis Function | A popular SVM kernel that handles non-linearly separable data |
| PSD | Power Spectral Density | Measurement of signal energy at each frequency |
| SPD | Symmetric Positive Definite | Mathematical property of covariance matrices required for Riemannian geometry |
| LSL | Lab Streaming Layer | Network protocol for real-time streaming of neuroscience data |
| IPC | Inter-Process Communication | Any method for passing data between separate running programs |
| µV | Micro-Volt | Unit of EEG amplitude (1 µV = 0.000001 Volt) |
| Hz | Hertz | Samples per second (250 Hz = 250 samples per second) |
| Epoch | — | A fixed-length segment of EEG data cut around an event |
| Softmax | — | Function that converts a vector of numbers into a probability distribution summing to 1.0 |
| Proba | — | Short for probability — the model's confidence in each class |
| Quadrant | — | One of four emotion zones in the Valence-Arousal 2D space |
| Geodesic | — | The shortest path between two points on a curved manifold |
| Tangent Space | — | Flat (Euclidean) local approximation to a curved manifold at a reference point |
| mDNS | Multicast DNS | Used by LSL to advertise streams on the local network without a central server |

---

## ❓ FAQ

**Q: Can this system actually read someone's mind?**
A: No — it reads the *electrical pattern associated with imagining a specific movement*. It cannot decode abstract thoughts, memories, or speech. It can only distinguish between the 4 pre-trained motor imagery classes.

**Q: Why 4 classes and not more?**
A: The BCI Competition IV 2a dataset was designed with these 4 classes. More classes reduce accuracy significantly (random chance drops from 50% for 2-class to 25% for 4-class). Research shows 4-class MI is the practical limit for non-invasive EEG with current technology.

**Q: Does changing the subject affect the emotion model?**
A: No — there is one global emotion SVM trained on the entire DEAP dataset (all 32 subjects pooled). Only the MI model switches per subject.

**Q: Why does the oscilloscope show 8 channels when the dataset has 22?**
A: The other 14 channels are streamed and used for inference, but showing all 22 on the oscilloscope would make the display unreadably crowded. The 8 displayed channels (Fz, FC3, FC1, FCz, FC2, FC4, C5, C3) are the most diagnostically informative for motor imagery.

**Q: What is the latency from thought to device action?**
A: Minimum theoretical latency is 4 seconds (epoch length) + ~50 ms (inference) + 800 ms (dashboard refresh) = ~4.85 seconds. In practice the sliding window means the system re-evaluates every 2 seconds, so the average latency from motor imagery onset to visual confirmation is approximately 2–3 seconds.

**Q: Why does the tongue command need to be predicted twice?**
A: Tongue imagery is the least distinct class in EEG — it produces signals similar to facial muscle activity. A single prediction could easily be a false positive. Requiring two consecutive tongue predictions makes accidental emergency activation extremely unlikely while still being usable.

**Q: Can I use a real EEG headset?**
A: Yes. Any headset with a Python LSL driver (e.g., OpenBCI, Emotiv, Muse via BlueMuse, g.tec) can replace the simulators. You would run the headset's LSL streamer instead of `lsl_simulator_mi.py`, keeping the stream name as `NeuroFusion_MI`. The inference engine and dashboard require no changes.

**Q: What happens if the inference engine crashes during a session?**
A: The dashboard shows the last known `state.json` values (frozen). The emergency state is preserved so no accidental deactivation occurs. Restarting the engine resumes from a fresh state (all devices reset to OFF).

---

## ⚠️ Known Limitations

1. **Calibration required per new user** — the trained models are calibrated on BCI Competition IV 2a subjects. A new physical user would need a calibration session (~5–10 minutes of recorded motor imagery) to achieve full accuracy.

2. **No artifact rejection at runtime** — the preprocessing pipeline rejects artifact-contaminated epochs offline. The live inference engine does not currently detect blinks, jaw clenches, or movement artifacts. A production system would add an online artifact rejection step.

3. **Simulated data only** — no real EEG hardware is currently connected. The system is architecturally ready (just swap the simulator), but clinical validation on real subjects has not been performed.

4. **Emotion labels biased toward calm** — the DEAP dataset's ground-truth ratings are known to be class-imbalanced (more calm/content trials). The SVM learned this bias, which is why `live_demo` mode artificially cycles through all 4 emotion quadrants to demonstrate the full system.

5. **No online adaptation** — EEG signals drift over a session (fatigue, electrode impedance changes). A production BCI would update the model parameters in real-time. NeuroFusion uses static trained models.

6. **Windows-primary** — the `start.bat` launcher and `venv\Scripts\activate` paths are Windows-specific. The Python code itself is cross-platform.

---

## 🔁 Data Flow — Annotated End-to-End

```
STEP 0: OFFLINE (done once before running the system)
────────────────────────────────────────────────────────
BCI IV 2a .gdf files         DEAP .dat files
        │                           │
        ▼                           ▼
preprocess_bci2a.py       preprocess_deap_emotion.py
        │                    extract_deap_features.py
        ▼                           │
A0xT_X.npy, A0xT_y.npy    deap_features_X.npy
        │                  deap_labels_quadrant.npy
        ▼                           │
build_combined_features             │
A0xT_combined_X.npy                │
        │                           │
        ▼                           ▼
train_subjectwise_combined  train_emotion_classifier
train_subjectwise_riemann           │
train_eegnet_mi                     ▼
        │                  emotion_svm_classifier.pkl
        ▼                  emotion_scaler.pkl
export_best_all_models
        │
        ▼
best_model_per_subject.json
best_models/A0xT_best.pkl

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1: RUNTIME (4 processes running simultaneously)
────────────────────────────────────────────────────────

Process 1:  lsl_simulator_mi.py
  Loads bci2a_all_subjects_X.npy (memory-mapped)
  Picks random trial every 4 seconds
  Injects Gaussian noise if noise_level > 0 (reads config.json)
  Pushes 250 samples/sec → LSL stream "NeuroFusion_MI"
  Side-writes first 8 channels → runtime/raw_eeg_buffer.json
                                              │
                                              │ (for oscilloscope)
                                              ▼
Process 2:  lsl_simulator_emo.py              ui/app.py reads it
  Loads deap_features_X.npy                  at 800ms intervals
  Broadcasts 1 feature vector/sec
  → LSL stream "NeuroFusion_Emo"

               │                   │
               ▼ LSL pull          ▼ LSL pull
         Process 3: inference_engine.py
           Pulls MI chunks → accumulates in mi_buffer
           At 1000 samples: extract epoch (22×1000)
             → Combined SVM: compute combined features → predict
             → Riemannian SVM: covariance → tangent space → predict
             → If max_prob ≥ threshold: apply_mi_command()
           Pulls Emo chunks every iteration:
             → Emotion SVM: scale → predict
           Every write: state.json (atomic .tmp → rename)
                │
                ▼
          runtime/state.json
                │
                ▼ polls every 800ms
         Process 4: ui/app.py (Streamlit)
           Reads state.json
           Reads config.json  ◄──── user changes sidebar controls
           Writes config.json ──────────────────────────────────►
           Writes manual_command.json (button press) ──────────►
           Renders: SVG, oscilloscope, gauges, probability bars,
                    event log, session stats
```

---

## 🧑‍💻 Code Quality & Architecture Patterns Used

| Pattern | Where Used | Why |
|---------|-----------|-----|
| **Lazy loading / cache dict** | `_models_cache`, `_keras_cache`, `_csp_cache` | Avoids re-loading large model files on every prediction call |
| **Atomic file write** | `state.json` via `.tmp` + rename | Prevents the UI reading a half-written JSON |
| **Mailbox / polling** | `manual_command.json` | Decouples UI button press from inference engine execution |
| **Memory-mapped arrays** | `np.load(..., mmap_mode='r')` | Large `.npy` files are not loaded fully into RAM; OS pages them on demand |
| **Hot-reload config** | `read_config(tick)` with `_CFG_INTERVAL` | Settings update live without restarting any process |
| **Prewarm** | `_prewarm_models()` at startup | Eliminates first-prediction latency spike |
| **Sliding window** | `mi_buffer[-(epoch_samples//2):]` | Continuous 50% overlapping inference without gaps |
| **Soft voting ensemble** | `p_prob + s_prob` | Combines two model outputs into a single more-robust prediction |
| **Separation of concerns** | 4 separate processes | Each process has exactly one responsibility; any can be replaced independently |
