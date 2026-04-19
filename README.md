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
