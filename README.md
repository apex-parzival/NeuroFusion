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

## 🏗️ The Architecture: How Does the Data Flow?
Since we don't have a real physical headset yet, we built "simulators" that pretend to be headsets. Here is the step-by-step journey of the data:

### Step 1: The Headset Simulators (`scripts/lsl_simulator_mi.py` & `emo.py`)
* **What it does:** These scripts load the giant files of pre-recorded brain waves (`.npy` files). They slice the data up and broadcast it over our computer's Wi-Fi network continuously, exactly like a real physical headset transmitting via Bluetooth.
* **Why it flows there:** To create a realistic environment. When we buy a real headset later, we just turn off these scripts and connect the real headset. Our application won't even know the difference!

### Step 2: The Data Buffer/Listener (Internal Pipeline)
* **What it does:** The main app "listens" to the network. Instead of looking at 1 millisecond of brain data (which is too short to understand), it gathers a "chunk" of time—like recording a 4-second video clip of the brain waves. This chunk is called an **epoch**.

### Step 3: Feature Extraction (Cleaning the Noise)
* **What it does:** Raw brain waves look like scribbles. We pass the 4-second "epoch" through filters (specifically, something called CSP - Common Spatial Pattern). 
* **Why it flows there:** This cleans off the "noise" and highlights the exact part of the brain wave that proves the person is thinking about their left hand.

### Step 4: The AI Brain (`pipeline/ui_model_loader.py`)
* **What it does:** The cleaned data goes into our Machine Learning models. We have trained complex math models (like EEGNet or Support Vector Machines) on hours of brain data. 
* **Why it flows there:** The AI looks at the shape of the cleaned wave and makes a guess: *"I am 95% sure this person is thinking about their feet!"*

### Step 5: The Smart Home Application (`ui/app.py`)
* **What it does:** The final guess from the AI arrives at the User Interface. 
* **Where it goes after:** The app translates the guess into a rule. If AI says "Feet" -> The app updates the TV to "ON". The user sees the TV turn on instantly on their dashboard. Loop complete!

---

## 🚀 How to Run the Project
Because we split this into a real-life architecture, you must run the **Simulators** first, and then the **Application**.

### 1. Start the Brain Simulators
Open a terminal, activate your Python environment, and start broadcasting Motor Imagery (Thoughts):
```bash
python scripts/lsl_simulator_mi.py
```
Open a **second** terminal and start broadcasting Emotions (Mood):
```bash
python scripts/lsl_simulator_emo.py
```

### 2. Start the Smart Home Hub
Open a **third** terminal and launch the primary User Interface:
```bash
streamlit run ui/app.py
```
Open your web browser, click "Next Brain Step", and watch as the simulated brain waves are captured, decoded by the AI, and used to control the virtual Smart Home!
