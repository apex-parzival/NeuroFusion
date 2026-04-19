"""
NeuroFusion Demo Launcher
Starts all backend processes and opens the dashboard with ONE command.
Usage: python demo.py
"""
import subprocess, sys, time, webbrowser, json, signal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
RUNTIME_DIR = PROJECT_ROOT / "runtime"
STATE_FILE = RUNTIME_DIR / "state.json"
PYTHON = sys.executable
processes = []

def cleanup(sig=None, frame=None):
    print("\n🛑 Shutting down NeuroFusion Demo...")
    for p in processes:
        try: p.terminate()
        except: pass
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup)

def launch(cmd_args, label):
    print(f"  ▶ Starting {label}...")
    p = subprocess.Popen([PYTHON] + cmd_args, cwd=PROJECT_ROOT)
    processes.append(p)
    return p

def wait_for_backend(timeout=30):
    print("  ⏳ Waiting for inference engine...")
    # Give some initial time for processes to start writing the file
    time.sleep(3)
    for _ in range(timeout * 2):
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                if data.get("state"): return True
            except: pass
        time.sleep(0.5)
    return False

if __name__ == "__main__":
    print("\n🧠 NeuroFusion Demo Launcher")
    print("=" * 40)
    
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    
    # Optional: empty state file if it exists to ensure freshness
    try:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
    except Exception: pass

    launch(["scripts/lsl_simulator_mi.py"], "Motor Imagery Simulator (250 Hz)")
    time.sleep(2)
    launch(["scripts/lsl_simulator_emo.py"], "Emotion Simulator (1 Hz)")
    time.sleep(1)
    launch(["-m", "pipeline.inference_engine"], "Real-time Inference Engine")
    time.sleep(3)

    if not wait_for_backend():
        print("⚠️  Backend slow to start — launching dashboard anyway.")

    streamlit_proc = subprocess.Popen(
        [PYTHON, "-m", "streamlit", "run", "ui/app.py",
         "--server.headless=true", "--server.port=8501"],
        cwd=PROJECT_ROOT
    )
    processes.append(streamlit_proc)
    time.sleep(3)

    print("\n✅ NeuroFusion is live!")
    print("   → Dashboard: http://localhost:8501")
    webbrowser.open("http://localhost:8501")
    print("\n   Press Ctrl+C to stop all processes.\n")

    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        cleanup()
