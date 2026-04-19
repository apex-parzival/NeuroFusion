@echo off
cd /d "%~dp0"

echo.
echo  ==========================================
echo   NEUROFUSION - Starting all services...
echo  ==========================================
echo.

echo [0/4] Checking dependencies...
call .venv\Scripts\activate
pip install -q plotly pandas streamlit-extras streamlit-autorefresh 2>nul
echo     Done.
echo.

echo [1/4] Starting MI Simulator...
start "NeuroFusion: MI Simulator" cmd /k ".venv\Scripts\activate && python scripts\lsl_simulator_mi.py"
timeout /t 4 /nobreak >nul

echo [2/4] Starting Emotion Simulator...
start "NeuroFusion: Emotion Simulator" cmd /k ".venv\Scripts\activate && python scripts\lsl_simulator_emo.py"
timeout /t 4 /nobreak >nul

echo [3/4] Starting Inference Engine...
start "NeuroFusion: Inference Engine" cmd /k ".venv\Scripts\activate && python pipeline\inference_engine.py"
timeout /t 6 /nobreak >nul

echo [4/4] Starting Dashboard...
start "NeuroFusion: Dashboard" cmd /k ".venv\Scripts\activate && cd ui && streamlit run app.py"

echo.
echo  All services started. Browser will open automatically.
echo  Close this window when done or press any key to exit.
echo.
pause
