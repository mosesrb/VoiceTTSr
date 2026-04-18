@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  VoiceTTSr Professional - Universal Installer
echo ============================================================
echo.

:: 1. Check Python
where.exe python.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.10 and add it to PATH.
    pause
    exit /b 1
)

:: 2. Setup GUI Environment (gui-env)
echo [1/5] Setting up GUI Environment (gui-env)...
if not exist "gui-env" (
    python -m venv gui-env
)
gui-env\Scripts\python.exe -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install GUI dependencies. Check your internet connection.
    pause
    exit /b 1
)

:: 3. Setup XTTS v2 Environment
echo.
echo [2/5] Setting up XTTS v2 Environment (xtts-env-py310)...
if not exist "xtts-env-py310" (
    python -m venv xtts-env-py310
)
xtts-env-py310\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
xtts-env-py310\Scripts\python.exe -m pip install TTS==0.22.0 transformers==4.37.2

:: 4. Setup Qwen3-TTS Environment
echo.
echo [3/5] Setting up Qwen3-TTS Environment (qwen-env-py310)...
if not exist "qwen-env-py310" (
    python -m venv qwen-env-py310
)
qwen-env-py310\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
qwen-env-py310\Scripts\python.exe -m pip install qwen-tts soundfile accelerate transformers huggingface-hub librosa

:: 5. Setup RVC v2 Environment
echo.
echo [4/5] Setting up RVC v2 Environment (rvc-env)...
call setup_rvc_env.bat

:: 6. Download Baseline Models
echo.
echo [5/5] Downloading Baseline RVC Resources & Models...
python download_resources.py

echo.
echo ============================================================
echo  INSTALLATION COMPLETE!
echo  Click 'VoiceTTSr.bat' on your desktop to launch.
echo ============================================================
pause
