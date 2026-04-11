@echo off
echo ============================================================
echo  VoiceTTSr - Qwen3-TTS Environment Setup
echo ============================================================
echo.

set QWEN_ENV=qwen-env-py310
set PYTHON=python

echo [1/4] Creating virtual environment: %QWEN_ENV%
%PYTHON% -m venv %QWEN_ENV%
if errorlevel 1 (
    echo ERROR: Failed to create venv. Make sure Python 3.10+ is installed.
    pause
    exit /b 1
)

echo.
echo [2/4] Installing PyTorch (CUDA 11.8)...
%QWEN_ENV%\Scripts\pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo.
echo [3/4] Installing Qwen3-TTS and dependencies...
%QWEN_ENV%\Scripts\pip install qwen-tts soundfile accelerate transformers huggingface-hub

echo.
echo [4/4] Installing optional resampling library...
%QWEN_ENV%\Scripts\pip install librosa

echo.
echo ============================================================
echo  Qwen env ready! → %QWEN_ENV%
echo  Qwen model will download on first use (~6 GB).
echo ============================================================
pause
