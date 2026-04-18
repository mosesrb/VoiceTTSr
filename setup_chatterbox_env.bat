@echo off
setlocal enabledelayedexpansion

echo ============================================
echo  VoiceTTSr - Chatterbox Environment Setup
echo ============================================
echo.

:: ── Check we're in the right folder ──────────────────────────────────────
if not exist "voice_cloner_gui.py" (
    echo ERROR: Run this bat file from your VoiceTTSr app folder.
    echo        e.g. E:\MachineApps\VoiceTTSr\
    pause
    exit /b 1
)

:: ── Find Python 3.11 from conda ───────────────────────────────────────────
set CONDA_PY=
for %%P in (
    "%USERPROFILE%\miniconda3\envs\chatterbox\python.exe"
    "%USERPROFILE%\anaconda3\envs\chatterbox\python.exe"
    "%USERPROFILE%\Miniconda3\envs\chatterbox\python.exe"
    "%USERPROFILE%\Anaconda3\envs\chatterbox\python.exe"
    "C:\ProgramData\miniconda3\envs\chatterbox\python.exe"
    "C:\ProgramData\anaconda3\envs\chatterbox\python.exe"
) do (
    if exist %%P (
        set CONDA_PY=%%P
        echo Found conda chatterbox env at: %%P
        goto :found_conda
    )
)

echo.
echo Could not auto-find your conda chatterbox env.
echo Please enter the full path to your conda chatterbox python.exe:
echo Example: C:\Users\Neongiant\miniconda3\envs\chatterbox\python.exe
echo.
set /p CONDA_PY="Path: "
if not exist "%CONDA_PY%" (
    echo ERROR: File not found: %CONDA_PY%
    pause
    exit /b 1
)

:found_conda
echo.

:: ── Wipe old broken env ───────────────────────────────────────────────────
echo [1/5] Removing old chatterbox-env-py311 if it exists...
if exist "chatterbox-env-py311" (
    rmdir /s /q "chatterbox-env-py311"
    echo       Removed old env.
) else (
    echo       No old env found, continuing.
)
echo.

:: ── Create fresh venv ─────────────────────────────────────────────────────
echo [2/5] Creating fresh virtual environment...
%CONDA_PY% -m venv chatterbox-env-py311
if errorlevel 1 (
    echo ERROR: Failed to create venv. Check your conda python path.
    pause
    exit /b 1
)

set PY=chatterbox-env-py311\Scripts\python.exe
if not exist "%PY%" (
    echo ERROR: python.exe not found in chatterbox-env-py311\Scripts\
    echo        Venv creation may have failed.
    pause
    exit /b 1
)
echo       Venv created OK.
echo.

:: ── Upgrade pip ───────────────────────────────────────────────────────────
echo [3/5] Upgrading pip...
%PY% -m pip install --upgrade pip --quiet
echo       pip upgraded.
echo.

:: ── Install PyTorch cu118 (matches RTX 2080 Super) ────────────────────────
echo [4/5] Installing PyTorch 2.6.0 cu118 (RTX 2080 Super)...
echo       This may take a few minutes (~2.5 GB download)...
%PY% -m pip install torch==2.6.0+cu118 torchaudio==2.6.0+cu118 torchvision==0.21.0+cu118 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo ERROR: PyTorch install failed.
    pause
    exit /b 1
)
echo       PyTorch installed.
echo.

:: ── Install chatterbox-tts and exact deps ────────────────────────────────
echo [5/5] Installing chatterbox-tts and pinned dependencies...
%PY% -m pip install chatterbox-tts
if errorlevel 1 (
    echo ERROR: chatterbox-tts install failed.
    pause
    exit /b 1
)

:: Pin the exact versions chatterbox-tts 0.1.7 needs
echo       Pinning compatible dependency versions...
%PY% -m pip install ^
    "numpy==1.26.4" ^
    "safetensors==0.5.3" ^
    "huggingface_hub==0.25.1" ^
    "diffusers==0.29.0" ^
    "transformers==4.43.3" ^
    "tokenizers==0.19.1" ^
    --force-reinstall --no-deps
echo       Dependencies pinned.
echo.

:: ── Verify install ────────────────────────────────────────────────────────
echo ── Verifying installation ──
%PY% -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
%PY% -c "from chatterbox.tts import ChatterboxTTS; print('chatterbox: OK')"
if errorlevel 1 (
    echo.
    echo WARNING: chatterbox import check failed.
    echo          Try running chatterbox_worker.py manually to see the error:
    echo          chatterbox-env-py311\Scripts\python.exe chatterbox_worker.py
) else (
    echo.
    echo ============================================
    echo  SUCCESS! Chatterbox env is ready.
    echo  Load the model from the GUI now.
    echo ============================================
)

echo.
pause
