@echo off
set "ENV=rvc-env"

echo ========================================================
echo SETTING UP RVC ENVIRONMENT (Option C)
echo ========================================================
echo.

if not exist "%ENV%" (
    echo Creating virtual environment...
    python -m venv "%ENV%"
)

echo.
echo [1/6] Bootstrapping pip to a version that handles legacy metadata...
REM pip >= 24.1 rejects omegaconf 2.0.x metadata. Pin to 23.3.2 for installs.
"%ENV%\Scripts\python.exe" -m pip install "pip==23.3.2" wheel "setuptools==69.5.1"

echo.
echo [2/6] Installing PyTorch (CUDA 11.8)...
"%ENV%\Scripts\python.exe" -m pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

echo.
echo [3/6] Installing audio and ML dependencies...
"%ENV%\Scripts\python.exe" -m pip install transformers librosa==0.9.1 scipy soundfile numpy==1.23.5 faiss-cpu==1.7.3 pydub loguru tqdm fsspec av ffmpeg-python torchcrepe python-multipart

echo.
echo [4/6] Installing omegaconf 2.1.1 (has get_ref_type, valid metadata, compatible with fairseq + rvc-python)...
"%ENV%\Scripts\python.exe" -m pip install "omegaconf==2.1.1"

echo.
echo [5/6] Installing fairseq...
"%ENV%\Scripts\python.exe" -m pip install "https://github.com/gdiaz384/fairseq/releases/download/v0.12.2.2024Feb07/fairseq-0.12.2-cp310-cp310-win_amd64.whl"
if errorlevel 1 (
    echo gdiaz384 wheel failed, trying Jmica HuggingFace mirror...
    "%ENV%\Scripts\python.exe" -m pip install "https://huggingface.co/Jmica/rvc/resolve/main/fairseq-0.12.2-cp310-cp310-win_amd64.whl"
)
if errorlevel 1 (
    echo WARNING: fairseq wheels failed. Worker will use direct Hubert checkpoint download as fallback.
)

echo.
echo [6/6] Installing rvc-python...
"%ENV%\Scripts\python.exe" -m pip install rvc-python --no-deps
"%ENV%\Scripts\python.exe" -m pip install praat-parselmouth pyworld

echo.
echo ========================================================
echo SETUP COMPLETE! Run the app and click "Load RVC Env".
echo ========================================================
pause