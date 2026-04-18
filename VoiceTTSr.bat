@echo off
setlocal enabledelayedexpansion
set "PY_PATH=gui-env\Scripts\python.exe"
cd /d "%~dp0"

if not exist "%PY_PATH%" (
    echo [!] WARNING: GUI Environment not found.
    set /p choice="Would you like to run the installer now? (Y/N): "
    if /i "!choice!"=="Y" (
        call install_all.bat
    ) else (
        echo [ERROR] Cannot launch without environment.
        pause
        exit /b 1
    )
)

:: Re-verify after potential install
if not exist "%PY_PATH%" (
    echo [ERROR] Installation failed or was skipped. Please run 'install_all.bat' manually.
    pause
    exit /b 1
)

start "" "%PY_PATH%" voice_cloner_gui.py
exit
