@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  VoiceTTSr Studio - One-Click Bootstrap Installer
echo ============================================================
echo.

:: 1. Check Python
where.exe python.exe >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Python not found.
    echo Please install Python 3.10 from https://www.python.org/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: 2. Check if we need to download the source
if exist "voice_cloner_gui.py" (
    echo [INFO] Found VoiceTTSr files in current directory.
    goto :INSTALL
)

echo [INFO] No application files found. Fetching Latest Release...

:: Try Git first
where.exe git.exe >nul 2>nul
if %errorlevel% == 0 (
    echo [1/2] Cloning repository via Git...
    git clone https://github.com/mosesrb/VoiceTTSr.git .
) else (
    echo [1/2] Git not found. Downloading ZIP via curl/powershell...
    curl -L https://github.com/mosesrb/VoiceTTSr/archive/refs/heads/main.zip -o voicetts_temp.zip
    if %errorlevel% neq 0 (
        echo [ERROR] Download failed. Check your internet connection.
        pause
        exit /b 1
    )
    
    echo [1/2] Extracting files...
    powershell -Command "Expand-Archive -Path 'voicetts_temp.zip' -DestinationPath 'temp_extract' -Force"
    
    :: Move files from the nested folder (GitHub ZIPs always nest) to current dir
    for /d %%D in (temp_extract\VoiceTTSr-*) do (
        xcopy "%%D\*" "." /E /Y /Q
    )
    
    :: Cleanup
    del voicetts_temp.zip
    rd /s /q temp_extract
)

:INSTALL
echo.
echo [2/2] Launching Environment Setup...
if exist "install_all.bat" (
    call install_all.bat
) else (
    echo [ERROR] Codebase download appears incomplete. 'install_all.bat' missing.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  BOOTSTRAP COMPLETE!
echo  Launch the studio using 'VoiceTTSr.bat'
echo ============================================================
pause
