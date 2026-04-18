@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  VoiceTTSr - Release Packager
echo ============================================================
echo.

:: Change directory to the app root (one level up from /tools)
cd /d "%~dp0.."

:: Get current version from code if possible
for /f "tokens=3" %%A in ('findstr /C:"VERSION =" voice_cloner_gui.py') do (
    set "VERSION=%%A"
    set "VERSION=!VERSION:"=!"
)

if "!VERSION!" == "" set "VERSION=Latest"

set "ZIP_NAME=VoiceTTSr_v!VERSION!_Light.zip"

echo [1/3] Cleaning up temporary files...
:: Remove pycache
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"

echo [2/3] Preparing ZIP: !ZIP_NAME!
echo (Excluding environments, large models, and Git history...)

:: Use powershell to zip the current directory while excluding specific patterns
powershell -Command ^
    "$exclude = @('*-env*', '.git*', 'voicetts_temp.zip', 'VoiceTTSr_v*.zip', 'voice_files', '_backup', 'Output', 'references', 'rvc_models'); ^
    Get-ChildItem -Path '.' -Exclude $exclude | Compress-Archive -DestinationPath '!ZIP_NAME!' -Force"

if %errorlevel% neq 0 (
    echo [ERROR] Zipping failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Package Created Successfully!
echo File: !ZIP_NAME!
echo.
echo You can now upload this ZIP to your GitHub Release under 'Assets'.
echo ============================================================
pause
