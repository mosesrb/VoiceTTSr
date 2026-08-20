"""
VoiceTTSr - Native Windows Launcher Builder
Compiles voice_cloner_gui.py into a lightweight, no-console native executable (VoiceTTSr.exe).
"""

import os
import sys
import subprocess
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ICON_PATH = os.path.join(ROOT_DIR, "icon.ico")
ENTRY_POINT = os.path.join(ROOT_DIR, "voice_cloner_gui.py")
DIST_DIR = os.path.join(ROOT_DIR, "dist")
BUILD_DIR = os.path.join(ROOT_DIR, "build")


def check_and_install_pyinstaller():
    try:
        import PyInstaller
        print("[INFO] PyInstaller is already installed.")
    except ImportError:
        print("[INFO] PyInstaller not found. Installing via pip...")
        cmd = [sys.executable, "-m", "pip", "install", "pyinstaller"]
        res = subprocess.run(cmd, cwd=ROOT_DIR)
        if res.returncode != 0:
            print("[ERROR] Failed to install PyInstaller.")
            sys.exit(1)


def build_executable():
    print("=" * 60)
    print(" VoiceTTSr — Compiling Native Windows Executable")
    print("=" * 60)

    check_and_install_pyinstaller()

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name",
        "VoiceTTSr",
    ]

    if os.path.isfile(ICON_PATH):
        cmd.extend(["--icon", ICON_PATH])

    # Include local packages & data
    for folder in ["docs", "dsp", "ui", "core"]:
        folder_path = os.path.join(ROOT_DIR, folder)
        if os.path.isdir(folder_path):
            cmd.extend(["--add-data", f"{folder_path};{folder}"])

    cmd.append(ENTRY_POINT)

    print(f"[BUILD] Running PyInstaller: {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=ROOT_DIR)
    if res.returncode != 0:
        print("[ERROR] PyInstaller build failed.")
        return False

    built_exe = os.path.join(DIST_DIR, "VoiceTTSr", "VoiceTTSr.exe")
    if os.path.isfile(built_exe):
        target_exe = os.path.join(ROOT_DIR, "VoiceTTSr.exe")
        try:
            shutil.copy2(built_exe, target_exe)
            print(f"\n[SUCCESS] Compiled Launcher ready at: {target_exe}")
        except Exception:
            print(f"\n[SUCCESS] Compiled Launcher ready at: {built_exe}")
        return True

    print("[WARN] Built executable not located at expected path.")
    return False


if __name__ == "__main__":
    success = build_executable()
    if not success:
        sys.exit(1)
