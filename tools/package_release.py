"""
VoiceTTSr - Unified Release Packager
Generates portable release zip and formatted release notes for GitHub Releases.
"""

import os
import sys
import zipfile
import re
import shutil

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
DIST_DIR = os.path.join(ROOT_DIR, "dist")
VERSION = "1.7.0"

# Find version dynamically if defined
gui_py = os.path.join(ROOT_DIR, "voice_cloner_gui.py")
if os.path.isfile(gui_py):
    with open(gui_py, "r", encoding="utf-8") as f:
        match = re.search(r'VERSION\s*=\s*["\']([^"\']+)["\']', f.read())
        if match:
            VERSION = match.group(1)

ZIP_NAME = f"VoiceTTSr_v{VERSION}_Portable.zip"
ZIP_PATH = os.path.join(DIST_DIR, ZIP_NAME)


def clean_pycache():
    print("[CLEAN] Removing pycache and temporary files...", flush=True)
    exclude_dirs = {"venv", ".venv", "gui-env", "xtts-env-py310", "qwen-env-py310", "chatterbox-env-py310", "rvc-env", ".git", ".pytest_cache", "build", "dist"}
    for root, dirs, files in os.walk(ROOT_DIR):
        dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.endswith("-env")]
        for d in dirs:
            if d == "__pycache__":
                shutil.rmtree(os.path.join(root, d), ignore_errors=True)


def generate_release_notes():
    notes_path = os.path.join(DIST_DIR, "RELEASE_NOTES.md")
    notes_content = f"""# VoiceTTSr v{VERSION} — Release Notes

## 🎙️ What's New in v{VERSION}

### ⚡ Highlights & New Features
* **Multi-Engine Voice Studio**: Seamless switching between Coqui XTTS v2, Qwen3-TTS (with emotion tags & voice design), Chatterbox (accelerated diffusion), and RVC v2 voice conversion.
* **First-Launch Ethical Use, Terms & Privacy Agreement**: Explicit biometric consent verification, 100% offline data guarantees, and in-app policy viewers.
* **Audio Health Analyzer**: Built-in FFT spectral analyzer, SNR estimator, DC offset detector, and automated normalization.
* **Bethesda Modding Toolkit**: Direct generation of `.lip` lip-sync and `.fuz` audio packing for Skyrim Special Edition.
* **High-DPI 4K Support**: Native per-monitor scaling on Windows 10/11.

---

### 📦 Installation
1. Download `{ZIP_NAME}` and extract to any folder.
2. Run `install_all.bat` to initialize the isolated engine environments.
3. Launch via `VoiceTTSr.exe` (or `VoiceTTSr.bat`).

*Note: On first engine launch, required AI weights are downloaded locally into your cache. Once downloaded, all voice cloning and synthesis operates 100% offline.*
"""
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes_content)
    print(f"[NOTES] Generated release notes at: {notes_path}", flush=True)


def create_portable_zip():
    print(f"\n[ZIP] Creating Portable Package: {ZIP_NAME}", flush=True)
    os.makedirs(DIST_DIR, exist_ok=True)

    # Fast directory exclusions (pruned directly from os.walk)
    excluded_dir_names = {
        ".git",
        ".github",
        ".pytest_cache",
        "build",
        "dist",
        "venv",
        ".venv",
        "gui-env",
        "voice_files",
        "_backup",
        "__pycache__",
        "temp_extract",
        ".temp_ag_kit",
        ".agent",
        "archive",
    }

    def _is_excluded(d_name: str) -> bool:
        if d_name in excluded_dir_names:
            return True
        if "-env" in d_name or d_name.endswith("venv") or d_name.startswith("."):
            return True
        return False

    excluded_files = {
        "voicecloner_config.json",
        "voicetts_temp.zip",
        "speaker_latents.pt",
        "docs/memories.md",
        "docs/worklogs.md",
        "docs/Universal_Software_Audit_Prompt.md",
    }

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(ROOT_DIR):
            # Prune directories in-place so os.walk skips descending into them
            dirs[:] = [d for d in dirs if not _is_excluded(d)]

            for file in files:
                rel_file = os.path.relpath(os.path.join(root, file), ROOT_DIR).replace("\\", "/")

                if rel_file in excluded_files:
                    continue
                if any(_is_excluded(part) for part in rel_file.split("/")):
                    continue
                if file.endswith(
                    (
                        ".pyc",
                        ".pyo",
                        ".log",
                        ".bak",
                        ".tmp",
                        ".partial",
                        ".wav",
                        ".mp3",
                        ".safetensors",
                        ".pth",
                        ".pt",
                        ".index",
                    )
                ):
                    if not rel_file.startswith("docs/assets/"):
                        continue

                zipf.write(os.path.join(root, file), rel_file)

    size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
    print(f"[SUCCESS] Packaged {ZIP_NAME} ({size_mb:.2f} MB)", flush=True)
    print(f"Archive ready at: {ZIP_PATH}", flush=True)


import subprocess
from tools.build_launcher import build_executable

def compile_installer():
    iscc_path = shutil.which("iscc") or shutil.which("ISCC")
    # Also check standard Inno Setup installation paths on Windows
    if not iscc_path:
        for p in [
            r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            r"C:\Program Files\Inno Setup 6\ISCC.exe",
        ]:
            if os.path.isfile(p):
                iscc_path = p
                break

    if iscc_path:
        print(f"\n[INSTALLER] Compiling Inno Setup Installer with: {iscc_path}", flush=True)
        iss_file = os.path.join(ROOT_DIR, "tools", "VoiceTTSr_installer.iss")
        res = subprocess.run([iscc_path, iss_file], cwd=ROOT_DIR)
        if res.returncode == 0:
            print(f"[SUCCESS] Setup installer compiled at: dist/VoiceTTSr_Setup_v{VERSION}.exe", flush=True)
            return True
        else:
            print("[WARN] Inno Setup compilation returned non-zero code.", flush=True)
    else:
        print("[INFO] Inno Setup (ISCC.exe) not found on system path; skipping installer .exe compilation.", flush=True)
    return False


def main():
    print("=" * 60)
    print(f" VoiceTTSr v{VERSION} — Release Packaging Suite")
    print("=" * 60)

    clean_pycache()
    
    # 1. Compile native launcher executable (VoiceTTSr.exe)
    print("\n[STEP 1/3] Compiling Native Windows Launcher...", flush=True)
    build_executable()

    # 2. Compile Inno Setup Windows Installer if ISCC is present
    print("\n[STEP 2/3] Building Inno Setup Installer...", flush=True)
    compile_installer()

    # 3. Create portable zip and release notes
    print("\n[STEP 3/3] Creating Portable Zip & Release Notes...", flush=True)
    create_portable_zip()
    generate_release_notes()

    print("\n" + "=" * 60)
    print(" RELEASE ASSETS READY FOR GITHUB RELEASES")
    print(f" 1. Portable Archive: dist/{ZIP_NAME}")
    if os.path.isfile(os.path.join(DIST_DIR, f"VoiceTTSr_Setup_v{VERSION}.exe")):
        print(f" 2. Setup Installer:  dist/VoiceTTSr_Setup_v{VERSION}.exe")
    print(f" 3. Standalone EXE:   dist/VoiceTTSr.exe")
    print(f" 4. Release Notes:    dist/RELEASE_NOTES.md")
    print("=" * 60)


if __name__ == "__main__":
    main()

