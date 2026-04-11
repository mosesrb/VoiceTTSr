# VoiceTTSr (Voice Text-To-Speech Renderer) 🎙️

![Version](https://img.shields.io/badge/version-1.6.0-blue.svg)

**VoiceTTSr (Voice Text-To-Speech Renderer)** is a high-performance, multi-engine voice cloning studio that provides a unified interface for **XTTS v2**, **Qwen3-TTS**, and **RVC v2**.

By leveraging a standalone subprocess worker architecture, VoiceTTSr eliminates "dependency hell" by running each ML engine in its own isolated Python environment while maintaining a responsive, real-time GUI.

---

## 🚀 1-Minute Quick Start

If you are a new user and want the fastest way to get started:

1.  **Open PowerShell** in the folder where you want to install.
2.  **Copy & Paste** this command:
    ```powershell
    curl -L https://github.com/mosesrb/VoiceTTSr/raw/main/install_VoiceTTSr.bat -o install.bat; .\install.bat
    ```
3.  **Launch** using `VoiceTTSr.bat`.

*Alternatively, you can manually download [install_VoiceTTSr.bat](https://github.com/mosesrb/VoiceTTSr/raw/main/install_VoiceTTSr.bat) and run it.*

---

## 🏷️ Versioning & Changelog

VoiceTTSr uses standard [Semantic Versioning](https://semver.org/). 

**Current Stable Release: v1.6.0**
- Refactored `_jobs_frame` layout to eradicate scrolling gaps
- Improved contrast/readability for "Generate" and Accent buttons
- Forced explicit OS-level process killing on "Global Stop" for immediate responsive halting
- Centralized Python-based worker environment configurations

---

## 🎨 Key Features

*   **XTTS v1/v2 Engine**: Stable, high-fidelity voice cloning across 17+ languages.
*   **Qwen3-TTS Engine**: State-of-the-art expressive synthesis with "Power Mode" for emotional acting.
*   **RVC v2 Integration**: Post-processing "reskinning" for maximum character accuracy and pitch control.
*   **Batch Processing**: Rapidly process groups of audio files with granular skip/stop controls.
*   **Skyrim SE Integration**: Dedicated utility for generating `.lip` and `.xwm` files for Bethesda modding.
*   **Audio Analyzer**: Integrated tools to check reference audio health and loudness.

---

## 🛠️ One-Command Installation

VoiceTTSr is designed to be "plug-and-play." To set up your local studio and all three AI engines:

1.  Ensure **Python 3.10** and **Git** are installed on your system.
2.  Run the master installer:
    ```cmd
    install_all.bat
    ```
    *This will automatically create 3 isolated virtual environments, install all ML dependencies (~8GB total), and download the required Hubert/RMVPE baseline models.*

---

## 🚀 Launching the App

Once the installation is complete, launch the studio via:
```cmd
VoiceTTSr.bat
```
*(Or use the created Desktop Shortcut)*

---

## 🏗️ Architecture

VoiceTTSr uses a **Subprocess Worker Architecture** to handle large machine learning models:
*   **GUI**: Minimalist Python environment (pygame/pydub/numpy).
*   **XTTS Worker**: Isolated venv (TTS/transformers/torch).
*   **Qwen Worker**: Isolated venv (qwen-tts/transformers/accelerate).
*   **RVC Worker**: Isolated venv (rvc-python/fairseq/rmvpe).

This design ensures that version conflicts between different engines are impossible and the GUI remains perfectly fluid during long generation tasks.

---

**VoiceTTSr (Voice Text-To-Speech Renderer)** — Rounding up the edge of Voice Synthesis.
