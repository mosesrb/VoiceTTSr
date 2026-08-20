# VoiceTTSr

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11-blue.svg)]()
[![Privacy: 100% Local](https://img.shields.io/badge/Privacy-100%25%20Offline-success.svg)]()

**VoiceTTSr** is an open-source, local-first voice cloning and speech synthesis studio. It gives creators, modders, and developers a unified desktop interface to generate high-fidelity character voices on their own hardware without cloud subscriptions, API rate limits, or privacy compromises.

---

## Key Capabilities

* **Multi-Engine Synthesis**: Switch seamlessly between **Coqui XTTS v2** (multilingual voice cloning), **Qwen3-TTS** (expressive emotion tags & voice design), and **Chatterbox** (high-speed flow-matching diffusion).
* **RVC Vocal Re-Skinning**: Integrated Retrieval-based Voice Conversion layer to apply character timbre, pitch shifts, and index matching over synthesized audio.
* **Bethesda Modding Toolkit**: Generate Bethesda-compliant lip-sync (`.lip`) and automatically pack audio into Skyrim Special Edition `.fuz` voice archives.
* **100% Offline & Private**: Zero telemetry, zero analytics, and zero cloud calls. All voice latents, recordings, and outputs remain strictly on your local PC.
* **Audio Health Analyzer**: Built-in FFT spectral analyzer, SNR estimator, DC offset detector, and automated normalization to ensure pristine reference audio.
* **Persistent Voice Profiles**: Save averaged voice embeddings into compact, safe `.safetensors` profile files for instant one-click recall.

---

## Quick Start

### Prerequisites
* **OS**: Windows 10 / 11 (64-bit)
* **GPU**: NVIDIA GPU with at least 6–8 GB VRAM (CUDA 11.8+ supported)
* **Python**: Python 3.10 installed with "Add Python to PATH" enabled

### 1. Clone & Setup
```bash
git clone https://github.com/mosesrb/VoiceTTSr.git
cd VoiceTTSr
install_all.bat
```

### 2. Launch Studio
Double-click `VoiceTTSr.bat` or run:
```bash
VoiceTTSr.bat
```

---

## Supported Neural Engines

| Engine | Primary Strength | Supported Languages | Typical VRAM |
| :--- | :--- | :--- | :--- |
| **Coqui XTTS v2** | High-fidelity voice cloning from short reference audio | 17 Languages | ~4.0 GB |
| **Qwen3-TTS** | Emotional acting & expressive styles (`[whisper]`, `[angry]`, etc.) | 10 Languages | ~6.0 GB |
| **Chatterbox** | Ultra-fast diffusion speech synthesis with exaggeration control | English | ~1.5 GB |
| **RVC v2** | Post-conversion timbre enhancement and character tuning | Universal | ~2.0 GB |

---

## Architecture

VoiceTTSr uses a **Modular Subprocess Worker Architecture**. The main Tkinter desktop GUI orchestrates independent worker processes over standard JSON-lines IPC. Each engine operates inside its own isolated Python environment, preventing library conflicts between differing PyTorch, CUDA, and transformer versions.

```
┌──────────────────────────────────────────────────────────┐
│              VoiceTTSr Studio (Tkinter GUI)              │
└────────────┬─────────────┬─────────────┬─────────────┬───┘
             │ (IPC)       │ (IPC)       │ (IPC)       │ (IPC)
             ▼             ▼             ▼             ▼
      ┌────────────┐┌────────────┐┌────────────┐┌────────────┐
      │XTTS Worker ││Qwen Worker ││Chatterbox  ││RVC Worker  │
      │ (xtts-env) ││ (qwen-env) ││ (cb-env)   ││ (rvc-env)  │
      └────────────┘└────────────┘└────────────┘└────────────┘
```

---

## Ethical Use & Guidelines

VoiceTTSr is built for personal voice cloning, authorized voice acting, and game modding. Users are responsible for ensuring they possess explicit, informed consent for any voice they clone. Unauthorized deepfakes, deceptive impersonation, fraud, or harassment are strictly prohibited. See [docs/VOICE_ETHICS.md](docs/VOICE_ETHICS.md) for details.

---

## License & Third-Party Notices

* VoiceTTSr code is licensed under the [GNU General Public License v3.0](LICENSE).
* Upstream model weights and binaries operate under their respective licenses (e.g. Coqui CPML non-commercial terms for XTTS v2, Bethesda Creation Kit EULA for modding tools). See [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) for full attributions.
