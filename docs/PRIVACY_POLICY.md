# VoiceTTSr — Local Privacy Policy & Data Guarantee

> **Certified 100% Offline & Local Execution Architecture**  
> *Last Updated: August 2026*

---

## 1. 100% Offline & Local Execution

VoiceTTSr is designed from the ground up as an entirely local, self-contained desktop studio. All text-to-speech synthesis, neural voice cloning latents, flow-matching diffusion, and digital signal processing execute strictly on your personal computer hardware.

---

## 2. Zero Telemetry & Zero Cloud Transmission

* **No Telemetry**: VoiceTTSr collects zero analytics, zero telemetry, zero metrics, and zero crash dumps.
* **No Cloud Transmission**: Your voice reference audio, generated output files, prompt text, custom models, and speaker embeddings are **NEVER** uploaded or transmitted to any external server or third party.
* **Full Offline Operation**: Once initial technical models are downloaded, VoiceTTSr functions with all internet and network connections completely disabled.

---

## 3. Local File Safety & Storage

* **Local Storage**: Voice profiles (`.safetensors`) and generated audio files (`.wav`, `.fuz`) reside exclusively in your local application directories or your configured output folders.
* **Recycle Bin Protection**: File deletion actions in the studio route safely through your operating system's Recycle Bin / Trash via `send2trash` rather than permanently deleting data without recovery options.
* **Transparent Configuration**: Studio settings (`voicecloner_config.json`) are stored locally on your disk in human-readable JSON format.

---

## 4. User Responsibility & Biometric Voice Data

Because voice embeddings and cloned profiles represent biometric characteristics of individuals:

* You are solely responsible for securing your local files and reference recordings.
* You must ensure that any audio recorded or stored on your system complies with all applicable privacy, consent, and biometric data protection regulations in your jurisdiction.
