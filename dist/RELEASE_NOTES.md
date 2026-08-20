# VoiceTTSr v1.7.0

Windows installer release for VoiceTTSr Studio.

## Highlights & What's New

* **Dedicated Windows Setup Wizard**: Single installer (`VoiceTTSr_Setup_v1.7.0.exe`) providing clean desktop and Start Menu integration.
* **Smart First-Run Onboarding**:
  * Automatic detection of AI engine environments upon initial launch.
  * Direct one-click prompt and persistent header button to run `install_all.bat` within a visible terminal window.
  * Subprocess safety preventing recursive launcher execution when virtual environments are uninitialized.
* **Explicit Model Download & Cache Logging**:
  * Added instant log detection for Hugging Face and Coqui weight downloads vs. cached local disk loads.
* **Rich Ethics & Policy Viewer**:
  * Integrated comprehensive Ethics, Privacy Policy (100% offline guarantee), and Third-Party License documentation with formatted headings, callouts, and bullet points.
  * Pixel-perfect layout with 32-bit alpha-transparent application icons.
* **Engine & Security Hardening**:
  * Enforced safe tensor verification (`weights_only=True`) by default when loading voice profiles.
  * High-pass filtering and normalization fallback for XTTS audio enhancement.
  * Expanded test suite with 32 unit tests passing at 100%.

## Installation & Getting Started

1. Download and run `VoiceTTSr_Setup_v1.7.0.exe`.
2. Launch VoiceTTSr from your Desktop or Start Menu.
3. On first launch, accept the Ethical Use Agreement and click **Initialize Engines** when prompted (or use the **`[ ⚡ Setup Engines ]`** button in the header) to set up local PyTorch and AI dependencies.
