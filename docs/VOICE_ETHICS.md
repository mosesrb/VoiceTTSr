# Voice Ethics & Acceptable Use

**Why this document exists:** a 2026-07-03 audit found that VoiceTTSr's first-run setup silently downloaded a voice-conversion model trained on a real, named public figure's voice (renamed on download to a generic `male_baseline.pth`, obscuring its origin), with no disclosure, consent record, or attribution. That default has been removed (see `implementation_plan.md`, Phase 0). This document explains the principle behind that fix so the same mistake doesn't get reintroduced.

## The core principle

VoiceTTSr is a general-purpose voice cloning and TTS tool. Like any tool that can reproduce a specific person's voice, it can be used well or badly. The project's responsibility is to not make the harmful path the default or the path of least resistance:

- **No real person's voice ships as a default or example asset without their informed consent**, regardless of how easy that person's voice is to find online or how good the resulting model sounds.
- **Bringing your own reference audio is the normal workflow**, not a "baseline" model. If someone wants to clone a specific voice, that should be a deliberate, disclosed choice they make, not something the installer does silently on their behalf.
- **Renaming a source file to obscure its origin is a disclosure failure**, even if unintentional. If an asset is derived from a real person, the filename, README, and UI should say so plainly.

## What this means in practice

- `download_resources.py`'s `RESOURCES` dict only contains non-voice technical assets (content encoders, pitch extractors) required for RVC to function at all — never ready-to-use voice models.
- `download_resources.py`'s `OPTIONAL_VOICES` dict exists for anyone who wants to maintain a fork or personal build with specific opt-in voices, but it ships empty by default and every entry requires a `source_note` describing provenance.
- Before running a voice-cloning or RVC operation on reference audio, the GUI should surface a brief, non-blocking reminder that the user is responsible for having the right to use the voice they're cloning (see the `implementation_plan.md` Phase 0 checklist for the consent-dialog task).

## Acceptable use

VoiceTTSr should not knowingly be used to:
- Clone a real person's voice without their consent, including public figures.
- Generate audio intended to impersonate someone for fraud, harassment, defamation, or to spread false statements attributed to them.
- Circumvent another product's or platform's voice-authentication or identity-verification systems.

Legitimate uses this project is built for include: cloning your own voice, using voices you have explicit permission to use, fully synthetic/original character voices (e.g. for game mods, as the Skyrim export feature is designed for), and other consented or fictional use cases.

## If you're contributing

If your contribution adds any bundled or auto-downloaded voice/persona asset, please:
1. Confirm and document its provenance (who/what it's derived from, and under what license or consent).
2. Do not rename source files in a way that obscures that provenance.
3. Default new assets to opt-in (`OPTIONAL_VOICES`, not `RESOURCES`), and flag the PR for maintainer review before merging.

This document is a living policy — update it if the project's approach to voice provenance changes, and cross-reference it from `project_status.md` if a related issue is open.
