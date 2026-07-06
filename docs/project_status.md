# VoiceTTSr — Project Status

**Last updated:** 2026-07-05
**Version:** v1.7.0 (base commit `8b48c24`) + unreleased audit remediation work
**Status:** 🟡 Code-complete on all release-blocking audit findings; **not yet verified on real GPU/Windows hardware**, so hold off on a tagged release until that verification pass happens.

This file is the living source of truth for where the project stands. Update it whenever a phase in `implementation_plan.md` moves, a release ships, or a major architectural decision is made. It's meant to be read in under two minutes by anyone (including future-you) trying to understand "where are we right now."

---

## 1. One-paragraph summary

VoiceTTSr is a Windows desktop GUI (Tkinter) that orchestrates four independent TTS/voice-conversion backends — XTTS v2, Qwen3-TTS, Chatterbox TTS, RVC v2 — each running in its own isolated Conda/venv subprocess, coordinated over a JSON-lines IPC protocol. It supports voice cloning from reference audio, "emotional acting" presets, RVC re-skinning, and export to Skyrim mod-ready `.fuz`/lip-sync formats. A 2026-07-03 audit found a trust & safety issue (an undisclosed real-person voice model shipped as a default) plus several packaging/security gaps; as of 2026-07-04, all three release-blocking phases of remediation (Phase 0-2) are code-complete and verified as far as a non-GPU, non-Windows environment allows. **The one thing standing between this and a clean release is a live verification pass on real GPU/Windows hardware** — see Section 3.

---

## 2. Current release readiness

| Gate | Status | Notes |
|---|---|---|
| Core functionality | 🟡 Code-complete, GPU-unverified | 4-engine architecture compiles clean, IPC protocol consistent; changes to 3 of 4 workers' model-loading code have not been run against real GPU model loads |
| Security — subprocess/injection | ✅ Good | No `shell=True`, no `eval`/`exec`/`os.system`; argument-list subprocess calls throughout |
| Security — filename handling | ✅ Good | Regex allowlist sanitization before all disk writes |
| Security — deserialization | 🟢 Fixed, 1 documented residual item | XTTS/Qwen no longer force `weights_only=False` globally; RVC fails closed instead of silently bypassing; downloads are SHA-256 verified. Chatterbox profile loading still uses `weights_only=False` with a documented, unresolved question about the exact class involved (see `implementation_plan.md` 1.1) |
| Trust & safety — voice provenance | ✅ Fixed | No voice model is downloaded by default; `docs/VOICE_ETHICS.md` and a one-time in-app consent gate added |
| Packaging — install reliability | ✅ Fixed | `requirements.txt`'s `numpy` pin relaxed; verified installing cleanly on Python 3.12 in this environment |
| Licensing documentation | ✅ Documented | `docs/THIRD_PARTY_NOTICES.md` added; also surfaced and fixed a previously-undocumented issue: `tools/FonixData.cdf` was bundled despite its own upstream saying it must not be redistributed -- removed from the repo |
| Test coverage | 🟡 CI added, no unit tests yet | `.github/workflows/ci.yml` added (syntax check + install check + download-script smoke test, all manually verified to pass); no `tests/` directory yet |
| Repo hygiene | ✅ Fixed | `.gitignore` restored from its last known-good version, with new entries for `.partial` downloads and the removed `FonixData.cdf` |
| Silent failure paths | ✅ Fixed | All 7 bare `except: pass` blocks resolved -- either narrowed to specific exceptions or upgraded to actually inform the user when silence would have hidden a real problem |

**Bottom line:** every release-blocking item from the audit is now code-complete. What's left is verification this sandbox genuinely cannot do (no GPU, no Windows, no network access to huggingface.co) -- see Section 3 for the specific list. Do not tag a release until that pass happens.

---

## 3. Audit remediation progress

Tracking against `implementation_plan.md`, phase by phase.

| Phase | Description | Status | Progress |
|---|---|---|---|
| 0 | Voice model disclosure/removal (trust & safety blocker) | ✅ Done | 5 / 5 tasks |
| 1 | Deserialization & download integrity | ✅ Done, 1 documented open item | 5 / 7 tasks (2 intentionally deferred, see below) |
| 2 | Packaging, dependency & licensing hygiene | ✅ Done | 8 / 10 tasks (2 minor items deferred, see below) |
| 3 | Code quality / maintainability | 🟡 Partial | 3.1 done; 3.2 (file split) and 3.3 (tests) not started |

**What's still open, specifically:**
- **1.2:** HF downloads use SHA-256 verification (the real integrity control) but aren't pinned to a specific commit revision -- this sandbox has no network access to confirm a real, current commit hash, and a maintainer should add this as defense-in-depth once they can verify one directly.
- **1.1:** `chatterbox_worker.py` still uses `weights_only=False` for its own profile format, with the exact object type needing live verification against an installed `chatterbox-tts` to resolve properly (see `implementation_plan.md`).
- **2.1/2.2:** README wasn't updated to link `docs/THIRD_PARTY_NOTICES.md` or state a minimum Python version explicitly -- functional fix (the actual `requirements.txt` pin) is done, this is just cross-referencing.
- **3.2/3.3:** Deliberately not started in this pass -- both would benefit from a human running the live GUI between refactor steps rather than being done blind.

**This is a snapshot from a single remediation pass (2026-07-04), not from tracked GitHub Issues yet.** If issues get filed under an `audit-2026-07` tag, switch this table to counting those instead.

---

## 4. What works today

- Multi-engine isolation: XTTS, Qwen, Chatterbox, and RVC each run in a dedicated interpreter, preventing dependency conflicts between engines.
- JSON-lines IPC protocol with `ping`/`generate`/`save_profile`/`quit` (and engine-specific actions like `enhance_refs`, `post_process`, `create_profile`, `infer`) is implemented consistently across all four workers.
- Environment variable sanitization (`PYTHONHOME`, `CONDA_*` stripped) prevents cross-environment contamination when spawning workers.
- Profile save/load (`.pt` / `.cbprof`) for recalling voice embeddings without re-uploading reference audio.
- Skyrim modding export path (`.fuz` + lip-sync phonemes via bundled `FaceFXWrapper.exe`/`xWMAEncode.exe`; `FonixData.cdf` must now be sourced by the user, see `docs/THIRD_PARTY_NOTICES.md`).
- "Mumble Guard" post-processing check with auto-retry on detected artifacts/silence.
- No voice model ships by default; users bring their own reference audio, with a one-time consent reminder before first generation.
- Downloaded technical assets (Hubert, RMVPE) are SHA-256 verified before use.
- CI runs syntax + install checks across Python 3.10-3.12 on every push/PR (pending first real run on GitHub Actions).

## 5. Known issues / open risks

1. `chatterbox_worker.py` still uses `weights_only=False` to load its own profile format -- lower risk than a downloaded checkpoint (these are app-generated files) but not fully resolved; needs a live install to verify the exact class and either allowlist it narrowly or switch to plain-tensor serialization.
2. HF downloads aren't pinned to a specific commit revision (SHA-256 verification is the real control here, but revision pinning would be good defense-in-depth) -- blocked on this sandbox not having network access to confirm a real hash.
3. No unit test suite yet (`tests/` doesn't exist) -- CI currently only catches syntax errors and install breaks, not logic regressions.
4. `voice_cloner_gui.py` is still a 3,853-line monolith -- Phase 3.2 (splitting it into a `gui/` package) hasn't started.
5. README doesn't yet link `docs/THIRD_PARTY_NOTICES.md` or state a minimum Python version explicitly.
6. None of the code changes in this remediation pass have been run on real GPU/Windows hardware -- see Section 3 for what specifically needs that verification.

## 6. Architecture reference

See `architecture.md` in this folder for the full system design, IPC protocol details, and data flow.

## 7. History

**2026-07-05 -- rvc-python dependency warning investigated.** A user reported seeing `pip`'s resolver warn that `rvc-python 0.1.5 requires fastapi/pydantic/uvicorn, which is not installed` while running `setup_rvc_env.bat`. Traced this by actually installing `rvc-python` from PyPI and reading its source: those three packages are only imported by `rvc_python/api.py` and `rvc_python/__main__.py` (rvc-python's own optional REST API server), never by `rvc_python.infer` (the only submodule `rvc_worker.py` uses). The warning is an expected consequence of `setup_rvc_env.bat` intentionally installing `rvc-python --no-deps` (to avoid its pinned dependency versions conflicting with this project's own torch/numpy/fairseq pins), and is safe to ignore. Neither the script nor the code explained this before, so added comments in both `setup_rvc_env.bat` and `rvc_worker.py` so this doesn't cause confusion again.

**2026-07-04 -- Audit remediation pass.** Implemented Phase 0 (removed the undisclosed real-person voice model from defaults, added `docs/VOICE_ETHICS.md` and an in-app consent gate), Phase 1 (narrowed or removed `weights_only=False` bypasses in 3 of 4 workers with reasoning specific to each; added SHA-256 verification to `download_resources.py` and moved its required assets to a more canonical source), Phase 2 (fixed the `numpy` pin breaking Python 3.12 installs, added `docs/THIRD_PARTY_NOTICES.md`, restored `.gitignore`, added CI), and Phase 3.1 (resolved all 7 bare `except: pass` blocks). Also surfaced and fixed a new issue not in the original audit: `tools/FonixData.cdf`, a Bethesda proprietary file, was bundled in violation of its own upstream tool's redistribution terms -- removed. All changes verified via `py_compile`, targeted functional tests, and manual replication of the new CI steps in a non-GPU, non-Windows sandbox with restricted network access; none of it has been run on the project's actual target hardware yet.

**2026-07-03 -- Initial audit.** Full code review of v1.7.0 (commit `8b48c24`) identified the findings remediated above. See `VoiceTTSr_Audit_Report.md`.

## 8. How to update this file

- When a Phase in `implementation_plan.md` completes, flip its row in Section 3 to ✅ Done and update Section 2's gate table.
- When a new audit, review, or remediation pass happens, append a dated entry to Section 7 rather than overwriting this snapshot.
- Keep Section 4/5 in sync with reality — if something in "known issues" gets fixed, move it out, don't just delete it silently (it should already be reflected in a History entry).
