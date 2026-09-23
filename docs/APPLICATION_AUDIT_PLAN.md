# VoiceTTSr Application Audit Plan

**Prepared:** 2026-08-24  
**Target:** VoiceTTSr desktop application and all supporting runtime, worker, tooling, and release surfaces  
**Purpose:** Define an evidence-driven audit that can support a defensible release decision. This document is a plan, not an audit result.

## 1. Source Documents Reviewed

The plan is based on the following project documents:

- `docs/memories.md` — current architecture, invariants, ADRs, engine matrix, and environment map.
- `docs/worklogs.md` — claimed v1.7.0 baseline, prior remediation, test history, and roadmap.
- `docs/Universal_Software_Audit_Prompt.md` — required audit dimensions and deliverables.
- `docs/PRIVACY_POLICY.md` — offline, zero-telemetry, local-storage, and deletion claims.
- `docs/VOICE_ETHICS.md` — consent, provenance, and acceptable-use requirements.
- `docs/THIRD_PARTY_NOTICES.md` — model, binary, asset, and redistribution obligations.
- `docs/archive/architecture-design.md` and `docs/archive/project-status.md` — historical baseline and claims that may reveal documentation drift.

Documentation is input evidence only. Claims such as “100/100,” “production hardened,” “100% offline,” “thread safe,” and “zero known RCE” must be independently verified against source, tests, binaries, runtime behavior, and release artifacts.

## 2. Audit Objectives

The audit will determine whether VoiceTTSr is safe and ready for its intended release tier by evaluating:

- correctness and functional completeness;
- architecture, maintainability, and technical debt;
- security, privacy, voice consent, provenance, and licensing;
- reliability, recovery, concurrency, and data-loss resistance;
- startup/runtime performance and CPU, RAM, VRAM, disk, and process lifecycle behavior;
- UI/UX, accessibility, localization, and high-DPI behavior;
- tests, CI, installation, packaging, updates, and rollback;
- Windows support and the accuracy of Linux/macOS compatibility claims;
- release artifact integrity, reproducibility, and operational readiness.

The result will include an A–F grade, a 0–100 readiness score, and one release recommendation: Not Ready, Internal Use Ready, Beta Ready, Production Ready, or Enterprise Ready.

## 3. System Surfaces in Scope

| Surface | Primary components | Main risks to audit |
|---|---|---|
| Desktop GUI/orchestrator | `voice_cloner_gui.py`, `ui/`, `core/` | Tk thread violations, god-class complexity, unsafe paths, state corruption, usability, accessibility |
| Worker supervisor and IPC | `_TtsWorker` and JSON-lines protocol | command validation, stale messages, hangs, process leaks, malformed output, cancellation |
| Neural TTS workers | `xtts_worker.py`, `qwen_worker.py`, `chatterbox_worker.py` | unsafe model/profile loading, GPU/CPU fallback, OOM recovery, output integrity |
| Voice conversion worker | `rvc_worker.py` | unsafe checkpoints, untrusted paths, model provenance, failure isolation |
| DSP and audio analysis | `dsp/` | malformed/corrupt audio, numeric edge cases, clipping, sample-format correctness, large files |
| Skyrim export pipeline | `skyrim_utils.py`, `tools/FaceFXWrapper.exe`, `tools/xWMAEncode.exe` | command injection, path quoting, proprietary assets, binary trust, partial output cleanup |
| Resource acquisition/setup | `download_resources.py`, `install_*.bat`, `setup_*_env.bat`, requirements files | supply-chain compromise, missing hashes, partial downloads, privilege and path handling |
| Configuration and local data | `voicecloner_config.json`, `Profiles/`, `references/`, `Output/`, `rvc_models/` | biometric-data exposure, unsafe permissions, corruption, path traversal, recoverability |
| Build, installer, and release | `tools/build_launcher.py`, `VoiceTTSr.spec`, Inno Setup, packager, `.github/workflows/` | non-reproducible builds, unsigned binaries, oversized/stale artifacts, missing notices, release drift |
| Policy and user safeguards | ethics dialog plus policy documents | unenforced consent, misleading privacy claims, absent attribution, commercial-use ambiguity |

Database-specific checks are expected to be not applicable because the documented design uses local JSON/files rather than a database. The audit must verify that assumption and record the evidence instead of silently omitting the area.

## 4. Known Claims and Drift to Verify First

These are audit hypotheses, not findings:

1. The documentation describes workers under `workers/`, while the inspected repository currently places the four worker scripts at the root.
2. Worklogs claim a lightweight release/launcher architecture and approximately 28 MB portable packaging, while the current root `VoiceTTSr.exe` is approximately 474 MB. Determine whether it is current, tracked, reproducible, and actually shipped.
3. The archived status reports 94/100 and 19 tests, while current worklogs report 100/100 and 32 tests. Recalculate from evidence and identify stale documents.
4. The privacy policy promises zero cloud transmission and full offline operation after initial downloads. Verify all runtime network paths, model-library behavior, update/download behavior, and telemetry defaults.
5. Ethics documentation references `implementation_plan.md` and `project_status.md`; verify that referenced controls exist in code and that references resolve to maintained documents.
6. Third-party notices identify licenses that can restrict commercial use or redistribution. Verify the exact versions, model cards, binary provenance, licenses, and contents of every release artifact.
7. Cross-platform compatibility is claimed, but the primary supported OS is Windows and native helper tools are Windows binaries. Test or narrow the claim.

## 5. Audit Method and Evidence Rules

Every claim and finding must be backed by at least one reproducible evidence item:

- exact file and line reference;
- test name and captured result;
- command and output;
- runtime log/process trace;
- screenshot or screen recording for UI behavior;
- artifact manifest, hash, signature, or software bill of materials (SBOM);
- dependency/model license or authoritative upstream record.

Positive statements require evidence. Each negative finding must state root cause, impact, reproduction/evidence, recommendation, and validation method. A prior worklog entry or passing test count alone is not proof that a control is effective.

Use synthetic or explicitly consented audio during testing. Do not introduce real-person voice models, publish generated impersonations, or upload reference audio to external services. Preserve user files; destructive and corruption tests must run only in disposable directories and configurations.

## 6. Execution Phases

### Phase 0 — Freeze Scope and Capture the Baseline

1. Record commit, branch, dirty-tree state, OS, Python versions, GPU/driver/CUDA versions, free disk, and environment locations.
2. Inventory tracked source, generated artifacts, binaries, models, configuration, and ignored files.
3. Hash release binaries and bundled native tools; record sizes and version metadata.
4. Build a requirements/model manifest for the GUI plus every worker environment.
5. Map documented components and claims to their actual current locations.
6. Create an evidence directory outside user-data folders and a finding register with stable IDs.

**Exit evidence:** baseline manifest, environment matrix, artifact hashes, architecture/component map, and documentation-drift list.

### Phase 1 — Fast Release-Blocker Triage

Run high-yield checks before expensive model execution:

1. Search all Python, batch, installer, and workflow files for unsafe deserialization, shell execution, unquoted paths, embedded secrets, unrestricted downloads, weak hashes, writable executable search paths, and unsafe temporary files.
2. Inspect all profile and checkpoint loaders, including every fallback path and compatibility patch.
3. Trace all network-capable code and imported SDK/model-library behavior against privacy claims.
4. Inspect release contents for private audio, profiles, proprietary data, build caches, credentials, and forbidden baseline voice models.
5. Verify that `FonixData.cdf` and any real-person voice model are absent from tracked and packaged content.
6. Run the existing test suite from a clean GUI environment and preserve the complete output.

**Stop condition:** immediately flag Critical findings involving code execution, credential exposure, non-consensual bundled voices, destructive data loss, or prohibited redistribution. Continue safe read-only analysis while the release decision remains blocked.

### Phase 2 — Architecture and Code Quality Audit

1. Produce a dependency map for GUI, core, UI, DSP, workers, Skyrim tooling, downloader, and packaging.
2. Verify the invariant that GUI startup imports no heavy ML/CUDA frameworks.
3. Quantify the remaining `voice_cloner_gui.py` monolith: responsibilities, complex methods, duplicated state handling, and hidden coupling.
4. Review dependency direction, cohesion, error boundaries, configuration ownership, logging, and typed immutable models.
5. Detect dead code, duplicate implementations, stale launch paths, and mismatches between root-level workers and documentation.
6. Score SOLID, DRY, KISS, separation of concerns, composition over inheritance, and dependency injection from 1–10 with evidence.

**Exit evidence:** dependency diagram, code-quality metrics, principle scorecard, refactor candidates, and architecture findings.

### Phase 3 — Security, Privacy, Ethics, and Licensing Audit

1. Threat-model assets, trust boundaries, and attackers: malicious profile/model/audio/config, compromised download, hostile text/path, local low-privilege user, and tampered helper binary.
2. Validate safetensors usage and `torch.load(weights_only=True)` enforcement. Prove that any temporary unsafe base-checkpoint load is narrowly scoped and cannot consume user-controlled input.
3. Fuzz JSON-line commands and worker stdout with malformed JSON, oversized fields, unexpected types, invalid actions, path traversal, control characters, and partial lines.
4. Test downloader TLS behavior, redirects, timeouts, atomicity, hash verification, resume/cleanup, destination confinement, and model-cache behavior.
5. Review subprocess construction in Skyrim and setup/build tooling for injection, DLL search-order issues, writable binary replacement, and unsafe privilege assumptions.
6. Monitor DNS, socket, and process activity during install, first run, model load, generation, analysis, and export. Reconcile every connection with the privacy UI.
7. Verify consent reminders exist at the actual cloning/conversion decision point, not solely on first launch; test acceptance persistence and reset behavior.
8. Build a complete component/model/license matrix. Confirm commercial-use restrictions, attribution, source offers/notices, binary redistribution permission, and watermark disclosures.
9. Generate dependency vulnerability reports independently for every isolated environment; distinguish exploitable runtime issues from development-only packages.

**Exit evidence:** threat model, attack-surface map, dependency vulnerability reports, network trace, license/provenance matrix, and security/privacy findings.

### Phase 4 — Correctness, Reliability, Concurrency, and Data Safety

1. Exercise start, ping, generate, chunk, completion, cancellation, restart, engine switch, and quit for every worker.
2. Inject worker crash, hang, malformed stdout, nonzero exit, CUDA OOM, missing model, corrupt profile, permission denial, full disk, and output-path loss.
3. Verify timeouts, cancellation boundaries, UI recovery, stale-queue clearing, orphan-process cleanup, partial-file cleanup, and useful error messages.
4. Stress rapid engine switching, repeated start/stop, concurrent UI edits, window close during generation, and long batch queues.
5. Instrument all Tk calls and state access to detect background-thread GUI access; test the immutable snapshot guarantee.
6. Validate config writes for atomicity, schema/type handling, corrupt JSON recovery, concurrent writes, migration/default behavior, and least-sensitive persistence.
7. Verify recycle-bin behavior and fallback semantics. Test collisions, locked files, network/removable drives, long paths, Unicode, and invalid output names.
8. Fuzz audio decoding and DSP with zero-length, truncated, huge, multichannel, unusual bit-depth/sample-rate, NaN/Inf, silent, clipped, and adversarial files.
9. Verify Skyrim outputs byte-for-byte where possible and confirm that failed steps do not leave misleading valid-looking `.fuz` files.

**Exit evidence:** fault-injection matrix, concurrency traces, recovery results, corrupt-input corpus results, and data-safety findings.

### Phase 5 — Functional, UI/UX, Accessibility, and Compatibility Audit

Use small synthetic fixtures and explicitly consented reference audio to test:

1. Fresh install and first launch; model onboarding; ethics/privacy flow; output configuration.
2. XTTS, Qwen, Chatterbox, and RVC happy paths plus validation failures and unsupported hardware paths.
3. Batch generation, profile create/load, playback, Audio Analyzer, CSV export, filters, deletion/recovery, and Skyrim export.
4. Keyboard-only operation, focus order, visible focus, shortcuts, screen-reader labels where feasible, color contrast, text scaling, resize/minimum-size behavior, and 100/125/150/200% DPI.
5. Long strings, Unicode, right-to-left text, unsupported languages, narrow screens, and error/status discoverability.
6. Windows 10 and 11 as supported targets. Perform clean smoke tests on Linux/macOS if environments are available; otherwise mark compatibility unverified and narrow release claims.
7. CPU-only, low-VRAM, recommended GPU, offline, slow network, and missing-optional-tool configurations.

**Exit evidence:** end-to-end checklist, screenshots, accessibility/compatibility matrix, and UI/UX findings.

### Phase 6 — Performance and Resource Audit

Measure cold and warm behavior for each engine and major workflow:

1. GUI startup time and proof that heavy ML packages are not loaded in the GUI process.
2. Worker/model load latency, time to first audio, real-time factor, throughput, and long-batch stability.
3. GUI CPU/RAM/handle/thread counts and worker CPU/RAM/VRAM before load, after load, during work, after cancellation, after switch, and after exit.
4. VRAM release and fragmentation across repeated engine switches; OOM behavior on constrained devices.
5. Disk usage for environments, caches, models, temporary files, profiles, outputs, installer, and portable release.
6. Audio Analyzer behavior on large collections and large individual files; UI responsiveness during scans and CSV export.
7. Worst-case input length, reference count, batch size, and output-directory growth; document enforced or safe limits.

**Exit evidence:** benchmark table, resource timelines, leak analysis, bottleneck list, and capacity recommendations.

### Phase 7 — Tests, CI/CD, Installation, and Release Audit

1. Run `python -m pytest tests/ -v` from a clean supported GUI environment; enumerate actual tests rather than relying on the documented count.
2. Map tests to risks and components. Identify missing integration, GUI, worker lifecycle, model smoke, installer, negative-security, and end-to-end coverage.
3. Add coverage measurement for code suited to unit testing; evaluate meaningful branch/behavior coverage rather than using pass percentage as “test coverage.”
4. Inspect CI permissions, pinned action versions, secrets, artifact provenance, cache poisoning risk, release triggers, retention, and failure behavior.
5. Build launcher, portable archive, and installer from a clean checkout using documented automation. Compare outputs, contents, sizes, hashes, and version strings.
6. Test clean install, paths with spaces/Unicode, non-admin install, upgrade, repair/reinstall, uninstall, retained user data, offline relaunch, and rollback.
7. Verify signatures or explicitly document their absence; generate checksums and an SBOM; confirm release notes and all required policies/notices are packaged.
8. Confirm `.gitignore` and packagers exclude environments, caches, local config, reference audio, profiles, outputs, and development secrets.

**Exit evidence:** test-risk matrix, CI review, reproducibility report, install/uninstall report, release manifest, checksums, and SBOM.

### Phase 8 — Synthesis and Release Decision

1. De-duplicate findings and assign stable IDs (`SEC-`, `REL-`, `CON-`, `PERF-`, `UX-`, `TEST-`, `RELENG-`, `LIC-`, `DOC-`).
2. Reproduce every Critical/High finding independently and record confidence.
3. Score likelihood and impact; create a priority-ordered remediation plan with effort, dependencies, expected impact, risk reduction, owner, and validation test.
4. Recalculate the release-readiness score from evidence; do not inherit 94/100 or 100/100 from documentation.
5. Issue the final grade and release recommendation, listing explicit blockers and conditions for the next tier.
6. After remediation, run focused regression tests plus the full release gate and publish a closure addendum.

## 7. Minimum Test and Evidence Matrix

| Area | Required automated evidence | Required runtime/manual evidence |
|---|---|---|
| GUI/core | model/config/IPC unit tests; import-boundary check | fresh launch, close/cancel, keyboard/DPI walkthrough |
| Each TTS worker | protocol, validation, secure-loading tests | model load, one generation, cancellation, crash/OOM recovery |
| RVC | secure-loading and parameter-bound tests | consented conversion, corrupt model, CPU/GPU failure behavior |
| DSP | format/property tests and corrupt-input fuzz corpus | analyzer/filter results checked against known fixtures |
| Skyrim | command/path and FUZ binary tests | tool-missing, permission failure, and valid export on Windows |
| Downloader/setup | URL/hash/path/atomicity tests | interrupted install/download and offline recovery |
| Privacy/ethics | policy resolution and acceptance-state tests | network trace and cloning-point consent UX |
| Packaging/release | manifest/content assertions | clean build, install, upgrade, uninstall, offline relaunch |

Full neural-model tests may be tagged and run in a GPU release-gate job rather than ordinary unit CI, but they cannot be omitted from release qualification.

## 8. Finding Severity and Risk Scoring

### Severity

- **Critical:** practical code execution, credential/private biometric disclosure, irreversible broad data loss, non-consensual voice distribution, or release-illegal artifact.
- **High:** likely serious compromise, repeated crash/data corruption, unusable primary workflow, or major privacy/license misrepresentation.
- **Medium:** constrained security/reliability issue, important UX/accessibility failure, material maintainability or performance risk.
- **Low:** limited-impact defect, hardening opportunity, minor usability/documentation drift.
- **Informational:** verified observation or improvement with no present material risk.

### Risk score

Use **Likelihood (1–5) × Impact (1–5)**:

- 20–25: Critical priority
- 12–19: High priority
- 6–11: Medium priority
- 1–5: Low priority

Severity may be raised where law, biometric privacy, consent, supply-chain integrity, or irreversible data loss warrants it. Record the rationale for overrides.

## 9. Required Finding Record

Each finding must contain:

| Field | Requirement |
|---|---|
| ID and title | Stable category prefix and concise description |
| Severity/category/component | Consistent taxonomy and exact affected surface |
| Status/confidence | Open, mitigated, accepted, false positive; High/Medium/Low confidence |
| Evidence/reproduction | File/line, command, fixture, log, screenshot, or artifact hash |
| Root cause | Underlying design or implementation cause, not only the symptom |
| Impact and affected users | Security, privacy, correctness, reliability, UX, release, or business impact |
| Likelihood/impact/risk | 1–5 values and calculated score |
| Recommendation | Smallest durable correction, including defense in depth where warranted |
| Effort/dependencies/owner | S/M/L or person-days, prerequisites, accountable owner |
| Validation | Exact regression, runtime check, or artifact inspection that closes it |

## 10. Release Gates

VoiceTTSr must not be rated Production Ready unless all of the following are true:

- no open Critical findings;
- no open High security, privacy, ethics, licensing, data-loss, or primary-workflow reliability findings;
- every shipped engine passes a supported-hardware smoke test and fails safely on unsupported hardware;
- clean install, launch, upgrade, uninstall, and offline relaunch pass on supported Windows versions;
- release contents, licenses, notices, checksums, and provenance are verified;
- privacy and offline claims match observed network behavior and are accurately qualified;
- no forbidden real-person/default voice asset or proprietary `FonixData.cdf` is shipped;
- cancellation, worker crash, corrupt config/profile/audio, permission denial, full disk, and engine-switch recovery are tested;
- automated tests and CI pass from a clean environment, with meaningful risk coverage documented;
- release artifacts are built from the audited commit and their hashes are recorded;
- user documentation reflects actual paths, artifact sizes, support boundaries, and known limitations.

Enterprise Ready additionally requires defined support/security response processes, signed release artifacts, documented update/rollback strategy, vulnerability disclosure handling, stronger accessibility evidence, and repeatable SBOM/provenance generation.

## 11. Planned Deliverables

1. Executive summary with grade, 0–100 score, strengths, risks, blockers, and release decision.
2. Verified system and trust-boundary overview.
3. Findings table across all 15 audit dimensions.
4. Risk matrix and release-gate status.
5. Test-risk traceability matrix and raw evidence index.
6. Dependency/model/license/provenance matrix and SBOM.
7. Performance/resource benchmark report.
8. Installation, packaging, and reproducibility report.
9. Documentation-drift report.
10. Priority-ordered remediation plan with effort, impact, dependencies, owner, and validation.
11. Post-remediation closure report when fixes are complete.

## 12. Suggested Execution Order and Effort

The recommended order is Phase 0, Phase 1, then parallelizable deep work across Phases 2–7, followed by Phase 8. A credible first audit is expected to require roughly 8–12 engineer-days, excluding large model downloads and cross-platform hardware availability:

| Work | Estimate |
|---|---:|
| Baseline and blocker triage | 1–1.5 days |
| Architecture/code quality | 1 day |
| Security/privacy/ethics/licensing | 2–3 days |
| Reliability/concurrency/data safety | 1.5–2 days |
| Functional/UI/accessibility/platform | 1–2 days |
| Performance/resource profiling | 1–1.5 days |
| CI/install/release reproducibility | 1–1.5 days |
| Synthesis and final report | 0.5–1 day |

If GPU hardware, a clean Windows VM, or Linux/macOS test hosts are unavailable, record those as evidence gaps. Do not convert an untested claim into a passing score.
