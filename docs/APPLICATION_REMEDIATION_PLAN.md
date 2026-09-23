# VoiceTTSr Application Remediation Plan

**Prepared:** 2026-08-24  
**Companion document:** `docs/APPLICATION_AUDIT_PLAN.md`  
**Target:** VoiceTTSr desktop application, isolated model workers, supporting tools, installers, and release pipeline  
**Status:** Initial plan — confirmed baseline issues are actionable; all other work remains evidence-gated until the audit confirms a finding.

## 1. Purpose and Operating Rule

This plan defines how VoiceTTSr will move from audit evidence to a defensible release state. It does not treat historical “94/100,” “100/100,” “production hardened,” or “security passed” statements as current proof.

Remediation follows one rule: **validate, fix the root cause, add a regression control, then re-test the release gate**. Audit hypotheses must not trigger speculative rewrites. Every remediation item must be linked to a confirmed finding, an owner, a validation method, and closure evidence.

## 2. Priority Definitions

| Priority | Meaning | Target response |
|---|---|---|
| **P0 — Release blocker** | Critical security/privacy/ethics/licensing issue, irreversible data-loss risk, unreproducible baseline, or inability to validate the product | Start immediately; no public release until closed |
| **P1 — Required for production** | High-risk reliability, security, primary-workflow, packaging, or compliance gap | Complete before Production Ready decision |
| **P2 — Required quality hardening** | Medium-risk maintainability, UX/accessibility, performance, compatibility, or test-depth issue | Complete for production where user impact is material; otherwise time-bound with owner |
| **P3 — Improvement** | Low-risk cleanup, documentation polish, optimization, or future scalability work | Schedule after release gates are satisfied |

## 3. Current Verified Baseline Issues

The following were directly observed while preparing this plan and can enter the remediation backlog without waiting for the full audit:

| ID | Priority | Verified issue | Evidence | Required outcome |
|---|---|---|---|---|
| **REM-001** | P0 | The documented default test command is not reproducible in the current workspace. `python` points to a missing Python 3.11 executable; `gui-env\Scripts\python.exe` points to a missing Python 3.10 executable. | Both attempts to run `python -m pytest tests -q` failed before test collection. | A clean, documented environment can install/resolve dependencies and run the complete test suite from the audited commit. |
| **REM-002** | P1 | Current documentation describes worker files under `workers/`, but the four worker scripts are at repository root. | `docs/memories.md`, archived architecture/status, and worklogs conflict with the current tree. | Choose one canonical layout and update code, launch paths, tests, packagers, and active docs atomically. |
| **REM-003** | P1 | Ethics documentation references `implementation_plan.md` and `project_status.md`, but those are not current resolvable active documents at the referenced paths. | `docs/VOICE_ETHICS.md` contains the references; only an archived project status exists. | Replace broken references with maintained, repository-valid documents and verify the described consent control in code/UI. |
| **REM-004** | P1 | Readiness and test claims conflict across maintained and archived documentation. “32/32 passed” is presented as “100% coverage,” while this is a pass count, not coverage measurement. | `docs/worklogs.md` reports 100/100 and 32/32; archived status reports 94/100 and 19/19; 32 test functions currently match the naming pattern. | Publish one evidence-backed status source, distinguish test pass rate from coverage, and mark historical documents clearly. |
| **REM-005** | P1 | A local root `VoiceTTSr.exe` is approximately 474 MB, while documentation describes a lightweight launcher/approximately 28 MB release architecture. The executable is not returned by `git ls-files`. | Local artifact size: 474,518,730 bytes; release claims appear in architecture/worklogs. | Establish which artifact is authoritative, reproduce it from a clean commit, document expected sizes, and ensure local artifacts cannot silently enter packages. |
| **REM-006** | P1 | Strong privacy/release claims are stated as certifications without attached current runtime evidence. | README and privacy policy claim zero cloud calls/100% offline; worklogs state production hardened/security passed. | Produce network/runtime evidence and qualify all statements to match observed install, model-download, cache, and generation behavior. |

These items do not establish that the application is unsafe; they establish that its current release claims and verification baseline are not yet fully reproducible.

## 4. Remediation Governance

### 4.1 Finding intake

Each confirmed audit finding becomes a remediation ticket containing:

- finding ID, severity, affected versions/components, and release gate;
- reproduction steps and immutable evidence reference;
- root cause and affected trust boundary or workflow;
- proposed correction and alternatives considered;
- owner, estimate, dependencies, and target milestone;
- tests and runtime checks required for closure;
- documentation, notice, migration, and rollback impact.

Critical and High findings require independent reproduction before implementation, unless delaying containment would expose users. False positives are closed with evidence rather than deleted from the record.

### 4.2 Change discipline

1. Use small remediation changes grouped by one risk or tightly coupled control.
2. Preserve VoiceTTSr’s documented architecture invariants unless the remediation explicitly replaces an invariant through an ADR.
3. Keep heavy ML imports outside the GUI process.
4. Keep UI access on the Tk main thread and snapshot worker inputs into immutable data.
5. Use safe tensor formats and never broaden unpickling to accept untrusted user files.
6. Use disposable fixtures for destructive, corrupt-file, and failure-injection tests.
7. Use only synthetic or explicitly consented voice material in tests and release assets.
8. Do not close a finding on code review alone; run its regression check and the appropriate release-gate suite.

### 4.3 Branch and evidence strategy

- Remediation branches should use `codex/` or the project’s agreed release prefix and reference finding IDs.
- Store concise test output, benchmark summaries, hashes, manifests, screenshots, and traces in the audit evidence index; do not commit private voice/audio data.
- Rebuild release artifacts from the remediated commit. Never reuse an artifact whose provenance cannot be tied to that commit.
- Update `docs/worklogs.md` only after validation, using exact commands and results rather than generalized readiness claims.

## 5. Remediation Workstreams

### Workstream A — Restore a Reproducible Engineering Baseline

**Priority:** P0  
**Primary item:** REM-001

1. Decide and document the supported GUI/orchestrator Python version.
2. Repair or recreate `gui-env` from a clean interpreter; do not patch its internal absolute paths manually as the durable solution.
3. Pin direct and transitive GUI test/build dependencies using the project’s selected lock strategy.
4. Provide one environment bootstrap command and one canonical test command that work in a clean Windows checkout.
5. Add a preflight script/check that reports missing or stale interpreters, worker environments, GPU prerequisites, helper binaries, and available disk.
6. Run and capture the 32 currently discoverable tests, then enumerate collected tests using pytest rather than filename-pattern counts.
7. Add CI coverage for clean environment creation so broken virtual-environment paths cannot be mistaken for a passing baseline.

**Closure criteria:** clean setup plus full test execution succeeds twice on a fresh Windows environment; CI repeats the process; the exact Python/dependency versions and command are documented.

### Workstream B — Security and Trust-Boundary Remediation

**Priority:** P0/P1 when findings are confirmed

1. Fix every unsafe deserialization path at the boundary where untrusted model/profile input enters. Preserve safetensors as the default and prove any base-model compatibility exception cannot load user-controlled data.
2. Apply strict schema/type/size/action validation to worker JSON-line requests and responses.
3. Constrain all input/output/profile/model paths to intended roots where appropriate; reject traversal, device paths, control characters, and unsafe overwrite cases.
4. Replace unsafe shell invocation with argument arrays; correctly quote Windows paths; make executable resolution explicit.
5. Require HTTPS, bounded timeouts, atomic writes, destination confinement, and cryptographic integrity for downloaded technical assets and models wherever stable hashes are possible.
6. Remove secrets, private files, reference audio, profiles, outputs, caches, and forbidden voice assets from tracked and packaged content; add automated package assertions.
7. Address dependency vulnerabilities independently in the GUI, XTTS, Qwen, Chatterbox, and RVC environments. Where an old dependency is unavoidable, document exploitability, compensating controls, and upgrade owner/date.
8. Harden native-helper execution against writable search paths and binary substitution; record hashes and provenance for bundled executables.

**Closure criteria:** all Critical/High security findings are closed; hostile-input regression tests pass; dependency exceptions are documented and approved; artifact content checks pass.

### Workstream C — Privacy, Consent, Provenance, and Licensing

**Priority:** P0/P1  
**Primary items:** REM-003, REM-006

1. Trace and test every potential network path: installers, resource downloads, model hubs/caches, libraries, update logic, crash handling, and generation.
2. Rewrite “100% offline,” “zero cloud calls,” and “certified” language to distinguish initial acquisition from offline inference and to disclose any unavoidable metadata/network behavior.
3. Make offline mode observable and testable. Fail with clear remediation instructions when required local models are missing rather than silently connecting.
4. Verify consent guidance appears at the action point for cloning and RVC, is accessible after first launch, and is not merely a persisted blanket waiver.
5. Verify that no named-person/default voice model ships or downloads automatically; add a test against resource manifests and package contents.
6. Create a versioned dependency/model/binary license and provenance matrix covering exact artifacts, source, license, redistribution, commercial-use limits, required notices, and watermark behavior.
7. Resolve redistribution rights for `FaceFXWrapper.exe` and `xWMAEncode.exe`; keep `FonixData.cdf` excluded and test that packaging fails if it appears.
8. Make XTTS commercial-use restrictions and Chatterbox watermark behavior visible at the relevant engine selection/export points when legally or ethically required.
9. Repair policy cross-references and assign a maintainer/review trigger for dependency or bundled-asset changes.

**Closure criteria:** observed network behavior matches policy; license/provenance matrix is complete; forbidden-asset checks pass; consent UX is verified; all required notices ship in every distribution.

### Workstream D — Worker, IPC, and Concurrency Reliability

**Priority:** P1

1. Add explicit worker states, startup/generation/quit timeouts, bounded queues, request correlation IDs, and deterministic cancellation semantics where missing.
2. Handle malformed/non-JSON stdout without losing subsequent valid protocol messages; separate machine protocol output from diagnostic logs.
3. Ensure stale responses cannot cross job boundaries after cancellation, restart, or engine switching.
4. Terminate full subprocess trees on stop/exit and verify no Python/CUDA child survives GUI shutdown.
5. Add recovery paths for worker crash/hang, missing model, CUDA OOM, corrupt profile, invalid output, full disk, and permission denial.
6. Instrument Tk calls during tests and remove every background-thread widget/variable access. Route UI updates through `after`/main-thread dispatch.
7. Validate engine-switch VRAM release under repetition and expose clear status when memory cannot be reclaimed.
8. Add lifecycle integration tests with lightweight fake workers plus tagged real-model smoke tests for release qualification.

**Closure criteria:** the fault-injection matrix passes; no orphan processes or stale events remain; cancellation and restart are deterministic; supported engines pass tagged smoke tests.

### Workstream E — File, Configuration, and Audio Data Safety

**Priority:** P1

1. Add a versioned configuration schema, strict type/range validation, atomic replacement, corrupt-file quarantine, and migration tests.
2. Never persist secrets or unnecessary biometric-derived data in configuration/logs. Document the storage and deletion lifecycle for reference audio and profiles.
3. Sanitize filenames and custom output paths; handle collisions, locked files, Unicode, long paths, removable/network drives, and missing directories.
4. Write generated/converted/exported files to temporary names and atomically publish only after validation; clean incomplete files on cancellation/failure.
5. Make recycle-bin fallback behavior explicit. Avoid silent permanent deletion when `send2trash` fails; ask or safely stop according to the workflow.
6. Add audio decoder/DSP limits and validation for zero-length, truncated, oversized, unusual format, NaN/Inf, silent, clipped, and multichannel inputs.
7. Validate generated WAV headers/sample formats and Skyrim `.fuz` boundaries before reporting success.
8. Provide recovery guidance for corrupt config, missing output, full disk, and failed exports.

**Closure criteria:** destructive/corruption tests use disposable paths and pass; interrupted operations leave no misleading final artifacts; recovery instructions are actionable.

### Workstream F — Architecture and Maintainability

**Priority:** P2  
**Primary item:** REM-002

1. Decide whether worker scripts remain at root or move into a `workers/` package. Update process-launch path discovery, packagers, installers, tests, and docs in the same change.
2. Decompose `voice_cloner_gui.py` by responsibility, beginning with low-coupling seams: worker supervision, configuration repository, job orchestration, output management, Audio Analyzer view, and Skyrim view.
3. Keep the Tk root/application composition layer thin and pass explicit services/state rather than reaching across widgets.
4. Centralize parameter validation and protocol models shared by GUI and workers without importing ML dependencies into the GUI.
5. Remove duplicate DSP/file/config helpers only after characterization tests preserve behavior.
6. Add type checking and focused lint/static-security checks with pinned configuration and an incremental adoption baseline.
7. Record architecture changes in ADRs and update `docs/memories.md` component maps immediately.

**Closure criteria:** documented paths match the tree; launch/package tests pass; key responsibilities have clear owners; no heavy ML package enters the GUI import graph.

### Workstream G — Test Strategy and CI Quality

**Priority:** P1/P2  
**Primary item:** REM-004

1. Replace “N/N tests equals 100% coverage” language with separate collected/pass counts and measured statement/branch coverage.
2. Map each release risk to unit, integration, end-to-end, manual, or tagged hardware evidence.
3. Add missing negative tests for IPC validation, secure loading, paths, downloads, cancellation, worker death, config migration, partial files, license/asset manifests, and packaging.
4. Add clean Windows GUI smoke, installer smoke, and artifact-content jobs. Run real-model GPU smoke tests on an approved release runner or document the manual gate.
5. Pin CI actions to reviewed immutable versions/commits, minimize token permissions, validate release triggers, and protect artifacts from untrusted cache/input paths.
6. Archive JUnit/coverage/security/SBOM outputs with the commit and release manifest.
7. Set practical, risk-based quality gates; do not inflate coverage with tests that only import modules or assert implementation details.

**Closure criteria:** CI is reproducible and least-privileged; test-risk traceability is complete; required suites pass from a clean commit; status docs report accurate metrics.

### Workstream H — UI/UX, Accessibility, and Platform Claims

**Priority:** P2

1. Fix blocked or ambiguous primary workflows first: setup/model readiness, engine selection, generate/cancel/retry, errors, output recovery, and consent.
2. Establish keyboard focus order, visible focus, shortcuts, semantic labels, readable contrast, and usable dialogs without a mouse.
3. Verify layouts at 100%, 125%, 150%, and 200% scaling; support resizing, text expansion, long paths, and long/localized strings.
4. Make current state, active engine, download/model readiness, GPU usage, generation progress, cancellation, and failure recovery visible.
5. Distinguish warnings from fatal errors and include a next action without exposing sensitive prompts/paths unnecessarily.
6. Test Windows 10/11 as supported targets. Test Linux/macOS in clean environments before claiming compatibility; otherwise document them as experimental/unsupported.
7. Provide CPU-only and insufficient-VRAM guidance and prevent users from entering impossible workflows.

**Closure criteria:** supported Windows UI matrix passes; primary workflows are keyboard operable; support claims match tested platforms/hardware.

### Workstream I — Performance and Resource Management

**Priority:** P2

1. Establish budgets for GUI cold start, worker load, time to first audio, real-time factor, idle CPU, RAM, VRAM, handles/threads, and disk/cache growth.
2. Profile before optimizing. Attribute time/memory to GUI, workers, model download/load, DSP, and packaging separately.
3. Prevent unbounded batch queues, logs, analyzer input sets, and temporary/cache growth; expose safe limits.
4. Verify model hibernation releases VRAM and subprocess resources after cancellation, engine switch, failure, and exit.
5. Keep long DSP, file scan, model, and export work off the Tk thread while maintaining bounded cancellation and progress updates.
6. Add benchmark scripts with synthetic/consented fixed fixtures and record hardware/software details so results are comparable.

**Closure criteria:** no confirmed leak remains; budgets are met or documented as known limitations; benchmark regressions are visible before release.

### Workstream J — Build, Installer, and Release Integrity

**Priority:** P1  
**Primary item:** REM-005

1. Identify the intended launcher and release formats; remove ambiguity among the local 474 MB executable, PyInstaller spec, installer, and portable package.
2. Build each supported artifact from a clean checkout and record commit, tool versions, inputs, contents, size, and SHA-256.
3. Ensure package scripts fail closed when local config, voice/reference/profile/output data, environments, caches, proprietary assets, or unexpected large files are included.
4. Make version metadata consistent across GUI, executable, installer, archive name, release notes, and documentation.
5. Test non-admin install, spaces/Unicode paths, first run, upgrade, repair/reinstall, uninstall, retained user data, rollback, and offline relaunch.
6. Produce an SBOM and all required third-party notices for every distribution.
7. Sign Windows artifacts or explicitly document signature absence and associated user warnings until signing is implemented.
8. Define release rollback/revocation, vulnerability disclosure, and supported-version policies.

**Closure criteria:** artifacts are reproducible from the audited commit, contain only allowlisted content, pass install lifecycle tests, and have recorded hashes/SBOM/notices.

### Workstream K — Documentation and Readiness Truthfulness

**Priority:** P1/P2  
**Primary items:** REM-002 through REM-006

1. Designate active versus archived documents and add archive banners/dates so historical scores cannot be read as current status.
2. Replace unsupported absolutes with measured, scoped claims and link each readiness assertion to evidence.
3. Correct directory maps, environment commands, test metrics, artifact sizes, supported platforms, and model/license details.
4. Keep one current status dashboard. Calculate readiness only after audit synthesis; do not manually set 100/100 because a checklist was completed.
5. Add documentation checks for broken local links, required policies/notices, and component-path existence.
6. Update worklogs with exact verification results after remediation; retain failed checks and known limitations.

**Closure criteria:** active docs agree with the audited tree and artifacts; links resolve; claims are current, qualified, and evidence-backed.

## 6. Sequenced Remediation Waves

### Wave 0 — Baseline Recovery and Containment

**Target:** 1–2 engineer-days after audit evidence is available.

- Close REM-001 and establish clean tests/CI.
- Triage all Critical/High findings.
- Quarantine forbidden or unproven release assets if found.
- Freeze production release claims until evidence is reproducible.
- Create the finding register and evidence index.

**Gate:** testing is reproducible; no unknown Critical issue remains untriaged.

### Wave 1 — Security, Privacy, Ethics, Licensing, and Data Safety

**Target:** 3–6 engineer-days, highly dependent on findings and third-party licensing review.

- Close all P0 and High security/privacy/ethics/license findings.
- Harden downloads, model/profile loading, paths, subprocesses, and package contents.
- Reconcile offline/privacy claims with network traces.
- Complete provenance/license matrix and consent controls.
- Add regression and package-content tests.

**Gate:** no open Critical or High issue in these categories; legal/provenance uncertainties block public distribution until resolved.

### Wave 2 — Reliability and Primary Workflow Stability

**Target:** 3–5 engineer-days plus real-model hardware time.

- Repair IPC lifecycle, timeouts, cancellation, restart, and shutdown.
- Close data corruption/loss and partial-output findings.
- Pass worker crash/OOM/corrupt-input/permission/full-disk scenarios.
- Run one real smoke test per shipped engine on supported hardware.

**Gate:** all primary workflows complete or recover safely; no orphan workers or stale job results; no open High reliability finding.

### Wave 3 — Release Engineering and Documentation Alignment

**Target:** 2–4 engineer-days.

- Resolve artifact architecture and reproducibility.
- Complete clean install/upgrade/uninstall/offline tests.
- Generate hashes, SBOM, notices, and accurate release manifest.
- Close documentation drift and publish evidence-backed readiness status.

**Gate:** artifacts derive from the audited commit and meet all Production Ready release gates.

### Wave 4 — Maintainability, UX, Accessibility, and Performance

**Target:** 4–8 engineer-days, suitable for incremental milestones.

- Decompose high-risk seams of the GUI monolith.
- Improve test depth and clean architecture boundaries.
- Close material accessibility/high-DPI/platform gaps.
- Meet performance/resource budgets and document constraints.

**Gate:** all remaining P2 findings either close or have an approved owner, deadline, user-impact disclosure, and rationale for deferral.

## 7. Prioritized Initial Backlog

| Order | Item | Priority | Estimate | Dependencies | Validation |
|---:|---|---|---:|---|---|
| 1 | Restore clean supported Python environment and canonical test command | P0 | 0.5–1 day | Supported Python decision | Fresh checkout runs full pytest twice and in CI |
| 2 | Establish current finding register, evidence index, and release freeze criteria | P0 | 0.5 day | Audit baseline | All Critical/High findings have owner/status |
| 3 | Verify package contents, local large executable provenance, and forbidden assets | P0/P1 | 0.5–1 day | Clean build tooling | Allowlist assertions, hashes, clean manifest |
| 4 | Validate model/profile loaders, download integrity, paths, and subprocess boundaries | P0/P1 | 1–3 days | Working test runtime | Hostile-input/security regressions pass |
| 5 | Produce network trace and correct privacy/offline claims | P1 | 1 day | Install/model fixtures | Offline/network matrix matches docs |
| 6 | Complete license/provenance matrix and consent control verification | P1 | 1–3 days | Exact artifact versions | Legal notices and forbidden-asset tests pass |
| 7 | Harden worker lifecycle, cancellation, failure recovery, and shutdown | P1 | 1–3 days | Fake worker harness | Fault-injection matrix passes |
| 8 | Protect config/output/audio workflows from corruption and partial files | P1 | 1–2 days | Disposable fixtures | Recovery and atomicity tests pass |
| 9 | Reproduce launcher, portable archive, and installer from clean commit | P1 | 1–2 days | Baseline/build dependencies | Manifest, hashes, install lifecycle pass |
| 10 | Correct active docs, references, metrics, layout, support, and artifact claims | P1 | 0.5–1 day | Outcomes of items 1–9 | Link/path checks and evidence review pass |
| 11 | Expand risk-based CI and tagged hardware/model smoke gates | P1/P2 | 1–2 days | Stable worker tests/runners | Risk traceability and CI artifacts complete |
| 12 | Incrementally decompose GUI and close material UX/accessibility/performance issues | P2 | 3–8 days | Characterization tests | Regression, UI matrix, and budgets pass |

Estimates are planning ranges, not commitments. Confirm them after the audit supplies reproduction detail and affected-code scope.

## 8. Closure Standard

A remediation is closed only when:

1. the root cause is corrected or a formally accepted risk documents why it is not;
2. a regression test or repeatable verification detects recurrence;
3. focused tests and the relevant full gate pass in a clean environment;
4. user data, migration, rollback, and compatibility effects are addressed;
5. policies, notices, architecture docs, setup instructions, and worklogs are updated where affected;
6. the evidence index records commit, command, environment, result, and artifact hashes/screenshots/traces;
7. a reviewer confirms that the original reproduction no longer succeeds and no equivalent bypass remains.

“Code changed,” “tests passed previously,” or “unable to reproduce on one machine” is not sufficient closure evidence.

## 9. Production Release Exit Criteria

Before VoiceTTSr can receive a Production Ready recommendation:

- REM-001 is closed and clean CI/test execution is reproducible;
- no Critical finding is open;
- no High security, privacy, ethics, licensing, data-loss, primary-workflow, or release-integrity finding is open;
- every shipped engine has passed a supported-hardware smoke test and safe-failure checks;
- privacy/offline statements match runtime network evidence;
- no non-consensual/default named-person voice or prohibited proprietary asset is distributed;
- dependency/model/binary provenance and license obligations are complete;
- cancellation, crash, OOM, corrupt files, permissions, full disk, and engine switching recover safely;
- supported Windows install, upgrade, uninstall, and offline relaunch pass from artifacts built from the audited commit;
- checksums, SBOM, notices, version metadata, and release manifest are present;
- active documentation is accurate and the final readiness score is recalculated from evidence.

Any deferred P2/P3 issue must have an owner, deadline, impact statement, and release-note disclosure when users could encounter it.

## 10. Reporting Cadence

- Update the finding/remediation register after each reproducible result or merged fix.
- Re-run focused tests immediately after each change.
- Run the full clean test gate at the end of every remediation wave.
- Run full install/model/release qualification before changing the release recommendation.
- Append a precise worklog entry only after validation, including failed checks and remaining limitations.
- Publish a post-remediation closure report mapping every original finding to evidence: Closed, Accepted, Deferred, or Not Reproducible.

