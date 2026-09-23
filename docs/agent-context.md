# Agent Context â€” VoiceTTSr

**Blueprint Version:** 1.1.0  
**Central Hub:** `E:/MachineApps/Project Directives/project-directives`  
**Last Synced:** 2026-09-23  

---

## Operating Invariants for this Repository

Any AI agent (or human developer) contributing to **VoiceTTSr** must follow the Project Directives blueprint:

1. **Before writing code**:
   - Inspect `project.yaml`, `project-status.md`, `roadmap.yaml`, and `memories.md`.
   - Read recent entries in `worklogs.md` to establish current trajectory.
   - Inspect Git status (`git status`, `git log -5`).
   - Read `DIRECTIVES.md` and `OPERATIONS.md` in the hub before non-trivial changes.

2. **Universal Ethics & Rules**:
   - Source of truth: Central rules in `DIRECTIVES.md` and `OPERATIONS.md`.
   - Never fabricate commit history, test outcomes, or benchmark claims.
   - Never commit secrets, credentials, or transient build artifacts.
   - Keep scope bounded to the active task; file non-immediate ideas as unresolved items.

3. **Commit Standards**:
   - All commits must conform to `COMMIT_CONVENTIONS.md` (e.g. `feat(scope): ...`, `fix(scope): ...`, `docs(.project): ...`).
   - Run tests and static checks before committing.

4. **Session Handoff & Closure**:
   - Append a structured session entry to `worklogs.md`.
   - Update `project-status.md` with current checkpoint, active blockers, and immediate next steps.
   - Update `roadmap.yaml` if phase or milestone completion progressed.
   - Commit documentation changes and ensure sync to the central hub.

