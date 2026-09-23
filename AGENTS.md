# AGENTS.md â€” Agent Operating Instructions for VoiceTTSr

> **NOTICE TO ALL AI AGENTS & PAIR PROGRAMMERS**:  
> This project is governed by the **Project Directives** centralized blueprint.  
> Before formulating plans or executing modifications, you MUST read and comply with the governing directives.

---

## 1. Governance Directives

- **Central Blueprint Repo**: `E:/MachineApps/Project Directives/project-directives`
- **Session Rules**: Read `E:/MachineApps/Project Directives/project-directives/DIRECTIVES.md`
- **Universal Operating Rules**: Read `E:/MachineApps/Project Directives/project-directives/OPERATIONS.md`
- **Commit Formatting**: Read `E:/MachineApps/Project Directives/project-directives/COMMIT_CONVENTIONS.md`

## 2. Local Project State

Before taking action, inspect the project's documentation:
- `project.yaml` â€” Identity, metadata, and current phase
- `project-status.md` â€” Current milestone, active blockers, and immediate tasks
- `roadmap.yaml` â€” Phased delivery plan
- `memories.md` â€” Durable architectural decisions and known invariants
- `worklogs.md` â€” Historical session records (append-only)

## 3. Commit Rules

All git commits must follow conventional commits:
- `feat(scope): ...` â€” New feature
- `fix(scope): ...` â€” Bug fix
- `docs(.project): ...` â€” Project governance / session closure
- `refactor(scope): ...` â€” Restructuring without logic change
- `test(scope): ...` â€” Adding/updating tests

Never use vague messages like "updates" or "fix bug".

## 4. Session Wrap-Up Requirements

Every session must conclude with:
1. Running project verification/tests
2. Appending a structured entry to `worklogs.md`
3. Updating `project-status.md`
4. Updating `roadmap.yaml` if progress was made
5. Staging and committing documentation changes

