# Entity Resolution Challenge — Reproducibility Layer

This directory is the persistent code/state layer for the project.

## Source of truth

- `dataset/` stays local and is intentionally ignored by Git.
- `src/` contains reusable preprocessing, metric, data-loading, and split utilities.
- `scripts/` contains reproducible phase scripts.
- `tests/` verifies critical behavior.
- `docs/PHASE_STATE.md` records measured results and architectural decisions.

## Workflow

Before a major experiment:

```powershell
git status
```

After a successful experiment:

```powershell
git add .
git commit -m "Checkpoint: <phase>"
git push
```

Do not store `.venv/` or the challenge dataset in GitHub.

## Current continuation point

Phase 10 Step 2 has completed on the current local fast-dev slice.
The next planned experiment is character TF-IDF candidate retrieval.

Do not tune against test data.
