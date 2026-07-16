# AGENTS.md

## Authoritative documents

Read before editing:

- `docs/IMPLEMENTATION_PLAN.md`
- `docs/ARCHITECTURE.md`
- `docs/DATA_CONTRACT.md`
- `docs/ROADMAP.md`
- `docs/STATUS.md`
- `docs/LEGACY_REFERENCE.md`
- `docs/DECISIONS.md`

## Legacy repository

The legacy repository is read-only:

`/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic`

Override its location with `MUSIC_CRITIC_LEGACY_ROOT`.

1. Never modify, format, stage, commit, reset, clean, or restore the legacy repository.
2. Never import legacy modules at runtime.
3. Treat legacy code as reference material, not the V2 specification.
4. Do not copy whole legacy modules.
5. Record adapted concepts and rejected assumptions in `docs/LEGACY_REFERENCE.md`.
6. V2 must run without the legacy checkout.

## Scientific and engineering rules

1. Implement only the requested roadmap phase.
2. Raw unlabeled MIDI inference must remain possible.
3. Theory labels are auxiliary targets unless a later recorded decision says otherwise.
4. Missing labels use masks and are never negative labels.
5. Gold semantic segmentation cannot be required at inference.
6. Canonical timing must be exact and must not rely on float equality.
7. Do not commit datasets, rendered audio, generated MIDI, caches, checkpoints, or outputs.
8. Every implementation change requires tests.
9. Do not silently add dependencies.
10. Preserve provenance, target availability, and confidence.
11. Update `docs/STATUS.md` after every task.
12. Update `docs/DECISIONS.md` when an architectural decision changes.

## Task protocol

Before editing, state the current phase, files to change, legacy files to inspect,
tests to add or run, and explicit non-goals.

At completion, report files changed, behavior implemented, exact test results,
legacy logic reused or rejected, unresolved issues, and the next phase.
