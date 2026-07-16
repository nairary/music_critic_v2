# Music Critic V2

Music Critic V2 is a raw-symbolic-first research project for reusable symbolic
music representation learning, theory analysis, and preference-aware quality
assessment.

This repository is a clean-room successor to the legacy Music Critic V1
repository. V1 may be inspected as read-only reference material, but this
package has no runtime dependency on it and must remain runnable when the legacy
checkout is absent.

## Current state

Phase 0 bootstrap only. The repository currently contains project structure,
architecture and migration documentation, a legacy snapshot, and contract
tests. It does not yet implement canonical schema classes, MIDI adapters,
graphs, neural models, training, or inference.

## Layout

- `src/music_critic/`: future production package;
- `docs/`: authoritative plan, architecture, contracts, decisions, and status;
- `configs/`: reserved for phase-owned configuration;
- `scripts/`: repository maintenance checks;
- `tests/`: lightweight repository-contract tests.

## Environment

Optional environment variables are documented in `.env.example`:

- `MUSIC_DATA_ROOT`: external dataset root;
- `MUSIC_CRITIC_LEGACY_ROOT`: read-only V1 checkout location.

Datasets remain outside Git. Never commit data, MIDI/audio corpora, generated
outputs, caches, or checkpoints.

## Commands

Run directly from a checkout:

```bash
PYTHONPATH=src python -c "import music_critic; print(music_critic.__version__)"
python -m pytest -q
python -m compileall src
make check
make legacy-check
```

An editable installation is optional:

```bash
python -m pip install -e .
```

See `docs/ROADMAP.md` for staged implementation work and
`docs/IMPLEMENTATION_PLAN.md` for the scientific specification.
