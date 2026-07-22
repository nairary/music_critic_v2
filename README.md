# Music Critic V2

Music Critic V2 is a raw-symbolic-first research project for reusable symbolic
music representation learning, theory analysis, and preference-aware quality
assessment.

This repository is a clean-room successor to the legacy Music Critic V1
repository. V1 may be inspected as read-only reference material, but this
package has no runtime dependency on it and must remain runnable when the legacy
checkout is absent.

## Current state

Phases 0 through 3A are implemented. The repository provides an exact immutable
canonical schema, generic MIDI and HookTheory adapters, diagnostic canonical
MIDI export, and a versioned raw-only PyG heterograph builder. The graph uses
mandatory `song`, `track`, `bar`, `beat`, `onset`, and `note` nodes and never
uses theory targets, gold semantic structure, split, or provenance as encoder
input or topology. HookTheory and generic MIDI therefore have graph-schema
parity; their raw observations and supervision are not expected to have general
data parity. Neural models, SSL objectives, corruption training,
preference training, and deployable scoring inference are not implemented yet.

## Layout

- `src/music_critic/data/`: canonical timing, schema, validation, serialization;
- `src/music_critic/adapters/`: generic MIDI and HookTheory conversion;
- `src/music_critic/exporters/`: output-only diagnostic MIDI rendering;
- `src/music_critic/graph/`: feature registry, relations, builder, validation,
  and deterministic graph serialization;
- `docs/`: authoritative plan, architecture, contracts, decisions, and status;
- `configs/`: reserved for phase-owned configuration;
- `scripts/`: audits, rendering/smoke tools, and graph benchmark;
- `tests/`: canonical, adapter, exporter, audit, integration, and graph tests.

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

Build a raw graph or run the small construction benchmark from canonical JSON:

```bash
PYTHONPATH=src python -c \
  "from music_critic.data import load_piece; from music_critic.graph import build_raw_graph; print(build_raw_graph(load_piece('tests/fixtures/data/canonical_piece_v2.json')))"
PYTHONPATH=src python scripts/benchmark_graph_builder.py \
  tests/fixtures/data/canonical_piece_v2.json --repeats 5
PYTHONPATH=src python scripts/benchmark_graph_builder.py \
  --synthetic-suite --repeats 1
```

`build_raw_graph` validates its `CanonicalPiece` by default. Callers that have
already run canonical validation may opt into the documented
`assume_valid=True` fast path. Structural timing remains exact rational data
through graph indexing and becomes `float32` only when feature tensors are
materialized. PyTorch/PyG imports are isolated to `music_critic.graph`, although
the project currently declares those packages as global installation
dependencies.

An editable installation is optional:

```bash
python -m pip install -e .
```

See `docs/ROADMAP.md` for staged implementation work and
`docs/IMPLEMENTATION_PLAN.md` for the scientific specification.
