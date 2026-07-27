# Music Critic V2

Music Critic V2 is a raw-symbolic-first research project for reusable symbolic
music representation learning, theory analysis, and preference-aware quality
assessment.

This repository is a clean-room successor to the legacy Music Critic V1
repository. V1 may be inspected as read-only reference material, but this
package has no runtime dependency on it and must remain runnable when the legacy
checkout is absent.

## Current state

Phases 0 through 5B.2 and the Phase 6A/6B representation baselines are
implemented. Phase 6C adds a reproducible supervised training harness for
those unchanged baselines.
The repository provides an exact immutable
canonical schema, generic MIDI and HookTheory adapters, diagnostic canonical
MIDI export, a production POP909-CL adapter, and a versioned raw-only PyG
heterograph builder. The POP909-CL adapter fingerprints the pinned 909-file
corpus, projects only channel-0 score content into raw canonical pieces, keeps
channel-1 chord evidence as masked target-only supervision, and streams corpus
conversion with typed missing-target and quarantine results. The graph uses
mandatory `song`, `track`, `bar`, `beat`, `onset`, and `note` nodes and never
uses theory targets, gold semantic structure, split, or provenance as encoder
input or topology. HookTheory and generic MIDI therefore have graph-schema
parity; their raw observations and supervision are not expected to have general
data parity. A versioned source-native target ontology, exact canonical target
alignment, tensor sidecars, and a production mixed-source PyG collator preserve
that raw-only boundary. Phase 5B.2 adds portable canonical caching, globally
manifested splits, a lazy Dataset, deterministic mixture sampling, and
worker-safe loading. Phase 6A adds comparable feature-only and local
relation-aware encoders, retained local outputs, 14 source-native
fully-supervised heads, inspectable losses, local reconstruction plumbing,
strict checkpoints, and CPU diagnostics. Phase 6B adds deterministic raw-edge
ownership, bar/track pooling, a per-sample coarse Transformer, contextual SONG
rows, top-down gated residual fusion, strict hierarchical checkpoints, and a
controlled three-way ablation. SSL, corruption training, preference training,
PLL, and deployable scoring inference are not implemented yet.

## Layout

- `src/music_critic/data/`: canonical timing, schema, validation, serialization;
- `src/music_critic/adapters/`: generic MIDI, HookTheory, and POP909-CL
  conversion;
- `src/music_critic/exporters/`: output-only diagnostic MIDI rendering;
- `src/music_critic/graph/`: feature registry, relations, builder, validation,
  and deterministic graph serialization;
- `src/music_critic/tasks/`: source-native ontology, exact alignment, versioned
  encodings, target tensorization, mixed-source collator, corpus loading, and
  statistics;
- `src/music_critic/models/`: Phase 6A raw feature/local-GNN encoders and the
  Phase 6B deterministic hierarchy, coarse Transformer, top-down fusion,
  source-native heads, losses, reconstruction, diagnostics, and checkpoints;
- `src/music_critic/training/`: Hydra configuration, non-mutating batch device
  transfer, split CLI, deterministic runners, metrics, and epoch checkpoints;
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
PYTHONPATH=src python scripts/benchmark_multisource_collator.py \
  --samples 32 --repeats 3
PYTHONPATH=src python scripts/benchmark_multisource_collator.py \
  --target-heavy --repeats 3
PYTHONPATH=src python scripts/benchmark_phase6a.py \
  --larger-repeats 4 --overfit-steps 40
PYTHONPATH=src python scripts/benchmark_phase6b.py \
  --larger-repeats 4 --overfit-steps 30
```

`build_raw_graph` validates its `CanonicalPiece` by default. Callers that have
already run canonical validation may opt into the documented
`assume_valid=True` fast path. Structural timing remains exact rational data
through graph indexing and becomes `float32` only when feature tensors are
materialized. PyTorch/PyG imports are isolated to `music_critic.graph`,
`music_critic.tasks`, and `music_critic.models`; the project already declares
those packages as global installation dependencies.

An editable installation is optional:

```bash
python -m pip install -e .
python -m pip install -e '.[training]'
```

See `docs/ROADMAP.md` for staged implementation work and
`docs/IMPLEMENTATION_PLAN.md` for the scientific specification.

Run the opt-in production acceptance against an installed pinned corpus:

```bash
PYTHONPATH=src python scripts/accept_pop909_cl_adapter.py \
  --root data/pop909-cl \
  --manifest tests/fixtures/pop909_cl/production_manifest.json \
  --output /tmp/music-critic-v2-phase4b-production-acceptance.json
```

POP909-CL production evidence and the completed Phase 4B contract are documented
in `docs/POP909_CL_FIELD_AUDIT.md` and
`docs/POP909_CL_ADAPTER_CONTRACT.md`. Original POP909 is retained separately
as lineage/possible ablation evidence in
`docs/POP909_ORIGINAL_FIELD_AUDIT.md`.
The Phase 5A/5B.1 sidecar and collator contracts are documented in
`docs/MULTISOURCE_TARGET_CONTRACT.md` and `docs/MULTISOURCE_COLLATOR.md`.
The Phase 5B.2 portable corpus index/cache, external split manifest, lazy
Dataset, deterministic mixture sampler, and worker-safe DataLoader contract is
documented in `docs/MULTISOURCE_DATASET.md`. Full corpus cache builds are
explicit opt-in commands; default tests use bounded/synthetic artifacts only.
The learned local baseline and its strict non-critic/non-SSL boundary are in
`docs/PHASE6A_BASELINE.md`.
The deterministic hierarchy, bar+track Transformer, contextual song row,
top-down fusion, and controlled three-way ablation are in
`docs/PHASE6B_HIERARCHY.md`.
Phase 6C cache/split preparation, one-batch acceptance, ordinary training,
device transfer, artifacts, and epoch-boundary resume are in
`docs/TRAINING.md`.
