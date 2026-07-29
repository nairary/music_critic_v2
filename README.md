# Music Critic V2

Music Critic V2 is a raw-symbolic-first research project for reusable symbolic
music representation learning, theory analysis, and preference-aware quality
assessment.

This repository is a clean-room successor to the legacy Music Critic V1
repository. V1 may be inspected as read-only reference material, but this
package has no runtime dependency on it and must remain runnable when the legacy
checkout is absent.

## Current state

Phases 0 through 5B.2, the Phase 6A/6B representation baselines, and the
Phase 6C reproducible supervised training harness are implemented and merged.
Phase 6D-A adds deterministic supervised checkpoint evaluation, train-only
trivial baselines, and bounded performance evidence for those unchanged
baselines. Phase 7A is implemented on draft PR #15 as a deterministic
GraphMAE2-inspired masked-graph SSL baseline over the unchanged raw-only graph.
It adds note pitch-group masking, owner-track-statistic and peer-note leakage
closure, shared stop-gradient full-view representation targets, contextual
decoder re-masking, bar/song latent prediction, target-free raw-cache loading,
and strict SSL checkpoint/resume and encoder-transfer contracts.
Decoder context mode
`online_owner_track_bar_song_temporal_neighbors` retains masked-online
owner/bar/song/temporal context after latent re-masking, avoiding predictions
that depend only on one shared mask token when a view is fully re-masked.
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
controlled three-way ablation. Hierarchical/adaptive SSL, corruption training,
preference training, PLL, and deployable scoring inference are not implemented
yet.
Phase 7A reconstruction is SSL representation plumbing only: it is not a
masked-note likelihood, PLL, critic, quality score, or full-scale effectiveness
claim. Hierarchical masking remains Phase 8, and PDMX-scale SSL evaluation
remains Phase 10.

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
- `src/music_critic/evaluation/`: candidate-first checkpoint evaluation,
  streaming metrics, train-only priors, and the opt-in bounded profiler;
- `src/music_critic/ssl/`: versioned field masks and overlays, deterministic
  MaskPlans and contextual decoder views, representation objectives, bounded
  and production-cache raw-only loading, masked hierarchical model,
  checkpoints, transfer, and training CLI;
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
PYTHONPATH=src python -m music_critic.ssl.run \
  experiment=one_batch model=hierarchical data=bounded device=cpu
```

For production-cache SSL, the target-free dataset loads each canonical cache
piece and rebuilds its raw graph without projecting supervised targets. SSL
reports separately identify the data source, whether a production cache was
read, whether the run was one-batch plumbing, and whether production or
full-corpus SSL training occurred.

`build_raw_graph` validates its `CanonicalPiece` by default. Callers that have
already run canonical validation may opt into the documented
`assume_valid=True` fast path. Structural timing remains exact rational data
through graph indexing and becomes `float32` only when feature tensors are
materialized. PyTorch/PyG imports are isolated to the graph/model-facing
packages (`music_critic.graph`, `music_critic.tasks`, `music_critic.models`,
`music_critic.training`, `music_critic.evaluation`, and `music_critic.ssl`);
the project already declares those packages as global installation
dependencies.

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
Phase 6D-A evaluation metrics, baseline provenance, fingerprint checks,
artifacts, test-split acknowledgement, and profiling are in
`docs/EVALUATION.md`.
The deterministic GraphMAE2-inspired Phase 7A mask, model, objective,
checkpoint, transfer, and bounded-science contracts are in
`docs/PHASE7A_SSL_BASELINE.md`.

Evaluate a checkpoint on fixed validation:

```bash
PYTHONPATH=src python -m music_critic.evaluation.run \
  checkpoint=/absolute/path/to/best.pt \
  data=mixed \
  data.index_paths='[/absolute/path/hooktheory.index.json,/absolute/path/pop909_cl.index.json]' \
  data.cache_roots='[/absolute/path/hooktheory,/absolute/path/pop909_cl]' \
  data.split_manifest=/absolute/path/global.split.json \
  split=validation \
  output_dir=/absolute/path/to/evaluation
```

Test evaluation additionally requires
`acknowledge_test_evaluation=true`. The detailed synthetic profiler is
disabled unless `enabled=true`; see `docs/EVALUATION.md` for its bounded
matrix command. Production cache paths are read-only inputs: evaluation never
rebuilds or mutates canonical artifacts.
