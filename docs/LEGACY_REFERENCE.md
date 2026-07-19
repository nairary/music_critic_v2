# Legacy Music Critic V1 Reference

This document is an audit map, not a runtime dependency list.

## Legacy identity

- Path: `/home/str/Fine-tune-text2midi-llm-with-gnn-theory-critic`
- Commit: `2d8281f31cc9ad9c8fecaf332da0c61e0e949415`
- Branch: `sections`
- Remote: `git@github.com:nairary/Fine-tune-text2midi-llm-with-gnn-theory-critic.git`
- State at capture: dirty, with pre-existing deleted, modified, and untracked
  files recorded exactly in `legacy_snapshot.json`.
- Python: 3.13.5

No V1 file is copied as production code. Exact paths below are relative to the
legacy root.

## Current V1 data flow

```text
HookTheory raw JSON + structure JSONL
→ src/data/preprocess_hooktheory.py
→ src/data/canonicalize_hooktheory.py
→ src/data/encode_teacher_features.py
→ src/dataloader/hooktheory_dataset.py
→ src/dataloader/utils_graph.py
→ src/models/teacher_gnn.py
→ src/training/train_teacher.py
```

The teacher inference entry point consumes encoded song JSON rather than raw
MIDI. Raw-MIDI scoring is provided through the separate observer pipeline.

## Actual V1 graph

Mandatory nodes:

```text
song, section, bar, onset, note, chord
```

Mandatory edges:

```text
song-contains_section-section
section-belongs_to_song-song
song-contains_bar-bar
section-next_section-section
section-contains_bar-bar
bar-in_section-section
bar-next_bar-bar
bar-contains_onset-onset
section-contains_onset-onset
onset-in_section-section
onset-next_onset-onset
onset-starts_note-note
onset-starts_chord-chord
section-contains_note-note
note-in_section-section
section-contains_chord-chord
chord-in_section-section
note-next_note-note
chord-next_chord-chord
chord-covers_note-note
```

The graph uses mixed float tensors. Note inputs include `sd_id`; chord inputs
include root, type, inversion, applied, borrowed, and chord-component fields;
song inputs include encoded key and meter IDs; section inputs include semantic
section labels.

## Component classification

| Legacy component | Classification | Files inspected | Useful ideas | Must not be copied | V2 direction and risks |
|---|---|---|---|---|---|
| HookTheory raw preprocessing | Adapt selected algorithms | `src/data/preprocess_hooktheory.py` | Source discovery, section attachment, diagnostic outputs | HookTheory-shaped top-level records and permissive float coercion | Implement a dataset adapter after the canonical contract exists; retain provenance and leakage-safe grouping. |
| HookTheory canonicalization | Reuse conceptually | `src/data/canonicalize_hooktheory.py` | Explicit normalization, reporter counts, raw-value preservation | The claim that this dataset-specific structure is a universal canonical schema | Map into V2 dataclasses with exact timing and source-aware annotations. |
| Teacher feature encoding | Reference only | `src/data/encode_teacher_features.py`, `metadata/specs/`, `metadata/vocabs/` | Vocabulary/unknown-ID mechanics and multihot encoding | Encoded theory fields as raw inputs and fixed HookTheory vocabularies | Future feature/target registries must separate observable fields from targets. |
| V1 graph layouts | Discard for V2 | `src/dataloader/graph_layouts.py` | A concrete inventory of leakage to guard against | Fixed positional mixed-float layouts, `sd_id`, chord theory IDs, key IDs, section labels | Use named categorical/continuous tensors generated from a registry. |
| V1 graph builder | Reference only | `src/dataloader/utils_graph.py` | Deterministic ordering, explicit empty stores, containment and temporal edges | Mandatory gold `section` and `chord` nodes, float timing, missing track/beat nodes | Build `song/track/bar/beat/onset/note` from raw evidence; semantic structure becomes optional supervision. |
| Masking logic | Adapt selected algorithms | `src/dataloader/utils_graph.py` | Preserve decoder targets and deterministic seedable selection | Masking theory labels that should not be raw inputs | Mask observable fields and later hierarchical units with explicit target preservation. |
| Theory-aware corruptions | Defer as ablation | `src/dataloader/song_corruptions.py`, `function_rules.py`, `theory_helpers.py` | Some musically meaningful robustness probes and metadata conventions | Corruption-heavy quality supervision as the primary critic objective | Reintroduce selected transformations only as evaluation or controlled ablations. |
| TeacherGNN | Reference only | `src/models/teacher_gnn.py` | Type-specific encoding, hetero message passing, local contexts, pooling hooks | Architecture tied to V1 node schema and annotation leakage | Implement a raw graph encoder plus hierarchy and long-context model only after data/graph phases. |
| Reconstruction heads | Adapt selected algorithms | `src/models/teacher_heads.py` | Separate heads and explicit valid-ID losses | Reconstruction of annotation IDs as if they were raw fields | SSL decoders reconstruct observable features; theory prediction uses masked supervised heads. |
| Local score heads | Defer as ablation | `src/models/teacher_heads.py`, `src/models/teacher_gnn.py` | Local diagnostic scoring and contextual aggregation | Treating corruption detection as synonymous with musical quality | Compare later against interpretable aspect heads trained from real preference evidence. |
| Graph score and ranking loss | Reuse conceptually | `src/training/teacher_losses.py` | Pairwise margins, intra/inter-batch comparisons, finite empty-task losses | Clean-versus-hand-corrupted ranking as the sole preference definition | Use group-aware pairwise preferences and calibrated aspect scores. |
| Observer pipeline | Discard for V2 | `src/observer/data_pipeline.py`, `dataset.py`, `cached_dataset.py`, `model.py`, `train_observer_distill.py` | Operational lessons for MIDI parsing, caching, and batch inference | Teacher-to-observer scalar distillation and teacher vocabulary coupling | V2 encoder itself accepts raw MIDI-derived graphs, making the observer workaround unnecessary. |
| Chord scorer | Reference only | `src/observer/chord_parser.py`, `chord_score_fitting.py` | Sonority extraction, candidate explanations, confidence/provenance concepts | Predicted chords as mandatory raw graph evidence | Candidate predictions may become optional features or evaluation baselines, never required inputs. |
| Training and checkpointing | Adapt selected algorithms | `src/training/train_teacher.py`, `dynamic_loss_weighting.py` | Seeding, staged execution, metrics JSONL, checkpoint metadata, batch limits | V1 data/model assumptions and Hydra-coupled global configuration | Rebuild generic phase-owned training infrastructure with schema and registry versions in checkpoints. |
| Hydra configuration | Reuse conceptually | `configs/` | Composable experiment groups and reproducible resolved configs | Copying the large V1 configuration surface before V2 interfaces exist | Introduce configuration incrementally when a phase owns runnable behavior. |
| Teacher inference CLI | Discard for V2 | `src/inference/infer_teacher_score.py` | Input validation and checkpoint-driven construction patterns | Requirement for encoded theory-rich song JSON | Public V2 inference starts from unlabeled MIDI or canonical raw symbolic data. |
| Observer inference CLI | Adapt selected algorithms | `src/inference/infer_observer_scores.py` | Batch ranking, normalized output rows, explicit errors | Required tonic/mode metadata and observer checkpoint semantics | Later raw-MIDI API should preserve robust batch behavior without requiring theory labels. |
| Evaluation utilities | Reuse conceptually | `src/evaluation/teacher_local_metrics.py`, evaluation-related tests and scripts | Structured JSON reports and example capture | Metrics centered only on synthetic corruptions | Expand to SSL, theory, preference, calibration, OOD, and ablation evaluation. |
| Tests and fixtures | Adapt selected algorithms | `tests/` | Deterministic tiny graphs, malformed-input checks, checkpoint and resume tests | Reliance on local datasets or production output directories | Keep synthetic fixtures and phase-specific contract/integration tests in the clean repository. |

## Major incompatibilities

### Theory-label leakage

V1 note, chord, song, and section inputs directly contain theoretical labels.
This is acceptable only as historical V1 behavior. V2 raw encoders cannot
consume labels unavailable from ordinary MIDI.

### Timing

V1 represents beat positions and durations as floats and groups onsets with
epsilon tolerances. V2 canonicalization will preserve exact rational
quarter-note timing and convert to floats only at tensor construction.

### Gold semantic structure

V1 graph construction makes `section` and `chord` nodes mandatory. Those
boundaries and labels are not available in unlabeled MIDI. V2 uses raw
candidate slots/direct heads and keeps semantic nodes optional.

### HookTheory assumptions

The teacher pipeline assumes lead-sheet melody plus annotated chord symbols,
encoded relative scale degrees, and a small fixed vocabulary. HookTheory is
useful theory supervision but is not a multitrack deployment distribution.

### Observer deprecation direction

The observer exists because the teacher cannot consume ordinary MIDI. V2 removes
that architectural split by making raw MIDI-derived graphs the shared encoder
input. Observer code remains reference material only.

## Genuinely reusable infrastructure concepts

- deterministic transformations and fixed seeds;
- structured validation and diagnostic reports;
- explicit provenance;
- staged training and batch limits;
- JSONL metrics and checkpoint metadata;
- robust CLI errors;
- tiny deterministic test fixtures;
- pair/group-aware ranking evaluation.

## Phase 2B.0 remediation classification

Retained from V1 only as documented or synthetic compatibility behavior:

- the historical major-fixed chromatic table and MIDI-72 absolute-octave
  reconstruction, which are no longer production semantics;
- support for diagnosing legacy root `8` as synthetic bVII compatibility input.

Rejected as source or upstream facts:

- treating MIDI 72 as an observed corpus pitch or Sheet Sage invariant;
- treating root `8` as observed (the corpus-wide count is zero) or accepted by
  upstream TheoryTab (sounding upstream roots are `1..7`);
- treating encoded IDs, V1 meter tokens such as `12/3`, or first-region summary
  fields as V2 canonical source semantics;
- requiring gold structure, chord annotations, or theory labels at inference.

Upstream Sheet Sage at commit
`bbdd7b7b6a5fb845828f82790acdceb03a197779` supplies separate evidence for
1-based beat conversion, beat-unit grouping, raw TheoryTab validation, and
partially available applied-chord behavior. Applied harmony remains
intentionally deferred from the V2 MVP.

## Phase 2B.1 production adaptation

The remediated production adapter rejects the V1 major-fixed pitch table and
MIDI-72 anchor in favor of pinned Sheet Sage scale steps, true accidental
offsets, and MIDI 60 for relative octave zero. It also rejects V1's uniform
float beat arithmetic: canonical qn time is integrated piecewise with one qn
per simple raw beat and one-half qn per compound raw beat. V1 remains useful
only for source discovery, grouping intent, and explicit synthetic root-8
compatibility tests.

Rejected at runtime: legacy imports, HTCanon input, encoded theory IDs, legacy
meter tokens, chord-note rendering, applied-harmony reinterpretation, and any
requirement for gold structure or theory targets at inference.
