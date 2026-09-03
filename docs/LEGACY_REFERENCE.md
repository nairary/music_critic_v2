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

## Phase 2B.2 renderer adaptation

No legacy renderer module or runtime logic was copied or imported. The generic
exporter is derived solely from the V2 canonical contract and low-level `mido`
events. V1 chord accompaniment concepts remain rejected: canonical chord and
key targets are DAW markers only, and applied, alternate, pedal, and voicing
semantics do not generate notes. Exact rational PPQ selection, explicit
quantization reporting, canonical-beat clicks, and canonical MIDI round-trip
tests are new V2 infrastructure.

## Phase 3A graph adaptation

The legacy graph builder was not re-opened, imported, or copied for Phase 3A;
the existing audit above supplied the bounded reference classification. V2
retains only the general ideas of deterministic node ordering, explicit empty
stores, chronological edges, and explicit reverse relations. It rejects V1
mixed float layouts, theory-bearing note/song/chord/section features, mandatory
gold chord and section nodes, epsilon onset grouping, and simultaneous-note
pairwise structure. The new graph is derived only from the accepted canonical
contract, adds track and denominator-unit beat levels, uses exact onsets, and
represents sustained activity through sparse note-to-beat incidence.

## Phase 4A POP909 evidence adaptation

No legacy repository file was opened, copied, imported, or modified for Phase
4A. No legacy runtime logic was reused. The installed processed POP909 mirror
was measured only as local corpus evidence and was rejected as a specification
source: it lacks version, license, documentation, and annotation assets, and
its `piano` plus `chords`/`MIDI 01` tracks are not equivalent to the official
`MELODY`/`BRIDGE`/`PIANO` contract. Legacy five-class chord compression,
track-order role guessing, exact float/beat snapping, and treating missing
labels as negatives remain rejected. The Phase 4B contract instead derives
from the pinned official POP909 repository/paper, exact V2 timing, explicit
provenance, masked auxiliary targets, and song-level version grouping.

### Phase 4A POP909-CL remediation

The preceding Phase 4A paragraph records the initial, now superseded corpus
classification. No legacy repository file was opened or changed during the
remediation, and no legacy logic was reused. A complete byte comparison against
the pinned POP909-CL repository established the local files as the production
`POP909_processed` corpus. The remediation retains rejection of legacy
five-class chord compression, track-order guessing, float timing equality,
target leakage, and missing-as-negative labels. It adds a stricter boundary:
the embedded channel-1 chord instrument is target-only and is removed before
canonical raw conversion, while channel-0 score content remains the inference
input. Original POP909 is lineage/ablation evidence only.

## Phase 4B production adaptation

No legacy repository file was opened, copied, imported, or modified for Phase
4B, and no legacy runtime logic was reused. The production adapter was derived
from the pinned POP909-CL contract, the independent Phase 4A audit, the V2
generic MIDI adapter, canonical schema, and raw graph leakage boundary.

Retained only as already documented general V2 engineering concepts:
deterministic conversion, exact timing, explicit provenance, structured
failures, and masked missing targets. Rejected throughout the implementation:
legacy five-class chord compression, track-order/name/pitch-range routing,
float snapping, mandatory semantic graph structure, chord-note rendering,
target-derived raw notes, and missing-as-negative behavior.

### Phase 6C POP909-CL identity remediation

No legacy repository file was opened, copied, imported, or modified for this
remediation, and no legacy runtime logic was reused. The source-record,
score-projection equivalence, lineage, graph-fingerprint, and target-bundle
identity policy derives only from the current V2 contracts and programmatic
full-corpus evidence for POP909-CL records 543 and 553.

## Phase 5A multi-source target contract

No legacy repository file was opened, copied, imported, or modified for Phase
5A, and no legacy runtime logic was reused. The existing audit classification
was sufficient. Phase 5A retains only previously accepted general concepts:
stable vocabularies, deterministic ordering, explicit empty stores/sidecars,
group-aware splitting, and masked task routing.

Rejected for the production contract remain V1 encoded theory IDs as graph
features, shared mixed-float target layouts, mandatory chord/section nodes,
missing-as-negative behavior, HookTheory-specific vocabularies presented as a
universal ontology, and float/epsilon target alignment. The new registry and
sample/batch shapes derive solely from current V2 adapters, canonical data,
raw graph contracts, and versioned bounded evidence.

## Phase 6A trainable local baseline

No legacy runtime file was opened, imported, copied, or modified for Phase 6A.
The existing audit classification was sufficient. V2 retains only the general
ideas of per-node-type feature encoders, explicit relation-aware message
passing, residual local layers, and local prediction/reconstruction hooks.

Phase 6A rejects V1 theory-bearing encoder inputs, mandatory semantic
chord/section topology, HookTheory-specific graph assumptions, mixed
unversioned feature layouts, global-only pooling, and graph-score or
quality-score interpretation. Heads gather current V2 target sidecars through
explicit local indices, and missing, PU, or open-vocabulary observations do not
become ordinary negative supervision. Hierarchy, critic, SSL, and likelihood
work remain owned by their later V2 phases.

## Phase 7A deterministic SSL adaptation

Exactly two legacy files were inspected read-only for Phase 7A:

- `src/dataloader/utils_graph.py`;
- `src/models/teacher_heads.py`.

No legacy file was modified, imported at runtime, copied wholesale, formatted,
staged, or used as the V2 specification. V2 remains runnable without the
legacy checkout.

Only broad ideas were adapted:

- select explicit rows and preserve separate reconstruction evidence, while
  replacing legacy process-global randomness with deterministic per-sample
  SHA-256 selection;
- keep reconstruction decoding separate from the encoder and from scoring;
- retain explicit valid-row/count handling instead of treating an absent row
  as a negative observation.

The following legacy assumptions were explicitly rejected:

- Python `random.sample`, `random.random`, and other process-global random
  choices for masks;
- `copy.deepcopy(graph)` masked-graph construction and mutation of graph
  feature tensors;
- writing numeric zero as an ambiguous mask sentinel;
- assuming that masking only selected-note fields closes leakage through
  unselected peer-relative pitch or owner-track pitch statistics;
- masking or reconstructing theory labels such as note scale degree and chord
  root/type/applied/borrowed IDs;
- mandatory gold chord/section nodes or theory-derived topology;
- treating teacher local/global score heads or corruption discrimination as
  an SSL, critic, aesthetic, or quality objective.

Phase 7A instead masks only raw-observable redundant pitch representations
through a versioned model-side overlay, hides availability evidence, closes
unselected-peer relative-pitch and owner-track aggregate leakage, reconstructs
detached full-view representations with masked-online owner/bar/song/temporal
decoder context, and keeps raw graph fingerprints unchanged. Its production
cache dataset rebuilds raw graphs without projecting supervised targets. No
legacy checkpoint, vocabulary, model class, or runtime module participates.

## Phase 8A hierarchy-aware masking

No legacy repository file was inspected for Phase 8A. No legacy code, planner,
randomness, hierarchy representation, objective, checkpoint, or configuration
was reused or adapted.

Phase 8A derives exclusively from current V2 raw relations, the accepted
Phase 7A `MaskPlan`/overlay/prepared-attestation contracts, and the Phase 6
hierarchical encoder. In particular, start-anchored onset/beat/bar/track
descendants, deterministic policy mixtures, structured unavailable evidence,
and the supplemental bounded oracle are new V2 contracts. The V2 package
continues to run without the legacy checkout.

## Phase 8B.2A scientific comparison protocol

No legacy repository file was inspected for Phase 8B.2A. No legacy trainer,
evaluation path, checkpoint, data schedule, statistical routine, test access,
or transfer logic was reused or adapted.

The end-to-end runner remediation also inspected and adapted no legacy file.
Subprocess DAG orchestration, metadata attestation, actual-schedule checking,
step-budget execution, fixed validation, per-piece sufficient statistics, and
paired-seed configuration selection are current V2 implementations.

The blocking production-path semantic-projection remediation likewise
inspected, reused, and adapted no legacy file. Source-neutral index/cache/split/
composition/membership projection and explicit test-access terminology are V2-
only contracts.

The comparison protocol composes only current V2 Phase 8B.1 SSL, Phase 6C
supervised training, Phase 6D candidate-first evaluation, and the accepted V2
encoder export. Natural/matched analyses, named paired seed domains,
validation-only ranking, the single-use test lock, piece bootstrap, and
immutable aggregate artifacts are new V2 contracts. V2 and its plan CLI
remain runnable without the legacy checkout.

## Phase 9A Dilemmadata evidence audit

No legacy repository file was opened, imported, copied, modified, formatted,
staged, or used for Phase 9A. No legacy runtime logic was reused. Evidence came
from the exact official Dilemmadata v1.0 snapshot, its processing source and
metadata, a clean upstream checkout, current V2 canonical/raw-graph contracts,
and new bounded synthetic fixtures.

Only already accepted general V2 principles carry forward: exact rational
timing, deterministic conversion, explicit provenance, masked missing targets,
target-blind raw graphs, transitive group-aware splitting, and structured
failures. Phase 9A explicitly rejects legacy theory-bearing graph inputs,
target-derived notes or topology, float snapping, missing-as-negative labels,
mandatory semantic voice/section/chord structure, unversioned shared label
IDs, and annotation-level record splits that separate equivalent inputs.

## Phase 9B.1 Dilemmadata production raw adapter

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or used for Phase 9B.1. No legacy runtime logic was reused
or adapted. The implementation derives only from the current V2 canonical,
graph, cache, loading, and SSL contracts; the accepted Phase 9A evidence; and
the pinned Dilemmadata v1.0 snapshot and its included processing documentation.

The same legacy assumptions remain rejected: theory-bearing graph inputs,
target-derived notes/topology, float timing or snapping, mandatory semantic
tracks/voices/chords/sections, missing-as-negative labels, and annotation-level
splits. The source-neutral track, exact tie/grace/meter/bar policies,
target-independent cache projection, structured quarantine, and Phase 8B raw
loader integration are new V2-only contracts.

The Phase 9B.1 blocking remediation likewise opened, searched, imported,
copied, modified, formatted, staged, and reused no legacy repository file or
logic. Strict policy validation, discovery-record binding, distinct
key-signature diagnostics, independent source cache reruns, and the committed
production manifest derive only from current V2 contracts and pinned
Dilemmadata evidence.

## Phase 9B.2C executable supervised smoke

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or used for Phase 9B.2C. No legacy runtime or training logic
was reused or adapted. The runner and verifier compose only the accepted V2
production cache, model, loss, checkpoint, and evaluation contracts. They
continue to reject theory-bearing model inputs, target-dependent candidate
generation, source conversion inside training, float timing, missing-as-
negative labels, and test access before an explicit later authorization.

## Phase 9C-A one-seed production pilot

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or used for Phase 9C-A. No legacy sampling, SSL, transfer,
training, evaluation, checkpoint, resume, plotting, or selection logic was
reused. The pilot composes only accepted current-V2 Phase 8B.2 and Phase 9B.2
boundaries. Legacy theory-bearing inputs, float timing, gold semantic topology,
missing-as-negative labels, corruption score interpretation, and observer
distillation remain rejected.

The pre-RTX blocking correction also opened and reused no legacy file or logic.
Exact-assignment split composition and fixed-update `last.pt` comparison derive
only from current V2 split, training-report, checkpoint, and evaluation
evidence.

The RTX-profile Hydra runtime remediation likewise opened, imported, and reused
no legacy file or logic. It changes only current-V2 Hydra override syntax and
fail-closed profile process/report handling.

## Phase 9C-B onset-BiGRU diagnostic

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused for Phase 9C-B. The decoder, raw-row sequence
isolation, ownership pooling, residual fusion, transfer fingerprints, matrix
runner, metrics, and verifier derive only from current V2 graph, hierarchy,
Dilemmadata supervision, training, evaluation, and Phase 9C-A contracts.

Legacy theory-bearing inputs, float/epsilon onset grouping, mandatory semantic
chord/section nodes, target-derived sequence order, missing-as-negative labels,
observer distillation, and corruption-score interpretation remain rejected.

The blocking Phase 9C-B profile remediation opened, imported, copied,
modified, formatted, staged, and reused no legacy file or logic. The shared
dataset-view and deterministic downstream schedule builder derives only from
current V2 target-cache, split, sampler, training-runtime, and Phase 8B.2
fingerprint contracts.

The second profile remediation likewise opened, imported, copied, modified,
formatted, staged, and reused no legacy file or logic. Typed checkpoint-model
reconstruction and explicit encoder-export validation derive only from current
V2 model, training-checkpoint, SSL-export, and evaluation contracts. Legacy
checkpoint guessing, partial state loading, and implicit artifact substitution
remain rejected.

## Phase 9C-C MLP convergence diagnostic

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused for Phase 9C-C. The applied-update telemetry,
mid-epoch checkpoint/resume boundary, fixed milestone evaluation, convergence
report and verifier compose only current V2 Phase 9C-B schedule, training,
checkpoint, transfer and evaluation contracts.

Legacy epoch reinterpretation, checkpoint-shape guessing, partial state loads,
target-derived scheduling, missing-as-negative labels, automatic plateau
claims and test access remain rejected.

## Phase 9E-A common harmonic projection

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused for Phase 9E-A. No legacy label mapping, chord
vocabulary, pitch conversion, model, checkpoint, graph, or audit logic was
adapted. The implementation derives only from current V2 raw/target contracts,
the accepted pinned Dilemmadata evidence, and the independently pinned official
AnalysisGNN repository used as an external MIT-licensed mapping reference.

Legacy theory-bearing inputs, missing-as-negative labels, Python-hash class
IDs, runtime vocabulary growth, implicit enharmonic collapse, inferred bass,
mandatory semantic segmentation, majority analysis selection, and
unversioned/shared label tables remain rejected. The frozen mapping state
machine, immutable derived target sidecar, explicit losses/divergences,
per-field local-key masks, proven-template pitch-class-set dependency, and
source-free audit manifest are new V2-only contracts. V2 runs and checks the
committed evidence without either the legacy or AnalysisGNN checkout.

The 2026-08-28 Phase 9E-A remediation likewise opened, imported, copied,
modified, formatted, staged, and reused no legacy file or logic. It corrected
only the current V2 frozen external reference schema after directly verifying
the pinned official AnalysisGNN `process_inversion_from_chord` evidence:
DLC `2` and `42` map to third inversion. AN ordinal `2` remains a distinct
source-task value mapped to second inversion. A shared token-only inversion
reference is rejected; no upstream module is vendored or imported at runtime.

## Phase 9E-B1 AnalysisGNN reconstruction

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused for Phase 9E-B1. No legacy model, graph,
preprocessing, split, training, evaluation, or checkpoint logic was adapted.
The isolated comparator derives from the user-supplied scientific
specification, current V2 exact raw/target contracts, unchanged Phase 9E-A
projection, pinned Dilemmadata evidence, and the official public AnalysisGNN
and GraphMuse histories.

Legacy theory-bearing inputs, float timing equality, target-derived topology,
missing-as-negative labels, semantic-segmentation inference requirements,
runtime legacy imports, checkpoint-shape guessing, and transposition leakage
remain rejected. No adapted legacy concept was accepted.

The 2026-08-29 overlapping-supervision remediation likewise opened, searched,
imported, copied, modified, formatted, staged, and reused no legacy repository
file or logic. Sparse note-to-source-entry memberships, equivalent-class
deduplication, conflict diagnostics, the all-719 structural preflight, and the
deterministic CUDA environment gate derive only from the current V2
AnalysisGNN experiment adapter and immutable Phase 9E-A sidecars. The rejected
assumptions are one-entry-per-note supervision, arbitrary point/interval
priority, deleting zero-duration entries, unavailable-row overwrite, and
silently resolving different available common classes.

The subsequent conflicting-label forensic audit also did not open, search,
import, copy, modify, format, stage, or reuse the legacy repository or its
logic. Exact external evidence came only from read-only Dilemmadata commit
`d60ee75b4a9495e932a4a7be39381578be17e222`,
`processing/utils.py::make_labeled_pitch_array`, and read-only official
AnalysisGNN commit `e115182fb29b74bdcb6bf3547ed427d967580947`:
`analysisgnn/utils/dcl_tsv_utils.py::{load_labeled_pitch_array,
create_graph_from_df,create_labels_dlc,process_inversion_from_chord}`,
`analysisgnn/data/datasets/dlc.py::DLCGraphDataset._process_single`, and
`analysisgnn/models/analysis.py::onsetwise_logit_aggregation`. These sources
were inspected as provenance only and remain absent from the V2 runtime.

The adapted concept is the official row-aligned DLC target interpretation:
one retained TSV row becomes one graph note with one quality/inversion label.
The audit rejects inferring point/interval, first/last, or grace precedence,
because the pinned code defines none. It also rejects training one V2 note row
against two different classes. No policy was implemented; exact V2 source-row
membership is only the evidence-backed recommendation for a later remediation,
with task-note masking retained as the conservative fallback if source-row
provenance cannot be contract-bound.

The exact source-row remediation likewise did not open, search, import, copy,
modify, format, stage, or reuse the legacy repository or its logic. It turns
the already recorded pinned Dilemmadata/AnalysisGNN row-alignment evidence into
an experiment-local checked contract. The implementation uses current V2
canonical note IDs, exact rational timing, source-native target entities, and
the unchanged common projection. Arbitrary precedence, legacy label logic,
runtime external imports, and fallback masking remain rejected; no legacy or
pinned external checkout is required for ordinary V2 inference.

## Phase 9E-B2 Dilemmadata raw coverage remediation

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused. No legacy preprocessing, tie, meter, graph,
target, split, model, training, or evaluation logic was adapted. The repair
policies derive only from the pinned Dilemmadata raw evidence, current V2 exact
timing/canonical/graph contracts, and the preceding read-only coverage audit.

The accepted V2 concepts are exact rational timing, target-blind raw
construction, stable source identity, local masks for missing or ambiguous
labels, and deterministic provenance. Float/epsilon equality, random tie
selection, target-derived topology, fabricated note duration, whole-record
exclusion for a local repair, mandatory gold segmentation, runtime legacy
imports, and using the AnalysisGNN 14-record selection as a raw-adapter filter
remain rejected. AnalysisGNN code, model, checkpoint, and labels were not used
to make any repair decision, and this phase makes no exact-reproduction claim.

## Phase 9E-B3 expanded multi-task contract

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused for B3. No legacy dataset selection, task mapping,
vocabulary, entity identity, split, metric, model, or training logic was
adapted. The legacy checkout is not required at runtime or for the source-free
audit check.

The only external scientific evidence inspected read-only was the official
AnalysisGNN paper and pinned commit
`e115182fb29b74bdcb6bf3547ed427d967580947`. It was used to document the
20-property paper inventory, 21 unique code heads, joint Roman-numeral
components, literal defects, aliases, and missing-as-class behavior. No
external module is imported in production.

Accepted V2 concepts are exact rational timing, B2 source components and raw
lineage, target-independent shared entities, independent masks, stable SHA-256
assignment, and explicit TEST sealing. Rejected assumptions include filename-
only grouping, task-specific chord IDs, missing-as-class-0, silent `+7`/`+M7`
collapse, malformed Roman literals, label-valued split objectives, target-
derived raw graphs, legacy runtime imports, and claims of exact reproduction.

The B3 scientific-contract remediation likewise opens, searches, imports,
copies, modifies, and reuses no legacy file or logic. Its read-only evidence is
limited to the already pinned official AnalysisGNN commit and the published
paper: quality-17 remains the corrected source-faithful V2 space, while a
separate serialized quality-15 projection reproduces the pinned `+7`/`+M7`
collapse only for comparison. The corrected harmonic-event metric is rejected
as a paper-compatible claim. A distinct unevaluated note-level contract follows
the paper's five semantic components while recording that pinned validation/NCT
and onset-test evaluator branches disagree. No external or legacy code is a
runtime dependency.

## Phase 9E-B4 class-balance audit

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused for B4. No legacy class-frequency, balancing,
sampling, metric, dataset, split, model, or training logic was adapted. The
audit derives only from the current V2 B3 registry, vocabularies, masks,
canonical source-row provenance, frozen components/split, and target-sidecar
contract. V2 and source-free B4 verification remain independent of the legacy
checkout.

Accepted concepts are V2-only: missing/masked targets stay outside classes;
entity repetition is separated from canonical annotation support; record and
component independence are explicit; TRAIN determines candidate weights; and
VALIDATION measures coverage without opening TEST. Rejected assumptions are
raw entity count as independent sample size, filename grouping, missing as a
negative class, validation-informed weighting, TEST distribution inspection,
silent rare-class merging, and automatic sampler/loss changes based on one
audit.

## Phase 9E-B5A transposition audit

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused for B5A. The legacy project supplies no V2
transposition contract. The official profile was reconstructed independently
from the separately pinned public AnalysisGNN commit `e115182...`; its source
files and hashes are evidence only and are not runtime dependencies.

Accepted V2 concepts are global pitch/key-consistent transposition, explicit
equivariant versus invariant targets, immutable source/component identity, and
TRAIN-only augmentation. Rejected assumptions are modulo octave wrapping,
per-note octave repair, arithmetic class-ID shifting, random enharmonic choice,
OOV fallback in corrected V2, view-level split assignment, counting variants
as independent components, and inferring quality/Roman/phrase support from
pitch-only augmentation.

## Phase 9E-B5B training-policy freeze

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused for B5B. Legacy loss weighting, dataset sampling,
task routing, model inputs, trainer, checkpoint, and metric implementations do
not define any B5B contract and remain unnecessary for source-free checks.

Official profile evidence was inspected only in the separate read-only public
AnalysisGNN commit `e115182...` and the existing B1 attestation. Preserved
official concepts include its 21 unique code heads, smoothed cross entropy,
learned uncertainty weighting, combined loader, materialized transpositions,
validation loss selection, and divergent joint evaluator branches. They are
isolated in `O` and are not imported at runtime or applied to corrected V2.

Corrected B5B concepts derive from the current V2 B3/B4/B5A evidence: masks,
canonical source-row counts, frozen components, safe transposition closure,
fixed head groups, bounded TRAIN-only weights, and explicit TEST locking.
Rejected assumptions include missing-as-zero, treating all 20 heads as equal,
training phrase/section from positive-only rows, weighting by broadcast note
volume, counting augmented views as records, validation/TEST-informed weights,
filename sampling, silent official-corpus substitution, and any performance
claim without a paired `C0`/`C1` run.

## Phase 9E-B5C corrected model and trainer

No file in the read-only legacy repository was opened, searched, imported,
copied, formatted, staged, or reused for B5C. The model is composed solely from
current V2 production encoder, hierarchy, onset-BiGRU, graph, collator, B3
sidecar, B5A transposition, and B5B policy APIs. The separately pinned public
AnalysisGNN evidence remains documentation only and is not a runtime
dependency.

Accepted V2 concepts are raw-only prediction, exact rational/entity routing,
independent task heads and masks, component-aware sampling, complete
checkpoint state, and explicit corrected-versus-paper metric identities.
Rejected legacy/official assumptions include logit fusion, target-derived
graph structure, missing-as-class behavior, learned uncertainty weights,
materialized augmented records, view-level split assignment, staff prediction,
cadence-corpus substitution, TEST evaluation, and claims of exact AnalysisGNN
reproduction.

## Phase 9E-B5D full-training screen

No legacy file was opened, searched, imported, copied, or changed for B5D.
The phase only schedules the already accepted V2 B5C runtime for a larger
paired budget. It reuses current V2 deterministic checkpoint/resume,
component sampling, safe transposition, identity validation, and TEST-lock
concepts. It rejects epoch-count reinterpretation, early stopping, profile O
substitution, new optimizer/model behavior, and single-seed statistical
claims.

## Phase 9E-B5E result seal

No legacy repository file was opened, searched, imported, copied, or modified
for B5E. The result seal uses only B5D GPU artifacts and current V2 contracts.
It preserves the negative C1 outcome rather than importing a legacy
augmentation expectation, and rejects treating one seed as proof that all
transposition strategies are harmful or beneficial.

## Phase 9E-B5F transposition correctness

No legacy repository file was opened, searched, imported, copied, modified,
formatted, staged, or reused for B5F. The phase uses only current V2 B5A
primitives, B5C/B5D runtime paths, B5E evidence, independent arithmetic
oracles, and frozen Dilemmadata artifacts. The detected tritone round-trip
defect is not compared with or repaired from legacy behavior.

## Phase 9E-B5G/H directed inverse and full orbit

No legacy file was opened, searched, imported, copied, modified, formatted,
staged, or reused for B5G/H. Directed physical identity, corpus evidence, C2
orbit enumeration, scheduler and runner use only current V2 B5A-B5F
contracts. The work rejects legacy materialized-view splitting, modulo pitch
wrap, OOV fallback, augmented-view-as-independent-work counting and any TEST
or performance claim before the declared C2 run.

## Multi-source EDA foundational contract

The configured local legacy checkout was absent, so legacy discovery used a
temporary read-only/no-runtime fallback at pinned commit
`2d8281f31cc9ad9c8fecaf332da0c61e0e949415`. It inspected only the historical
HookTheory timeline audit, preprocessing/dataset/evaluation helpers, field and
vocabulary specs, observer schemas, and their small tests. No legacy module was
copied, imported by V2, or modified, and V2 does not depend on that temporary
checkout.

Only generic audit ideas were retained: bounded structured reason/example
buckets, deterministic detail-row order, explicit source grouping, preserved
raw values, machine-readable field/vocabulary references, and tiny malformed
fixtures. The common envelope, capability registry, typed count denominators,
availability partition, source-native composite identities, extension
validation, semantic SHA-256, approved-projection gate, and pre-loader TEST
guard are current V2 contracts rather than adapted legacy code.

Rejected legacy behavior includes HookTheory-only aggregate schemas; mixed or
implicit record/note/chord/section denominators; eager TEST theory reads and
TEST coverage; zero-on-empty metrics; null/unknown/empty conflation;
unversioned token IDs; filename-derived grouping; same-token cross-dialect
merging; random corruption balancing as class balance; float timing;
target/theory-bearing graph fields; noncanonical/unfingerprinted JSON; PDMX
metadata as targets; and any audit-driven change to sampling, loss, model, or
training policy.
