# Multi-source Target Ontology and Batching Contract

Status: **ACCEPTED PHASE 5A CONTRACT**

Ontology version: `1.0.0`

Implementation: `music_critic.tasks`

## 1. Scope and semantic layers

Phase 5A defines evidence and interfaces. It does not implement a production
dataset, sampler, PyG collator, target tensorizer, cache, split assignment,
model, loss, renderer, or training path.

The contract keeps these layers distinct:

1. **source-native annotation**: fields emitted by one production adapter;
2. **derived normalized target**: a deterministic target-only transformation
   with its own method version, availability mask, and provenance;
3. **raw graph input**: raw-only `HeteroData` built without targets,
   annotations, provenance, dataset/group/split identity, or confidence;
4. **model supervision**: future sidecar tensors routed by task, source view,
   supervision context, and availability;
5. **probabilistic chord/accompaniment model**: a future normalized model of
   observed notes or pitch sets, not a classifier alias;
6. **quality critic**: a future preference/aspect model requiring independent
   quality evidence.

HookTheory provides melody-conditioned harmonic supervision. POP909-CL
provides score-conditioned harmony-recognition supervision. Generic/raw MIDI
remains role-agnostic raw input with no required harmonic targets. Identical
output tokens do not erase these contexts or authorize a shared loss.

## 2. Public API and registry invariants

`music_critic.tasks` exports:

- `TARGET_ONTOLOGY_VERSION`, `TARGET_FAMILIES`, and
  `TARGET_FAMILY_BY_ID`;
- `CROSSWALKS` and `CROSSWALK_BY_ID`;
- `AlignmentPolicy`, `TargetFamilySpec`, and `CrosswalkSpec`;
- deterministic `ontology_contract_dict`, `dumps_ontology_contract`, and
  `ontology_contract_fingerprint`;
- `SampleTarget`, `TaskAvailability`, `MultiSourceSample`;
- future-container contracts `BatchTarget` and `MultiSourceBatch`;
- `GroupAssignment`, `DatasetSamplingWeight`,
  `validate_group_assignments`, and `deterministic_group_order`;
- `build_multisource_sample`, which creates target/provenance/diagnostic
  sidecars around an opaque raw graph without modifying it.

Every target-family specification records its stable source task ID, registry
version, semantics, canonical dtype, vocabulary or open value-space rule,
entity/granularity, exact timing and interval semantics, supervision context,
source adapter/view, missing-value meaning, required availability/provenance,
confidence policy, candidate/alignment policy, and cross-source-sharing
permission. Existing adapter task IDs are not renamed.

All Phase 5A families require per-entry availability masks and provenance.
Available confidence may be null when a source supplied no number. A false
mask requires null value, confidence, source, and provenance and never encodes
a negative class.

## 3. Exact source inventory

### HookTheory

Statistics below are reproducible bounded-fixture counts, not corpus-wide
claims. The audit converts 18 usable real-source excerpts from the 19-case
`hooktheory_golden_v1` manifest; `missing_payload` is accounted for but cannot
produce a piece. Timing is exact rational quarter-note time. Chord/key spans
are half-open; melody scale degree aligns by canonical note identity. Direct
adapter target provenance is `prov:annotation`, with its source/conversion
ancestors retained in the sample sidecar. Numeric confidence is not supplied.

| Stable task ID | dtype / value space | available | masked | Known limitations |
|---|---|---:|---:|---|
| `theory.melody.scale_degree` | categorical string; degrees `1..7` with `b`, `#`, `bb`, `##` | 3 | 0 | invalid/resting/unresolved notes are omitted or unavailable, never negative |
| `theory.local_key.tonic_pc` | categorical string `0..11` | 21 | 0 | local-key tonic is not a chord root |
| `theory.local_key.mode` | open normalized mode string | 21 | 0 | unresolved mode is unavailable |
| `theory.chord.presence` | categorical `false`, `true` | 11 | 0 | explicit rest/chord span; absence of annotation is not `false` |
| `theory.chord.root_degree` | categorical `0..6`, `bVII` | 10 | 1 | functional degree; zero/invalid/rest roots are unavailable |
| `theory.chord.extent` | categorical `5`, `7`, `9`, `11`, `13` | 10 | 1 | not a POP909-CL quality |
| `theory.chord.inversion` | categorical ordinal `0..3` | 10 | 1 | not root-to-bass semitone distance |
| `theory.chord.adds` | multi-label `4`, `6`, `9` | 10 | 1 | source-native decoration |
| `theory.chord.omits` | multi-label `3`, `5` | 10 | 1 | source-native decoration |
| `theory.chord.alterations` | multi-label `b5`, `#5`, `b9`, `#9`, `#11`, `b13` | 10 | 1 | source-native decoration |
| `theory.chord.suspensions` | multi-label `2`, `4` | 10 | 1 | source-native decoration |
| `theory.chord.borrowed` | open `none`, `mode:*`, `pcset:*`, `unknown:*` | 10 | 1 | applied/borrowed cross-source semantics are unresolved |

Empty multi-label tuples in an available entry mean that the source supplied
no members of that decoration family. They are not missing values.

### POP909-CL

These statistics come from production manifest `1.0.0`, generated by the
already accepted streaming 909-file run. Phase 5A does not repeat that run.
Chord and coverage spans use exact `tick/PPQN` rational time, intersected with
raw duration only where required by canonical schema. Boundary is the exact
span-start event. Direct block boundary/bass provenance is human-corrected,
expert-reviewed annotation evidence; root/quality/inversion and `N` use
explicit derivation provenance. Numeric confidence is not supplied.

| Stable task ID | dtype / vocabulary | available | masked | Known ambiguous/unsupported behavior |
|---|---|---:|---:|---|
| `pop909_cl.chord.boundary` | categorical `present` | 116,055 | 2 | directly observed even if normalization is ambiguous/unsupported |
| `pop909_cl.chord.root` | categorical `C..B` pitch-class names | 109,668 | 6,389 | ambiguous and unsupported roots are masked |
| `pop909_cl.chord.quality` | 13-class categorical quality | 109,800 | 6,257 | ambiguous remains available only if all candidates agree |
| `pop909_cl.chord.bass` | categorical `C..B` pitch-class names | 116,055 | 2 | directly observed lowest pitch class; independent mask |
| `pop909_cl.chord.inversion` | categorical semitone distance `0..11` | 109,668 | 6,389 | depends on unambiguous root; independent from bass mask |
| `pop909_cl.chord.no_chord` | categorical `N` | 947 | 153 | only positive leading/internal gaps are `N`; 151 trailing spans and two missing-instrument spans remain masked |

The manifest records 5,801 ambiguous blocks and 586 unsupported blocks.
Boundary and bass remain available for those blocks. `367` and `658` supply
one explicitly masked entry for every family. Missing annotation, a masked
entry, unsupported normalization, ambiguous normalization, trailing uncovered
time, and available `N` are six distinct states.

## 4. Conservative crosswalk

There are no accepted `exact_shared` or `derived_lossless_subset` mappings in
ontology `1.0.0`. A future model may route multiple source-native heads into a
shared representation, but that is a model policy, not a claim that labels are
identical.

| Potential crosswalk | Status | Reason |
|---|---|---|
| HookTheory `root_degree` ↔ POP909-CL absolute `root` | `incompatible` | functional degree and absolute pitch class have different meanings |
| HookTheory `extent` ↔ POP909-CL `quality` | `incompatible` | extent does not losslessly determine pitch-class-set quality |
| HookTheory ordinal `inversion` ↔ POP909-CL semitone `inversion` | `incompatible` | ordinal position and root-to-bass interval are different value spaces |
| HookTheory chord `presence` ↔ POP909-CL `boundary` | `incompatible` | span presence and an exact boundary event are different tasks |
| HookTheory chord `presence/rest` ↔ POP909-CL `no_chord` | `incompatible` | rest/absence/trailing masked and positive internal `N` are not interchangeable |
| HookTheory absolute-root derivation ↔ POP909-CL `root` | `deferred` | applied and borrowed semantics plus a lossless key/degree rule are missing |
| Source-chord pitch-class-set rendering ↔ POP909-CL normalized quality | `deferred` | renderer, decoration, applied, and borrowed semantics are not defined |
| Unpaired key, melody, decoration, borrowed, and bass families | `source_specific` | no current production target has proven equivalent semantics |

A future `derived_lossless_subset` entry must state exact prerequisites,
algorithm/version, unsupported-case masking, and provenance parents, with
tests for ambiguous and unsupported inputs. Until then, no automatic
conversion is performed. Phase 5A implements no chord renderer, applied
harmony, borrowed reinterpretation, or target-derived notes.

## 5. Alignment contract

Alignment remains a sidecar operation:

- all source spans retain exact `RationalTime`; no float equality or
  nearest-neighbor snapping is permitted;
- `note_identity_v1` maps only an exact canonical note entity ID;
- `half_open_anchor_span_v1` tests onset point times and beat/bar start anchors
  with exact half-open containment, `span.start_qn <= candidate_time <
  span.end_qn`; positive interval intersection is not an alignment rule;
- a candidate exactly at a span end belongs to the following span;
- when several source spans address one typed candidate, equal available
  values merge deterministically; conflicting available values are masked
  with diagnostic code `multisource.alignment_conflict`;
- `span_start_boundary_v1` represents a point at the exact span start; Phase
  5B may use only an exact-time raw candidate or retain the source event with
  entity index `-1`, a false entity-index mask, and null node type;
- every aligned index carries an explicit node type; coincident node types do
  not use an implicit priority;
- `coverage_span_v1` aligns only explicitly available gap spans; trailing
  masked coverage is never converted to `N`;
- ambiguous and unsupported scalar targets remain masked while raw candidate
  evidence and diagnostics remain sidecars;
- an entirely empty task family is represented with zero entries, not a
  fabricated negative;
- unannotated positions are not enumerated as negatives.

`pop909_cl.chord.boundary` is a positive-unlabeled event-detection target.
Observed span starts carry only the class `present`; non-boundary candidates
remain unlabeled and ontology `1.0.0` defines no synthetic `absent` class or
derived negative rule. A future Phase 5B loss must preserve that objective or
introduce a separately versioned, evidence-backed negative policy.

Raw graph stores contain neither alignment indices nor target values. Future
entity indices, values, masks, confidence, provenance, and diagnostics remain
outside `HeteroData`. Graph construction and fingerprinting therefore stay
invariant under target, annotation, provenance, group, lineage, and split
changes.

## 6. Phase 5B sample and batch API

`MultiSourceSample` is the accepted future sample shape:

```text
raw_graph                    opaque raw-only HeteroData
dataset_id, piece_id
source_group_id, lineage_group_id
target_bundle                sorted source-native SampleTarget sidecars
target_availability          all registry tasks; absent differs from masked
target_provenance_sidecar    referenced records plus provenance ancestors
diagnostics                  CPU-side QualityFlag records
```

Each `SampleTarget` retains task/view/alignment, unique non-empty entity IDs,
values validated against the task value space, availability mask, optional
finite confidence in `[0, 1]`, source, and provenance IDs. `TaskAvailability`
requires a registered task, non-negative counts, and zero counts for a family
declared absent.
`build_multisource_sample` performs this projection but does not load data,
align nodes, tensorize, batch, or sample.

`MultiSourceBatch` is an immutable shape contract only:

```text
raw_graph_batch              PyG Batch containing raw graph stores only
target_batches               sorted BatchTarget sidecars
dataset/piece/source/lineage identities
diagnostics_cpu              per-sample CPU metadata
```

Each `BatchTarget` reserves separate values, availability mask, entity indices,
entity-index mask, explicit entity node types, sample indices, optional
confidence, and CPU provenance/diagnostics. All leading dimensions equal
`entry_count`; aligned indices are non-negative and typed for the task,
whereas unaligned retained events use index `-1`, false index mask, and null
node type. `MultiSourceBatch` rejects empty sample metadata, duplicate or
unsorted task sidecars, out-of-range sample indices, a false raw-only marker,
and target/provenance fields embedded in graph stores.
`entry_count=0` with empty per-entry metadata is the canonical completely empty
family. If a dataset lacks a task in a mixed batch, no negative label and no
loss entry are created.

Phase 5B must implement and test actual tensor dtypes/shapes, PyG batching,
worker seeding, mixture weights, and collator performance without changing
this raw/sidecar boundary.

## 7. Grouping and deterministic sampling

- Authoritative lineage is read from canonical provenance. A caller-supplied
  lineage is an assertion: it must be non-empty and exactly match that value.
  When provenance has no lineage, the explicit fallback is
  `piece.source_group_id`; no unrelated POP lineage is invented.
- HookTheory therefore uses provenance `ori_uid` when present and otherwise
  its stable canonical source-group fallback.
- POP909-CL uses `pop909-cl:<song-id>` as `source_group_id` and
  `pop909-lineage:<song-id>` as cross-version lineage.
- Matching CL/original POP909 lineage cannot cross splits.
- Split assignment occurs at source and lineage group level. Phase 5A
  validates assignments but creates no final project splits.
- Exact duplicate assignment rows are rejected. Repeated `(dataset_id,
  piece_id)` identities cannot disagree on source or lineage.
- `deterministic_group_order` first computes atomic transitive components:
  pieces connected through either a shared source group or a shared lineage
  group remain contiguous and cannot be separated by ordering. Components are
  hashed with an explicit integer seed; pieces within each component use
  stable identity order. The result is input-order invariant, while a
  different seed changes component order without changing component contents.
- `DatasetSamplingWeight` reserves positive future per-dataset weights without
  implementing a sampler.
- One POP909 song/piece is one sample. Its chord blocks are target entries, not
  116,055 independent training samples.

## 8. Machine-readable evidence

Run:

```bash
python scripts/audit_multisource_targets.py \
  --check tests/fixtures/multisource/target_contract_manifest.json
```

The deterministic artifact records canonical/graph/adapter/ontology versions,
all target specifications and crosswalk statuses, bounded HookTheory counts,
accepted POP909-CL manifest counts, vocabularies, contract-source SHA-256
values, and explicit scan policy. It contains no corpus record, MIDI, cache,
generated output, or detailed corpus report.

No full corpus scan was needed: bounded real HookTheory excerpts exercise every
current family and the accepted POP909-CL manifest already contains the
required family/ambiguity/mask aggregates. Corpus-wide HookTheory target counts
remain intentionally unclaimed.

## 9. Deferred blockers before Phase 5B

Phase 5B still must choose concrete target tensor encodings, implement exact
entity-index generation for each policy, validate PyG batch offsets, implement
the dataset/collator and deterministic worker seeds, and define configurable
dataset mixture weights. It must not promote any deferred crosswalk without a
separate evidence-backed ontology version.

Applied harmony, borrowed-chord cross-source semantics, pitch-class-set
rendering, actual accompaniment likelihood, final splits, and model head
routing remain later decisions.
