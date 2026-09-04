# Multi-source EDA contract

## Status and scope

This document is the authoritative contract for source-specific exploratory
data analysis implemented after the common foundation commit. It defines
derived, read-only evidence. It does not define a production audit result,
change corpus membership, or authorize reading held-out supervision.

The public API is `music_critic.eda`. Its implementation is standard-library
only and does not discover a corpus, open a target sidecar, build a graph, or
import a source adapter as an import side effect.

The contract deliberately separates two report types:

- `RawCorpusEDA@1.0.0` contains raw/source-structural and proven target-free
  graph evidence;
- `SupervisionEDA@1.0.0` contains source-native supervision evidence for
  TRAIN and VALIDATION only.

Both use `MultiSourceEDAEnvelope@1.0.0`. The fixed capability registry is
`MultiSourceEDACapabilityRegistry@1.0.0`. Decoders require exact schema
versions and fail closed on unknown or missing fields. The version policy ID
is `semantic_versioning_fail_closed_v1`: incompatible semantics require a
version change, and consumers must not assume that a newer version is
accepted merely because it is SemVer-shaped.

## Capability registry

The runtime source of truth is `EDA_CAPABILITIES`; use
`corpus_eda_capability()`, `capability_registry_dict()`, and
`capability_registry_fingerprint()` rather than reproducing it.

| Corpus ID | RawCorpusEDA | SupervisionEDA |
| --- | ---: | ---: |
| `dilemmadata` | yes | yes |
| `hooktheory` | yes | yes |
| `pdmx` | yes | no |
| `pop909_cl` | yes | yes |

Capability means that a source may implement the contract. It does not mean
that a production audit has run. PDMX is a raw/SSL source at this boundary;
`SupervisionEDA`, its target access guard, and even an empty placeholder
supervision report are rejected for PDMX.

## Common envelope

`ReportEnvelope` binds every result to:

- report schema name/version and `ReportKind`;
- `CorpusId`, versioned source/release identity, and versioned producer/adapter
  identity;
- a lowercase 40- or 64-hex Git `repository_commit`;
- evidence, execution, completeness, and split scopes;
- exactly the `ObservationUnit` values used by the semantic payload;
- at least one versioned input-manifest identity for an observed evidence
  scope, with optional normalized repository-relative POSIX path; truthful
  `unknown`/`unavailable` reports may be manifest-free;
- structured invariants, warnings, and unavailable reasons;
- optional operational metadata.

`VersionedIdentity` is always the triple `identity`, SemVer `version`, and
lowercase SHA-256 `fingerprint`. Its fingerprint is an externally bound input:
the producer must derive or obtain it from the identified artifact/contract and
verify it before constructing the report. It is not one of the constructor-
computed report hashes. Input manifest `(role, identity)` pairs are unique. Raw
reports accept only manifests whose `target_free` flag is true and reject the
roles `target`, `target_sidecar`, and `supervision`.

`operational_metadata` is an exact, closed exception to the semantic hash. Its
only permitted keys and value types are:

- `absolute_path`, `cwd`, `finished_at`, `hostname`, `host_name`, `started_at`,
  and `timestamp`: non-empty strings;
- `pid`: a non-negative integer (not a boolean);
- `duration_seconds`, `wall_clock_duration`, and `wall_clock_seconds`: finite,
  non-negative integers or floats (not booleans).

Any other key is rejected rather than silently excluded from the fingerprint.
This prevents a source adapter from hiding semantic configuration in an
operational mapping. Free-form semantic mappings reject operational aliases in
keys and absolute paths in keys or values; typed identity, provenance, schema,
row, task, and count-name channels also reject operational aliases in their
values. This includes run/execution/processing duration aliases in camel,
kebab, or snake form. Repository-relative manifest paths, URLs, harmonic syntax
such as `V/ii`, and truthful relative source-domain paths/timestamps remain
semantic and allowed.

Evidence and execution are coupled exactly:

| `EvidenceScope` | Required `ExecutionMode` |
| --- | --- |
| `fixture` | `synthetic_fixture` |
| `manifest_replay` | `manifest_replay` |
| `bounded` | `bounded_scan` |
| `production` | `production_scan` |
| `unknown` | `not_executed` |
| `unavailable` | `not_executed` |

`unknown` evidence requires `unknown` completeness; `unavailable` evidence
requires `unavailable` completeness. Fixture, replay, and bounded evidence
cannot be represented as production execution. Not running a calculation is
not evidence for a numeric zero. A report whose completeness is
`not_computed`, `unavailable`, or `unknown` requires at least one structured
unavailable reason and cannot contain observed raw metrics, observed graph
evidence, observed task rows, or populated extension rows. At the nested
level, `UnitCount`, `MetricCoverage`, and `TaskFamilyEvidence` cannot be
`observed` under `unknown` or `unavailable` evidence;
`AvailabilityCounts` is forbidden under those two evidence scopes; and a
`SourceExtension` under either scope may contain only explicit non-observed
metric rows whose payload/count summaries are empty.

A `production` report additionally rejects fixture/replay/bounded/synthetic
markers in typed evidence-attestation channels. These include source,
producer, manifest, vocabulary, mapping and work identities; manifest roles
and paths; provenance and reason/code fields; metric/count names; and typed
namespace, schema, row, task, dialect, annotation, and granularity fields.
This is deliberately not a content-word filter: source-native categories and
values, warning/reason prose, projected values, and the complete namespaced
`ExtensionRow.payload` subtree may truthfully contain words such as `Replay`
or `synthetic` and remain fingerprinted domain evidence. In particular, a
payload leaf named `provenance` is source-native domain data; the extension's
evidence attestation remains the separate typed `SourceExtension.provenance`.

The semantic states `available`, `masked`, `missing`, and `unsupported` are
distinct from the computation/access states `observed`, `unknown`,
`not_computed`, `not_applicable`, and `locked`. `EDAReasonCode` provides the
common reasons `eda.manifest_unavailable`, `eda.metric_not_computed`,
`eda.not_applicable`, `eda.production_not_run`, `eda.source_unavailable`,
`eda.target_free_unproven`, `eda.test_targets_locked`, and
`eda.work_identity_unproven`. A source may use a more specific stable reason
identifier where the typed field permits it; the reason must never be encoded
as a fabricated value.

## Observation units and denominators

`ObservationUnit` distinguishes split-assignment rows, source files, records,
target-access attempts, logical and canonical works, excerpts, events, onsets,
notes, bars, beats,
tracks, parts, tempo and meter events, instruments, programs, target rows,
label occurrences, augmented pairs, sampler presentations, optimizer updates,
graph nodes and edges, and raw-identity collisions. The public enum value for
the first unit is `ObservationUnit.SPLIT_ASSIGNMENT` (`split_assignment`), and
callback-attempt counters use `ObservationUnit.TARGET_ACCESS_ATTEMPT`
(`target_access_attempt`) rather than the musical `event` unit.

`UnitCount` records a name, observation unit, nullable value, explicit
denominator and denominator unit, split and evidence scopes, provenance,
computation status, and optional reason. An observed value and denominator are
non-negative integers. A non-observed value is null and carries a reason.
An `unknown` or `unavailable` evidence scope cannot carry an observed count.
`sum_unit_counts()` accepts only observed counts with an identical denominator,
units, scopes, provenance, and status, then sums numerators while preserving
that one shared denominator; it therefore cannot silently combine populations,
records with notes, target rows with presentations, or TRAIN with VALIDATION.

`MetricCoverage` uses one population unit and enforces, when observed:

```text
observed_count + unknown_count == denominator
```

For a non-observed metric, both counts are null; the denominator may remain
known or may also be null. Provenance sequences are non-empty, unique, and
canonicalized. A logical or canonical work count may be observed only when a
versioned work identity is supplied. A work ID must not be inferred silently
from a filename. Without that identity, a work-unit value and work-unit
denominator are both null; a known work population is itself a work-identity
claim.

The typed-count requirement applies to population cardinalities, not every
integer in source-native JSON. Exact ratio objects (for example a musical
time-signature numerator/denominator), physical measurements with units, and
source-native probability, weight, or confidence summaries may remain domain
payload. A field or container that denotes a population count, frequency,
cardinality, total, or denominator cannot use those forms to evade
`UnitCount` and its explicit comparison population.

## RawCorpusEDA

`RawCorpusEDAPayload` contains exactly one `RawMetricEvidence` row for every
entry in `RAW_METRIC_CATALOG`, one `GraphEvidence`, and zero or more source
extensions. Unsupported or uncomputed common metrics remain explicit catalog
rows with non-observed coverage and no summary; an adapter must not delete the
row or place a replacement in an extension.

The frozen common catalog is exposed with its public `RawMetricSpec` value type
and `MetricSummaryKind` enum:

| Metric ID | Summary | Population / value or measurement |
| --- | --- | --- |
| `accepted_records` | count | record / record |
| `bars` | numeric | record / `bars_per_record` |
| `beats` | numeric | record / `beats_per_record` |
| `conversion_outcomes` | categorical | record / record |
| `cross_split_raw_identity_collisions` | count | record / raw identity collision |
| `density` | numeric | record / `pitched_notes_per_quarter_note` |
| `discovered_records` | count | record / record |
| `duplicate_candidates` | count | record / record |
| `duration` | numeric | record / `quarter_note` |
| `empty_records` | count | record / record |
| `graph_edge_counts` | categorical | record / graph edge |
| `graph_node_counts` | categorical | record / graph node |
| `graph_size_distribution` | numeric | record / `nodes_plus_edges_per_record` |
| `instruments` | categorical | record / instrument |
| `invalid_records` | count | record / record |
| `meter` | categorical | record / meter event |
| `meter_changes` | numeric | record / `meter_changes_per_record` |
| `notes` | numeric | record / `notes_per_record` |
| `onsets` | numeric | record / `onsets_per_record` |
| `oversize_records` | count | record / record |
| `parse_outcomes` | categorical | record / record |
| `parts` | numeric | record / `parts_per_record` |
| `percussion_presence` | categorical | record / record |
| `pitch_range` | numeric | record / `midi_note_number` |
| `polyphony` | numeric | record / `simultaneous_note_count` |
| `programs` | categorical | record / program |
| `quarantined_records` | count | record / record |
| `reason_codes` | categorical | record / record |
| `tempo` | numeric | record / `beats_per_minute` |
| `tempo_changes` | numeric | record / `tempo_changes_per_record` |
| `tracks` | numeric | record / `tracks_per_record` |
| `version_candidates` | count | record / record |

Non-observed metrics contain no summary. A known-empty count population
(`denominator == observed_count == unknown_count == 0`) has exactly one
explicit typed zero rather than omission or an unavailable value. If
`observed_count == 0` but `unknown_count > 0`, every metric kind has no summary:
the unobserved population cannot be converted into a zero. A known-empty
numeric or categorical metric likewise has no invented summary. With a
positive observed population, numeric and count metrics contain exactly one
summary of the catalog kind, while a categorical metric contains its category
rows. A multi-occurrence categorical metric may truthfully have no category
rows when the observed records contain no such occurrences.

Count and category rows bind the catalog units, denominator, split, evidence,
and provenance exactly to their coverage. Every count summary, including each
categorical row, has `UnitCount.name == metric_id`; the separate category field
identifies the bucket. Single-valued record categories sum to observed
coverage. Multi-occurrence categories such as meter events, instruments,
programs, graph nodes, and graph edges need not; `reason_codes` is also
explicitly multi-valued because one record may contribute more than one
reason.

For a count whose value unit equals the coverage population unit, the value is
at most `observed_count`, never the observed-plus-unknown denominator.
When all three inventory counts are observed, their complete `MetricCoverage`
values are identical, including population, observed/unknown counts, split,
evidence scope, and provenance. Their typed counts share the same comparison
population and provenance, `discovered_records.value` equals that shared
coverage's `observed_count`, and:

```text
discovered_records = accepted_records + quarantined_records
```

Raw reports accept source-structural units only: split assignments, source
files, records, proven works, excerpts, musical events/onsets/notes/bars/beats,
tracks/parts, tempo/meter events, instruments/programs, graph nodes/edges, and
raw-identity collisions. Target-access attempts, target rows, label
occurrences, augmented pairs, sampler presentations, and optimizer updates are
supervision/training units and are rejected from numerator and denominator
positions in raw extensions and envelopes.

`NumericDistribution` requires a named measurement unit, finite minimum,
maximum, and mean with `minimum <= mean <= maximum`, plus optional unique
quantiles. Quantile probabilities are reduced rational pairs in `[0, 1]`, are
sorted by probability, have nondecreasing finite values, and lie within the
declared minimum/maximum. An emitted probability-zero quantile equals the
minimum and an emitted probability-one quantile equals the maximum, as required
by R7. For one observed scalar, minimum, maximum, mean, and every quantile value
are identical. For `n > 0`, the reported extrema and mean must be realizable by
an `n`-value sample:

```text
maximum + (n - 1) * minimum <= n * mean
                                  <= minimum + (n - 1) * maximum
```

Integer means are checked exactly; binary64 feasibility receives at most one
ULP of the reported mean before multiplication by `n`. All common raw numeric
measurements are non-negative, and observed tempo is strictly positive. MIDI
pitch bounds lie in inclusive `0..127`. Bounds are integer-valued for bars, beats, graph
size, meter changes, notes, onsets, parts, pitch range, polyphony, tempo
changes, and tracks; their means may still be fractional. The frozen policy
IDs are
`finite_binary64_json_shortest_roundtrip_v1` and
`r7_linear_interpolation_sorted_finite_v1`.

Observed graph metrics are valid only when `GraphEvidence` has
`target_free=true` and versioned identities for the graph schema, graph
builder, feature registry, and validator. Those four values must exactly equal
`APPROVED_RAW_GRAPH_CONTRACT`; arbitrary self-consistent identities are not an
attestation. The approved entries pin the existing raw graph schema/version
and the tracked builder, feature-registry, and validator file hashes. The node,
edge, and graph-size metrics have identical `MetricCoverage` values, and their
coverage status equals the `GraphEvidence` status. If target-freedom or an
exact approved binding is not proven, graph evidence and all three graph
metrics are non-observed with a structured reason; unavailable graph evidence
cannot assert `target_free=true`.

For an observed positive record population, let `n` be the shared
`observed_count` and let `S` be the sum of all graph-node category occurrences
plus all graph-edge category occurrences. Graph size is coupled to those
categories exactly:

```text
graph_size_distribution.mean == S / n
maximum + (n - 1) * minimum <= S
                                  <= minimum + (n - 1) * maximum
```

Integer means use exact rational comparison, including values beyond binary64
integer precision; a float mean must be the frozen binary64 representation of
the exact fraction. The second relation prevents aggregate totals that cannot
coexist with the claimed finite-sample extrema.

The frozen graph bindings are:

| Field | Identity/version | Fingerprint and tracked source |
| --- | --- | --- |
| `graph_schema` | `music_critic.graph.raw_schema@1.0.0` | `e0be8d4c522147036418501b230411ac5fc2eafa5284bab44bbc3e6ee3059fc8` — `src/music_critic/graph/relations.py` |
| `graph_builder` | `music_critic.graph.build_raw_graph@1.0.0` | `ccf423169631d4bb12295b92b4403625902eb1ded9478165f2ebc23d836fab65` — `src/music_critic/graph/builder.py` |
| `feature_registry` | `music_critic.graph.raw_feature_registry@1.0.0` | `a041e2c4a221bc0bc722ff3015423230b9e5d5cf56a6efbc4dc71aab351df6f7` — `src/music_critic/graph/feature_registry.py` |
| `validator` | `music_critic.graph.validate_raw_graph@1.0.0` | `8de80cbf5929507da727293751aaba723d4256a5bc65aa0309b968873ffafa99` — `src/music_critic/graph/validation.py` |

Raw manifests and extensions are target-free. Raw semantic payloads reject
target, label, theory, class-support, projection, co-occurrence, and related
target-derived field names recursively, including source-specific compound
keys whose normalized tokens expose those meanings. The check covers source
and manifest identities, envelope invariants/warnings/unavailable detail and
provenance, common metric category/count/reason/provenance channels, graph
reasons, and extensions. The exact typed `eda.target_free_unproven` reason code
is retained as a narrow exception; its detail and provenance are still checked.
A raw report with
non-computed, unavailable, or unknown completeness cannot hide observed
evidence in a common metric, graph binding, or extension row.

## SupervisionEDA

`SupervisionEDAPayload` requires at least one `TaskFamilyEvidence`, one
`TestTargetLockEvidence`, and optional extensions. Its envelope split is
`train`, `validation`, or `train_validation`; each task row is specifically
TRAIN or VALIDATION. TEST and all-split supervision reports fail validation.

Each task row binds:

- corpus, source task ID, dialect, and annotation namespace;
- a versioned vocabulary identity/fingerprint;
- label granularity and categorical or multilabel value type;
- target-row observation unit, split, evidence, provenance, and computation
  status;
- native availability, optional proven work identity, available-only class
  support, optional approved projection evidence, and an unavailable reason
  when not observed.

Supervision TEST-token validation covers every typed task attestation field,
including `source_task_id`, dialect, annotation namespace, vocabulary identity,
`label_granularity`, work identity, reasons, and nested provenance. None may
select TEST/held-out supervision under a locked report.

Rows are unique by `(corpus, source_task_id, dialect, split_scope)`. If the
same corpus/task/dialect appears in both TRAIN and VALIDATION, it keeps one
schema identity across splits: annotation namespace, exact versioned
vocabulary, label granularity, label value type, observation unit, and optional
work identity are identical. Class-support rows across those split instances
also use one common unique-work observation/denominator unit; a task cannot use
logical work in TRAIN and canonical work in VALIDATION.

An observed task has `AvailabilityCounts` in target rows and enforces:

```text
available + masked + missing + unsupported == denominator
```

A non-observed task has no availability, class support, projections, or
empty-multilabel zero; it carries a reason instead.

`SourceValueIdentity` is computed from corpus, source task ID, dialect, source
value, value kind, and identity-contract version. Surface equality therefore
does not merge values from different tasks or dialects. Scalar, multilabel,
and available-empty-multilabel values are distinct kinds. A scalar class value
cannot be null, an empty string, a list, or a mapping: null/empty source values
are availability states rather than classes. A non-empty multilabel value is
normalized by canonical-JSON ordering of unique, non-empty, stripped string
members, so
input order does not create a second identity. An empty list uses the distinct
`empty_multi_label` kind and is counted outside class support.

`ClassSupport` is always `available_only=true`. It records label-occurrence,
unique-record, and unique-work counts with identical split, evidence, and
provenance. Their names are exactly `occurrence_count`,
`unique_record_count`, and `unique_work_count`. Occurrence count units are label-occurrence over target-row;
unique-record units are record over record; unique-work observation and
denominator units are the same explicit `logical_work` or `canonical_work`.
Occurrence and record counts are observed, unique-record support cannot exceed
occurrences, and the two are zero or nonzero together. Observed unique-work
support cannot exceed unique-record support, and those two are likewise zero
or nonzero together. Occurrence counts use the available target-row count as their
denominator and bind the exact task, dialect, split, and evidence scope.
Categorical occurrence totals equal available rows. Multilabel occurrences
may exceed row count, but every non-empty available row contributes at least
one occurrence. Multilabel class-support rows use one scalar, non-empty,
stripped string `SourceValueIdentity` per vocabulary label, never a whole-set
identity; each label's occurrence count is at most the number of non-empty
available rows. An available empty label set is represented by the typed
`UnitCount` in `empty_multilabel_available_count`, never a bare integer. It
uses target-row over target-row units, the available-row denominator, and the
same task split/evidence/provenance; it produces no label occurrence and is
neither missing nor masked. If no work identity is proven, every unique-work
count is `not_applicable` and null.

## Approved projections

Source-native evidence is primary. `ProjectionEvidence` never edits or
repairs it. A projection row binds the full `SourceValueIdentity`, exact
versioned mapping registry, common task identity, native availability state,
mapping state, projected value, and provenance. Its source identity must
already occur in native class support and its native state is therefore
`available`; masked, missing, and unsupported native populations remain in
`AvailabilityCounts` rather than becoming projection classes.

`APPROVED_PROJECTION_REGISTRIES` currently contains only:

```text
identity:    music_critic.dilemmadata.common_harmonic
version:     1.0.0
fingerprint: bb50920808b6ad3a19fb32b8315a417a837b2ab008efd7bee71e71d120e2ee2e
corpus:      dilemmadata
```

`DILEMMADATA_COMMON_TASK_IDS` freezes the six permitted common tasks:

- `dilemmadata.common.chord.bass_pc`;
- `dilemmadata.common.chord.inversion`;
- `dilemmadata.common.chord.pitch_class_set`;
- `dilemmadata.common.chord.quality`;
- `dilemmadata.common.chord.root_pc`;
- `dilemmadata.common.key.local`.

The validator also checks the approved source/common-task pair and requires
`an_joint` for `dilemmadata.an.*` and `dlc` for
`dilemmadata.dlc.*`. Task-specific row validation then checks exact quality,
inversion, and root/bass transformations, the local-key output shape, and the
sorted non-empty pitch-class-set representation. Quality values use the
registry's source-native spelling, including `major triad` and `minor triad`.
Possession of the registry fingerprint alone is insufficient. A
non-projecting state still requires an approved source/common task and dialect
row. The nested static mapping tables are deeply immutable, so a process cannot
change an approved inversion row without changing versioned code.

`ProjectionAvailabilityCounts` is separate from native
`AvailabilityCounts`. Its exact fields are `corpus`, `source_task_id`,
`dialect`, `mapping_registry`, `common_task_identity`, `observation_unit`,
`denominator`, `exact`, `coarsened`, `ambiguous`, `unsupported`, `invalid`,
`missing`, `masked`, `split_scope`, `evidence_scope`, and `provenance`. It is
keyed by one approved registry/common-task pair, counted in target rows, and
partitions its explicit denominator exactly:

```text
exact + coarsened + ambiguous + unsupported + invalid + missing + masked
    == denominator
```

Its corpus/source-task/dialect, target-row denominator, split, and evidence
scope bind the enclosing native task; its provenance is a separate non-empty
canonical sequence. The aggregate denominator equals the native availability
denominator. Projection state counts are otherwise independent of native state
counts and class-support totals: context- or dependency-sensitive inversion,
local-key, and pitch-class-set projection means projection `missing` and
`masked` need not equal native `missing` and `masked`. An aggregate row may
stand alone. `ProjectionEvidence` class rows are optional, but every emitted
class row must have a matching registry/common-task availability row. Static
quality, inversion, root-PC, and bass-PC class rows are exact verified registry
rows. Local-key and pitch-class-set class rows are dynamic: validation proves
approved routing, mapping state, and projected-value shape, but does not claim
to verify the adapter's derivation context.

An exact or coarsened mapping requires a projected value. Ambiguous,
unsupported, and invalid class rows carry no projected value. Missing and
masked are aggregate `ProjectionAvailabilityCounts` states and are forbidden
as `ProjectionEvidence` class rows. Similar names, equal tokens in different
dialects, PDMX metadata, and unregistered crosswalks are not projections.

## TEST-target lock

All supervision aggregation must pass target-free split assignments through
`load_supervision_train_validation_only()` or an equivalent use of
`SupervisionTargetAccessGuard.load_train_validation()`.

The guard performs a complete first pass before calling either callback. It
reads a TEST row's `split` token and discards the row before reading a record
ID, descriptor, path, or other field. TRAIN and VALIDATION rows require corpus,
record ID, split, a lowercase assignment-manifest SHA-256, and
`target_free=true`. Their exact field allowlist is
`assignment_manifest_fingerprint`, `corpus`, `record_id`, `split`, and
`target_free`; every additional field—not only a known target/path name—is
rejected. Unknown or malformed rows fail before any callback. The TEST branch
intentionally reads only `split`, so even a TEST row containing an unsafe path
is counted and discarded without inspecting that path.
All allowed TRAIN/VALIDATION rows must carry the same assignment-manifest
fingerprint; disagreement fails during this same callback-free preflight.
Duplicate retained `(corpus, record_id)` rows also fail before either callback,
including a record assigned once to TRAIN and once to VALIDATION. The retained
split plan is therefore mutually exclusive. TEST record IDs are deliberately
never read, so a TEST assignment is counted solely from its split token rather
than inspected for identity.

The callbacks are:

```python
resolve_descriptor(record_id: str, split: SplitScope) -> Descriptor
load_target(descriptor: Descriptor, split: SplitScope) -> LoadedTarget
```

The result is `(loaded_train_validation_rows, TestTargetLockEvidence)`. The
guard creates that evidence through `TestTargetLockEvidence.from_guard(...)`;
adapters do not hand-assemble scalar audit counters. All five counter fields
are `UnitCount` values:

- `test_assignment_count` observes `split_assignment`;
- `test_descriptor_resolution_count` and `test_target_loader_call_count`
  observe `target_access_attempt` values;
- `test_target_records_opened` observes `record`;
- `test_target_rows_loaded` observes `target_row`.

Every counter has `denominator_unit=split_assignment`, `split_scope=test`, the
same report evidence scope and non-empty provenance, and the common denominator
`test_assignment_count.value`: the number of TEST split-assignment rows counted
without reading their record IDs. The assignment counter's value equals that
denominator; the other four values are observed zero. Lock evidence also
carries the common assignment-manifest fingerprint, requires both pre-access
gates, and requires false values for:

- `test_targets_read`;
- `test_targets_used_for_eda`;
- `test_targets_used_for_model_evaluation`;
- `test_class_distributions_emitted`;
- `test_coverage_emitted`;
- `test_cooccurrence_emitted`.

There is no unlock parameter. Raw TEST evidence is allowed only through the
same proven target-free raw contract; otherwise it is structured unavailable
evidence.

`SupervisionTargetAccessGuard` and the functional facade accept optional
`evidence_scope` and `provenance`. Their convenience defaults are `fixture`
and `("supervision-target-access-guard",)`. A production child, and any child
emitting another report scope, must explicitly pass its report evidence scope
and report provenance; `SupervisionEDA` verifies that the lock counter scope
equals the envelope evidence scope.

The guard requires at least one TRAIN or VALIDATION assignment; an empty or
TEST-only projection is rejected. A supervision report requires exactly one
target-free input manifest with role `split_assignment`, and its identity
fingerprint must equal `TestTargetLockEvidence.assignment_manifest_fingerprint`.
It also requires at least one other target-bearing input manifest. The sole
exception is a truthful manifest-free `unknown` or `unavailable` non-evidence
report: it uses `TestTargetLockEvidence.not_executed(...)`, a null assignment
manifest fingerprint, and five null `locked` counters instead of inventing
manifest evidence. Any extension row with observed coverage is observed
supervision evidence and therefore also requires an observed guard attestation,
even when every common task row is non-observed. An explicit non-observed empty
row does not.

The lock object is a validated attestation of the prescribed adapter path, not
a cryptographic or process-wide ban on unrelated Python code opening a target
file. Every source adapter must actually use the guard and retain a
descriptor/loader spy regression that fails on any TEST invocation. Merely
constructing an all-false lock object is not sufficient operational evidence.

## Source adapter and extension boundary

`SourceEDAAdapter` is the runtime-checkable raw-only protocol with `corpus`,
versioned `adapter_identity`, sorted source-owned `extension_namespaces`, and
`build_raw_eda(request)`. `SupervisionSourceEDAAdapter` extends it with
`build_supervision_eda(request)`. Dilemmadata, HookTheory, and POP909-CL must
implement the supervision protocol to register; PDMX may and should implement
only the raw protocol. `EDAAdapterRegistry.build_supervision()` rejects PDMX
before adapter dispatch. The request is opaque so source-local configuration
does not change this shared schema.

Create a process-local `EDAAdapterRegistry`, call `register(adapter)`, then use
`build_raw()` or `build_supervision()`. Registration rejects duplicates and
requires `extension_namespaces` to be a tuple of strings. It also rejects
namespaces that are unsorted, duplicated, or do not start with
`<corpus>.`. Returned report corpus and producer identity must equal the
registration. An extension namespace used by a report must have been declared
by the adapter. Corpus, adapter identity, and namespace declarations are
snapshotted immutably at registration; mutating the adapter object later
cannot rewrite what the registry accepted.

`SourceExtension` binds corpus, namespace, extension schema/version, explicit
split scope, evidence scope, non-empty provenance, `target_free`, sorted
`ExtensionRow` values, extension contract `1.0.0`, and an automatically
computed fingerprint over all of those semantic fields. Each `ExtensionRow`
is exactly one source-native metric: `row_id` is that metric's stable identity
within its extension namespace/schema, and the row has mandatory
`MetricCoverage`, JSON-safe source-native summary payload, and optional typed
`UnitCount` summary components.

Row coverage has the extension's exact split scope, evidence scope, and
provenance. Every typed count is observed and has those same three bindings,
the exact coverage denominator, and the coverage population unit as its
denominator unit. A count in that same population cannot exceed
`coverage.observed_count`. If row coverage is non-observed, both payload and
counts are empty. If `observed_count == 0` with an unknown remainder, payload
and counts are both empty. Only fully known-empty coverage with denominator
zero may carry exact typed zero counts (or no counts), and it never carries a
payload summary. In raw reports, row coverage uses only a source-structural raw
unit, and that unit participates in the envelope's exact observation-unit set.

Extension rows cannot redefine common envelope fields, contain operational
metadata keys, hide bare population `count`/`denominator` fields, or replace
common catalog rows. Envelope/wrapper keys (including envelope schema/version
policy), fixed raw metric IDs, and common task structures are reserved
recursively. A typed extension `UnitCount.name` is subject to the same
collision rule. Namespaced nested source-schema leaves such as `name`,
`status`, `category`, `mean`, `provenance`, `payload`, and `value` remain
allowed. Exact ratio objects, physical measurements, and source-native
probability/weight/confidence summaries (for example a
`confidence_histogram`) remain domain payload, but cannot disguise a
population count. Probability-, normalized-, or confidence-named numeric
containers are accepted as summaries only when every numeric leaf is finite
and lies in `[0, 1]`; out-of-range bucket frequencies and integer-valued
weight histograms belong in typed `UnitCount` rows. A categorical mapping with
only non-negative integer leaves is likewise treated as a count summary unless
its field explicitly denotes an identifier/code or a recognized exact ratio,
measurement, physical unit, or MIDI/program value.

When a source provides confidence/calibration evidence with no common catalog
field, preserve it in its corpus namespace under this extension contract; do
not discard it or reinterpret it as native availability or a common
projection.

Unknown or unavailable extensions may contain only non-observed metric rows
with empty payload/count summaries. Raw extensions additionally require
`target_free=true`, bind a split within the raw envelope and the exact report
evidence scope, and reject target-derived content. Supervision extensions bind
exactly TRAIN or VALIDATION within the report scope, match its evidence scope,
reject TEST-named or TEST-selecting fields recursively, including bare plural
`tests`/`testsets`, singular/plural `heldout`, `held_out`, `holdout`, and nested
`held`/`hold` + `out` aliases, and bind their row coverage/counts within the
extension scopes.

TEST/target token checks also cover
manifest, source, task, namespace/schema, work/row identity, count, and
provenance channels, including envelope invariant/warning/reason fields and
task/work-count reasons. The exact typed `eda.test_targets_locked` reason code
is the narrow exception; its detail and provenance are still checked.
Ordinary non-TEST scope metadata such as `train` or a
source-release partition remains allowed.

An extension row coverage or count that supplies a known logical/canonical-work
population or value requires the extension's own versioned `work_identity`. So
does any nested payload field whose normalized name
claims a work, logical-work, or canonical-work ID/identifier/key/UID/UUID,
including compact and plural aliases. For raw extensions, target-free
validation covers namespace, schema name, extension and row-coverage
provenance/reason, count names/reasons/provenance, and payload. Supervision
TEST validation covers the corresponding row-coverage channels. The same
namespace may occur once per split; duplicate `(namespace, split_scope)`
entries are rejected, while TRAIN and VALIDATION instances remain distinct
semantic evidence. Across those instances, one namespace retains the same
schema name/version, work identity, and `target_free` declaration. A stable
`row_id` retains one `coverage.observation_unit` across splits. Its observed
instances retain one typed-count component schema: the same component names
with the same `(observation_unit, denominator_unit)` pairs.

## Canonical serialization and fingerprints

Use the public report API rather than hand-writing JSON or fingerprints:

- `report_dict()` returns the validated JSON-safe report;
- `report_fingerprint()` recomputes the semantic SHA-256;
- `dumps_report()` emits deterministic Unicode JSON without a terminal
  newline;
- `canonical_report_bytes()` emits UTF-8 with exactly one terminal newline;
- `dump_report()` writes those bytes failure-atomically;
- `report_from_dict()`, `loads_report()`, and `load_report()` strictly decode,
  cross-validate, and verify stored fingerprints.

The implementation reuses `dumps_canonical_json()` and
`canonical_json_sha256()` from `music_critic.data.serialization`: mapping keys
are sorted, Unicode is retained, compact hashes have no insignificant
whitespace, and NaN, infinity, or lone Unicode surrogates fail closed. Every
string and mapping key must be valid UTF-8 scalar text. Structural identifiers,
schema fields, provenance strings, and mapping keys reject Unicode control and
format categories (`Cc`/`Cf`). Opaque domain/prose fields such as source
categories, warning messages, unavailable detail, and scalar source values may
retain meaningful interior tabs/newlines and emoji; fields that represent one
scalar class string still must be non-empty and stripped. Semantically unordered
rows are sorted by their typed identities before hashing; meaningful ordered
values remain ordered. Negative floating zero is normalized to `0.0`. Strict JSON
loading also rejects duplicate object keys at every nesting level rather than
silently keeping the last occurrence. Validated arbitrary JSON is recursively
frozen in memory: nested mappings are read-only and sequences are tuples in
extension payloads and projected values. Public serializers materialize fresh
JSON mappings/lists, so caller mutation cannot stale an existing report
fingerprint.

The semantic fingerprint covers the schema/envelope identity—including corpus,
source, producer, repository commit, evidence, split, manifests, invariants,
warnings, and unavailable reasons—and the semantic payload. It excludes the
entire `operational_metadata` mapping and the fingerprint field itself.
Absolute paths, hostnames, process IDs, timestamps, and wall-clock durations
belong only in operational metadata. Changing them does not change the
fingerprint; changing semantic evidence does. Report semantic fingerprints,
extension fingerprints, and source-value identities are constructor-computed
and must never be injected. `VersionedIdentity.fingerprint` is different: it
is a required externally supplied and verified binding for a release,
producer, manifest, vocabulary, work contract, graph contract, or mapping
registry, and remains inside the semantic fingerprint domain.

## Non-goals and immutable boundaries

This foundation does not scan any real corpus, recompute corpus checksums,
convert MIDI, build graphs, inspect checkpoints, use a GPU, train a model, or
materialize production distributions. It changes no data, cache, graph,
sidecar, membership, split, grouping, vocabulary, projection, mask, model,
head, loss, sampler, SSL protocol, training configuration, checkpoint, or
existing production manifest. Source-specific work must preserve those
boundaries and report its evidence scope literally.
