# Multi-source EDA source-branch handoff

## Immutable base

Every Dilemmadata, POP909-CL, HookTheory, and PDMX EDA worktree must be created
from this exact foundation commit:

```text
<EDA_CONTRACT_SHA>
```

`<EDA_CONTRACT_SHA>` is intentionally a handoff token in this tracked file. A
Git commit cannot contain its own exact object ID: replacing the token changes
the tree and therefore changes the commit ID. The single-commit requirement is
preserved by publishing the resolved SHA in the post-commit Codex handoff.
Child tasks must copy that resolved 40-hex value verbatim, verify it with
`git rev-parse HEAD`, and branch from the SHA rather than the moving branch
name. No source branch may replace this token or edit this shared handoff.

Example, after substituting the delivered SHA and source-specific names:

```bash
git worktree add -b audit/<source>-eda /tmp/music-critic-v2-<source>-eda <EDA_CONTRACT_SHA>
git -C /tmp/music-critic-v2-<source>-eda rev-parse HEAD
git -C /tmp/music-critic-v2-<source>-eda status --short
```

## Frozen shared API

The report schemas are:

```text
MultiSourceEDAEnvelope@1.0.0
RawCorpusEDA@1.0.0
SupervisionEDA@1.0.0
MultiSourceEDACapabilityRegistry@1.0.0
```

Shared implementation paths:

- `src/music_critic/eda/contracts.py`: types, enums, capability matrix,
  validators, metric catalog, and semantic fingerprints;
- `src/music_critic/eda/serialization.py`: strict load/dump and atomic file
  serialization;
- `src/music_critic/eda/access.py`: pre-open TEST-target guard;
- `src/music_critic/eda/adapters.py`: adapter protocol and process-local
  registry;
- `src/music_critic/eda/__init__.py`: supported public imports;
- `src/music_critic/data/serialization.py`: reused canonical JSON primitives;
- `tests/fixtures/multisource/eda_contract_cases.json`: bounded synthetic
  four-corpus input evidence;
- `tests/eda/`: executable shared-contract tests;
- `tests/data/test_serialization.py`, `tests/data/test_schema.py`, and
  `tests/test_import.py`: directly related serialization/schema/import
  regressions;
- `tests/audit/test_multisource_target_audit.py`,
  `tests/audit/test_phase9eb4_class_balance_audit.py`, and
  `tests/tasks/test_multisource_contract.py`: related audit/task contracts;
- `tests/test_repository_contract.py`: repository-policy verification, run as
  its own command;
- `docs/MULTISOURCE_EDA_CONTRACT.md`: normative semantics.

Source branches must not change these modules, the shared fixture/tests, this
handoff, or common authoritative documents. A genuine common-schema defect or
required semantic change stops the source task and returns to a separately
reviewed foundation/version-bump decision.

Task-specific child override: although the repository-wide task protocol
normally asks every task to update shared status and legacy-reference files, a
source-adapter child must not edit `docs/STATUS.md` or
`docs/LEGACY_REFERENCE.md`. It returns exact proposed status and legacy
reuse/rejection notes in its final report for the foundation owner to record.
This narrow override does not relax any other repository instruction.

## Capability assignment

| Source branch | Required raw report | Allowed supervision report |
| --- | ---: | ---: |
| Dilemmadata | yes | yes |
| POP909-CL | yes | yes |
| HookTheory | yes | yes |
| PDMX | yes | no |

PDMX must not create an empty `SupervisionEDA`. Capability support is not a
claim that a production audit already exists.

## Adapter registration

Implement adapters outside the shared `music_critic.eda` package. Every source
implements the raw-only `SourceEDAAdapter` surface:

```python
corpus: CorpusId
adapter_identity: VersionedIdentity
extension_namespaces: tuple[str, ...]

def build_raw_eda(request: object) -> RawCorpusEDA: ...
```

Dilemmadata, HookTheory, and POP909-CL additionally implement
`SupervisionSourceEDAAdapter`:

```python
def build_supervision_eda(request: object) -> SupervisionEDA: ...
```

Register and execute it through the shared gate:

```python
from music_critic.eda import EDAAdapterRegistry

registry = EDAAdapterRegistry()
registry.register(adapter)
raw_report = registry.build_raw(adapter.corpus, raw_request)
supervision_report = registry.build_supervision(adapter.corpus, supervision_request)
```

Call `build_supervision()` only for the three labeled corpora. Registration
requires their adapters to satisfy `SupervisionSourceEDAAdapter`. A PDMX
adapter needs no forbidden supervision stub: it satisfies only
`SourceEDAAdapter`, and the registry rejects PDMX supervision before dispatch.
Adapter requests are source-local and opaque; do not add their configuration
to the common schema.

Registration validates a unique corpus, exact producer identity, and sorted,
unique extension namespaces beginning with `<corpus>.`. The registry also
checks that every emitted namespace was declared. It snapshots corpus,
adapter identity, and namespaces at registration, so later mutation of the
adapter object cannot change the accepted declaration.

## Required report construction

For every report:

1. Build exact `VersionedIdentity` values for source release, adapter, input
   manifests, vocabulary/work identities where applicable, and graph contract
   bindings where graph metrics are observed. Derive or obtain each external
   fingerprint from the identified artifact/contract and verify it; do not
   invent an attestation hash.
2. Use `ReportEnvelope` with the literal evidence/execution/completeness/split
   scopes and exactly the observation units used by the payload.
3. Use repository-relative POSIX manifest paths. Keep machine-local paths and
   timing only in `operational_metadata`. That mapping accepts only the frozen
   keys `absolute_path`, `cwd`, `duration_seconds`, `finished_at`, `hostname`,
   `host_name`, `pid`, `started_at`, `timestamp`, `wall_clock_duration`, and
   `wall_clock_seconds`, with the types specified by the contract; all other
   keys are rejected. Free-form semantic mappings and identity/provenance
   channels reject operational aliases and absolute paths in keys or values.
   Truthful repository-relative/source-domain paths and timestamps, URLs, and
   harmonic syntax such as `V/ii` remain allowed.
4. Preserve provenance on every count, metric, availability row, class row,
   projection, invariant, warning, and unavailable reason.
5. Represent unavailable, unknown, not-computed, not-applicable, or locked
   evidence with a null value and reason; never with a fabricated zero.
6. Let constructors canonicalize rows and compute report semantic fingerprints,
   extension fingerprints, and source-value identities. Never inject those
   computed values. `VersionedIdentity.fingerprint` is instead the verified
   external binding supplied in step 1 and remains semantic evidence.

For a `production` report, do not use fixture/replay/bounded/synthetic markers
in typed attestation channels: source/producer/manifest/vocabulary/mapping/work
identities, roles and manifest paths, provenance/reason/code fields,
metric/count names, namespace/schema/row/task/dialect/annotation/granularity.
Do not apply that filter to source-domain content: categories, warning/reason
prose, source/projected values, and the entire namespaced extension `payload`
may truthfully contain the same words. A `payload.provenance` leaf is domain
content; typed extension evidence provenance remains
`SourceExtension.provenance`.

If report completeness is `not_computed`, `unavailable`, or `unknown`, add a
structured unavailable reason and emit no observed metric/task/graph evidence
or populated extension rows. `unknown` and `unavailable` evidence scopes can
never carry observed nested evidence.

Every `RawCorpusEDA` must emit all rows in `RAW_METRIC_CATALOG`. Fill metrics
that the source cannot soundly compute with typed non-observed
`MetricCoverage`. Bind every count summary and categorical-row count name
exactly to its `metric_id`; keep the bucket identity in `CategoryCount.category`.
Non-observed metrics have no summary. A fully known-empty count population
(`denominator == observed_count == unknown_count == 0`) has one explicit typed
zero; a known-empty numeric/categorical metric has no invented summary. When
`observed_count == 0` but `unknown_count > 0`, every kind has no summary—the
unknown population is not zero. With a positive observed population,
multi-occurrence categorical metrics may still have no buckets when there were
no occurrences.
`reason_codes` is multi-valued, as are the source occurrence families whose
value unit differs from the record population.

If discovered, accepted, and quarantined counts are observed, all three use
identical `MetricCoverage` and count comparison bindings;
`discovered_records.value == coverage.observed_count` and
`discovered_records == accepted_records + quarantined_records`. Common numeric
summaries are non-negative; tempo is positive; pitch lies in `0..127`; the
discrete metric bounds named by the contract are integer-valued. R7 endpoint
quantiles equal the extrema, a singleton has identical extrema/mean/quantiles,
and every mean/extrema combination is realizable for its observed sample size.

Raw target-token checks cover envelope invariant/warning/unavailable fields,
common metric categories/counts/reasons/provenance, identities, manifests, and
extensions. Only the exact standard `eda.target_free_unproven` reason code is
excepted; its detail and provenance remain checked. Only emit graph summaries
after a complete target-free graph
attestation whose identities exactly match `APPROVED_RAW_GRAPH_CONTRACT`. Raw
manifests and raw extensions must be explicitly target-free.

The graph-node, graph-edge, and graph-size rows use exactly identical coverage
and the `GraphEvidence` status. For positive shared `observed_count = n`, let
`S` be all node plus edge category occurrences. Require exact
`graph_size_distribution.mean == S / n` and realizable extrema:

```text
maximum + (n - 1) * minimum <= S
                                  <= minimum + (n - 1) * maximum
```

Do not synthesize those graph identities. Import the constant; it pins:

```text
graph_schema     music_critic.graph.raw_schema@1.0.0
                 e0be8d4c522147036418501b230411ac5fc2eafa5284bab44bbc3e6ee3059fc8
graph_builder    music_critic.graph.build_raw_graph@1.0.0
                 ccf423169631d4bb12295b92b4403625902eb1ded9478165f2ebc23d836fab65
feature_registry music_critic.graph.raw_feature_registry@1.0.0
                 a041e2c4a221bc0bc722ff3015423230b9e5d5cf56a6efbc4dc71aab351df6f7
validator        music_critic.graph.validate_raw_graph@1.0.0
                 8de80cbf5929507da727293751aaba723d4256a5bc65aa0309b968873ffafa99
```

Every `SupervisionEDA` must preserve corpus/task/dialect/source-value identity,
versioned vocabulary, target-row availability, and available-only class
support. Use unique-work support only when a versioned work identity is
proven; otherwise emit a `not_applicable` null work count. Keep an available
empty multilabel set separate from missing/masked and from label occurrences.
Rows are unique by `(corpus, source_task_id, dialect, split_scope)`. Repeating
one corpus/task/dialect in TRAIN and VALIDATION requires an identical
annotation namespace, vocabulary identity/version/fingerprint, label
granularity, label value type, observation unit, and optional work identity.
All class-support rows for that task family also retain one unique-work
observation/denominator unit across splits; do not switch between logical and
canonical work.

Scalar class values cannot be null or empty. Standalone non-empty multilabel
set identities canonically order unique, non-empty, stripped string members.
For a MULTI_LABEL task, however, each `ClassSupport` row uses one scalar,
non-empty, stripped vocabulary label; never use a whole-set identity there.
Class support counts must use label-occurrence/target-row, record/record, and
work/work units; share split, evidence, and provenance; and obey
`unique_work <= unique_record <= occurrence` when work support is observed.
Name the three counts exactly `occurrence_count`, `unique_record_count`, and
`unique_work_count`. Each multilabel class occurrence is at most
`available - empty_multilabel_available_count`.
Projection rows bind available native class-support identities; keep all
masked, missing, and unsupported native populations in `AvailabilityCounts`.
Represent `empty_multilabel_available_count` as a target-row-over-target-row
`UnitCount` with the available-row denominator and exact task scopes and
provenance; never pass a bare integer.

## Source extensions

Use `SourceExtension` only for evidence that has no common catalog field. An
allowed extension:

- has a namespace beginning with the canonical corpus ID and declared in
  `adapter.extension_namespaces`;
- has its own schema name and SemVer;
- declares `split_scope`, `evidence_scope`, and non-empty provenance;
- supplies a versioned `work_identity` before observing any logical- or
  canonical-work observation/denominator or emitting a payload work-ID alias;
- uses deterministic unique `row_id` values, each identifying exactly one
  source-native extension metric;
- gives every row mandatory `MetricCoverage` and stores that metric's count
  summary components as `UnitCount`, never as bare payload population counts;
- cannot shadow recursively reserved envelope/wrapper fields, fixed common
  metric IDs, or common task structures, and cannot carry operational keys;
- is `target_free=true` and contains no target-derived field for raw reports.

`coverage` is a required keyword-only constructor argument:

```python
row = ExtensionRow(
    row_id="<stable-source-metric-id>",
    payload=source_native_summary,
    counts=typed_summary_counts,
    coverage=metric_coverage,
)
```

Each row coverage exactly matches the extension split, evidence scope, and
provenance. Every row count is observed and exactly matches that coverage's
denominator, denominator population unit, split, evidence scope, and
provenance. A same-population numerator cannot exceed
`coverage.observed_count`. Non-observed coverage requires empty payload and
counts. Coverage with zero observed rows and a positive unknown remainder has
neither payload nor counts. Fully known-empty coverage also has no payload and
permits only no counts or exact typed zero counts over the zero denominator.
Raw row coverage and every raw row count numerator/denominator use
source-structural raw units only. They never use target-access-attempt,
target-row, label-occurrence, augmented-pair, sampler-presentation, or
optimizer-update units.

Nested namespaced source-schema leaf fields such as `name`, `status`,
`category`, `mean`, `provenance`, `payload`, and `value` remain allowed.
Exact ratio objects, physical measurements, and genuine source-native
probability/weight/confidence summaries such as `confidence_histogram` remain
payload. They cannot be used to disguise population cardinality, frequency,
total, or denominator fields that belong in `UnitCount`. Numeric
probability-, normalized-, and confidence-named containers are summaries only
when every leaf is finite and within `[0, 1]`; out-of-range bucket frequencies
and integer-valued weight histograms must use typed `UnitCount` rows. A
categorical mapping whose leaves are all non-negative integers is also treated
as a count summary unless the field explicitly denotes an identifier/code or
a recognized exact ratio, measurement, physical unit, or MIDI/program value.
If the source exposes confidence/calibration evidence without a common field,
preserve it in its corpus-owned extension rather than dropping it or coercing
it into availability/projection evidence.

The extension scope must sit inside the envelope scope and match its evidence
scope. Unknown/unavailable extensions may contain only explicit non-observed
rows with empty payload/count summaries. A supervision extension is
specifically TRAIN or VALIDATION and cannot contain TEST-named or TEST-selecting
payload fields, including bare plural `tests`/`testsets`, singular/plural
held-out/holdout, and nested `held`/`hold` + `out` aliases.

Raw extension namespace, schema name, extension and row-coverage
provenance/reason, count names/reasons/provenance, row/work identity, and
payload all pass target-free checks.
Supervision TEST-token checks likewise cover manifest/source/task identities,
task dialect/annotation/vocabulary/label-granularity/work/reason fields,
namespace/schema, row/work identity, row coverage, counts, and nested
provenance. Truthful envelope invariant/warning/reason and task/work-count
reason channels are also
checked; only the exact standard `eda.test_targets_locked` reason code is
excepted, while its detail/provenance remain checked. Truthful non-TEST scope
metadata and relative semantic paths remain allowed. One namespace may be
emitted separately for TRAIN and VALIDATION; uniqueness is by
`(namespace, split_scope)`. Across splits, that namespace keeps one schema
name/version, work identity, and `target_free` declaration. A stable `row_id`
keeps one coverage observation unit; all observed instances keep the same
typed-count component names and each component's observation/denominator units.

Payload aliases for work/logical-work/canonical-work ID, identifier, key, UID,
or UUID—including compact, plural, and nested forms—require the extension's
versioned `work_identity`; a filename or unversioned local key is not proof.

Do not use an extension to rename, omit, weaken, or override a common metric,
availability rule, projection rule, TEST lock, or fingerprint field.

## Projection rule

Native evidence is always emitted first. Import
`APPROVED_PROJECTION_REGISTRIES` and use an exact contained
`VersionedIdentity`; do not reconstruct or approximate the registry identity.
At this foundation SHA, the only approved entry is
`music_critic.dilemmadata.common_harmonic@1.0.0` with fingerprint
`bb50920808b6ad3a19fb32b8315a417a837b2ab008efd7bee71e71d120e2ee2e`.
It applies only to Dilemmadata. POP909-CL and HookTheory branches may report
native evidence but must not invent a common projection. PDMX metadata must
never become supervision.

Use `DILEMMADATA_COMMON_TASK_IDS` for the allowed common task set. The
constructor validates the exact source task/common task pair and AN/DLC
dialect, then applies task-specific validation to the mapping state and
projected value. Exact quality/inversion/root/bass transforms and the required
local-key/pitch-class-set shapes are enforced. For AN quality fixtures and
source-native rows, use `major triad` and `minor triad`. Possession of the
registry fingerprint alone is insufficient.

For every approved common task represented, emit a separate
`ProjectionAvailabilityCounts` target-row partition. Populate exactly its
corpus, source task, dialect, registry, common task, observation unit,
denominator, seven state counts, split, evidence scope, and provenance fields.
Its exact, coarsened, ambiguous, unsupported, invalid, missing, and masked
counts sum exactly to the denominator and remain separate from native
`AvailabilityCounts`.
`ProjectionEvidence` is emitted only for available native class-support
identities. Static quality/inversion/root/bass mappings are exact registry
checks. Dynamic local-key and pitch-class-set rows attest approved routing,
state, and output shape, not hidden derivation context.

Bind each projection aggregate to the enclosing task, its native target-row
denominator, split, and evidence scope, while preserving the aggregate's own
non-empty provenance. Its seven state counts remain independent of native
state counts and class-support totals: in particular, projection missing and
masked need not equal native missing and masked when projection depends on
context or another source field. An aggregate row may stand alone. Class rows
are optional; if emitted, each must have a matching registry/common-task
aggregate and an available native class-support identity. Do not emit missing
or masked `ProjectionEvidence` rows.

## Mandatory TEST gate

Before resolving a target descriptor or opening a sidecar, pass the complete
assignment sequence to:

```python
from music_critic.eda import load_supervision_train_validation_only

loaded, test_lock = load_supervision_train_validation_only(
    corpus,
    assignments,
    resolve_descriptor=resolve_descriptor,
    load_target=load_target,
    evidence_scope=report_evidence_scope,
    provenance=report_provenance,
)
```

The assignment projection itself must be target-free. For TRAIN/VALIDATION its
exact allowed field set is `assignment_manifest_fingerprint`, `corpus`,
`record_id`, `split`, and `target_free`; any other field fails before callbacks.
TEST is discarded after reading only `split`. A source regression test must
install descriptor and loader spies that raise on TEST and prove that neither
is invoked. All TRAIN/VALIDATION rows must bind the same assignment-manifest
fingerprint before callbacks. Duplicate retained `(corpus, record_id)` rows
are also rejected before callbacks, including one record assigned to both
TRAIN and VALIDATION; the retained split plan is mutually exclusive. TEST
record IDs are deliberately never read. The emitted
`TestTargetLockEvidence` must retain
five typed TEST counters built by `TestTargetLockEvidence.from_guard(...)`.
`test_assignment_count` observes `split_assignment`; descriptor-resolution and
target-loader-call counts observe `target_access_attempt`; opened target records observe
`record`; and loaded rows observe `target_row`. All five are `UnitCount` values
over the same TEST split-assignment-row count, use
`denominator_unit=split_assignment` and `split_scope=test`, and share report
evidence scope and provenance. The assignment count equals its denominator;
the four access/load counts are observed zero. Keep the false
read/EDA/evaluation/distribution/coverage/co-occurrence flags as well. Do not
add an unlock switch or call a dataset abstraction whose construction opens
TEST targets as a side effect.

The guard and facade default to fixture evidence and provenance
`("supervision-target-access-guard",)` for fixture usage. Production and other
source children must pass their actual report evidence scope and provenance,
as shown above; do not let the fixture defaults leak into a production report.

The guard rejects empty and TEST-only assignment sets. Bind exactly one
target-free `InputManifestRef` with role `split_assignment` and the guard's
`assignment_manifest_fingerprint` in `SupervisionEDA`, plus at least one other
target-bearing input manifest. A manifest-free `unknown` or `unavailable`
non-evidence report is the explicit exception: construct its lock with
`TestTargetLockEvidence.not_executed(...)`, which leaves the manifest
fingerprint null and all five typed counters null/`locked`; do not fabricate
manifests. The lock is a validated attestation, not a
cryptographic ban on unrelated file access; the child adapter must exercise
the guard and prove that path with descriptor and loader spies.
Any extension row with observed coverage also requires an observed guard
attestation, even when the common task rows are all non-observed. An explicit
non-observed empty metric row does not.

## Serializer and fingerprint API

Use only the public API:

```python
from music_critic.eda import (
    canonical_report_bytes,
    dump_report,
    dumps_report,
    load_report,
    loads_report,
    report_dict,
    report_fingerprint,
    report_from_dict,
)
```

`dumps_report()` is canonical JSON without a newline;
`canonical_report_bytes()` has exactly one newline; `dump_report()` is atomic.
Loaders enforce exact fields, all cross-field validators, and the stored
semantic fingerprint. Operational metadata is serialized but excluded from
the fingerprint. Schema identity, repository commit, source/producer/manifest
identities, semantic warnings/reasons, and payload remain inside it. JSON with
duplicate object keys is rejected at every nesting level. Every string and
mapping key must be valid UTF-8 scalar text; lone surrogates fail closed.
Structural identifiers/provenance and mapping keys reject Unicode `Cc`/`Cf`;
opaque category/prose/source-value fields may preserve meaningful interior
tabs/newlines and emoji. A scalar class string remains non-empty and stripped.
Nested extension
payloads and projected values are recursively frozen after validation;
serializers return fresh JSON containers, so a caller cannot mutate a report
behind its stored fingerprint.

## Targeted verification

Run the foundation regressions, then the directly related pre-existing
contracts. Do not run the full repository suite in a source-specific task.

```bash
MUSIC_CRITIC_PYTHON=/home/str/music-critic-v2/.venv/bin/python
PYTHONPATH=src "$MUSIC_CRITIC_PYTHON" -m pytest -q \
  tests/eda \
  tests/data/test_serialization.py \
  tests/data/test_schema.py \
  tests/test_import.py \
  tests/audit/test_multisource_target_audit.py \
  tests/audit/test_phase9eb4_class_balance_audit.py \
  tests/tasks/test_multisource_contract.py
PYTHONPATH=src "$MUSIC_CRITIC_PYTHON" -m pytest -q tests/test_repository_contract.py
"$MUSIC_CRITIC_PYTHON" -m compileall -q \
  src/music_critic/eda \
  src/music_critic/data/serialization.py
git diff --check
```

Add focused adapter tests and synthetic/source-bounded fixtures in the child
branch. A production scan must be an explicit source-task deliverable; never
promote fixture, manifest-replay, or bounded evidence to `production`.

## Source-specific final report format

Each child task must report:

1. exact `<EDA_CONTRACT_SHA>`, child branch, child HEAD, and separate worktree;
2. source/release and adapter identities and fingerprints;
3. capability used (`RawCorpusEDA`, and `SupervisionEDA` only when allowed);
4. evidence scope, execution mode, completeness, split scope, and bound input
   manifests;
5. raw metric coverage with observation units, denominators, observed and
   unknown populations, plus structured unavailable reasons;
6. target-free graph attestation or the reason graph evidence is unavailable;
7. for labeled sources, task/dialect/vocabulary identities, native
   availability partition, typed empty-multilabel count, available-only class
   support, and work-identity status;
8. any projection rows with the exact pre-approved registry binding, reported
   with per-common-task projection availability separately from native
   availability;
9. TEST gate ordering, descriptor/loader spy evidence, and every required
   zero/false lock field;
10. source extension namespaces, per-row metric coverage/count bindings, and
    extension fingerprints;
11. report semantic fingerprints and output paths, using repository-relative
    identities in semantic evidence;
12. exact targeted test commands and results;
13. whether any production scan ran and confirmation that the full suite,
    GPU, training, and unrelated artifact reads did not run;
14. changed files, unresolved issues, exact proposed shared STATUS note, exact
    legacy concepts reused/rejected note, and confirmation that shared
    schema/docs (including STATUS/LEGACY), data, membership, splits, graphs,
    targets, vocabularies, projections, models, training, checkpoints, and C2
    artifacts stayed unchanged.
