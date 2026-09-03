# POP909-CL source-specific EDA

## Status and evidence boundary

This source adapter implements the common EDA contract rooted at
`65eb32fb948efde0fa117d7d27d19d8f16fa25b4`. It does not rerun the POP909-CL
corpus audit or the 909-file Phase 4B acceptance. It exposes four deliberately
different evidence surfaces:

- `RawCorpusEDA` is a `manifest_replay` of the tracked target-free raw EDA
  projection in `tests/fixtures/pop909_cl/eda_raw_manifest.json`;
- `Pop909ClPhase4EvidenceReplay` validates the exact tracked Phase 4A audit and
  Phase 4B production-manifest bytes, then exposes their pre-split aggregate
  evidence without recasting it as TRAIN/VALIDATION evidence;
- `SupervisionEDA` is `synthetic_fixture` evidence over three retained
  TRAIN/VALIDATION records and one split-only TEST assignment;
- a second formal `SupervisionEDA` uses `unknown`/`not_executed` scope to make
  absent production split-by-class evidence explicit.

The repository has accepted corpus-wide native availability totals, but it has
no tracked TRAIN/VALIDATION class rows. The fixture report therefore tests the
native class/mask/split contract without claiming that its distributions
describe the 909-record release. The unknown report prevents consumers from
treating fixture evidence as a production substitute. No production scan ran
for this increment.

## Source and adapter identities

The source release remains POP909-CL `POP909_processed` at upstream commit
`be9094392903c471a930519e1c0bacf8b6be5d62`, adapter/corpus manifest version
`2.0.0`, and corpus content fingerprint
`b34f07d9a2678abdb6f0dcf5db1c3aec3f35caca813f1fac80c0717cfc8e0c65`.
The EDA producer is `music_critic.adapters.pop909_cl_eda@1.0.0`; its
fingerprint is the SHA-256 of its source module and is recomputed when the
adapter is instantiated.

The source-owned API is:

```python
from music_critic.adapters.pop909_cl_eda import (
    Pop909ClEDAAdapter,
    Pop909ClPhase4EvidenceRequest,
    Pop909ClRawEDARequest,
    Pop909ClSupervisionEDARequest,
    Pop909ClUnavailableSupervisionEDARequest,
    dumps_pop909_cl_phase4_evidence,
    replay_pop909_cl_phase4_evidence,
    validate_pop909_cl_identity_splits,
)
```

`Pop909ClEDAAdapter` is registered through `EDAAdapterRegistry`. The request
types keep their source-local paths/configuration outside the shared schema.
All four request types require an explicit `repository_commit`; there is no
foundation-SHA default and the adapter never derives the current HEAD. This
prevents a newly changed producer from being mislabeled as evidence produced at
the earlier contract commit.
The module is not imported by `music_critic.eda` and has no corpus-discovery or
report-generation import side effect.

## Raw manifest replay

The raw input is a target-free projection with schema
`Pop909ClRawEDAManifest@1.0.0`. It preserves the distinct inventory meanings:

| Measure | Manifest-replay value | Population |
| --- | ---: | --- |
| Discovered logical records | 909 | all logical records |
| Accepted records | 908 | all logical records |
| Quarantined records | 1 | all logical records |
| Quarantined identity | `172` | documented meter case |
| Channel-0 score instruments | 909 | all logical records |
| Raw duplicate candidates | 2 | 908 accepted records |
| Raw-equivalence groups | 907 | 908 accepted records |
| Cross-split raw-identity collisions | 0 | 908 accepted assignments |
| Source-group split collisions | 0 | 908 accepted assignments |
| Lineage split collisions | 0 | 908 accepted assignments |
| TRAIN assignments | 701 | 908 accepted assignments |
| VALIDATION assignments | 101 | 908 accepted assignments |
| TEST assignments | 106 | 908 accepted assignments |

`discovered_records = accepted_records + quarantined_records` is enforced by
the common schema. The conversion distribution is 908 accepted and one
quarantined; the one reason row is
`midi_adapter.meter_change_inside_bar`.

Every common raw metric is present. The manifest can soundly replay inventory,
conversion outcome, channel-0 instrument, duplicate, reason, and accepted
split-collision rows. Numeric/music-content distributions absent from the
tracked aggregate projection remain `not_computed` with a known 909-record
denominator; they are not reported as zero.

The raw extension namespace is `pop909_cl.raw_manifest`, schema
`Pop909ClRawManifestExtension@1.0.0`. It uses typed counts for:

- 909 type-1 MIDI records at PPQN 480;
- 909 empty conductor/meta tracks;
- 909 tempo events, 911 meter events, and 1,065 key-signature events, all from
  conductor track 0;
- 910 AppleDouble installation-noise files over 1,819 installation files;
- 126,163 score warning events, including 123,439 same-pitch-overlap warning
  events, with the four exact warning-code occurrence/affected-record counts,
  908 observed and one unknown record;
- the two records in the sole non-trivial raw-equivalence cluster;
- zero source-group and zero lineage split collisions;
- the exact 701/101/106 accepted-record assignment partition.

The documented song-172 meter offsets, 600 and 480 ticks, are source-native
measurement payload, not population counts. The high warning number is an
event total and must not be interpreted as a failed-record count.

The tracked target-free audit also establishes a partial score-note
distribution over 908 converted records: minimum 175, median 1,655, p95 2,403,
and maximum 4,233, with the quarantined record unknown. The common `notes`
metric requires an arithmetic mean, which the tracked evidence does not retain,
so the formal common row remains `not_computed`; the known partial summary is
preserved by the source extension and Phase 4 replay and is not padded with an
invented mean.

The score-only warning distribution is likewise partial but exact: minimum 3,
median 123, p95 282, and maximum 966 warnings over the 908 converted records;
its arithmetic mean is not reconstructed. Unsafe complete-file diagnostics
that mix channel-1 chord notes into the score are deliberately excluded from
raw statistics.

## Identity and leakage result

Source records `piece:pop909-cl-543` and `piece:pop909-cl-553` retain distinct
record identities but share
`pop909-cl-score:4585134e3f7a70c105a3bb678a04ab2bc4522c04e11183f6fd6c59046be25286`.
The accepted joint split places both in TRAIN. The replay binds the accepted
split-manifest semantic fingerprint
`b0546316acb225bb95439dab78fab95232b0a7a758316b69b85dc87f733c384d`.

The source helper checks record identity, source-group identity, lineage, and
optional canonical-work identity. These identities close transitively before
split comparison, so alternate record IDs cannot evade leakage checking by
bridging through different identity families. The accepted source-group and
lineage checks pass. A canonical-work ontology is not established by the
tracked evidence, so canonical-work identity and unique-work support remain
explicitly `not_computed`/`not_applicable`; neither is inferred from a
filename, numeric song ID, or lineage label.

## Graph evidence

The accepted Phase 4B checks prove raw equality under hidden annotation and
validate individual graphs, but the target-free EDA projection has no complete
node/edge/graph-size aggregate. `GraphEvidence` and all three coupled graph
metrics therefore remain `not_computed` with
`eda.target_free_unproven`. No graph contract identity or zero distribution is
fabricated, and no graph was built for this task.

## Validated Phase 4 aggregate replay

`replay_pop909_cl_phase4_evidence()` accepts the tracked raw projection plus
the historical Phase 4A audit and Phase 4B production manifest. It first
checks the complete audit and production file SHA-256 values:

- Phase 4A audit:
  `46e7254f8a451f64a009d54cceec5a16703eb3ca80b88984127a643c73f9105a`;
- Phase 4B production manifest:
  `bc9c4118c72cb39bc1393fd2d250db577835a837fc53f8fe8c1238c7b13a8031`.

It then cross-checks the release fingerprint, upstream commit, inventory,
instrument, block, normalization, gap, mask, and structural totals between the
manifests. Byte drift fails closed before any value is replayed.

The replay distinguishes 909 logical source records from a canonical-work
ontology. There are 908 accepted records; `172` remains the only quarantined
record and is not reconstructed. There are 907 source records with a chord
instrument, of which 906 are accepted canonical records with chord evidence;
`367` and `658` are accepted with missing target instruments. Canonical-work
identity remains `not_computed`.

The pre-split source target-row partition is:

| Task | Denominator | Available | Masked | Missing | Unsupported |
| --- | ---: | ---: | ---: | ---: | ---: |
| bass | 116,055 | 116,055 | 0 | 0 | 0 |
| boundary | 116,055 | 116,055 | 0 | 0 | 0 |
| inversion | 116,055 | 109,668 | 5,801 | 0 | 586 |
| no-chord | 1,098 | 947 | 151 | 0 | 0 |
| quality | 116,055 | 109,800 | 5,669 | 0 | 586 |
| root | 116,055 | 109,668 | 5,801 | 0 | 586 |

These are target-row denominators, so the two records with no target instrument
are not invented as pseudo-blocks. At record unit the accepted population is
908: boundary and bass have exact available support in 906 records and the two
missing-instrument records are `367` and `658`. Per-task available record
support for root, quality, inversion, and no-chord is absent from tracked
evidence and remains `not_computed`; the replay does not infer it from row
totals. The historical all-corpus cohort has 907 records with chord blocks,
including quarantined `172`, while the accepted cohort has 906.

For the five chord-block families, the denominator is 116,055 exact source
blocks. Ambiguous root and inversion rows are masked; unsupported shapes remain
unsupported. Quality is
available for 132 ambiguous blocks whose candidates agree, leaving 5,669
ambiguous quality rows masked. The no-chord denominator is 947 positive
leading/internal `N` spans plus 151 masked trailing uncovered spans. This
partition does not create a negative `not-N` class.

The 151 trailing uncovered durations retain minimum 1, median 401, p95 3,361,
and maximum 12,861 ticks. The replay also preserves 691 overlapping blocks, 87
repeated-pitch blocks, 313 mixed-end blocks, zero duplicate onsets, and eight
exact pairing anomalies. The 909-record block-count distribution is 0/124/185/
278 for minimum/median/p95/maximum. It records 261 raw pitch-class sets and 340
selected root/quality/bass labels, but no frequency map is retained.

These are historical all-corpus source-evidence counts. They include channel-1
evidence from quarantined `172` and therefore are not a claim of accepted
training-row support. The tracked manifests do not retain per-class occurrence
maps by split, per-class unique-record support, full task co-occurrence, or
TRAIN/VALIDATION distributions. Class concentration, record support,
co-occurrence, and TRAIN/VALIDATION shift consequently remain explicit
`not_computed`, not zero. No canonical-work or per-class work support is
inferred from numeric song IDs, source groups, or lineage.

## Native supervision fixture

The supervision fixture binds target ontology `1.0.1`, fingerprint
`86ea17b016eafb7109fe050f9332c57f8e0f3399046debc01f4d8ac5d19d9613`.
It emits separate TRAIN and VALIDATION rows for all six source-native families:

- `pop909_cl.chord.bass`;
- `pop909_cl.chord.boundary`;
- `pop909_cl.chord.inversion`;
- `pop909_cl.chord.no_chord`;
- `pop909_cl.chord.quality`;
- `pop909_cl.chord.root`.

Every task uses its exact registry vocabulary and granularity. Availability is
partitioned into available, masked, missing, and unsupported target rows.
Class support consumes available rows only and retains label-occurrence and
unique-record denominators. A masked root/inversion, missing no-chord row, and
unsupported quality row produce no class. Boundary `present` and no-chord `N`
remain positive-unlabeled source-native families. No common projection,
crosswalk, absent class, or negative `not-N` class is emitted.

The fixture is complete only relative to its three retained records. It is not
evidence for the accepted corpus-wide availability totals, class imbalance,
or production co-occurrence. Those require an explicitly authorized future
TRAIN/VALIDATION manifest replay or scan.

## Formal unavailable production supervision

`Pop909ClUnavailableSupervisionEDARequest` produces twelve non-observed task
rows: all six families for TRAIN and VALIDATION. Its envelope is
`evidence_scope=unknown`, `execution_mode=not_executed`, and
`completeness_status=unknown`; it has no manifests, availability rows, class
support, projections, or work identity. Structured reasons separately record
the unavailable split rows, class concentration/record support, co-occurrence,
TRAIN/VALIDATION shift, and canonical-work identity.

Its split lock is built with `TestTargetLockEvidence.not_executed()`. Every
counter is null and `locked`; none is presented as an observed zero. This
report is the production-facing answer until a future authorized split-aware
target replay exists.

## TEST lock

The split file contains two TRAIN rows, one VALIDATION row, and one TEST row
whose complete content is only `{"split": "test"}`. The adapter validates
record/source-group/lineage identity rows, projects retained assignments to the
shared exact allowlist, and then calls
`load_supervision_train_validation_only()` before either callback.
Only the target-free split/identity manifest is opened before that call. The
target-bearing fixture is opened lazily inside the first allowed `load_target`
callback, after complete guard preflight and descriptor resolution, then cached
for the remaining TRAIN/VALIDATION records. Its exact SHA-256 is
`6babf2150d4f3799dd5201af3e649e7e3eae33c7f08eb70f350ee27cb4f2318e`;
schema, ontology binding, record inventory, and bytes all fail closed there.

The emitted lock has one TEST split-assignment row, zero descriptor calls,
zero loader calls, zero opened records, zero loaded target rows, and all six
read/use/distribution/coverage/co-occurrence flags false. Regression spies
raise on any TEST callback and observe only the three retained records. The
target-bearing fixture contains no TEST record or TEST supervision.

## Manifests, serialization, and fingerprints

Tracked source-specific inputs are:

| Input | Role | Scope |
| --- | --- | --- |
| `tests/fixtures/pop909_cl/eda_raw_manifest.json` | `raw_projection` | manifest replay, target-free |
| `tests/fixtures/pop909_cl/eda_split_assignments.json` | `split_assignment` | synthetic, target-free |
| `tests/fixtures/pop909_cl/eda_supervision_fixture.json` | `fixture_rows` | synthetic, target-bearing TRAIN/VALIDATION only |
| `tests/fixtures/pop909_cl/audit_manifest.json` | Phase 4A aggregate replay | historical pre-split mixed evidence |
| `tests/fixtures/pop909_cl/production_manifest.json` | Phase 4B aggregate replay | historical pre-split mixed evidence |

The adapter computes file identities from exact bytes and lets the shared
constructors compute extension/report semantic fingerprints. Reports are
serialized only through `dump_report`/`dumps_report`; generated JSON is not
tracked. Because repository commit is semantic evidence, a report created with
the child HEAD intentionally has a different semantic fingerprint from the
same request bound to the foundation SHA.

Example:

```bash
: "${REPOSITORY_COMMIT:?set the exact 40- or 64-hex evidence commit}"
PYTHONPATH=src .venv/bin/python - "$REPOSITORY_COMMIT" <<'PY'
import sys
from pathlib import Path
from music_critic.adapters.pop909_cl_eda import (
    Pop909ClEDAAdapter,
    Pop909ClPhase4EvidenceRequest,
    Pop909ClRawEDARequest,
    Pop909ClSupervisionEDARequest,
    Pop909ClUnavailableSupervisionEDARequest,
    dumps_pop909_cl_phase4_evidence,
    replay_pop909_cl_phase4_evidence,
)
from music_critic.eda import EDAAdapterRegistry, dump_report

root = Path.cwd()
repository_commit = sys.argv[1]
adapter = Pop909ClEDAAdapter()
registry = EDAAdapterRegistry()
registry.register(adapter)
raw = registry.build_raw(
    "pop909_cl",
    Pop909ClRawEDARequest(
        root / "tests/fixtures/pop909_cl/eda_raw_manifest.json",
        repository_commit=repository_commit,
    ),
)
supervision = registry.build_supervision(
    "pop909_cl",
    Pop909ClSupervisionEDARequest(
        root / "tests/fixtures/pop909_cl/eda_split_assignments.json",
        root / "tests/fixtures/pop909_cl/eda_supervision_fixture.json",
        repository_commit=repository_commit,
    ),
)
unavailable_supervision = registry.build_supervision(
    "pop909_cl",
    Pop909ClUnavailableSupervisionEDARequest(
        repository_commit=repository_commit,
    ),
)
phase4 = replay_pop909_cl_phase4_evidence(
    Pop909ClPhase4EvidenceRequest(
        root / "tests/fixtures/pop909_cl/eda_raw_manifest.json",
        root / "tests/fixtures/pop909_cl/audit_manifest.json",
        root / "tests/fixtures/pop909_cl/production_manifest.json",
        repository_commit=repository_commit,
    )
)
dump_report(raw, "/tmp/pop909_cl_raw_eda.json")
dump_report(supervision, "/tmp/pop909_cl_supervision_eda.json")
dump_report(
    unavailable_supervision,
    "/tmp/pop909_cl_unavailable_supervision_eda.json",
)
Path("/tmp/pop909_cl_phase4_evidence.json").write_text(
    dumps_pop909_cl_phase4_evidence(phase4, indent=2),
    encoding="utf-8",
)
print(phase4.semantic_fingerprint)
PY
```

## Verification and immutable boundaries

Source-focused verification:

```bash
PYTHONPATH=src .venv/bin/python -m pytest -q \
  tests/adapters/test_pop909_cl_eda.py
```

The full foundation selection and repository-contract command are those in
`docs/MULTISOURCE_EDA_HANDOFF.md`. This task does not run the full repository
suite.

No real corpus traversal, source checksum pass, MIDI conversion, graph build,
production acceptance, GPU operation, training, or checkpoint/large-artifact
read ran here. The shared EDA schema/serialization/access guard,
common docs and fixtures, POP909-CL source data, corpus membership, splits,
graphs, targets, vocabularies, crosswalks, models, training state, checkpoints,
and C2 artifacts remain unchanged.
