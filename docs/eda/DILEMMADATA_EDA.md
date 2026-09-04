# Dilemmadata source-specific EDA

## Scope

`music_critic.adapters.dilemmadata_eda` implements the immutable
`RawCorpusEDA@1.0.0` and `SupervisionEDA@1.0.0` contracts from foundation
commit `65eb32fb948efde0fa117d7d27d19d8f16fa25b4`. It changes no common EDA
schema, serializer, access guard, corpus data, split, target, graph, model,
sampler, or training artifact.

Both reports replay committed evidence and are intentionally `partial`:

| Report | Evidence scope | Execution mode | Split scope |
| --- | --- | --- | --- |
| raw | `manifest_replay` | `manifest_replay` | `all` |
| supervision | `manifest_replay` | `manifest_replay` | `train_validation` |

This task did not traverse the production corpus, open a raw MIDI source,
materialize graphs, run training, or access TEST targets. Corpus-scale facts
come from already tracked B3/B4/B5 evidence. Nine guarded TRAIN/VALIDATION
surface cases exercise source-native identity, availability, and projection;
they are not presented as a corpus distribution.

## Identities and bound evidence

The source is `johentsch.dilemmadata.release@1.0.0`, fingerprint
`8f1161ad7cdbd979845012ffc6150cd82c5e91ab1197ed97385fffce57a0f312`,
at upstream release commit `d60ee75b4a9495e932a4a7be39381578be17e222`.
The adapter is `music_critic.adapters.dilemmadata_eda@1.0.1`, fingerprint
`efc89198d4a2e644e746ea7fe173ce60ae7ab7b9b646cca75396a76c78ec96c4`.

Raw replay verifies exact SHA-256 bindings for:

- `tests/fixtures/dilemmadata/audit_manifest.json` —
  `c321a75064abec81e1357690c256e16a16af2eb8d4a3e50e3cbb624b2c3d52aa`;
- `tests/fixtures/dilemmadata/production_manifest.json` —
  `58e0c28ea7f88dcea3d3e1e453b11d1cfd44635dcc8af4b91f04250c026bcbac`;
- `tests/fixtures/dilemmadata/eda_population_manifest.json` —
  `62b9e87eea6e1c4f6bd3612b8457e4c20834e986b8a48b822e2f2644f9c2047b`.

Supervision replay binds the target-free assignment projection at semantic
fingerprint
`17a9191d6260fb100548164d39bf95773eff44e58f8693c6e7d73412676abaa9`
and verifies these exact file hashes only after the common pre-open TEST gate:

| Evidence | SHA-256 |
| --- | --- |
| guarded surface cases | `ff28fa99b6a577cc970e75b07d9196e1bb7f17a0b3db5b065d6211aaf25b86a7` |
| native-family manifest | `f13a8017ee6d3618a9a177c387618c9e1631abdab01bbecf044f2eeb45ac0318` |
| common-harmonic manifest | `ef899a8a8f77f5387d0e952b1eaf94fa32cf2a9dea494bfcb35b4729f1cb40e2` |
| B3 corrected multitask | `a32c0dbf6d9a6c55da31a1296a736101d5cb2408f0a9d75d8a42007d7223f806` |
| B4 class balance | `fb9b41a0e6c985f9753f5609374bea0b168a9b131f21acd5cb9e8d343fb359c1` |
| B5A safe transposition | `3d6625381f170d1419bb0cacf1f6bb6f8c21bb641a11c8d0adc299f1803d734d` |
| B5B training policy | `12b8b812e138af45e0ca2f7926bf2de3a53872368f165b3ead24b6aa9140dd34` |
| B5E observed runs | `1d573a158666a9b258641a80a44b803b68bcffa514ccc5abbf38d84553097470` |
| B5H historical full-orbit planning snapshot | `69fddbf6aab4c1e49940343463cdeada05eefd2fee3aef0b5e487cda7cdaf74d` |

The native and common manifests are used only to validate the frozen family
and approved-registry bindings. Their historical aggregate TEST target facts
are not emitted. B3/B4/B5 evidence is accepted only with its explicit
zero-TEST-access locks and consistent semantic-fingerprint lineage.
The B5H input role is `historical_full_orbit_planning_snapshot`, bound to
snapshot evidence fingerprint
`28a77c929c9e5b006ce6b37d226428814cf503bcc06e15626aa52d4756c25df6`.
It does not supply the current C2 run-state.

## Three populations remain separate

The report never treats the 719-record subset as the corrected multitask
population:

| Population | AN | DLC | Total | TRAIN/VALIDATION/TEST records | TRAIN/VALIDATION/TEST components |
| --- | ---: | ---: | ---: | --- | --- |
| full raw inventory | 353 | 1,280 | 1,633 | not reassigned here | 1,507 total |
| corrected paper candidate | 353 | 1,266 | 1,619 | 1,295 / 162 / 162 | 1,209 / 147 / 151 |
| common-projection subset | 108 | 611 | 719 | 577 / 71 / 71 | 565 / 71 / 71 |

The corrected population excludes 14 raw DLC records without the required
paper-candidate metadata. The older 719-record population is the exact subset
bound to the approved common-harmonic projection. Both component populations
use explicit versioned component identity; no filename is promoted to a
canonical work ID.

The common raw catalog still reports 1,633 discovered, 719 accepted, and 914
quarantined records because those are the source adapter's conversion
outcomes. Source extensions carry the separate corrected and common-subset
universes. All 32 raw catalog rows exist. Per-record numeric distributions and
an approved target-free graph attestation are absent from the compact
manifests, so they remain `not_computed`, never fabricated as zero.

## Native supervision and projection semantics

All 22 source-native families are represented for TRAIN and VALIDATION: nine
AN and thirteen DLC families, for 44 task rows. The guarded surface replay has
split distributions for dialect-specific chord quality and inversion. The
other 18 families per split remain explicit `not_computed` rows because no
tracked split-native distribution can be emitted without reading TEST or
rerunning the audit.

TRAIN surface availability is:

| Dialect/task | Available | Masked | Missing | Unsupported |
| --- | ---: | ---: | ---: | ---: |
| AN quality | 3 | 0 | 0 | 0 |
| AN inversion | 2 | 1 | 0 | 0 |
| DLC quality | 3 | 0 | 0 | 0 |
| DLC inversion | 3 | 0 | 0 | 0 |

VALIDATION has one fully available AN case. DLC has one fully available row,
one missing quality row, and one masked inversion row. Available-only class
support keeps label-occurrence, record, and proven canonical-work units
separate.

AN inversion `2`, DLC inversion `2`, and DLC surface spelling `42` retain
three distinct `SourceValueIdentity` values. The corrected interpretation is:

- AN `2` maps exactly to second inversion;
- DLC native `2` maps exactly to third inversion;
- DLC surface `42` first normalizes to the frozen native value `2`, then maps
  exactly to third inversion.

Thus `42` is not left unsupported and is not injected as a seventh DLC native
class. Optional common rows use only the approved
`music_critic.dilemmadata.common_harmonic@1.0.0` registry, fingerprint
`bb50920808b6ad3a19fb32b8315a417a837b2ab008efd7bee71e71d120e2ee2e`.
Native evidence remains primary.

## Corpus-scale B3/B4 findings

Corrected five-component harmonic co-occurrence has 98,715 TRAIN events from
1,170 records and 1,091 components, versus 10,507 VALIDATION events from 149
records and 134 components. The paper-compatibility view is deliberately
separate: 187,548/20,465 note rows correspond to 98,438/10,477 canonical
harmonic rows. Event, note, record, canonical-source-row, and component units
are never summed together.

Corrected quality support preserves entity, canonical source-row, record,
component, dialect, and effective-component evidence. Examples:

| Class/split | Events/source rows | Records (AN/DLC) | Components (AN/DLC) | Effective components |
| --- | ---: | --- | --- | ---: |
| augmented major tetrachord TRAIN | 145 | 24 (10/14) | 24 (10/14) | 16.5943 |
| augmented major tetrachord VALIDATION | 9 | 4 (3/1) | 4 (3/1) | 3.8571 |
| augmented seventh chord TRAIN | 245 | 40 (6/34) | 38 (6/34) | 16.8373 |
| augmented seventh chord VALIDATION | 77 | 6 (1/5) | 6 (1/5) | 3.0296 |
| augmented triad TRAIN | 2,403 | 238 (67/171) | 213 (64/171) | 76.9603 |
| augmented triad VALIDATION | 345 | 32 (13/19) | 29 (13/19) | 16.1040 |

All 20 corrected heads retain exact TRAIN/VALIDATION observed-vocabulary
counts and TRAIN concentration summaries. Roman-numeral evidence makes the
shift explicit: 178 of 184 classes occur in TRAIN, 113 occur in VALIDATION,
and `vii%9`, `N+7`, `bV+7`, and `#v7` occur in VALIDATION but not TRAIN.
Head roles are advisory training policy, not data truth: eight primary, ten
auxiliary, and two deferred (`phrase`, `section`) because negative
supervision is missing.

## B5 augmentation and exposure

For the 1,295 TRAIN records, 1,231 support all 12 safe shifts and 64 support
2–11. The tracked transform audit contains 6,408 mapping rows: 5,956 valid and
452 invalid. The complete orbit has 15,540 nominal record/shift pairs, 15,389
eligible pairs, 151 exclusions, and 1,295 identity pairs.

Augmented variants are sampler presentations, not independent musical works.
They preserve source component/work identity and cannot inflate split or
class-support work counts. VALIDATION remains identity-only.

Configured and observed exposure is explicit for both presentations and
optimizer updates. C0/C1 are observed B5E results. C2 is deliberately labeled
as the historical Phase 9E-B5H planning snapshot rather than current execution
evidence:

| Profile | Presentations at cited snapshot | Updates at cited snapshot | Evidence scope |
| --- | ---: | ---: | --- |
| C0 | 20,000 configured / 20,000 observed | 10,000 configured / 10,000 observed | completed B5E seed-17 screen; selected corrected baseline |
| C1 | 20,000 configured / 20,000 observed | 10,000 configured / 10,000 observed | completed B5E seed-17 screen; experimentally deferred |
| C2 | 240,000 configured / 0 observed | 120,000 configured / 0 observed | historical B5H planning snapshot; `configured_untrained` at that snapshot |

Both C2 rows carry `snapshot_phase=9E-B5H`,
`run_state_scope=historical_b5h_planning_snapshot`,
`current_run_state_included=false`, and the pinned snapshot evidence
fingerprint. Their zero-valued components end in `_at_b5h_snapshot`; they are
not current counters. The structured warning
`dilemmadata.b5h_historical_planning_snapshot` points to
`docs/EXPERIMENT_LEDGER.md`, which is the sole source of current experiment
run-state.

The official O profile remains only a partial, non-runnable reproduction
contract; it is not substituted with corrected V2 data.

## TEST lock and leakage boundary

The target-free split projection exposes 162 identity-redacted TEST
assignments. It contains no TEST record ID or path. The shared gate sees all
162 split tokens before descriptor resolution. Its five counters report 162
assignments and zero TEST descriptor calls, loader calls, records opened, and
target rows loaded; read, EDA, evaluation, distribution, coverage, and
co-occurrence flags all remain false.

The split projection itself is bound by file SHA-256
`dd34263ec9dde70a134a6b987114e4d8db027cc27d744cb91179672caa958ea3`.
Before any callback or target-bearing open, parsing requires the exact frozen
top-level and assignment field sets, literal fingerprint policy, exact JSON
container/scalar types, integer lock count, and self-consistent repeated
fingerprints. Extra or missing fields, policy drift, file drift, and bool or
float substitutions fail closed.

Descriptor and loader spies prove that callbacks receive only the nine
TRAIN/VALIDATION surface cases. Target-bearing tracked evidence is verified
only after that gate. The surface replay additionally checks split atomicity
for canonical work, source group, and every lineage identity.

## Verification and non-goals

Focused verification is:

```bash
PYTHONPATH=src /private/tmp/music-critic-v2-eda-venv/bin/python -m pytest -q \
  tests/adapters/test_dilemmadata_eda.py
```

The immutable handoff selection and repository-contract test are run exactly
as prescribed by `docs/MULTISOURCE_EDA_HANDOFF.md`, followed by `compileall`
and `git diff --check`. Generated JSON reports are derived outputs and are not
committed; their semantic fingerprints include the final repository commit.

Non-goals are a production/raw scan, TEST target access, a graph build,
training or GPU/C2 execution, corpus/split mutation, target or vocabulary
changes, common-projection changes, cache/checkpoint/output commits, and any
claim that replayed evidence is a fresh production audit.
