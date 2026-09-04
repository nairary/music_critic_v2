# HookTheory source-specific EDA

Status: source-specific Phase 2B bounded evidence for the frozen
`RawCorpusEDA@1.0.0` and `SupervisionEDA@1.0.0` contracts. This implementation
does not scan the production corpus and does not change a shared EDA schema,
dataset membership, split, target, vocabulary, graph, or model contract.

## Identities and evidence boundary

The adapter is `music_critic.adapters.hooktheory_eda@1.0.0`, fingerprint
`58562fe45afbbfb2890dc28e5c2be4e8396157da4c45d63ddf0e8c6c9e80f26c`.
It binds the audited raw source release
`map.hooktheory.raw_release@1.0.0`, fingerprint
`8ab601050d0b8c8752c3b6bf190d63edefa5fce07735ce823bca6a3922dff833`.
The adapter is outside `music_critic.eda` and registers both the raw and
supervision capabilities through `EDAAdapterRegistry`.

Bounded evidence is intentionally distinct from production status:

- raw evidence covers the 19 tracked Phase 2B golden cases; 18 cases have a
  convertible excerpt and one missing-payload case is quarantined;
- the target-free raw split inventory contains 18 TRAIN cases and one TEST
  case;
- supervision evidence covers only the 17 convertible TRAIN excerpts;
- the TEST assignment is split-only: it contains no record ID, descriptor, or
  target path;
- no tracked machine-readable production availability/split table is bound to
  this adapter. Production requests therefore return `unknown` scope,
  `not_executed` mode, null populations, and no fabricated zeros.

The bounded inputs and their exact SHA-256 bindings are:

| Role | Repository path | SHA-256 | Target-free |
| --- | --- | --- | ---: |
| raw summary | `tests/fixtures/hooktheory/eda_raw_bounded_manifest.json` | `3b7246de6662ded92a177ebc7530506875c82d2ec585c926a87861baf60cdafb` | yes |
| split assignment | `tests/fixtures/hooktheory/eda_split_assignments.json` | `7302b00e5b946d9b021f484da9276dfba8bb87672db32297c592f6c48db200be` | yes |
| bounded rows | `tests/fixtures/hooktheory/eda_supervision_manifest.json` | `2ebf774e712b54ba5b2cec763dee3de7b624103f4d27d8e4266490e8d64e5745` | no |

All three manifests and every retained case file are fingerprint-bound and
fail closed on drift. The production source files named in the historical
field audit are not opened by this implementation.

## Raw report

The bounded raw report has `bounded` evidence scope, `bounded_scan` execution,
`partial` completeness, and `all` split scope. Inventory counts are 19
discovered, 18 accepted, one quarantined/invalid missing-payload record, nine
converted excerpts with no emitted melody note, and zero oversize records.
Structural distributions cover the 18 converted excerpts with a 19-record
denominator and one unknown. They include duration, notes, onsets, bars, beats,
tracks, parts, density, polyphony, meter changes, and tempo changes. Meter,
instrument, program, and percussion categories retain their own typed units.

Pitch range, tempo distribution, duplicate candidates, version candidates,
cross-split raw-identity collisions, and all three graph metrics are explicitly
`not_computed`. No exact graph-build attestation exists for the bounded raw
summary, so `GraphEvidence.target_free` remains null; graph values are not
inferred from converted pieces.

The source extension `hooktheory.raw_cases` carries only the typed 18 TRAIN / 1
TEST case inventory. It is target-free and does not expose case IDs or source
annotations.

## Supervision report

The source-native dialect is `hooktheory.theorytab`. The report contains one
TRAIN row and one explicit, non-computed VALIDATION row for each of 12 native
families:

| Source task | Value semantics | TRAIN availability A/M/missing/U | Available empty sets |
| --- | --- | ---: | ---: |
| `theory.melody.scale_degree` | categorical, closed | 8/0/0/0 | — |
| `theory.local_key.tonic_pc` | categorical, closed | 20/0/0/0 | — |
| `theory.local_key.mode` | categorical, open | 20/0/0/0 | — |
| `theory.chord.presence` | categorical, closed | 17/0/0/0 | — |
| `theory.chord.root_degree` | categorical, closed | 15/1/0/1 | — |
| `theory.chord.extent` | categorical, closed | 16/1/0/0 | — |
| `theory.chord.inversion` | categorical, closed | 16/1/0/0 | — |
| `theory.chord.adds` | multilabel, closed | 16/1/0/0 | 15 |
| `theory.chord.omits` | multilabel, closed | 16/1/0/0 | 15 |
| `theory.chord.alterations` | multilabel, closed | 16/1/0/0 | 14 |
| `theory.chord.suspensions` | multilabel, closed | 16/1/0/0 | 14 |
| `theory.chord.borrowed` | categorical, open | 16/1/0/0 | — |

Here A is available, M is masked, and U is unsupported. Missing, masked,
unsupported, and available empty multilabel sets remain separate states.
Multilabel class support uses one scalar label occurrence per row rather than
whole-set identities. Open values are preserved verbatim, including the mode
`dorian`, borrowed mode/pitch-class-set forms, and `unknown:super:2`.

Source-native identity is primary. This report deliberately emits no
Dilemmadata or POP909-CL crosswalk/projection rows. Partial `ori_uid` evidence
does not prove a complete logical-work identity, so unique-work support is
null and `not_applicable`, never substituted with record counts.

## TEST target lock

The full target-free split assignment inventory is passed to
`load_supervision_train_validation_only` before any descriptor resolution or
target open. Only the 17 TRAIN descriptors/loaders run. For the one TEST
assignment, descriptor resolutions, loader calls, opened records, and loaded
target rows are all typed zero; every TEST-use/distribution/coverage flag is
false. The supervision manifest and case files are opened lazily only from the
allowed loader callback.

## Construction

Every request requires the checked-out commit explicitly. There is no default:
silently binding a report to the older foundation commit would claim code that
did not exist at that revision. Pass the exact commit when producing reports:

```python
from pathlib import Path

from music_critic.adapters.hooktheory_eda import (
    HookTheoryEDAAdapter,
    HookTheoryProductionStatusEDARequest,
    HookTheoryRawEDARequest,
    HookTheorySupervisionEDARequest,
)
from music_critic.eda import EDAAdapterRegistry, dump_report

root = Path("/path/to/music-critic-v2")
commit = "<checked-out-40-hex-commit>"
registry = EDAAdapterRegistry()
adapter = HookTheoryEDAAdapter()
registry.register(adapter)
raw = registry.build_raw(
    adapter.corpus,
    HookTheoryRawEDARequest(root, repository_commit=commit),
)
supervision = registry.build_supervision(
    adapter.corpus,
    HookTheorySupervisionEDARequest(root, repository_commit=commit),
)
dump_report(raw, "/tmp/hooktheory_raw_bounded_eda.json")
dump_report(supervision, "/tmp/hooktheory_supervision_bounded_eda.json")

production_status = HookTheoryProductionStatusEDARequest(
    repository_commit=commit
)
```

`HookTheoryProductionStatusEDARequest` builds separate truthful non-evidence
reports without touching local corpus files. It is not a production scan.
