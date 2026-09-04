# PDMX source-specific EDA

## Scope and status

`music_critic.adapters.pdmx_eda` implements the raw-only PDMX capability from
foundation commit `65eb32fb948efde0fa117d7d27d19d8f16fa25b4`.  It is a
`manifest_replay`, not the Phase 10 PDMX adapter and not a new bounded or
production scan.  It opens only the tracked compact manifest and the dedicated
`tests/fixtures/pdmx/phase2a1_midi_evidence.json` evidence capsule whose
SHA-256 the manifest pins.

PDMX remains a raw/SSL corpus.  The adapter intentionally has no
`build_supervision_eda()` method; `EDAAdapterRegistry` rejects that operation
before source dispatch.  PDMX metadata, ratings, score annotations, and
notated key signatures are not converted into supervised classes.

## Evidence boundary

The replay binds `tests/fixtures/pdmx/eda_raw_manifest.json` to one immutable,
compact projection of the historical Phase 2A.1 evidence.  The adapter hashes
and reads that capsule at runtime; it does not hash mutable HEAD copies of
`docs/STATUS.md`, `scripts/smoke_midi_adapter.py`, or
`tests/integration/test_real_midi_adapter.py`.

The capsule pins the implementation commit
`32d68e8cb446d9b5dd57bfea1d28b94ccce46274` and its direct closure commit
`7508f96f3a09ddfd15a29c915a8a78beb25eb881`.  It binds the closure commit's
exact 32-line, 1,374-byte STATUS excerpt.  Extraction starts at the exact
inclusive marker

```text
- PDMX root: `/home/str/music-critic-v2/data/pdmx/mid`.
```

and stops immediately before the exact exclusive marker
`## Phase 2A.1 scope confirmation`.  The excerpt SHA-256 is
`91c744a8e176611875e1c55dcc56d6d39e13b8d7d9f72d2c8ae112e0bd431d58`.
The historical runner SHA-256 is
`61d5cfb778f3ba8d26ee8edddfcae7bdf02011e436455538e90c45ca401ddd6b`, and
the strict integration-test SHA-256 is
`109e75dce52bea2a750e09e2bf85626855a359740d8374b2aa2bafac5b15ff1a`.
Those digests name blobs at the pinned implementation commit; they are not
live dependencies on their current working-tree versions.

The capsule also keeps the evidence gaps explicit: the primary raw-run
artifact is not tracked, its external availability is unknown, and the exact
invocation and execution log are untracked.  The counts below therefore remain
a historical repository claim, not a reconstruction of primary run evidence.

The report source identity is explicitly
`pdmx.phase2a1_local_midi_snapshot_evidence@1.0.0`.  Its fingerprint is the
exact manifest-file SHA-256.  This is an identity for the local historical
evidence projection, not an upstream PDMX release identity.

The repository describes PDMX as a public-domain raw symbolic SSL candidate
and cites `https://arxiv.org/abs/2409.10831`, but it does not contain a bound
upstream release commit, release checksum, or license artifact for the local
MIDI tree.  Those fields are therefore `not_computed`, never inferred from the
paper title.  A versioned canonical/logical-work identity is likewise absent.

## Preserved findings

The only corpus-wide fact in the tracked evidence is recursive discovery of
`254,035` MIDI source files in the complete branched tree.  Production
acceptance, parse, conversion, duplicate, leakage, content-distribution, and
graph populations were not computed.

The separate historical deterministic spread diagnostic covered 100 files:

| Measure | Value | Unit |
| --- | ---: | --- |
| attempted | 100 | record |
| converted | 99 | record |
| failed | 1 | record |
| warning occurrences | 378 | event |
| notes in converted records | 47,459 | note |
| tracks in converted records | 246 | track |
| MIDI type 0 | 0 | record |
| MIDI type 1 | 99 | record |
| selected parent directories | 100 | source-tree directory |
| selected path depth | 3..3 | path component |

The sole failure was the already accepted generic-MIDI MVP rejection
`midi_adapter.meter_change_inside_bar` for
`2/31/QmcmH3b8xr1N9KSEu5zS4HG7f6Beq1fENiy3bdZ9D3FXrE.mid`, at tick `8970`
under active meter `75/4`.  The historical triage recorded no unreadable or
corrupt file, MIDI type 2, SMPTE/non-PPQN timing, invalid meter value,
metric-grid safety rejection, canonical validation failure, serialization
round-trip failure, or unexpected exception in that sample.

The extension preserves exact sample ratios—99/100 conversion, 47,459/99
notes, 246/99 tracks, and 378/99 warning occurrences per converted record.
They are SSL/resource scale estimators only.  The adapter does not extrapolate
them into fabricated corpus totals or treat the 100 records as independent
work identities.

## Explicitly unavailable evidence

All common metrics remain present.  Duration, notes/onsets per record,
bars/beats, tracks/parts per record, density, polyphony, pitch range, tempo,
meter, instruments/programs, percussion, empty/invalid/oversize records,
versions, duplicates, split collisions, and graph sizes are `not_computed`
for the 254,035-file population.  Aggregate note/track totals from the bounded
diagnostic live only in the PDMX extension with their 100-record denominator;
they are not numeric distributions.

No graph was built.  `GraphEvidence` and the three graph catalogue rows carry
`eda.target_free_unproven` without graph-contract identities.  No production
manifest/cache/adapter exists yet, no artifact or work identity was inferred
from filenames, and no duplicate/domain-gap scan ran.

## Public use

```python
from pathlib import Path

from music_critic.adapters.pdmx_eda import PDMXEDAAdapter, PDMXRawEDARequest
from music_critic.eda import EDAAdapterRegistry

adapter = PDMXEDAAdapter()
registry = EDAAdapterRegistry()
registry.register(adapter)
report = registry.build_raw(
    "pdmx",
    PDMXRawEDARequest(
        Path("tests/fixtures/pdmx/eda_raw_manifest.json"),
        repository_commit="<current-child-commit>",
    ),
)
```

Serialize only with the shared `dump_report()`/`dumps_report()` API.  Generated
reports are derived outputs and remain outside Git.  Their semantic
fingerprint includes the exact child commit supplied by the caller.  The
request therefore requires `repository_commit` explicitly and never defaults
to the older foundation commit.

## Verification and non-goals

Focused verification is:

```bash
PYTHONPATH=src "$MUSIC_CRITIC_PYTHON" -m pytest -q \
  tests/adapters/test_pdmx_eda.py
```

The immutable foundation and repository-contract selections remain those in
`docs/MULTISOURCE_EDA_HANDOFF.md`.  This source increment does not run a real
PDMX scan, convert MIDI, build graphs, create Phase 10 cache/windows, inspect
large artifacts, use GPU, train SSL, calculate a cross-dataset gap, or run the
full repository suite.  It changes no corpus data, membership, splits,
canonical cache, graph, model, objective, sampler, checkpoint, or common EDA
schema/document.

Future Phase 10 must first pin source and license artifacts, establish
artifact/version/work identities and duplicate-safe splits, then execute a
separately authorized bounded acceptance before a production structural EDA.
