# WS1 raw input correctness — Phase 9E / Phase 2B

This opt-in input correction is separate from frozen C0/C1/C2 experiment
protocols. Synthetic CPU tests establish implementation mechanics only. They
do not measure the effect of either issue on a trained model, corpus readiness,
augmentation benefit, or musical quality.

## Complete transposition

Historical `transpose_raw_graph_view` and
`transpose_raw_graph_view_directed` in `analysisgnn/transposition.py` remain
unchanged, including their tensor behavior, contract fingerprints and runtime
consumers. They update note pitch/class/octave and relative pitch but retain
the original track mean/min/max. A regression intentionally reproduces this
disagreement with a canonical rebuild. The old B5F tritone inverse diagnostic
and the already implemented B5G directed inverse are preserved historical
evidence; WS1 does not rediscover or replace B5G.

The separate `analysisgnn/transposition_inputs.py` exposes:

```python
from music_critic.experiments.analysisgnn.transposition import (
    canonical_directed_transposition,
)
from music_critic.experiments.analysisgnn.transposition_inputs import (
    complete_transposition_contract,
    transpose_complete_raw_graph_view,
)

contract = complete_transposition_contract()
forward = canonical_directed_transposition(6)  # (6, +6)
view = transpose_complete_raw_graph_view(raw_graph, transform=forward)
restored = transpose_complete_raw_graph_view(view, transform=forward.inverse())
```

`CompleteRawGraphTransposition@1.0.0` binds the complete current raw feature
registry, graph schema/builder versions, directed token schema, field roles,
availability and range policies into a deterministic SHA-256. Future consumers
must record this fingerprint and the directed token with their own view and
experiment identity. Graph metadata remains the existing strict raw schema;
no semantic/protocol annotation is added to model input. No current trainer
or frozen profile opts in automatically.

The accepted input is one validated, target-free raw graph using the default
registry, not a PyG batch or a graph with target stores. The transform moves
non-percussion MIDI pitches by signed integer addition and rejects the entire
view if any pitch leaves `[0, 127]`. No wrap, clipping or octave folding occurs.
Pitch class is modulo 12 and octave is integer floor division by 12.

| Field | Builder-compatible rule |
| --- | --- |
| `note.pitch`, `pitch_class`, `octave` | Shift pitched notes; percussion pitches stay unchanged. |
| `track.mean_pitch`, `min_pitch`, `max_pitch` | Recompute from every actual owned note, including percussion. |
| `track.pitch_std` | Population standard deviation from the same note list. |
| `note.track_relative_pitch` | `(pitch - mean) / std` when std is positive. |

Each note must have exactly one track owner, and percussion flags must agree
with that track. Ownership relations and canonical note positions determine
statistics, never entity-name conventions. Empty tracks have unavailable zero
statistics. Nonempty constant-pitch tracks have available mean/min/max and
available zero standard deviation, but unavailable zero relative pitch. These
are the existing raw-builder rules, not new missingness semantics.

All tensors are detached copies; the source remains unchanged even on a
range/contract error. IDs, exact-time identity, candidate slots, ownership,
topology, non-pitch features, and their availability remain unchanged. The
implementation uses the builder's float64 `fmean`/population-std arithmetic
before float32 packing. Synthetic oracle and inverse comparisons require exact
full graph serialization/fingerprint equality; no tolerance is needed in the
supported path. This API is correctness-oriented and has no corpus-scale or
GPU throughput evidence.

## HookTheory raw timeline

The current adapter is explicitly `HOOKTHEORY_ADAPTER_VERSION = "2.0.0"`.
Earlier pieces retained conversion provenance version `1` and the corpus index
adapter label `2B.1`. The new version changes conversion semantics; it does not
upgrade old canonical artifacts in place.

Raw duration is derived from declared source duration and observable melody
timing (including rests), tempo and meter events. Chord end times and unused
late key-annotation onsets cannot enlarge duration, bars, beats or graph
geometry. Existing key/scale interpretation needed to reconstruct relative
melody pitches remains in force. Removing or changing that interpretation can
change pitches and is outside this guarantee.

Target intervals are intersected with the raw timeline. A partially overlapping
annotation retains supervision on the overlap with a
`hooktheory.chord_clipped_to_raw_timeline` diagnostic. A wholly out-of-range
annotation becomes an empty endpoint span, all its targets are masked, and
`hooktheory.chord_outside_raw_timeline` records the original extent and reason.
Local-key spans use the corresponding `hooktheory.local_key_*` diagnostics.
The adapter retains target IDs/provenance and never invents a negative target
from absent or out-of-range annotations.

No sounding melody produces an explicit `hooktheory.empty_melody` diagnostic.
Source duration/rests/observable tempo and meter timing may still define a raw
timeline. With no positive observable duration, duration is zero, the melody
track is empty, bars/beats are empty, and chord-only content cannot create them.
Targets then have empty masked spans. This is a valid empty observation, not
evidence of negative harmony.

`hooktheory_raw_input_bytes(piece)` and
`hooktheory_raw_input_fingerprint(piece)` expose mandatory raw projection
`HOOKTHEORY_RAW_INPUT_VERSION = "1.0.0"`. The projection contains exact timing,
raw notes/tracks (including source track index), metric/tempo events and stable
IDs. It excludes targets, annotations, split, diagnostics, optional semantic
metadata and semantic provenance. It accepts new-version adapter output only;
historical pieces are not relabeled with current semantics.

Changing, removing, or extending chords and toggling `include_targets` preserves
mandatory raw bytes/fingerprint and the entire raw graph. Full canonical
serialization may differ because it also carries annotation values, target
masks, source provenance and diagnostics. These are separate artifacts.

The HookTheory corpus builder passes this raw fingerprint through the existing
`CanonicalCorpusInput.cache_input_sha256` boundary. Its
`adapter_config_fingerprint` now describes only the raw configuration:
dataset name, adapter version and raw projection version. Target visibility
is retained as `include_targets` in canonical conversion provenance; it is
excluded from the raw cache configuration. No generic cache helper or schema
changes.

The index still retains the complete physical source-row SHA/source manifest
fingerprint and full canonical artifact SHA. The existing immutable artifact
path combines the raw cache key with canonical SHA: two label versions can
share one raw key while retaining different target-bearing artifacts. Changed
melody pitches/timing change the raw key. New builds never rewrite historical
pieces or records, and a historical index is still explicitly historical,
not silently marked as a new conversion. A fresh new-version build uses a new
versioned key and the raw projection helper rejects old-version pieces.

## Compatibility and next gate

The default raw graph builder/schema, feature registry, vocabularies, target
ontology, trainer, frozen profiles and experiment records do not change.
Legacy source was neither inspected nor reused. Historical caches and terminal
experiment evidence remain immutable; old artifacts are never proof that a
new conversion or transposition path was exercised.

Any adoption in training needs a separately registered experiment/protocol
with the new input identities, fresh artifact namespace and appropriate
TRAIN/VALIDATION-only acceptance. No real corpus, TEST loader, SSH session or
trained checkpoint is used in WS1. The influence on historical C2 remains
unknown; no automatic rerun is authorized by this correction.
