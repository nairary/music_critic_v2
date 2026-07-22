# POP909 Adapter Contract

## Status and scope

This document specifies Phase 4B from Phase 4A evidence. It is not an
implementation. Phase 4B may implement only behavior stated here or record a
new decision before changing it.

The supervised adapter accepts the official song-directory layout pinned by
the audit manifest: one three-digit song directory, its same-named primary
MIDI, optional `versions/*.mid`, and the five named annotation files. A
flattened processed MIDI mirror may still enter through the generic MIDI
adapter, but it must not be presented as the official annotated POP909 source.
Missing required assets, duplicate IDs, path escapes, and malformed rows are
structured per-file/per-row failures; they are never silently skipped.

## Identity, discovery, and grouping

- Discover in normalized relative-path sort order and retain corpus/version
  identity, relative source path, content hash, and parser outcome.
- Use `pop909:<three-digit-song-id>` as `source_group_id` for the primary,
  every annotation view, and every alternative version.
- Treat alternative versions as views of one source, never independent split
  units. Do not assign dataset splits in the adapter.
- The later split API is conceptually
  `split_source_groups(sorted_group_ids, seed, ratios)`. It must validate that
  no group crosses splits and persist its inputs and result.
- Exact duplicate hashes are a diagnostic. They do not override song grouping
  or license/provenance identity and must be considered by the later splitter.

## Raw MIDI and timing

Use the generic MIDI adapter as the raw conversion boundary. Preserve every
source track separately, including the empty conductor track. Preserve exact
PPQN tick-to-quarter-note timing, all tempo and meter events, source indices,
channels, programs, and drum flags under the accepted canonical contract.

Annotation coordinates remain source `Decimal` seconds. Map them through the
piecewise MIDI tempo map with exact rational arithmetic. Never compare floats
for equality, round seconds to beat indices, or silently snap to a nearest
canonical beat. If a consumer requests aligned indices, it must provide a
named, versioned tolerance and receive the original seconds, signed error,
absolute error, match status, and outside-duration status. The audit's 100 ms
tolerance is diagnostic only and is not a default truth threshold.

Chord and key spans use half-open `[start,end)` in V2 while retaining both
source endpoints. This is an explicit V2 convention because upstream endpoint
inclusivity is undocumented. Audio- and MIDI-aligned annotations remain
separate `annotation_view_id` values.

Official `043` currently fails the generic adapter because a meter change is
inside an active bar. Phase 4B must either retain that explicit failure or add
a general, tested canonical rule for partial-bar meter changes. A one-song
special case is forbidden.

## Role targets and leakage boundary

For official primaries only, resolve one case-normalized track name each:

| MIDI name | Target value |
| --- | --- |
| `MELODY` | `melody` |
| `BRIDGE` | `secondary_melody` |
| `PIANO` | `accompaniment` |

Track order corroborates but never establishes a role. A role is available
only when its documented name occurs exactly once. Otherwise its `TargetArray`
mask is false; do not infer it from order, pitch, channel, or file naming.
Alternative versions and the processed mirror therefore have masked roles
unless future dataset evidence establishes their semantics.

Role targets use `source="dataset"`, `confidence=None`, and provenance that
identifies POP909, the source path/hash, the exact evidence field, and adapter
version. They are auxiliary labels. No role, mask, annotation, provenance,
confidence, dataset ID, group ID, or split field may enter the raw graph's
features or topology.

## Annotation targets

Phase 4B parses these source schemas:

- `beat_audio`: `time_seconds beat_order`;
- `beat_midi`: `time_seconds downbeat_simple downbeat_compound`;
- `chord_audio` and `chord_midi`: `start_seconds end_seconds raw_label`;
- `key_audio`: `start_seconds end_seconds raw_label`.

Every family becomes a separately masked auxiliary `TargetArray`; absence or
failure means unavailable, never a negative label. Use
`source="algorithm", confidence=None` for these five families. Preserve raw
labels, source decimals, source line, view ID, parser status, and any alignment
diagnostic. The professionally prepared MIDI uses `source="human"` and the
manually labeled tempo evidence uses `source="human"`, both with unknown
confidence unless the source later supplies calibrated values.

Chord parsing preserves `N` and the lossless grammar
`ROOT:QUALITY[(MODIFIER,...)][/BASS_DEGREE]`. Store the raw label even when a
structured parse succeeds. Unknown future labels remain raw with a false
structured-parse availability mask; they do not collapse to a catch-all chord.
Key parsing preserves `TONIC:maj|min` and raw spelling. Multiple key spans are
normal and must not be reduced to a single song key.

## Validation and reproducibility

The implementation must:

- account for every discovered primary as converted or explicitly failed;
- preserve all audit/parser exceptions with stable categories and relative
  paths;
- distinguish warning occurrences from affected-file counts;
- validate canonical pieces and deterministic serialization round trips;
- expose deterministic complete-corpus inventory and bounded-sample modes;
- reproduce `tests/fixtures/pop909/audit_manifest.json` when explicitly given
  the pinned official snapshot;
- keep the opt-in full-corpus test behind
  `MUSIC_CRITIC_RUN_REAL_POP909_TESTS=1` and require
  `MUSIC_CRITIC_POP909_ROOT`;
- never write under the source root or commit MIDI, annotations, caches,
  reports, generated media, or outputs.

Production acceptance additionally requires an explicit decision for song
`043`, regression tests for all golden cases, target hiding/masking tests, raw
graph leakage invariance, annotation-view separation, group-safe split tests,
and exact/tolerant timing diagnostics. Phase 4A does not authorize graph,
dataset, model, SSL, training, or inference implementation.
