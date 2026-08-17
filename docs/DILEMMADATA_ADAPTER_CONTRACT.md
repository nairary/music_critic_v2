# Dilemmadata Phase 9B.1 Raw Adapter Contract

## Status and boundary

Phase 9B.1 implements the evidence-backed production raw adapter described by
this contract. It adds runtime conversion, cache/index/split/loading, and an
SSL-ready path while leaving every theory sidecar, encoding, head, loss,
supervised training/evaluation, and ontology change for later work.

The answer to “can Dilemmadata produce a canonical piece without theory
labels?” has two levels:

- **Yes for a target-independent MIDI-compatible note-event projection:** all
  1,633 audited records provide exact rational onset/duration and
  MIDI-compatible pitch
  without reading theory columns.
- **Yes as a production raw-only `CanonicalPiece`:** Phase 9B.1 implements
  bar/meter events, required percussion/default policy, ties, zero-duration
  grace notes, and structured failures without reading theory labels.

Raw unlabeled MIDI inference remains possible through the generic MIDI adapter
and cannot depend on Dilemmadata staff, voice, spelling, TPC, or theory labels.

## Source discovery and dialects

The adapter accepts an explicit root; deployment wrappers may use
`MUSIC_CRITIC_DILEMMADATA_ROOT`; it never embeds a machine-specific path.
Discovery is sorted and restricted to:

- AN primary records: `pitch_arrays/AN/{split}/*_joint.tsv`;
- DLC primary records: `pitch_arrays/DLC/{collection}/*.tsv`.

AN `_slices.tsv`, `dataset_summary.tsv`, score files, processing metadata, and
merged summaries are auxiliary evidence, not additional samples. Unknown TSVs
below `pitch_arrays` fail closed until classified. Input files are opened as
strict UTF-8, tab-delimited, uncompressed streams. Output/cache/report roots
must be outside the corpus.

`DilemmadataAdapterConfig` is failure-closed. Its track, tie, grace, meter, and
default policy fields each accept only the one versioned value implemented by
the runtime; bools, other types, empty values, and unknown values are rejected.

Discovery applies `DilemmadataDiscoveryRecordBinding@1.0.0` to every record.
The binding includes corpus identity/version, logical record/piece/dataset/
dialect, canonical corpus-relative path, raw projection and grouping
fingerprints, source/lineage component IDs, split hint, resolution, optional
score identity, and original physical source identity. Conversion verifies the
binding before parsing. Mutation of any bound record field is
`dilemmadata.record_binding_mismatch`; it cannot change cache or split input.
A theory-only file change after discovery remains valid when the raw projection
is identical, while a raw change remains `dilemmadata.raw_fingerprint_mismatch`.

## Raw input contract

The mandatory target-blind note projection is:

```text
RawNoteObservation(
    exact_onset_qn,
    exact_duration_qn,
    midi_pitch,
    optional_spelling,
    optional_source_part_staff_voice,
    source_tie_onset_flag,
    source_meter_measure_evidence,
    provenance,
)
```

Dialect mappings are exact:

| Value | AN `*_joint.tsv` | DLC `*.tsv` |
|---|---|---|
| onset qn | `Fraction(s_offset_frac)` | `Fraction(quarterbeats_playthrough)` |
| duration qn | `Fraction(s_duration_frac)` | `4 * Fraction(duration)` |
| corroborating divisions | `onset_div`, `duration_div` | same |
| pitch | `int(s_midi)` | `int(pitch)` |
| spelling | `s_step`, `s_alter` | `step`, `alter` |
| source identity | `s_part_id`, `s_voice_id` | `staff`, `voice` |
| tie onset | `s_isOnset` | `is_note_onset` |
| meter | `ts_beats`, `ts_beat_type` | same |

`onset_beat`, `beat_float`, and `s_beat_float` are diagnostics only. They may
not become exact coordinates. Per-record divisions must imply exactly one
positive proportional resolution and agree with the rational columns.

Spelling and source voice identity are optional observations. They may enrich
canonical metadata when valid, but their deletion must not make the sample
unusable or alter raw-compatible graph topology. Source staff/voice is not a
voice-role target. Velocity, channel, program, articulation, dynamics, rests,
and semantic roles are unavailable and may not be fabricated.

Tempo is absent. Phase 9B may use only the canonical contract's existing
explicit default tempo, with provenance that it was inserted by the adapter.
It may not infer tempo from annotations or wall-clock assumptions.

## Note, tie, grace, meter, and bar rules

All musical comparisons use normalized exact rational values. Canonical IDs
must derive from stable source identity and deterministic order, not floats,
absolute paths, Python hashes, or target content.

The source tie-onset flag is retained. Phase 9B.1 merges a continuation into
one canonical note only for a unique exact contiguous same-pitch/source-voice
predecessor. It never sums or clips durations outside that rule.

A source-zero duration is a grace-note candidate. Mapping it to
`is_grace=true` is permitted only as an explicit versioned adapter rule with
tests against available score evidence; contradictory rows are quarantined.
Inventing a positive duration is forbidden.

Meter observations must agree for every simultaneous onset. Phase 9B.1
derives meter events and contiguous canonical bars from exact measure evidence,
including pickups, incomplete bars, and silent changes. A meter event may not
be placed merely at the first later note if the source proves an earlier bar
boundary. If the TSV lacks enough evidence, the record is quarantined or the
missing structure is represented only through an already accepted canonical
default with explicit provenance.

## Theory target boundary

All harmony, key, cadence, phrase, section, scale-degree, and derived analysis
fields are target-only. They may exist as repeated note-row columns but must be
parsed into provenance-bearing sidecars before any graph/model-input build.
The primary states `available`, `masked`, `missing`, and `unsupported` are
mutually exclusive and exhaustive. Missing/false gates mask; malformed
non-empty gates are unsupported and never also masked. `ambiguous` is an
additional family-local diagnostic and may overlap only `available`. Missing is
never negative.

The initial family mapping is intentionally conservative:

| Family | Phase 9A status | Alignment form |
|---|---|---|
| Global/local key | source-specific | piece/repeated run sidecar |
| Tonal region | deferred | do not duplicate local key as a normalized class |
| Chord boundary | lossless positive subset | exact observed note-onset point |
| Roman numeral/quality/inversion | source-specific | source annotation run |
| Root/bass | deferred crosswalk | source annotation run |
| Applied/secondary harmony | source-specific | source annotation run |
| Borrowed harmony | unavailable/deferred | fully masked |
| Cadence/phrase/section boundary | DLC-only source-specific | exact observed note-onset point |
| Note degree | source-specific | exact source note row |
| Voice role | incompatible/unavailable | source staff/voice is not a semantic role |

No common class ID is assigned merely because two columns have similar names.
Raw spelling, raw label, analyst/reviewer, alternative-label state, validation
gate, source URL/version, and unknown confidence are retained. Numeric
confidence remains `None` because no calibrated confidence field was observed.
`alt_label` is row-level alternative-label evidence retained as a
provenance/diagnostic sidecar; absent family-local evidence, it does not mark
each target family ambiguous.

In future Phase 9B.2, repeated chord/key values may be compressed only by exact
source annotation identity and exact adjacent coordinate evidence. It must not
infer an unobserved final end or use float snapping. A label boundary lost
between note rows remains unaligned/unsupported unless separate source evidence
restores it. Overlap policy must preserve distinct annotation views;
priority-based destructive selection is forbidden.

## Grouping and split contract

`source_group_id` is assigned after transitive closure over a versioned,
bounded-memory MIDI-compatible note-event multiset grouping fingerprint,
identical score bytes, and explicit `merged_summary.tsv` links. Composer/title
metadata alone is diagnostic and does not establish identity. The narrow
fingerprint includes only exact onset, exact duration, and MIDI pitch. It
deliberately excludes voice/track identity, tie state, meter/bar structure,
spelling, and other possible canonical inputs, so it is conservative split
evidence—not raw canonical, full-input, or model-input identity.

Every candidate multiple-analysis group over one MIDI-compatible note-event
multiset and every AN/DLC representation in one component must share one final
split. Exact alternative-analysis identity must be redefined and rechecked in
Phase 9B using the versioned canonical/model-input fingerprint. Release split
hints are inputs to diagnostics, not
final authority: five audited components contain conflicting hints. Phase 9B
must assign the component once, surface the conflicts, and fail closed if a
split manifest separates any component.

Record identity, source-group identity, raw canonical identity, target-bundle
identity, graph fingerprint, and model-input fingerprint are separate domains.
Alternative targets may change only the target-bundle identity.

## Leakage and fingerprint requirements

The production pipeline order is mandatory:

```text
discover raw record
  -> parse target-blind raw observations
  -> build/validate CanonicalPiece
  -> build raw graph and model-input fingerprint
  -> parse and align target sidecars
  -> route masked targets to supervised consumers
```

Theory columns, validation gates, target availability, analyst metadata,
alternative-analysis identity, provenance, diagnostics, confidence, and split
identity are forbidden in raw nodes, edges, features, positional structure,
or model-input fingerprints.

Phase 9B acceptance must mutate theory independently by deletion, replacement,
and reordering and prove exact equality of:

1. raw canonical projection and serialization;
2. every raw graph store and availability mask;
3. graph fingerprint;
4. model-input fingerprint.

The target bundle must change when a meaningful target changes. All analyses
of an equivalent input must retain one source group and split component.
Target-derived pitch arrays may never create canonical notes unless the same
note has separate raw score provenance.

## Structured quarantine

The adapter fails per record with stable categories and corpus-relative paths.
At minimum Phase 9B covers:

- missing/duplicate required columns and row-width mismatch;
- invalid rational timing, negative onset/duration, pitch outside `[0,127]`;
- inconsistent source resolution or non-monotonic source order;
- malformed tie flags or contradictory grace evidence;
- inconsistent simultaneous meter evidence or unresolvable bar coverage;
- inconsistent simultaneous key-signature evidence;
- forged or stale discovered-record binding;
- unknown primary dialect/layout;
- target boundary or span that cannot be exactly aligned;
- split component assigned to more than one final split.

Malformed targets mask or quarantine only the affected target view when raw
music remains valid. Malformed raw structure quarantines the record. Errors
never expose absolute paths or unbounded source payloads. The Phase 9A release
scan found zero structural record quarantines; synthetic fixtures exercise the
failure surface.

## Exact Phase 9B.1 scope

Phase 9B.1 implements the two versioned streaming dialect parsers, target-free
`CanonicalPiece` construction, tie/grace/meter/bar/default provenance,
transitive grouping and split manifests, raw-projection cache identity,
structured quarantine, leakage/raw mutation tests, environment-gated
full-corpus integration, and one official existing-Phase-8B optimizer smoke.

`DilemmadataAdapter@1.0.1` adds observable config and record-binding rejection.
`DilemmadataAcceptanceReport@1.1.0` and
`DilemmadataProductionManifest@1.1.0` require two independent source builds and
exact manifest equality. The adapter patch changes cache keys because adapter
version is part of generic cache identity; old artifacts remain immutable but
are not reused. Raw projection `1.0.0`, canonical schema, graph/model inputs,
grouping, ontology, targets, and HookTheory/POP909-CL contracts are unchanged.

Phase 9B.2A now implements source-native target sidecars and exact alignment as
the separate boundary documented in `DILEMMADATA_TARGET_SIDECARS.md`. The raw
adapter/canonical/cache/graph semantics and their versions remain unchanged;
the only narrow accepted-output addition is
`DilemmadataRawTargetAlignmentEvidence@1.1.0`, whose self-fingerprint is a
corruption check rather than origin proof. The target adapter independently
reconstructs this evidence from the pinned raw source with the same closed raw
parser/tie-merger, requires the exact canonical serialization, and compares
the full ordered row semantics before reading target or metadata values.
Theory heads, new losses, supervised training/evaluation, CUDA lifecycle
changes, new Phase 8 objectives, PDMX/Phase 10, and model-quality claims remain
out of scope.
