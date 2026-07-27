# POP909-CL Adapter Contract

## Status and evidence boundary

This document specifies the implemented Phase 4B production adapter from the
remediated Phase 4A evidence. The production corpus is
`POP909_processed` from POP909-CL repository commit
`be9094392903c471a930519e1c0bacf8b6be5d62`. Original POP909 is lineage and
possible future ablation evidence only.

The local installed 909 MIDI files are byte-for-byte equal to that pinned
upstream directory. Phase 4B must validate the recorded content fingerprint
and retain upstream repository, commit, MIT license, source path, and file hash
provenance. Absence of README/Git files from an extracted installation does not
weaken provenance when the complete content comparison succeeds.

## Production API

Runtime adapter version `2.0.0` is implemented in
`music_critic.adapters.pop909_cl` and exposed minimally through
`music_critic.adapters`.

The primary API is:

- `Pop909ClCorpusIdentity` and `discover_pop909_cl_corpus`;
- `Pop909ClAdapterConfig`;
- `Pop909ClCorpusRecord`;
- `convert_pop909_cl_file`;
- `iter_pop909_cl_corpus`;
- `Pop909ClAccepted`, `Pop909ClExpectedTargetAbsence`, and
  `Pop909ClQuarantine`;
- `Pop909ClAdapterError`, `Pop909ClCorpusIdentityError`, and
  `Pop909ClConversionError`;
- `pop909_cl_piece_id`, `pop909_cl_raw_input_group_id`,
  `pop909_cl_source_group_id`, and `pop909_lineage_group_id`.

The module also exposes typed chord-block, candidate, coverage, pairing, track,
and instrument evidence records. Production code does not import
`scripts/audit_pop909_cl.py`. Corpus manifest version `2.0.0` and acceptance
schema `2.0.0`, including bounded raw-duplicate evidence, are implemented by
the opt-in streaming acceptance in
`scripts/accept_pop909_cl_adapter.py` and
`tests/fixtures/pop909_cl/production_manifest.json`.

## Discovery and identity

- Accept a direct `POP909_processed` directory or the observed nested
  `POP909_processed/POP909_processed` extraction layout.
- Exclude `__MACOSX` and `._*` AppleDouble files from the corpus count and
  corpus-content fingerprint. They may appear only in a separate installation
  noise inventory and installation fingerprint.
- Normalize a filename stem only by trimming surrounding whitespace and then
  requiring exactly three digits. Preserve the exact relative path, including
  `043 .mid`, alongside the logical ID.
- Missing, duplicate, malformed, and unexpected identifiers are structured
  failures. Do not silently choose among duplicate logical IDs.
- Use dataset identity `pop909_cl`. Source-record identity is
  `piece:pop909-cl-<three-digit-song-id>`. Split-atomic raw-input equivalence
  is `pop909-cl-score:<score-projection-sha256>` and is computed only from the
  score-only projection. Cross-corpus song lineage remains
  `pop909-lineage:<three-digit-song-id>`.
- Assign no final split. A later group splitter must keep a CL song together
  and, if original POP909 is also used, keep matching lineage IDs in the same
  split.

## Instrument contract

The pinned upstream documentation defines the combined musical score as the
instrument on MIDI channel 0 and corrected chord blocks as the instrument on
MIDI channel 1. Time-signature and key-signature changes are MIDI meta-events.

Phase 4B must resolve instruments from channel-bearing MIDI events under that
documented contract. Track order and names such as `piano`, `chords`, or
`MIDI 01` are corroborating evidence only. Exactly one score instrument is
required. Missing or multiple score/chord instruments, mixed channels, or
other note-bearing channels produce structured observations; pitch range and
track order must never repair them silently. Unexpected observations are fatal.

The measured exceptions `367` and `658` have a channel-0 score but no channel-1
chord instrument. Their per-task chord availability is entirely masked, not
negative, and these two pinned exceptions are expected rather than fatal. Song
`658` demonstrates why a `chords` track name cannot override channel evidence:
its sole note-bearing track is channel 0 and is therefore the score.

## Raw-input leakage boundary

Channel 1 is target-bearing annotation, not raw music. Phase 4B must construct
the canonical raw piece from a score-only projection that:

- retains channel-0 score notes;
- retains required conductor, tempo, meter, and key metadata;
- excludes the complete channel-1 instrument, its notes, track record, name,
  end time, and other annotation-dependent events;
- exists in memory or temporary storage outside the dataset root;
- never changes as chord blocks are modified, removed, or replaced.

Consequently channel-1 evidence must never enter canonical musical tracks or
notes, raw note statistics, graph nodes/edges/features, graph serialization,
or graph fingerprints. The full CL MIDI passed to the current generic adapter
is explicitly unsafe because chord pitches become ordinary canonical notes.
That path is diagnostic only and cannot be used for training or inference.

Raw graph leakage tests must compare identical score projections after chord
mutation and require identical canonical score tracks/notes and graph
fingerprints. Annotation evidence must change independently.

Strict `graph_fingerprint` hashes the complete validated deterministic graph
serialization, including every `entity_id`. It is authoritative for canonical
piece ↔ graph binding, external-graph verification, post-preparation mutation
detection, and integrity diagnostics.

`model_input_fingerprint` contract `1.0.0` separately hashes the validated
numerical model input: global schema/feature/builder versions and `raw_only`;
ordered feature names; categorical/continuous values and availability;
candidate slots; and ordered edge types/indices. It excludes all entity IDs,
targets, provenance, source paths, grouping, and split. It must never replace
the strict binding fingerprint. POP split closure is authoritative only from
the score-projection `source_group_id`.

For downstream use alongside HookTheory, the shared auxiliary-target,
actual-accompaniment, role-agnostic inference, and future PLL boundaries are
defined in [`HARMONIC_SUPERVISION.md`](HARMONIC_SUPERVISION.md). That document
does not change this Phase 4B evidence, instrument, mask, provenance,
acceptance, or leakage contract: channel 0 is the combined polyphonic score;
channel 1 remains target-only expert-reviewed/human-corrected annotation and is
not actual performed/score accompaniment.

## Exact chord-block evidence

Within the uniquely resolved channel-1 instrument, pair note-on/off messages
at exact integer ticks and group notes by identical onset tick. Every block
must preserve before normalization:

- onset and end tick plus file PPQN;
- the complete sorted MIDI pitch multiset and individual note end ticks;
- pitch-class set, lowest source pitch, and bass pitch class;
- source track index, channel, track-name evidence, exact source path, and file
  SHA-256;
- pairing, repeated-pitch, mixed-end, overlap, and gap diagnostics.

Every dangling note-on and unmatched note-off must retain category, exact tick,
pitch, velocity and channel where present, chord-note-event ordinal, source
track/path/hash, and references to affected blocks/spans plus an explicit
affected interval. Aggregate counts cannot replace this evidence.

`N` is not encoded by a special MIDI note. Match upstream event semantics:
retain positive-duration leading and internal gaps as derived `N`, but do not
label uncovered time after the final chord as `N`. Trailing uncovered time is
a separate masked/unannotated span with null value/source/provenance. Overlaps
are diagnostics and are not truncated. Unsupported and ambiguous pitch-class
sets retain all raw evidence.

The pinned upstream normalization checks exact seventh patterns before exact
triad patterns while trying roots in ascending pitch-class order. Phase 4B may
record the upstream-selected root/quality/bass, but must also preserve all
matching candidates. Symmetric shapes can therefore be ambiguous, and an
unmatched set remains `unsupported`; neither case may be compressed silently.

Suggested auxiliary targets are separate masked arrays for boundary, root,
quality, bass, inversion, and no-chord state, aligned to exact rational
`tick/PPQN` annotation spans. Masks are task-specific:

- directly observed boundary and bass remain available even when normalization
  is ambiguous or unsupported;
- ambiguous root and inversion are unavailable single-label targets while all
  candidates remain preserved (a future version may explicitly choose a
  multi-label representation);
- quality is available for an ambiguous block only when every candidate agrees;
- unsupported root, quality, and inversion are unavailable;
- leading/internal `N` is available derived evidence, while trailing uncovered
  time and missing chord instruments are unavailable.

Raw chord blocks use source `human`, details `human_corrected` and
`expert_reviewed`, and null numeric confidence. Normalized root/quality/
inversion and inferred `N` use source `derived` with explicit provenance chains
through the corresponding pinned upstream normalizer/gap-event rule. Directly
observed boundary and bass reference raw-block provenance. Curated expert
evidence must not be described as infallible or unqualified human gold.

The stable Phase 4B task IDs are:

- `pop909_cl.chord.boundary`;
- `pop909_cl.chord.root`;
- `pop909_cl.chord.quality`;
- `pop909_cl.chord.bass`;
- `pop909_cl.chord.inversion`;
- `pop909_cl.chord.no_chord`.

They align through target-alignment spans of type `pop909_cl.chord` and use
annotation view `pop909_cl.channel_1`. For `367` and `658`, one full raw-piece
alignment span and six one-entry arrays have `mask=false` with null
value/confidence/source/provenance. This is explicit target unavailability,
not an empty negative class.

Exact source chord onset/end ticks and PPQN remain lossless in
`Pop909ClChordBlock` and raw-block provenance. When an upstream chord block
extends beyond the channel-0 raw score duration, its canonical target-alignment
span is deterministically intersected with the raw piece interval because
schema `2.0.0` forbids an annotation beyond `piece.duration_qn`. The exact
unclipped interval remains in structured evidence and provenance, and target
content never extends raw duration.

## Meter case and validation

POP909-CL song `172` changes from 4/4 to 6/8 at tick 85,080 with PPQN 480.
The previous 4/4 boundary is 84,480 and the next is 86,400, so the event is 600
ticks inside the active bar. The current generic adapter correctly rejects the
score-only projection. The Phase 4B MVP must retain `172` as the documented
quarantine, yielding 908/909 accepted conversion coverage. A later phase may
adopt a general tested partial-bar meter rule through a new recorded decision.
It must not special-case this song or silently move the event.

Production acceptance requires:

- every logical ID accounted for and the pinned fingerprint reproduced;
- unique score/chord resolution or an explicit structured failure;
- score-only generic conversion and deterministic canonical round trips;
- exact chord evidence and complete vocabulary/coverage reporting;
- unavailable masks for missing/unsupported targets;
- chord-mutation raw/canonical/graph invariance;
- group and lineage-group split leakage tests;
- the locked MVP quarantine for `172`, with no silent meter-event movement;
- no writes under source roots and no committed data, reports, caches, MIDI,
  generated media, or outputs.

The historical Phase 4A audit's `evidence_contract_ready` status remains
independent of its then-false `production_adapter_ready` field. Phase 4B
production readiness is now established by adapter-backed manifest checks, not
by rewriting that historical audit result.

The fresh streaming production acceptance completed in 1,249.285 seconds:
909 logical files, 908 validator-clean accepted pieces, only `172`
quarantined, 907 chord instruments, `367` and `658` explicitly masked,
116,055 blocks, 109,668 root/inversion targets, 109,800 quality targets,
116,055 boundary/bass targets, 5,801 ambiguous blocks, 586 unsupported blocks,
947 derived `N` spans, and 151 trailing masked spans. All 908 accepted pieces
passed deterministic visible/hidden canonical round trips, raw equality, and
raw graph fingerprint equality. The anomaly fingerprint is
`d1aee48a2bade9d545794a16e327c8304b718a30699e4b5328e9393d961e4051`.

## Full-corpus identity remediation

The Phase 6C full-cache build exposed one score-content collision: source
records `543` and `553` have distinct source file SHA-256 values
`7dc63700fb5e58d2d12b580aa53614413317232caa151920d6079ad2440b662b`
and
`618b99761e750edfaffb4053cc3ad073661fd5c969bfea840481f466a03ec07a`,
but their score-only projections are byte-identical with SHA-256
`4585134e3f7a70c105a3bb678a04ab2bc4522c04e11183f6fd6c59046be25286`.
Their canonical raw content and node/edge counts match after excluding record
identity/path/lineage/provenance/targets. Strict graph fingerprints differ:
543 is
`0a4fa698ed7748ebee855424f38c967bd04cf6b10e792b8b6a4e0aceb9230ed6`
and 553 is
`605072317c4029380d14d73a45be8f506a8edc45b6dca841ebb5b6e5d8920531`.
Their common model-input fingerprint is
`2c03b1a37a722173a72ce6fd0ce74a58f3a03627907ac4fd04702ddee07b9c7f`.

They are not a full corpus duplicate. Both have 163 chord blocks. Boundary
and empty no-chord values and masks agree. Bass values differ while its
all-available mask agrees; root, quality, and inversion values and masks
differ (154 versus 152 available rows). Target-bundle fingerprints therefore
differ. The accepted description is “multiple observed target views for one
score input,” not an unproven alternative-harmonization claim.

Both records are retained as `piece:pop909-cl-543` and
`piece:pop909-cl-553`, share one `pop909-cl-score:...` source group, retain
their distinct song lineage IDs, and must be assigned to one split. Production
evidence is 908 accepted record IDs, 907 raw-input-equivalence groups, one
two-record duplicate cluster `[543, 553]`, and the unchanged quarantine
`["172"]`. The resulting full index fingerprint is
`b2008221fa59ddd0df31289561b22341db9c2eac527e1a503eac57b74da27daf`.

Phase 4B does not authorize model, SSL, training, preference, or inference
implementation.
