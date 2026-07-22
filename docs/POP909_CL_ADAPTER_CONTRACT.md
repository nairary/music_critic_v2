# POP909-CL Adapter Contract

## Status and evidence boundary

This document specifies Phase 4B from the remediated Phase 4A evidence. It is
not a production implementation. The production corpus is
`POP909_processed` from POP909-CL repository commit
`be9094392903c471a930519e1c0bacf8b6be5d62`. Original POP909 is lineage and
possible future ablation evidence only.

The local installed 909 MIDI files are byte-for-byte equal to that pinned
upstream directory. Phase 4B must validate the recorded content fingerprint
and retain upstream repository, commit, MIT license, source path, and file hash
provenance. Absence of README/Git files from an extracted installation does not
weaken provenance when the complete content comparison succeeds.

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
- Use dataset identity `pop909_cl`, source group
  `pop909-cl:<three-digit-song-id>`, and cross-corpus lineage group
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
other note-bearing channels produce structured failures; pitch range and track
order must never repair them silently.

The measured exceptions `367` and `658` have a channel-0 score but no channel-1
chord instrument. Their chord targets are unavailable, not negative. Song
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

`N` is not encoded by a special MIDI note. Positive-duration gaps before,
between, or after chord blocks are retained explicitly as implicit no-chord
spans. Overlaps are diagnostics and are not truncated. Unsupported and
ambiguous pitch-class sets retain all raw evidence.

The pinned upstream normalization checks exact seventh patterns before exact
triad patterns while trying roots in ascending pitch-class order. Phase 4B may
record the upstream-selected root/quality/bass, but must also preserve all
matching candidates. Symmetric shapes can therefore be ambiguous, and an
unmatched set remains `unsupported`; neither case may be compressed silently.

Suggested auxiliary targets are separate masked arrays for boundary, root,
quality, bass/inversion, and no-chord state, aligned to exact rational
`tick/PPQN` annotation spans. Available `TargetArray` entries use canonical
source `human`, with `human_corrected` and `expert_reviewed` recorded as
provenance details and numeric confidence left null unless upstream supplies
it. This provenance means curated expert evidence; it must not be described as
infallible or unqualified human gold. Unsupported
normalizations have unavailable structured targets while their raw annotation
evidence remains preserved.

## Meter case and validation

POP909-CL song `172` changes from 4/4 to 6/8 at tick 85,080 with PPQN 480.
The previous 4/4 boundary is 84,480 and the next is 86,400, so the event is 600
ticks inside the active bar. The current generic adapter correctly rejects the
score-only projection. Phase 4B must either quarantine `172` and retain
908/909 conversion coverage or adopt a general tested partial-bar meter rule.
It must not special-case this song or silently move the event.

Production acceptance requires:

- every logical ID accounted for and the pinned fingerprint reproduced;
- unique score/chord resolution or an explicit structured failure;
- score-only generic conversion and deterministic canonical round trips;
- exact chord evidence and complete vocabulary/coverage reporting;
- unavailable masks for missing/unsupported targets;
- chord-mutation raw/canonical/graph invariance;
- group and lineage-group split leakage tests;
- explicit handling of `172` under a general policy;
- no writes under source roots and no committed data, reports, caches, MIDI,
  generated media, or outputs.

Phase 4B does not authorize model, SSL, training, preference, or inference
implementation.
