# POP909-CL Field Audit

## Scope and authoritative identity

Phase 4A is a read-only evidence gate. The production corpus is
`POP909_processed` from
`https://github.com/AndyWeasley2004/POP909-CL-Dataset` at commit
`be9094392903c471a930519e1c0bacf8b6be5d62`. The associated BACHI paper is
`https://arxiv.org/abs/2510.06528`. The upstream MIT license SHA-256 is
`fe6064d631bdf4ce46028ef3aa7bc4eac285b8a1000c46682795f26448d29288`.

The installed `data/pop909-cl` tree is an extracted POP909-CL installation,
not an anonymous processed original-POP909 mirror. A complete logical-ID,
relative-path, and SHA-256 comparison found 909/909 exact matches against the
pinned upstream `POP909_processed` directory, zero mismatches, zero missing or
extra IDs, and no duplicates. The exact upstream filename `043 .mid` is
preserved while its logical ID is normalized to `043`.

The content fingerprint over only the 909 exact corpus-relative paths and MIDI
hashes is
`b34f07d9a2678abdb6f0dcf5db1c3aec3f35caca813f1fac80c0717cfc8e0c65`.
The installation contains 1,819 files and has fingerprint
`af623705a375c419751e4ba6456224b8b700f50fc1a09a32af57e1620d1ff4dd`;
910 are `__MACOSX`/`._*` AppleDouble files. Those 910 files are installation
noise and do not enter the corpus count or content fingerprint.

Original POP909 at commit `d83e6ed...` remains separately documented in
`docs/POP909_ORIGINAL_FIELD_AUDIT.md`. Its external annotations, alternative
versions, three-track roles, vocabulary, and song `043` failure are lineage
facts only.

## MIDI and instrument contract

All 909 CL files are type-1 MIDI at PPQN 480. Upstream documents channel 0 as
the combined musical score and channel 1 as corrected chord annotations;
time/key signatures are MIDI meta-events. Measured channel-bearing tracks are:

| Evidence | Files/tracks |
| --- | ---: |
| Empty conductor/meta track | 909 |
| Unique channel-0 score instrument | 909 |
| Unique channel-1 chord instrument | 907 |
| Missing channel-1 chord instrument | 2 (`367`, `658`) |
| Ambiguous/mixed/other-channel instruments | 0 |

All global metadata is on conductor track 0: 909 tempo events, 911
time-signature events, and 1,065 key-signature events. The score projection
retains these exact messages and rejects required global metadata found only on
an excluded annotation instrument.

Observed track names are `piano` 908 times, `chords` 454 times, and `MIDI 01`
454 times. Names are not authoritative: `658` has only a channel-0 score, but
that track is named `chords`. Song `367` has a normally named channel-0
`piano` score and no chord instrument. Both produce the structured observation
`missing_chord_instrument`, but the pinned contract classifies these known
cases as expected masked target unavailability rather than fatal corpus
failures. Score projection remains possible because channel-0 evidence is
unique.

## Leakage-safe score crosswalk

The audit constructs one temporary score-only MIDI at a time, retains the
conductor/meta and channel-0 score tracks, drops every channel-1 instrument,
and passes only that projection to the existing generic adapter. Nothing is
written under the dataset root.

Score-only conversion accounts for all files: 908 convert and song `172` is
explicitly quarantined after `midi_adapter.meter_change_inside_bar`. It is the
sole documented quarantine, not a fatal Phase 4A evidence failure. Sixteen
spread-selected canonical JSON round trips are equal. Converted score note
counts range from 175 to 4,233 (median 1,655; p95 2,403).

| Score-only warning | Occurrences | Affected files |
| --- | ---: | ---: |
| `EMPTY_TRACK` | 908 | 908 |
| `INCOMPLETE_FINAL_BAR` | 908 | 908 |
| `OVERLAPPING_SAME_PITCH_NOTES` | 123,439 | 907 |
| `PIECE_TRAILING_SILENCE` | 908 | 908 |

The 126,163 score-only warnings are event/entity diagnostics, not failures.
Per converted file the total ranges 3–966, with median 123 and p95 282. The
high same-pitch count remains in the combined channel-0 score, where multiple
voices are flattened into one score instrument; it cannot be attributed
wholesale to chord annotations.

For comparison only, passing each complete CL MIDI to the generic adapter
produces 126,605 warnings: 123,873 same-pitch overlaps plus four dangling
note-ons and four unmatched note-offs. The audit preserves all eight pairing
events with exact tick, pitch, velocity, channel, chord-event ordinal,
track/path/hash, affected block onsets, affected span IDs, and an explicit
affected interval. They occur one pair each in `076`, `084`, `086`, and `088`.
The target-bearing chord instrument
therefore contributes 434 additional overlap occurrences and eight pairing
warnings, and it also raises canonical note counts from score-only median
1,655 to unsafe median 2,049. These complete-file values are explicitly unsafe
diagnostics, not production score statistics, because chord pitches become raw
musical notes and graph observations.

Synthetic invariance tests modify, replace, and delete only channel-1 chord
blocks. The score projection bytes, canonical score tracks/notes, and raw graph
fingerprint remain identical, while chord evidence and unsafe complete-file
note counts change.

## Embedded chord blocks

Chord annotations require no external text sidecars. The audit pairs channel-1
notes at exact MIDI ticks, groups equal onsets, and preserves for every block:
onset/end tick, all note end ticks, complete pitch multiset, pitch-class set,
lowest source pitch, bass pitch class, track/channel/name evidence, exact
relative source path, and file SHA-256.

Across the 907 available chord instruments there are 116,055 blocks. Per-file
block count ranges 0–278 (median 124; p95 185). Pairing finds four dangling
note-ons and four unmatched note-offs, one each in songs `076`, `084`, `086`,
and `088`. Structural diagnostics find:

| Diagnostic | Count |
| --- | ---: |
| Upstream-compatible leading/internal positive-duration `N` spans | 947 |
| Trailing masked/unannotated spans | 151 |
| Blocks starting while an earlier block remains active | 691 |
| Duplicate block onsets after onset grouping | 0 |
| Blocks with a repeated source pitch at one onset | 87 |
| Blocks whose notes have mixed end ticks | 313 |

`N` is not a MIDI note. The upstream exporter emits it only before the first
chord or in an internal positive-duration gap, never after the final chord.
The audit therefore retains 947 leading/internal `N` spans and represents 151
trailing uncovered intervals separately with `available=false` and null
value/source/provenance. Their durations range from 1 to 12,861 ticks (median
401; p95 3,361). Missing chord instruments remain a separate file-level
availability condition.

## CL vocabulary and normalization

The complete audit report contains frequency maps for all 261 observed raw
pitch-class sets and 340 selected root/quality/bass labels. The pinned upstream
normalizer tests exact seventh patterns before exact triads and tries possible
roots in ascending pitch-class order. The audit preserves all candidates
rather than discarding symmetry:

| Normalization status | Blocks |
| --- | ---: |
| Unambiguous supported | 109,668 |
| Ambiguous supported | 5,801 |
| Unsupported | 586 |

Selected upstream normalization coverage is 99.4951%; unambiguous coverage is
94.4966%. Unsupported blocks retain their full pitch evidence. Ambiguous
blocks retain every candidate plus the upstream order-selected value.

Task masks are field-specific. Directly observed boundary and bass targets are
available for all 116,055 blocks. Root and inversion are available for the
109,668 unambiguous supported blocks and masked for 5,801 ambiguous plus 586
unsupported blocks. Quality is available for 109,800 blocks because 132
ambiguous blocks have candidates that agree on quality; the remaining 6,255
quality entries are unavailable. Candidate sets remain evidence when a
single-label task mask is false.

Roots and basses cover all 12 sharp-spelled pitch classes. The complete quality
vocabulary is `M`, `m`, `o`, `+`, `sus2`, `sus4`, `D7`, `M7`, `m7`, `/o7`,
`o7`, `mM7`, and `+7`. Frequencies are retained in the temporary report and
reproducibly emitted by `scripts/audit_pop909_cl.py`; they are not compressed
to the legacy five-class vocabulary.

## Provenance qualification

The upstream repository describes expert-reviewed chord annotations, removal
of algorithmic curation tracks, metadata correction, processing logs, and
manual verification. The paper describes human-corrected chord, beat, key, and
time-signature labels. Raw channel-1 chord blocks therefore use source `human`,
details `human_corrected` and `expert_reviewed`, and unknown numeric confidence.
Normalized root, quality, and inversion use source `derived`, with a chain
from the raw block through pinned `process_pop909.py:get_chord_quality`
semantics to the candidate-preserving audit representation. Inferred `N` uses
a separate derived chain through upstream gap-event construction. Directly
observed boundary and bass retain raw human provenance. This is curated expert
evidence, not a claim of infallible or unqualified human gold; upstream also
records known concerns for songs `518` and `620`.

## Song 172 meter evidence

Song `172` has PPQN 480 and these conductor meter events:

| Tick | Previous meter | New meter | Previous boundary | Next boundary | Offset inside bar |
| ---: | --- | --- | ---: | ---: | ---: |
| 0 | 4/4 default | 4/4 | 0 | 0 | 0 |
| 85,080 | 4/4 | 6/8 | 84,480 | 86,400 | 600 |
| 101,400 | 6/8 | 4/4 | 100,920 | 102,360 | 480 |

At tick 85,080 the active 4/4 bar length is 1,920 ticks, so the first change is
600 ticks after the prior boundary. The current generic adapter rejects this
general unsupported condition before reaching the second mid-bar change. The
upstream processing log records time-signature changes plus a start-beat shift
for this song. Phase 4A records `172` as the quarantine, giving accepted score
coverage of 908/909. Phase 4B may retain it or adopt a general tested
partial-bar meter policy. No one-song exception or silent event movement is
allowed.

## Readiness

The strict report exposes independent statuses. `evidence_contract_ready` is
true: the pinned evidence is complete, `367`/`658` are expected masked target
absence, and `172` is documented quarantine rather than an unclassified
failure. `production_adapter_ready` is false because Phase 4B has not
implemented the adapter and the general partial-bar policy remains unresolved.

## Grouping and golden evidence

Each CL song uses `pop909-cl:<song-id>`. Original POP909 uses
`pop909-original:<song-id>`. A later multi-corpus splitter must additionally
bind matching derivatives with `pop909-lineage:<song-id>` so they share one
split. Phase 4A assigns no final split.

`tests/fixtures/pop909_cl/audit_manifest.json` pins corpus/installation hashes,
the complete upstream comparison, aggregate warning/chord diagnostics, the
exact pairing-evidence fingerprint, and eleven
bounded cases: ordinary `001`, filename `043`, unsupported/pairing/overlap
case `076`, overlap case `088`, meter failure `172`, ambiguous case `262`,
missing-chord cases `367` and `658`, maximum score-warning case `802`, and
maximum implicit-gap case `857`, plus maximum trailing-unannotated case `246`.
No MIDI, annotations, or generated report is committed.
