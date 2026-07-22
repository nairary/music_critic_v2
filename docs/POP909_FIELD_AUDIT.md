# POP909 Field Audit

## Scope and evidence

Phase 4A is a read-only evidence gate. No dataset file was changed, and this
audit does not implement a production adapter. Measurements were made with
`scripts/audit_pop909.py` over both the installed processed mirror and a pinned
checkout of the official POP909 repository. Dataset meaning comes from the
official repository README and the POP909 paper, not from the processed
directory name.

The official evidence snapshot is repository commit
`d83e6edba6872a704f5d3b8b32f5cb540088dae6` from
`https://github.com/music-x-lab/POP909-Dataset.git`. Its deterministic
path-and-content fingerprint is
`3822c50d7a964cb5ee747888c646a6ff52d38b230e8bb602520f7eb6b3866114`.
It contains 7,446 files: the root README and MIT license plus 7,444 files under
`POP909/`, including `index.xlsx`. There are 909 song directories, 909 primary
MIDI files, 1,989 MIDI files under `versions/`, and 909 files in each of five
annotation families.
There are no missing or unexpected contract assets and no duplicate song or
version identifiers. Twenty-two groups of files have identical content hashes;
content equality is retained as evidence and never used to split versions.
No standalone tempo or structural annotation family exists in this snapshot;
tempo evidence is carried by MIDI tempo events and the documented alignment
process.

The installed root `data/pop909-cl` is not a checkout and has no README,
license, version marker, or annotations. It contains 1,819 files: 909 actual
MIDI files under `POP909_processed/POP909_processed/` and 910 AppleDouble
resource files under `__MACOSX`; one MIDI is named `043 .mid`. Its fingerprint
is `af623705a375c419751e4ba6456224b8b700f50fc1a09a32af57e1620d1ff4dd`.
The MIDI structure and hashes differ from the official corpus. The local name
alone is therefore not evidence that this is any separately documented
human-corrected POP909 release. It is an annotation-free processed mirror and
is insufficient for supervised Phase 4B ingestion.

## MIDI crosswalk

Both roots were audited in deterministic song order with every primary
accounted for.

| Evidence root | Attempted | Converted | Explicitly failed | Type / PPQN |
| --- | ---: | ---: | ---: | --- |
| Official pinned snapshot | 909 | 908 | 1 (`043`) | type 1: 909; PPQN 480: 903, 960: 6 |
| Installed processed mirror | 909 | 908 | 1 (`172`) | type 1: 909; PPQN 480: 909 |

Both failures are `MidiAdapterError` with category
`midi_adapter.meter_change_inside_bar`: official `043` changes meter at tick
2,489 under active 4/4; processed `172` does so at tick 85,080. The generic
adapter correctly retains rather than hides these unsupported cases. All 16
spread-selected serialization round trips passed for each corpus.

For the 908 converted official primaries, canonical notes range 175–4,233
(median 1,655; p95 2,403), duration ranges 55.077–785.252 quarter notes and
44.657–471.151 seconds, and every result has four source/canonical tracks. The
processed mirror has 235–4,758 notes (median 2,049; p95 2,887), 2–3 tracks, and
27.501–393.001 seconds.

### Warning classification

| Code | Official occurrences | Official files | Processed occurrences | Processed files |
| --- | ---: | ---: | ---: | ---: |
| `EMPTY_TRACK` | 908 | 908 | 908 | 908 |
| `INCOMPLETE_FINAL_BAR` | 905 | 905 | 908 | 908 |
| `MID_BAR_TEMPO_CHANGE` | 4,694 | 287 | 0 | 0 |
| `OVERLAPPING_SAME_PITCH_NOTES` | 7,395 | 777 | 123,873 | 907 |
| `PIECE_TRAILING_SILENCE` | 908 | 908 | 908 | 908 |
| `midi.meter_conflict` | 82 | 82 | 0 | 0 |
| `midi.tempo_conflict` | 4 | 4 | 0 | 0 |
| `midi.dangling_note_on` | 0 | 0 | 4 | 4 |
| `midi.unmatched_note_off` | 0 | 0 | 4 | 4 |

Warnings count events or entities, not rejected files. In the processed
mirror, 123,873 of 126,605 warnings (97.84%) are same-pitch overlap pairs,
mostly arising from its synthesized `chords` tracks. Warning count per
converted file is 3–966 (median 124; p95 282). The exact 100-file spread used
by the earlier Phase 2A smoke sums to 14,475 warnings, 209,228 notes, and 300
tracks. Thus 14,475 was neither a file count nor 14,475 corrupt pieces; it was
an event-level diagnostic total dominated by intentional/polyphonic processed
track structure. The official snapshot has 14,896 total warnings and a much
lower per-file median of 9 (p95 47, maximum 934).

## Tracks and roles

The official paper defines three musical tracks: `MELODY` is the vocal lead,
`BRIDGE` is the secondary melody/lead-instrument bridge, and `PIANO` is the
accompaniment. Every one of 909 primary files has exactly one track with each
of those exact case-normalized names, at source indices 1, 2, and 3 after an
empty conductor track. All are non-drum program 0 on channels 0, 1, and 2.

None of the 1,989 alternative versions resolves the complete exact mapping.
Observed alternative names include `Melody`, `Cau`, `Piano`, `Grandeur 1`, and
`Grandeur 2`; order alone is not sufficient evidence. The processed mirror
also resolves zero complete role triples: its names are lowercase `piano` plus
either `chords` or `MIDI 01`, and it has no official melody/bridge separation.
Phase 4B must preserve every raw track, expose roles only as masked track-level
targets backed by the exact primary names, and leave every ambiguous or absent
role masked.

## Annotation files and time bases

All official annotation files decode strictly as UTF-8 plain text.
`beat_audio`, both chord families, and `key_audio` use tabs on every observed
row; `beat_midi` uses non-tab whitespace. The parser accepts either without
changing decimal tokens. Every coordinate is in seconds, not ticks, quarter
notes, or beat indices.

| Family | Columns | Files | Records |
| --- | --- | ---: | ---: |
| `beat_audio` | time, beat order | 909 | 348,870 |
| `beat_midi` | time, simple-meter downbeat, compound-meter downbeat | 909 | 303,894 |
| `chord_audio` | start, end, raw label | 909 | 98,384 |
| `chord_midi` | start, end, raw label | 909 | 124,805 |
| `key_audio` | start, end, raw label | 909 | 1,107 |

All 909 files in every family parsed with zero malformed rows, exact duplicate
records, non-monotonic starts, interval overlaps, or interval gaps. The source
does not document endpoint inclusivity. Phase 4B will retain source seconds
and use half-open `[start,end)` only as an explicit V2 interval convention.

Canonical comparison converts exact MIDI ticks through the complete tempo map
to rational seconds, compares source `Decimal` seconds without beat rounding,
and reports the nearest canonical denominator-beat start. This is an alignment
diagnostic, not authority to snap annotations:

| Family | Compared points | abs median / p95 / max (s) | signed median (s) | >100 ms | outside duration |
| --- | ---: | --- | ---: | ---: | ---: |
| `beat_audio` | 348,268 | 0.178779 / 0.420000 / 177.315450 | 0.023432 | 246,633 | 1,171 |
| `beat_midi` | 303,287 | 0.169024 / 0.402816 / 1.595743 | 0.046875 | 226,538 | 232 |
| `chord_audio` endpoints | 196,504 | 0.178886 / 0.423263 / 183.495065 | 0.034709 | 139,082 | 2,116 |
| `chord_midi` endpoints | 249,024 | 0.173000 / 0.406849 / 3.191486 | 0.042857 | 187,337 | 671 |
| `key_audio` endpoints | 2,212 | 0.196295 / 1.271128 / 177.318557 | 0.050159 | 1,596 | 177 |

The extreme audio values are terminal/coverage discrepancies, not evidence of
exact beat identity. Across all annotation points in the 365 successfully
converted pieces with more than one tempo event, absolute error has median
0.183269 s, p95 0.428350 s, and maximum 25.201785 s; 323,229 of 437,235 points
exceed 100 ms and 1,521 lie outside MIDI duration. Tempo changes therefore do
not justify a silent constant-tempo or nearest-beat conversion.

With a diagnostic 100 ms greedy match, audio/MIDI beat views match 275,711
points (absolute median 0.054004 s, p95 0.086495 s), leaving 73,159 audio and
28,183 MIDI points unmatched. Chord views match 163,114 boundaries (median
0.052989 s, p95 0.086901 s), leaving 33,654 audio and 86,496 MIDI boundaries
unmatched. Of 59,298 chord segments whose two endpoints match within 100 ms,
32,992 labels agree and 26,306 differ. The two views are distinct observations
and must never overwrite one another.

## Chord and key vocabularies

The complete deterministic report contains frequency maps for all 910 audio
chord labels, 320 MIDI chord labels, and 930 labels in their union. All 223,189
chord records parse losslessly; 6,202 are the sole no-chord token `N`.

The proposed grammar preserves the raw label and parses
`ROOT:QUALITY[(MODIFIER,...)][/BASS_DEGREE]`. Roots cover the 12 observed
spellings `A Ab B Bb C C# D E Eb F F# G`. The 19 qualities are `11`, `7`, `9`,
`aug`, `dim`, `dim7`, `hdim7`, `maj`, `maj6`, `maj7`, `maj9`, `min`, `min11`,
`min6`, `min7`, `min9`, `minmaj7`, `sus2`, and `sus4` (plus `N` outside the
grammar). Extensions are `2,4,6,7,9,11,13`; the observed alteration is `b7`;
suspensions are `sus2,sus4`; bass intervals are `2,3,4,5,6,7,b2,b3,b5,b6,b7,#4`.
This is deliberately not compressed to the legacy five-class vocabulary.

All 1,107 key records parse losslessly. The 24 labels are every observed
major/minor pair over `A, Ab, B, Bb, C, D, Db, E, Eb, F, G, Gb`; 152 pieces
contain more than one key interval. The audit JSON retains exact label
frequencies.

## Provenance, grouping, and golden evidence

The paper describes the MIDI performances as professional arranger work with
review and correction, and tempo curves as manually labeled for alignment.
It describes beat, chord, and key labels as MIR/algorithmic products. Phase 4B
must therefore use `source="human", confidence=None` for arrangement and tempo
evidence; `source="algorithm", confidence=None` for all five annotation
families; and `source="dataset", confidence=None` for exact track-role targets.
Algorithmic chord and key labels are auxiliary targets, not human gold.

The stable group ID is `pop909:<three-digit-song-id>`. A primary MIDI, every
annotation, and all versions of that song belong to the same group. Versions
are never independent examples. A later splitter accepts sorted group IDs plus
an explicit seed and ratios and returns one split per group; Phase 4A assigns
no train/validation/test split.

`tests/fixtures/pop909/audit_manifest.json` pins 11 official cases covering an
ordinary role-resolved piece, tempo and key changes, complex chords, no-chord
regions, alignment extremes, alternative versions, unusual/empty tracks, and
all important warning categories. It stores only relative paths, hashes, and
expected audit facts. The explicit primary failure `043` is pinned separately
in the same manifest.

## Gate result

The evidence audit is complete, but the production contract is intentionally
not yet strict-clean: official primary `043` needs an explicit mid-bar-meter
policy, and alternative role targets are ambiguous. These are Phase 4B design
inputs, not reasons to fabricate conversions or labels.
