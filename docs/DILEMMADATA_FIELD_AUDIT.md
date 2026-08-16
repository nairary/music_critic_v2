# Dilemmadata v1.0 Field Audit

## Scope and evidence identity

This document records the Phase 9A evidence audit. It is not a production
adapter specification and it does not authorize theory heads, losses, or
training changes.

The audited installation is an exact regular-file copy of the official
`johentsch/dilemmadata` `v1.0` tag at commit
`d60ee75b4a9495e932a4a7be39381578be17e222`. A clean checkout and the local
installation had the same 2,743 files and hashes, with no mismatched,
local-only, or upstream-only regular files. The installed directory has no
nested `.git` directory and does not contain populated corpus submodules; it
is a release source snapshot containing processed pitch arrays, processing
code, metadata, and the available AN score snapshots.

The official repository is <https://github.com/johentsch/dilemmadata>. Release
metadata in `.zenodo.json` identifies version `v1.0`, publication date
2026-02-04, creators Johannes Hentschel and Emmanouil Karystinaios, and license
identifier `CC-BY-NC-SA-4.0`. The release gitlinks identify AugmentedNet commit
`ec6cfe78fe252098ecdedd96bb300ad131830cc6` and Distant Listening Corpus commit
`3a3152b5ee2448359497bc02794cd39b937d4118` as source lineage.

No standalone license-text file and no dataset-specific `CITATION.cff` or
BibTeX file is present. The README cites the upstream corpora. Consequently,
downstream users must preserve dataset and upstream lineage, respect the
non-commercial/share-alike license identifier, and independently verify the
appropriate citation and redistribution terms. The repository commits only a
new synthetic fixture, never corpus rows.

## Audit method and reproducibility

`scripts/audit_dilemmadata.py` discovers records deterministically and scans
TSV files row by row with strict UTF-8 and strict tabular parsing. It keeps
only bounded vocabularies and record-level aggregates in memory. It does not
import or execute release processing modules. Corpus-relative paths are used
in reports, output below the corpus root is rejected, and runtime/platform
values are excluded from the semantic fingerprint.

The complete installation contains 884,619,898 bytes and has content
fingerprint:

```text
8f1161ad7cdbd979845012ffc6150cd82c5e91ab1197ed97385fffce57a0f312
```

Two independent complete runs emitted byte-identical 8.9 MiB JSON reports:

```text
report SHA-256:       2c5a05a439a5b18c6f98a353e88c078a1245a62b922b5abe5ef90b02707b522c
semantic fingerprint: 056de3ce37c0a22393038c60017dd5bff318469764e04ca44f5bfdbe5b38978d
```

The committed compact projection is
`tests/fixtures/dilemmadata/audit_manifest.json`. Reproduce it with:

```bash
MUSIC_CRITIC_DILEMMADATA_ROOT=/path/to/johentsch-dilemmadata-d60ee75 \
python scripts/audit_dilemmadata.py \
  --output /tmp/dilemmadata-audit.json \
  --check
```

`--root` overrides the environment variable. `--limit N` is a bounded
diagnostic and deliberately marks evidence incomplete. `--upstream-root`
enables an optional full file/hash comparison with a clean checkout.

## Installation and record inventory

The regular-file suffix inventory is 2,344 `.tsv`, 329 `.mxl`, 24
`.musicxml`, 24 `.krn`, 13 `.py`, three `.json`, three extensionless files,
and one each of `.csv`, `.md`, and `.txt`.

Primary records are not one homogeneous TSV format:

| Dialect | Discovery rule | Records | Note rows | Primary header shapes | Companion evidence |
|---|---|---:|---:|---:|---|
| `an_joint` | `pitch_arrays/AN/{split}/*_joint.tsv` | 353 | 758,555 | 1 | 353 scores and 353 derived `_slices.tsv` files |
| `dlc` | `pitch_arrays/DLC/{collection}/*.tsv` | 1,280 | 2,122,168 | 13 | processed metadata; source-score submodule is not populated |
| Total | — | 1,633 | 2,880,723 | 14 | 354 auxiliary pitch-array TSVs |

There are 46 collection identifiers. Metadata resolves 1,619 records and is
missing for 14. Metadata absence is retained, not converted into a false
value and not treated as a parser error.

Primary TSVs are uncompressed UTF-8 text with a tab delimiter and one header
row. `.mxl` companions are ZIP containers; `.musicxml` companions are XML.
The complete deterministic header list, per-field presence/missingness,
bounded vocabulary, and bounded lexical-type evidence are emitted by the full
JSON report. The 14-shape fingerprint is:

```text
cb0ec4b13d211cd2fd95580daaba20f3259e2782cc6a37e892f5ffbd0d3d1959
```

Recognized null spellings are empty string, `<NA>`, `NA`, `NaN`, `nan`,
`None`, and `null`. Missing values are counted separately and never interpreted
as negative labels.

## Raw musical representation

The primary rows are score-derived symbolic note arrays, not raw MIDI. The
release's own processing boundary constructs note, timing, spelling,
meter/measure, staff/voice, and tie-continuation columns before it merges
annotations. Phase 9A therefore classifies the following as raw observations:

| Concern | AN evidence | DLC evidence | Contract conclusion |
|---|---|---|---|
| Exact onset | `s_offset_frac` | `quarterbeats_playthrough` | exact quarter-note `Fraction` |
| Exact duration | `s_duration_frac` | `duration * 4` | exact quarter-note `Fraction` |
| Integer corroboration | `onset_div`, `duration_div` | same | one positive proportional resolution in every record |
| Absolute pitch | `s_midi` | `pitch` | MIDI-compatible integer pitch `[0,127]` |
| Spelling | `s_step`, `s_alter` | `step`, `alter` | optional score observation; 0 pitch/spelling mismatches |
| Source voice | `s_part_id`, `s_voice_id` | `staff`, `voice` | optional source identity, never semantic role |
| Tie continuation | `s_isOnset` | `is_note_onset` | false means continuation of a previous tied note |
| Meter/measure | per-note meter and measure columns | per-note meter and measure columns | observable but silent meter/bar events require 9B validation |
| Tempo | absent | absent | only the existing explicit provenance-bearing default is permitted |

The audit parsed a target-independent MIDI-compatible note projection for all
1,633 records with zero structural parse quarantines. Every record had exactly
one integer source-resolution candidate. It observed 74,773 tied-continuation
rows in 1,449 records and 23,314 zero-duration rows in 864 records. A zero
duration is a source-derived grace-note candidate; Phase 9B must map it to
`is_grace=true` under an explicit tested rule and quarantine contradictory
evidence. It must not invent a positive duration.

Velocity, channel, program, percussion identity, rests, articulation, and
dynamics are absent from the primary arrays. Staff/voice/spelling/TPC are not
requirements of the raw-MIDI inference contract. A target-free exact note
projection is therefore proven, but a production-complete `CanonicalPiece`
from every TSV is not yet proven: bar/meter event reconstruction,
percussion/default policy, tie identity, and grace handling belong to Phase
9B. Raw unlabeled MIDI inference continues to use the generic MIDI adapter.

## Target inventory

Counts use `available / masked / missing / ambiguous / unsupported` at the
note-row evidence level. `masked` includes absent source fields, invalid label
gates, and false positive-only boundary rows. `ambiguous` is additional
evidence on an available row and can overlap `available`. Source-entry counts
deduplicate repeated per-note labels by release annotation identity plus raw
value; they are not inferred semantic spans.

| Family | AN counts | DLC counts | Cross-source status |
|---|---:|---:|---|
| Global key | `0 / 758555 / 0 / 0 / 0` | `2122168 / 0 / 0 / 39009 / 0` | source-specific; absent in AN |
| Local key | `758555 / 0 / 0 / 0 / 0` | `2122168 / 0 / 0 / 39009 / 0` | source-specific |
| Tonal region | `758555 / 0 / 0 / 0 / 0` | `2122168 / 0 / 0 / 39009 / 0` | deferred alias/crosswalk, not a new native label |
| Chord boundary | `280821 / 477734 / 0 / 0 / 0` | `904869 / 1217299 / 0 / 0 / 0` | lossless positive subset of `a_isOnset` |
| Roman numeral | `758030 / 525 / 0 / 0 / 0` | `2094680 / 27488 / 0 / 38824 / 0` | source-specific grammar |
| Chord root | `758030 / 525 / 0 / 0 / 0` | `2094680 / 27488 / 0 / 38824 / 0` | deferred semantic crosswalk |
| Chord quality | `758030 / 525 / 0 / 0 / 0` | `2094680 / 27488 / 0 / 38824 / 0` | source-specific vocabulary |
| Bass | `758030 / 525 / 0 / 0 / 0` | `2094680 / 27488 / 0 / 38824 / 0` | deferred semantic crosswalk |
| Inversion | `758030 / 525 / 0 / 0 / 0` | `1085143 / 27488 / 1009537 / 26218 / 0` | source-specific |
| Applied/secondary harmony | `91135 / 525 / 666895 / 0 / 0` | `357733 / 27488 / 1736947 / 14149 / 0` | source-specific |
| Borrowed harmony | `0 / 758555 / 0 / 0 / 0` | `0 / 2122168 / 0 / 0 / 0` | unavailable/deferred |
| Cadence | `0 / 758555 / 0 / 0 / 0` | `40996 / 2081172 / 0 / 0 / 0` | DLC-only source-specific point evidence |
| Phrase boundary | `0 / 758555 / 0 / 0 / 0` | `67235 / 2054933 / 0 / 0 / 0` | DLC-only source-specific point evidence |
| Section boundary | `0 / 758555 / 0 / 0 / 0` | `9693 / 2112475 / 0 / 0 / 0` | DLC-only source-specific point evidence |
| Note degree | `758030 / 525 / 0 / 0 / 0` | `2094680 / 27488 / 0 / 0 / 0` | source-specific note-row label |
| Voice role | `0 / 758555 / 0 / 0 / 0` | `0 / 2122168 / 0 / 0 / 0` | incompatible: source voice is not semantic role |

No calibrated numeric confidence field was observed. Producer, analyst,
proofreader, source URL/version, validation gates, alternative-label evidence,
and raw label spellings remain provenance/diagnostic sidecars. None is a model
input merely because it exists in a pitch array.

## Alignment, overlap, grouping, and leakage findings

The exact alignment coordinate is the dialect-specific rational note onset.
No float equality or nearest-neighbour snapping is allowed. Repeated per-note
harmony labels must become source-identity/run sidecars; Phase 9A does not
invent an end time after the last evidenced boundary. The DLC outer merge can
forward-fill harmony and can lose annotation boundaries that fall between
notes, so such spans cannot be advertised as exact without additional source
evidence.

Split grouping is the transitive closure of:

1. identical target-independent MIDI-compatible note multiset;
2. identical AN score bytes;
3. explicit cross-source links in `processing/merged_summary.tsv`.

Composer/title similarity alone never joins records. The result is 1,507
components, including 126 multi-record components, 98 explicit AN/DLC
overlaps, and 30 exact-input clusters covering 60 records with different
target fingerprints. All alternative analyses must remain in one component.

Five components conflict with release split hints: three AN `training` and two
AN `validation` records link to DLC `test` records. Phase 9B must ignore the
conflicting per-record hints and assign a single split only after component
closure. This is an evidence warning and a production blocker, not a reason to
discard the records.

Raw observation and target-sidecar fingerprints are separate. A synthetic
target replacement changes the target fingerprint while preserving both the
raw-observation fingerprint and MIDI-compatible projection. Phase 9B must
extend this mutation matrix through canonical serialization, graph stores,
graph fingerprint, and model-input fingerprint.

## Quarantine result and unresolved evidence

The audited release has zero structurally quarantined primary records. The
parser still emits deterministic categories for empty files, duplicate or
missing headers, row-width mismatches, malformed raw notes/tie flags,
non-monotonic onset order, inconsistent source resolution, and exact
division-coordinate mismatch. Synthetic fixtures cover missing required
columns and malformed row widths.

The following are deliberately unresolved for Phase 9B rather than silently
normalized:

- full bar/meter event reconstruction from per-note evidence;
- an evidence-backed percussion/default policy for required canonical fields;
- tie continuation to canonical note-identity semantics;
- zero-duration grace validation;
- between-note target boundaries and final span ends;
- cross-source semantic normalization for roots, bass, inversions, applied
  harmony, and tonal regions;
- group-aware split reassignment for the five conflicts.
