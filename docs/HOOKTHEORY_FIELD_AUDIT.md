# HookTheory Field and Legacy-Behavior Audit

Status: **Phase 2B.0 evidence report**. This audit is an executable-specification
gate for a future adapter; it is not an adapter implementation.

## Audit basis

The audit used the local sources listed below, existing HTCanon outputs, and the
read-only legacy checkout at commit
`2d8281f31cc9ad9c8fecaf332da0c61e0e949415`. The deterministic standard-library
tool is `scripts/audit_hooktheory_legacy.py`. Its JSON report contains bounded
examples only and records paths relative to the dataset roots.

The legacy files inspected were:

- `src/data/preprocess_hooktheory.py`
- `src/data/canonicalize_hooktheory.py`
- `src/data/build_preprocess_song_timelines.py`
- `src/data/encode_teacher_features.py`
- `src/data/render_encoded_song_to_midi.py`
- `src/dataloader/theory_helpers.py`
- `src/dataloader/hooktheory_dataset.py`
- `src/dataloader/utils_graph.py`
- `tests/test_canonicalize_hooktheory.py`
- `docs/music_critic_v1/hooktheory_processed.txt`
- `docs/music_critic_v1/hooktheory_selected_field_types_documentation.txt`
- `docs/music_critic_v1/FIELDS_DECODE.txt`

The documentation files are under `docs/music_critic_v1/` because of
pre-existing changes in the legacy worktree. Nothing in that checkout was
modified.

## Discovered source inventory

`Hooktheory_Raw.json` is a directory, not a JSON file. Its primary merged raw
object is `Hooktheory_Raw.json/4_merged.json`. The audit reader accepts complete
top-level JSON objects and the legacy fragment form without outer braces.

| Relative path | Role / structure | Records | Bytes | SHA-256 |
|---|---:|---:|---:|---|
| `data/HookTheory/Hooktheory_Raw.json/4_merged.json` | raw legacy JSON object | 26,178 | 1,503,859,489 | `8ab601050d0b8c8752c3b6bf190d63edefa5fce07735ce823bca6a3922dff833` |
| `data/HookTheory/Hooktheory.json` | selected alternate-schema JSON object | 26,175 | 308,953,124 | `5e7457df5640170337c6e320d32fe90d6355b5ab96f15dbd3567180a05be9c08` |
| `data/HookTheory/Hooktheory_Train_Segments.json` | train manifest JSON object | 13,560 | 1,831,047 | `f2601eb544f2e5028ffad54d3827912865578fb9ad96e6768b35e4714d5c7207` |
| `data/HookTheory/Hooktheory_Valid_Segments.json` | val manifest JSON object | 1,333 | 180,037 | `12526962f77c2eb41cd117c8effa678b39c2b350384a7b048b327aff287b0c48` |
| `data/HookTheory/Hooktheory_Test_Segments.json` | test manifest JSON object | 1,480 | 200,130 | `72be80045d4d28842352383e605e8712d50b3437a07b15faa541ee9d17283d5a` |
| `data/HookTheory/HookTheoryKey.train.jsonl` | train key-label JSONL | 9,429 | 1,979,009 | `e2370d0a56a7dda797e22bd2d2dffd8988c0ee52f924f1618c13b3256ced9ddd` |
| `data/HookTheory/HookTheoryKey.val.jsonl` | val key-label JSONL | 920 | 193,156 | `04dbee929ec6b0257f759f46834f5534c3eebdb81aa63b249d0aa667a6efa986` |
| `data/HookTheory/HookTheoryKey.test.jsonl` | test key-label JSONL | 1,084 | 227,469 | `97bee618059a402dcd9494ed42d0c18ff95cdabc6d1591636aafd3fe91164a1f` |
| `data/HookTheory/HookTheoryStructure.train.jsonl` | train structure JSONL | 9,498 | 3,176,413 | `d14e246d03dd4f092ef40c8b53570dab7d52a374f5047481ce7ccff30dc5e33e` |
| `data/HookTheory/HookTheoryStructure.val.jsonl` | val structure JSONL | 927 | 310,442 | `003e29d0140696182d824e18b0a6fc2301f9ef63b5929e2dea8f8843949d564f` |
| `data/HookTheory/HookTheoryStructure.test.jsonl` | test structure JSONL | 1,090 | 364,753 | `481347a5b6c042d112c526706cf6647d593ef0b18e9ac5d705e7cdcf937210bc` |
| `data/HookTheory/FIELDS_DECODE.txt` | field documentation | — | 6,956 | `398564ac4534624297f37ae9f2ac2863a27a9015d74ecd39c1da0fdc3ea2b7e2` |

The raw split counts are train 21,233, val 2,184, and test 2,761. Three
train records have no `json` payload, leaving 26,175 selected, processed, and
canonicalized records.

## HTCanon inventory

| Relative path | Role | Records | Bytes | SHA-256 |
|---|---:|---:|---:|---|
| `data/HTCanon/HK_processed/hooktheory_processed.json` | selected legacy fields | 26,175 | 331,239,381 | `18421660eada680a223666f8e9af6b193900d91292b2ea7148e5c0687d2d42fe` |
| `data/HTCanon/HK_processed/hooktheory_processed_structured_only.json` | structured subset | 11,515 | 152,861,716 | `41cc04c29905be024339540889af56db32b4cbfc9b9dafab708b348413028261` |
| `data/HTCanon/HK_processed/original_songs_timeline.json` | original-audio timelines | 7,179 | 5,054,670 | `41c235663d37893b7cc14cb93db643cf65d12136fcd8d31dfb9a58906237c215` |
| `data/HTCanon/HK_processed/canonical_full/hooktheory_canonical.json` | canonicalized legacy fields | 26,175 | 372,383,691 | `2b78e7d90bd81bd6a9d9ce946bc1ebff259d6967dcda1ad7b139bfbc5a5d8dc8` |
| `data/HTCanon/HK_processed/canonical_structured_only/hooktheory_canonical.json` | canonical structured subset | 11,515 | 170,648,473 | `90502208e1705b0e1f15cbbb21a4c278255cc0f11ab8c05915e0f95f35badad5` |
| `data/HTCanon/HK_processed/encoded_full/teacher_encoded.json` | model-era encoded full set | 26,175 | 560,132,745 | `bdf5f23ae3b13e7a497c23a89115fb9c07bdeea07566e97e73b1873c1d98edec` |
| `data/HTCanon/HK_processed/encoded_structured_only/teacher_encoded.json` | model-era encoded subset | 11,515 | 252,011,028 | `97e789e0f132e3a8e00c0b92975ead059c9a307b263dcceb2bd04f7afab18db9` |
| `data/HTCanon/encoded_full/teacher_encoded.json` | duplicate encoded full artifact | 26,175 | 560,132,745 | `bdf5f23ae3b13e7a497c23a89115fb9c07bdeea07566e97e73b1873c1d98edec` |

Encoded IDs are historical model inputs, not source observations or future V2
raw features.

## Field evidence

The 26,178 raw records contain 1,338,346 note events, 449,072 chord events,
27,279 key regions, 26,540 tempo regions, and 27,217 meter regions.

| Field | Observed runtime types | Confirmed domain summary |
|---|---|---|
| `beat` | integer 1,054,162; number 760,456; null 17 | 1-based symbolic coordinates; fractional values are common |
| `duration` | integer 652,182; number 1,135,236 | fractional values are common |
| `sd` | string 1,338,346 | exact legacy tokens include `1..7`, flats, sharps, and `bb1` |
| `octave` | integer 1,338,338; null 8 | observed `-4..4` |
| `isRest` | boolean 1,338,346 | rest notes produce no note |
| chord `root` | integer 449,072 | `0..7` plus 20 negative malformed values; raw `8` was not observed |
| chord `type` | integer 449,072 | `5`, `7`, `9`, `11`, `13` |
| `inversion` | integer 449,072 | `0..3` |
| `applied` | integer 449,072 | `0..7`; nonzero occurrences are retained only as deferred raw evidence |
| `adds`, `omits`, `alterations`, `suspensions` | array only | empty and nonempty arrays observed |
| `borrowed` | null 328,325; string 114,290; array 6,457 | null, empty, known mode, unknown `super:2`, and pitch-class arrays observed |
| `alternate` | string 449,072 | empty string and `_`; semantics unresolved |
| `pedal` | null 449,072 | no non-null example observed |
| `beatUnit` | integer 27,217 | values `1` (27,106) and `3` (111); meter-denominator mapping unresolved |
| `numBeats` | integer 27,217 | `2,3,4,5,6,8,9,12` |

All raw chord list fields were present. The eight null octaves, 17 null beat
values, 20 negative roots, and three missing `json` payloads are deterministic
malformed-value candidates. `alternate="_"` occurs 14 times. No directly
observed MIDI pitch exists in this raw schema.

## Confirmed executable behavior

Symbolic HookTheory beats are 1-based. V2 conversion must use the raw numeric
lexeme as an exact decimal/rational input and calculate:

```text
canonical_onset_qn = Fraction(raw_beat_decimal_string) - 1
```

The exact legacy scale-degree offsets are `1:0`, `b1:11`, `#1:1`, `2:2`,
`b2:1`, `#2:3`, `3:4`, `b3:3`, `#3:5`, `4:5`, `b4:4`, `#4:6`, `5:7`,
`b5:6`, `#5:8`, `6:9`, `b6:8`, `#6:10`, `7:11`, `b7:10`, `#7:0`, and
`bb1:10`. A non-rest pitch is derived as:

```text
72 + 12 * octave + tonic_pitch_class + scale_degree_chromatic_offset
```

Its provenance method is `hooktheory_sd_octave_to_midi_v1`. Applied harmony is
ignored. Missing inputs or a result outside MIDI `0..127` diagnose and omit the
note; clamping is forbidden. The current corpus has no derived out-of-range
example, so that guard remains a contract-level not-observed category.

Raw chord roots `1..7` map to functional values `0..6`; raw `0` is rest/empty
and never tonic. The legacy code defines raw `8` as special bVII, but the audited
raw corpus contains no root `8`, so no real-data bVII fixture can be fabricated.
Negative roots are diagnosed and normalize to unavailable.

Borrowed null/empty normalize to `none`; known modes normalize to `mode_name`
with their template; arrays normalize modulo 12, sorted and unique as `pcset`;
unknown strings remain `unknown` and diagnose. No stringified list or
non-string/non-array unexpected type occurs in this corpus.

Legacy region normalization sorts by beat and removes exact consecutive
duplicates using the complete region key. No exact duplicate region was found
in the audited raw corpus. Multiple valid regions are common and must all be
preserved; they cannot be reduced to `main_key`, `main_bpm`, or `main_meter`.

## Structure join, grouping, and leakage

The verified join key is normalized split plus `Path(audio_path).stem`; `valid`
normalizes to `val`.

| Split | Symbolic clips | Structure rows / IDs | Matched | Symbolic only | Structure only | Duplicate IDs |
|---|---:|---:|---:|---:|---:|---:|
| train | 21,233 | 9,498 | 9,498 | 11,735 | 0 | 0 |
| val | 2,184 | 927 | 927 | 1,257 | 0 | 0 |
| test | 2,761 | 1,090 | 1,090 | 1,671 | 0 | 0 |

No structure row lacks `ori_uid`. There are 2,714 `ori_uid` groups containing
multiple clips. `ori_uid` is the expected V2 `source_group_id`; it must never be
invented for the 14,663 symbolic-only clips.

The audit found 23 `ori_uid` values spanning more than one split: 16 span
train/val, six span train/test, and `IxszlJppRQI` spans all three. This is an
explicit leakage finding. A future adapter or dataset split policy must group
by `ori_uid` and resolve these conflicts before training.

Structure `segment_start`, `segment_end`, and `duration` are audio seconds. They
remain raw evidence with
`section_alignment_status=unresolved_audio_seconds`. Phase 2B.0 creates neither
a section `TargetArray` nor a symbolic `AnnotationSpan` from them.

## Golden evidence and limitations

The selected fixture IDs are listed in
`tests/fixtures/hooktheory/golden_manifest.json`. Together they cover integer
and fractional timing, rests and derived pitches, major/minor/modal keys,
multiple key/tempo/meter regions, roots and malformed root zero, chord types and
decorations, borrowed variants, matched and unmatched symbolic structure,
shared `ori_uid`, applied raw evidence, and a missing payload.

Not observed and therefore not fabricated: raw root `8`, borrowed stringified
list, unexpected borrowed runtime type, a derived out-of-range pitch, a
non-null pedal, an exact duplicate region, a duplicate structure clip ID, an
unmatched structure row, or a missing structure `ori_uid`.

This audit establishes field behavior, traceability, joins, and bounded
examples. It does not establish the semantics of `beatUnit`, `alternate`,
`pedal`, applied harmony, or audio-to-symbolic section alignment, and it does
not construct a `CanonicalPiece`.
