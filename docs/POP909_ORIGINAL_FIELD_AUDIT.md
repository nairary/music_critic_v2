# Original POP909 Lineage Field Audit

## Status

This audit is retained as lineage, comparison, and possible future ablation
evidence. It does not define the production Phase 4B corpus or adapter. The
production corpus is POP909-CL `POP909_processed`; see
`docs/POP909_CL_FIELD_AUDIT.md`.

The original audit is reproducible with
`scripts/audit_pop909_original.py`. Its real-corpus test and environment are
explicitly original-only:

- `tests/integration/test_pop909_original_audit_real.py`;
- `MUSIC_CRITIC_RUN_REAL_POP909_ORIGINAL_TESTS=1`;
- `MUSIC_CRITIC_POP909_ORIGINAL_ROOT`.

## Pinned original corpus

The original POP909 snapshot is repository commit
`d83e6edba6872a704f5d3b8b32f5cb540088dae6` from
`https://github.com/music-x-lab/POP909-Dataset.git`. Its deterministic
path/content fingerprint is
`3822c50d7a964cb5ee747888c646a6ff52d38b230e8bb602520f7eb6b3866114`.
It contains 909 primary MIDI files, 1,989 alternative MIDI files, and 909
files in each external `beat_audio`, `beat_midi`, `chord_audio`, `chord_midi`,
and `key_audio` annotation family.

The generic adapter converts 908 of 909 original primaries. Original song
`043` is the explicit failure because a meter change at tick 2,489 occurs
inside an active 4/4 bar. This is an original-corpus issue, not the POP909-CL
production blocker.

All 909 original primaries uniquely expose `MELODY`, `BRIDGE`, and `PIANO`
track names. None of 1,989 alternatives has the complete exact-name mapping,
so any future original-corpus adapter must mask ambiguous alternative roles.
These roles are not POP909-CL roles: POP909-CL combines the musical score on
channel 0 and carries chord annotations on channel 1.

## Original external annotations

All 877,060 original external annotation records parse. The chord-label union
contains 930 text labels across 223,189 chord records, including 6,202 explicit
`N` records. The key vocabulary contains 24 labels across 1,107 records, and
152 pieces contain multiple key spans. Audio and MIDI chord views disagree
materially and remain distinct lineage views.

These counts and the original Harte-like text grammar do not describe the
embedded POP909-CL chord blocks. POP909-CL must be audited from channel-1 MIDI
pitch multisets, exact ticks, bass notes, and implicit gaps.

## Identity and leakage policy

Original POP909 uses `pop909_original` as dataset identity and
`pop909-original:<song-id>` as its source group. POP909-CL uses a separate
identity and source group. If both derivative corpora are used later, matching
song IDs additionally share `pop909-lineage:<song-id>` so they cannot cross
train/validation/test splits.

The earlier complete-file audit of the installed CL files observed 123,873
same-pitch overlap warnings. That count is preserved only as an unsafe generic
crosswalk diagnostic: it included target-bearing chord-instrument notes and is
not a score-quality statistic. The corrected score-only and chord-structure
counts are recorded in `docs/POP909_CL_FIELD_AUDIT.md`.

The bounded original golden evidence remains in
`tests/fixtures/pop909_original/audit_manifest.json`. No original MIDI,
annotation contents, or generated report is committed.
