# Original POP909 Lineage Adapter Notes

## Status

These notes describe only a possible future original-POP909 ablation adapter.
They are superseded for production Phase 4B by
`docs/POP909_CL_ADAPTER_CONTRACT.md` and the ADR that supersedes ADR-030.

An original-only adapter would accept the pinned original song-directory
layout, preserve `MELODY`, `BRIDGE`, and `PIANO` as separate raw tracks, keep
the five external annotation families as separate auxiliary views, and retain
their decimal seconds without silent snapping. Exact primary track names may
provide masked dataset role targets; track order alone may not. Alternative
versions with incomplete names remain masked.

Use `pop909_original` as dataset identity,
`pop909-original:<song-id>` as the original source group, and
`pop909-lineage:<song-id>` as the cross-corpus split group shared with
POP909-CL. Alternative versions are never independent split units.

The original external chord grammar, 930-label vocabulary, audio/MIDI
alignment findings, and song `043` conversion failure are original-only facts.
They must not enter the POP909-CL contract.

The original real-data regression remains gated by
`MUSIC_CRITIC_RUN_REAL_POP909_ORIGINAL_TESTS=1` plus
`MUSIC_CRITIC_POP909_ORIGINAL_ROOT`, and uses
`tests/fixtures/pop909_original/audit_manifest.json`.
