# HookTheory Phase 2B.0 golden fixtures

These 19 cases are bounded excerpts from local real data. They specify audited
raw evidence, selected upstream simplified-schema crosswalks, legacy
selection/canonicalization, and the remediated Phase 2B.1 contract. Every real fixture points to
`data/HookTheory/Hooktheory_Raw.json/4_merged.json`; source record hashes use
deterministic sorted compact JSON bytes.

Static tests require no local corpus. Set
`MUSIC_CRITIC_RUN_HOOKTHEORY_AUDIT=1` to verify hashes and excerpts against
`data/HookTheory` and `data/HTCanon`. Verification is read-only and never
rewrites a fixture.

`legacy_*_expected` describes existing V1/HTCanon evidence. It is not the V2
schema. In particular, encoded theory IDs are never V2 raw features. Structure
timestamps are audio seconds and have unresolved symbolic alignment.

The corpus-wide semantic meter crosswalk resolves `numBeats` as the canonical
numerator and maps `beatUnit=1` to denominator 4 and `beatUnit=3` to denominator
8. It found zero value mismatches across 27,216 paired regions and one raw-only
meter region omitted by the simplified schema.

Phase 2B.1 remediation maps compound raw beats to half-qn, uses compound
felt-pulse tempo, and derives scale-aware pitch with MIDI 60 for relative octave
zero. Changed fixture expectations carry an evidence note with the old/new
values, reason, source, and classification; source excerpts and hashes remain
unchanged.

Categories absent from the audited corpus are listed in the manifest rather
than fabricated. In particular, root 8 to bVII and the MIDI-72 pitch anchor are
explicit Music Critic V1 compatibility behaviors, not corpus observations or
upstream Sheet Sage invariants. There is no real root-8 fixture.
