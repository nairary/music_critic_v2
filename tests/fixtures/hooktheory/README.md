# HookTheory Phase 2B.0 golden fixtures

These are bounded excerpts from local real data. They specify audited legacy
selection/canonicalization and the future V2 contract without implementing an
adapter. Dataset paths are repository-relative; source record hashes use
deterministic sorted compact JSON bytes.

Static tests require no local corpus. Set
`MUSIC_CRITIC_RUN_HOOKTHEORY_AUDIT=1` to verify hashes and excerpts against
`data/HookTheory` and `data/HTCanon`. Verification is read-only and never
rewrites a fixture.

`legacy_*_expected` describes existing V1/HTCanon evidence. It is not the V2
schema. In particular, encoded theory IDs are never V2 raw features. Structure
timestamps are audio seconds and have unresolved symbolic alignment.

Categories absent from the audited corpus are listed in the manifest rather
than fabricated.
