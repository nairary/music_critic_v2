# Dilemmadata Phase 9B.1 Production Raw Adapter

## Accepted boundary

Phase 9B.1 introduces the production, target-independent Dilemmadata v1.0 raw
adapter. It discovers the pinned 353 AN and 1,280 DLC primary records, converts
each record to a validated raw-only `CanonicalPiece`, writes the Phase 5B
canonical cache/index, plans one transitive group-safe split manifest, and can
feed the accepted Phase 8B SSL runtime.

The adapter is not a theory adapter. Every harmony, key, cadence, phrase,
section, degree, validation-gate, analyst, and alternative-analysis column is
outside the raw projection. Canonical `annotations` and `targets` are empty.
Phase 9B.2 target sidecars, encodings, heads, losses, supervised training,
PDMX, Phase 10, and any effectiveness claim remain future work.

## Public contracts

The public API is exported from `music_critic.adapters`:

- `DilemmadataCorpusIdentity`, `DilemmadataAdapterConfig`,
  `DilemmadataCorpusRecord`, and `DilemmadataCorpusDiscovery`;
- `DilemmadataAccepted`, `DilemmadataQuarantine`, and stable adapter errors;
- `discover_dilemmadata_corpus`, `convert_dilemmadata_record`, and
  `iter_dilemmadata_corpus`.

The adapter, pinned-corpus identity, raw projection, grouping, discovered-record
binding, acceptance report, and production manifest contracts are independently
versioned. The runtime adapter is `1.0.1`, the acceptance report and manifest
are `1.1.0`, and the identity/raw/grouping/record-binding contracts remain
`1.0.0`. Full iteration fails before record conversion unless the v1.0 release
identity matches the pinned commit, 2,743-file inventory fingerprint, and
1,633-record dialect counts.

Every policy field in `DilemmadataAdapterConfig` accepts exactly its implemented
value. Booleans, non-strings, empty strings, and unknown policy identifiers fail
as `dilemmadata.config_invalid`; provenance and cache metadata therefore cannot
claim a policy that runtime did not execute.

## Identity domains

The implementation deliberately keeps these identities separate:

| Identity | Meaning | Permitted inputs |
|---|---|---|
| record/piece | stable logical source record | dialect, collection, relative logical name |
| physical source SHA-256 | byte-level inventory evidence | complete TSV bytes, including target columns |
| raw projection SHA-256 | complete adapter/cache input | normalized raw fields only |
| narrow grouping fingerprint | conservative split evidence | exact onset, duration, MIDI pitch multiset |
| source/lineage group | transitive split component | narrow grouping, identical AN score bytes, explicit overlap links |
| canonical/graph/model fingerprint | downstream raw evidence | accepted target-free canonical projection |

Discovery seals every record with `DilemmadataDiscoveryRecordBinding@1.0.0`.
The seal covers corpus identity, record/piece/dataset/dialect, canonical
relative path and opaque discovered-path locator, raw/grouping identities,
source and lineage groups, suggested split, resolution, optional score
identity, physical source identity, and discovery statistics. Conversion checks
the seal before parsing or constructing a piece. A forged record therefore
cannot change a split component. A post-discovery target-only byte change is
accepted when the raw projection is unchanged and is rebound to its new
external physical SHA; a raw mutation remains
`dilemmadata.raw_fingerprint_mismatch`.

The physical SHA-256 remains in the external index and corpus inventory. It is
not embedded in `CanonicalPiece`, because changing or deleting a theory column
must not change canonical bytes. Dilemmadata cache keys use the raw projection
SHA-256 through `CorpusCacheInputIdentity@1.0.0`; legacy HookTheory and POP909-CL
cache-key semantics and the generic cache/index versions remain unchanged.

## Raw mappings and defaults

Both TSV dialects are read as strict UTF-8 tabular streams. Musical time uses
exact `Fraction` values and the integer division columns must corroborate one
positive per-record resolution. MIDI pitch, optional spelling, source
part/staff/voice observations, tie state, key signature, and meter/measure
evidence are raw. Theory columns are never read by conversion.

Each record produces one source-neutral pitched track. Staff and voice remain
optional note observations; they do not select tracks, create semantic roles,
or alter graph topology. Velocity, program, channel, and instrument are
unknown. The schema-required percussion flag is conservatively `false` with a
warning and explicit default provenance; it is not an instrument claim.
Tempo is absent and receives the existing 500,000 microseconds-per-quarter
canonical default with provenance and a quality flag.

## Tie, grace, meter, and bar policy

- A tie continuation merges only when exactly one earlier note with the same
  MIDI pitch and source voice ends at the continuation onset. Missing,
  ambiguous, zero-duration, or inconsistent predecessors quarantine the
  record; durations are never clipped or guessed.
- A source-zero duration is retained as an `is_grace=true` zero-duration note.
  A zero-duration tie continuation is contradictory and quarantined.
- Simultaneous meter observations must agree. Meter changes use exact
  measure-anchor evidence when present; they are not float-snapped to nearby
  notes.
- Bars and beats are generated on an exact rational grid. Pickups use the
  evidenced first short measure, final short bars are explicitly incomplete,
  and anchors outside the reconstructed grid quarantine the record.
- Key-signature fifths are raw observations; mode remains `unknown`. Conflicting
  simultaneous observations use `dilemmadata.key_signature_conflict`, not a
  meter diagnostic. Harmony or local/global key labels never supply canonical
  key events.

All inserted/defaulted facts have dedicated provenance and quality flags.

## Discovery, grouping, cache, and loading

Discovery includes only AN `pitch_arrays/AN/{split}/*_joint.tsv` and DLC
`pitch_arrays/DLC/{collection}/*.tsv` primary records. Slices, summaries,
scores, processing metadata, and overlap tables are auxiliary evidence rather
than samples. Paths stored in records, pieces, quarantine rows, and reports are
corpus-relative.

`build_dilemmadata_corpus_cache` writes raw-only canonical artifacts and a
standard Phase 5B index. `scripts/build_multisource_cache.py dilemmadata`
exposes the production builder. The `dilemmadata` Hydra data group routes the
index/cache/split through `IndexedSSLRawDataset`; the raw loader does not
project targets. Split planning closes source and lineage links transitively,
so release split hints can be diagnosed but cannot separate a component.

## Failure and acceptance policy

Malformed raw records yield exactly one `DilemmadataQuarantine` with stable
`dilemmadata.*` categories and bounded messages. A full-corpus identity failure
is fatal before iteration. Target-column mutations do not quarantine raw music.

`scripts/accept_dilemmadata_adapter.py` performs the release gate outside the
corpus root and supports deterministic `--check` against
`tests/fixtures/dilemmadata/production_manifest.json`. It records
accepted/quarantined counts by dialect, stable failure
categories with bounded samples, processed bytes, peak RSS, note/tie/grace/
meter/bar/beat and graph totals, warnings/default provenance counters, cache
miss/hit evidence, grouping/split evidence, candidate raw-identity rechecks,
and deterministic cache mutation probes. The second build repeats the official
`discover -> convert_dilemmadata_record -> cache` path; cached pieces are not
used as its source. Readiness requires a byte-identical index, identical full
quarantine identities/categories and source-build semantic projection, and an
unchanged size/mtime/SHA snapshot for every immutable artifact. It then runs
one official CPU Phase 8B optimizer step on exactly two real AN and two real
DLC singleton components and requires finite loss, one attempted/applied and
zero skipped updates, finite nonzero online-encoder gradients/changes, zero
theory-target access, and zero retained prediction/CUDA tensors.

The committed manifest contains only contract versions, pinned identity,
outcome/category counts, accepted totals, grouping/cache/split fingerprints,
and Boolean/count SSL mechanics. It excludes corpus bytes, paths, duration,
RSS, caches, checkpoints, and the concrete loss value. `ready=true` requires
both intrinsic checks and exact manifest equality.

The official check form is:

```bash
python scripts/accept_dilemmadata_adapter.py \
  --root "$MUSIC_CRITIC_DILEMMADATA_ROOT" \
  --work-dir /tmp/dilemmadata-acceptance-work \
  --output /tmp/dilemmadata-acceptance.json \
  --check
```

The report is operational evidence only. It makes no model-quality,
likelihood, critic, theory-supervision, or scaled-training claim.

## Pinned v1.0 acceptance result

The Phase 9B.1 release run discovered all 1,633 primary records and produced
exactly one outcome each: 719 accepted and 914 quarantined, with zero fatal
failures. AN contributed 108 accepted and 245 quarantined; DLC contributed 611
accepted and 669 quarantined. Quarantine categories were 416 unresolvable bar
grids, 438 missing tie predecessors, 133 ambiguous exact tie predecessors, and
11 zero-duration tie-continuation contradictions. A record may carry more than
one category. These records were not coerced through invented meter or tie
rules.

The accepted cache has 719 raw-only pieces and 707 components. The first source
build recorded 0 hits/719 misses; the independently rediscovered and converted
second source build recorded 719 hits/0 misses. Both emitted byte-identical
indices at
`c0451976b6b6eab88cb90aa6c47d6afdba1b81ce9b588f0f84daa846154adb0e`;
quarantine projections and source-build semantic fingerprints matched, and no
immutable artifact changed. The group-safe split fingerprint is
`58ac7720f65f7fd3102248fb39d89291a78d65c06fc2ab9a16d78a6ee1666a3e`.
The semantic acceptance fingerprint is
`92187b3b10e27662536870b4fce9d683065a32bc20bf970184a2a7b33727287a`;
the complete committed manifest fingerprint is
`f14f4456bf11dc9f3096bfe0d119877c192cacbeb44e40a1e7347982f467124e`.
The opt-in local run took 1,351.434 seconds with peak process RSS
2,076,991,488 bytes. Generated cache, split, SSL checkpoint, and full report
artifacts remain outside the repository and are not committed.

The official Phase 8B smoke used exactly two accepted AN and two accepted DLC
singleton components. One CPU optimizer step was applied, the final finite
loss was `0.4853225231170654`, 308 online-encoder parameters had nonzero finite
gradients and changed, four mask-plan fingerprints were retained as host
evidence, and retained prediction/CUDA tensor counts were both zero. This is
data-path mechanics evidence, not an effectiveness result.
