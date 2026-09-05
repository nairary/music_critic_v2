# PDMX source manifests and window cache — WS3

Status: Phase 10 engineering foundation, tested only with generated synthetic
fixtures. This is not completed Phase 10, corpus readiness, an SSL run, or
evidence of improved music quality. The historical `pdmx_eda` replay and frozen
C0/C2 profiles remain unchanged.

## Supported source profile and evidence

`musicrender-metrical-json-raw-v1` is based on primary source definitions at
upstream PDMX commit `5c6a25b613e4497469e855e8bd1423b1d834ff45`:

- [reading/music.py](https://github.com/pnlong/PDMX/blob/5c6a25b613e4497469e855e8bd1423b1d834ff45/reading/music.py),
  Git blob `32b03c7ce42c318e595ff8af65935fd3f29c5030`: `to_dict` (lines 52–69),
  MusicRender metrical/absolute timing (83–106), `save` (556–626), and the
  upstream loader's fields (928–1064).
- [reading/classes.py](https://github.com/pnlong/PDMX/blob/5c6a25b613e4497469e855e8bd1423b1d834ff45/reading/classes.py),
  Git blob `6f91117a5a380bf88579424527aa3a5079018d02`: TimeSignature, Beat,
  Barline (118–183), Note (234–263), Track (297–341).

These immutable source files were read through the GitHub connector; no
upstream code was executed. The code revision identifies a format, **not the
release identity, license, completeness, or schema distribution of any supplied
PDMX corpus**. No actual source release was inspected in WS3; its identity stays
unknown. Tests construct small dictionaries matching the supported fields.

The adapter reads uncompressed UTF-8 JSON with a plain top-level mapping.
Nested `__class__.__name__` strings are inert data; no object reconstruction,
dynamic imports, MusicRender dependency, pickle, or compressed input is used.
Required `resolution` is positive integer ticks per quarter. `absolute_time`
must be false; absence uses a recorded metrical default. Each note has integer
`time`, `duration`, `pitch`; timing is `RationalTime(ticks, resolution)` with no
quantization or float equality. Tempo QPM decimals are parsed exactly, then
rounded half-up to the existing integer microseconds/quarter canonical field.

Raw tracks preserve optional program/name/percussion and notes preserve
optional velocity/spelling plus original track/notes-array ordinal IDs. Voices,
staff and roles are not inferred. The existing raw graph therefore uses
`next_in_track` when voice evidence is absent. No chord-array expansion,
Roman/chord pseudo-labels or theory targets occur. Key signatures with explicit
fifths remain observable key events; root alone is insufficient spelling
evidence. Optional notation/lyrics/expressive objects are ignored. They neither
set duration nor define the graph. Missing raw optional fields remain optional;
meter 4/4, tempo 120 QPM, and other applied defaults have explicit provenance.

Upstream `song_length` incorporates annotations/lyrics and an extra tick, so it
is deliberately excluded from raw extent. Extent comes from note ends and raw
barline/beat/meter/tempo onsets. Barline and downbeat boundaries constrain the
exact canonical denominator-unit meter grid; source performance beat pulses
are not substituted for that grid. Supported: exact meter changes, first
grid-aligned pickup, final incomplete bar and truncation at a meter change.
Internal irregular short bars and fractional-beat pickups outside the current
canonical contract reject with `unsupported_metric_geometry`. Shared schema
exceptions are not added. A grace-only point score gets a flagged metric
container while grace duration remains zero. Zero-duration non-grace notes,
empty/chord-only scores, absolute-time inputs and malformed fields are explicit
rejections. A failed invariant in the adapter output propagates as a programming
error rather than being mislabeled invalid source.
Unrepresentable decimal JSON numbers reject as `invalid_json` at the decoder
boundary; unknown programming exceptions outside that boundary still propagate.
Meters outside the existing raw feature vocabulary reject as `unsupported_meter`
before cache publication. Pinned adapter bounds are checked against the actual
registry by tests, preserving the adapter's dependency-free canonical layer.

## Source, selection and splits

`SourceManifest@1.0.0` contains explicit record IDs, portable relative paths,
SHA-256, license identity/evidence, optional rating, synthetic declarations,
supplied lineage links with evidence, schema profile, and optional release and
license-artifact fingerprints. It has no directory discovery or sampling by
filename. Source IDs and paths must be unique, paths remain below the explicit
root, and JSON rejects duplicate keys/non-finite numbers.

`mode=fixture` requires every record to be explicitly synthetic. Unknown
release fields stay null. `mode=production` rejects synthetic declarations and
requires a release identity, release SHA-256, license-artifact SHA-256 and
license evidence filtering. Those bindings are operator-supplied declarations;
the library does not independently authenticate release/license claims. A
future production invocation requires separate authorization and the planned
experiment record described below. No production source was opened here.

`SelectionConfig@1.0.0` requires a license allowlist; optional rating threshold
and record budget are explicit filters. Rating is never a gold quality label.
There is no built-in 1,000-record limit or mandatory sample policy. Source
records are ordered by source ID. Filtering has stable per-record reasons
(`pdmx.filter_license`, `pdmx.filter_license_evidence_missing`,
`pdmx.filter_rating_missing`, `pdmx.filter_rating`, `pdmx.filter_record_budget`).
The contract stores selected IDs and every exclusion.

All source records are grouped **before filtering and before windowing**.
Union-find components combine exact declared source hashes and supplied
lineage edges transitively. Actual source hashes are verified before an
accepted source is parsed/cached or reused. Filtered bridge records do not
erase leakage evidence. Components are identities of known evidence, not a
claim of complete deduplication or independent compositions; filenames do not
establish work identity. Unknown cross-corpus links remain `unknown` and must
be resolved before future mixed-corpus scientific splits.

The explicit `source_id -> train|validation` mapping covers the complete
manifest exactly, and each component has only one split. Every other split,
including TEST in a filtered record, is rejected before source/cache readers
or cache writes. This guard also precedes adapter reads, lazy dataset callbacks
and publication-index reads. No TEST unlock is implemented. All windows
inherit their pre-window component and split. The output uses the existing
`create_split_manifest`/`validate_split_manifest` contract; there is no new
window-level random split or replacement policy.

## Window semantics

`WindowConfig@1.0.0` has configurable bar count, stride, maximum windows per
piece and maximum notes per window. A trailing short window remains. Each
result carries global source piece/component, original bar IDs and half-open
bar ordinal range, exact global offset, original note IDs aligned to projected
notes, clipped-note count, rejection reason and `evaluation_unit=source_piece`.
Window IDs are separate deterministic piece IDs. Unclipped entity IDs remain;
clipped notes and synthesized start-context events get deterministic IDs and
provenance referencing the original entity/conversion record.

Ordinary notes overlap a half-open window and are clipped at both ends.
Their canonical times are local; retained source tick/second fields describe
the **original** source note. Provenance records original onset/duration and
left/right clipping. The exact transform is `global_qn = local_qn + offset`.
Crossing notes are never silently lost. Grace notes at an interior boundary
belong to the window starting there; terminal grace notes belong to the last
window. Grace durations remain zero. Bars retain pickup/incomplete flags and
metric offset; bar indices and event times become local. Effective start
tempo/meter/key context is carried forward. Events exactly at a nonterminal
window end belong to the next window. Source-terminal events remain for
terminal grace semantics and full-source window equivalence.

Empty windows and per-window note-budget failures produce reason-bearing
results without cache samples. The whole-source window budget is checked before
emission. All tracks are retained for raw context. No gold segmentation or
annotations are needed. VALIDATION overlap rejects by default; explicit
`allow_validation_overlap=true` acknowledges that subsequent evaluation must
aggregate original pieces. The loader itself does not perform that aggregation
or implement objective-specific SSL eligibility/weighting. Window counts and
overlapping clipped-note occurrences are never independent-work denominators.

## Streaming, artifacts and resume

`PdmxPipeline@1.0.0`, shard and journal versions `1.0.0` compose the existing
`CanonicalCorpusInput`, `cache_canonical_corpus`, `CorpusCacheConfig`,
`make_corpus_index` and `load_cached_piece`; shared helpers are not modified.
Each source is bounded by explicit byte/note/track/bar/beat budgets. Windowing
is an iterator. The thread worker queue holds at most `max_in_flight` tasks
and results; no process copies or corpus-wide list of decoded scores/graphs is
created. Workers only change execution scheduling. Results are emitted and
indexed in deterministic source order. Thread throughput gains are unmeasured.

Memory is bounded for decoded payloads by worker and per-source limits; global
manifest/index/report metadata is **O(records + windows)**, not constant space.
Per-source window metadata is bounded by the window budget. Optional graph
statistics build and release one fixture graph at a time per worker. The 60-item
generated queue test and 64-bar iterator test verify bounded consumption, not
corpus-scale RAM or throughput. There is no stress benchmark claim.

The output root contains `pdmx-v1.0.0/<contract SHA-256>/`:

- `contract.json`: sealed source/selection/split/config and all current shared
  canonical/graph/target-cache plus PDMX parser/projection/window versions;
- `canonical/`: existing content-addressed canonical JSON window artifacts;
- `shards/`: hashed per-source index, window provenance, graph counts and
  quarantine result, with observed-versus-declared source hash status;
- `journal/`: atomic, individually sealed completed-source references;
- `publication.json`: sealed ordered corpus index, standard split manifest,
  source-shard fingerprints and deterministic report. Written only after all
  selected sources reach a result.

Canonical source paths are relative, so moving roots or changing workers keeps
artifacts byte-identical. Publication uses same-directory temporary files,
file flush/fsync and atomic rename. Concurrent preprocess invocations targeting
one namespace and power-loss durability of directory metadata are not claimed.
Use one writer per output contract. Reader-visible publication is the completion
marker; unreferenced partial files are ignored. Corrupted referenced journal,
shard, canonical artifact or publication fails closed and is never a hit or
silently overwritten. Retain damaged evidence and select a new output root
after investigation. Changed bindings create a new namespace, preserving old
artifacts. Unexpected programming/runtime exceptions propagate and stop final
publication while committed records remain resumable.

Resume rechecks source bytes and all cached canonical hashes, skips successful
parsing/windowing, and validates current contracts. Unchanged bounded pre-read
rejections are re-observed without claiming a source hash or cache hit; an
accepted source changing bytes in place fails closed.

If a formerly unreadable, hash-mismatched or oversize source now verifies,
resume fails with `pdmx.resume_source_changed`. The previous unverified
quarantine, journal, shard and publication remain byte-identical; successful
verification cannot relabel that old attempt as a cache hit. Processing the
restored source requires an explicit retry with a new output root.

Corrupt existing artifacts are rejected rather than repaired in place. Resume avoids retained
canonical objects and reads one window at a time for verification. Callback
interruption after a committed record is covered by byte-identical resumed
versus uninterrupted fixture artifacts. Cache-hit counts are execution
observations on `PipelineResult`/CLI stdout, not part of deterministic artifact
identity. They distinguish source records and windows; filtered records never
become hits.

## Loader and reporting

`PdmxWindowDataset.from_output(root, split=...)` validates current contracts and
the complete split manifest using metadata only. Direct construction also
checks the matching root contract and current shared corpus index. Indexing
reads one canonical artifact through `load_cached_piece`, verifies raw and
split identity, builds the existing raw graph and returns `SSLRawSample` for
`collate_ssl_samples`. No target sidecar is opened. Existing corpus indices
also remain usable at the ordinary multisource boundary; the PDMX loader is
the path enforcing the additional PDMX versions and split locks.

Reports separate input records/components, selected/filtered/accepted/quarantined
records, window-bearing components, windows, rejection reasons, resource
rejections and clipping occurrences. Accepted means parsed with a window result,
not necessarily a nonempty/SSL-eligible window. Graph node/edge p50/p90/p99 use
exact nearest-rank values and a denominator of graphs actually built; otherwise
they are `not_computed`. Fixtures are identified as synthetic. Corpus
distribution, throughput and production readiness are not inferred.

## Explicit CLI and next acceptance

Importing the package or CLI performs no source/cache scan. All paths are
required; `configs/pdmx/synthetic.json` is a bounded fixture configuration.

```text
PYTHONPATH=src <python> scripts/preprocess_pdmx.py \
  --source-root <explicit-root> --manifest <source-manifest.json> \
  --splits <source-splits.json> --config <config.json> --output <ignored-output-root>
```

The future real preprocessing acceptance requires separately authorized scope
and a tracked planned experiment record before execution; verified supplied
release/license artifacts and schema compatibility; explicit source selection,
duplicate/lineage and split manifests; budgets/geometry; retained full logs,
hashes and report; and terminal registration before scientific decisions.
Known cross-corpus lineage gaps and SSL eligibility/mixture/evaluation policies
must be addressed before mixed-corpus SSL. The historical 1k gate is one possible
engineering budget, not a limit. TEST, real-data preprocessing/evaluation,
training, checkpoints, SSH, MXL ingestion, probabilistic PLL/ranking objectives
and effectiveness claims were not part of WS3.
