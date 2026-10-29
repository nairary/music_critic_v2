# HIERGRAPH launch preparation: independent review

2026-10-28. Review of local WIP against [spec.md](spec.md), the accepted
[E1–E5 campaign](../../2026-10-27/hierarchical-graph-analyzer/campaign.md),
AGENTS.md and the repository data/evidence contracts.

Repository: `/Users/aavalsamakin/.codex/worktrees/harmonic-event-pilot/music-critic-v2`.
Comparison base: detached HEAD `287bf856643f9e555352e1661a7bf0563772178a`.
The reviewed package, CLI and review tests are untracked WIP. Their actual files
were inspected directly; an empty tracked diff was not treated as review evidence.
No original-checkout edits, Git mutations, corpus reads, pretrained-checkpoint
reads, external calls or scientific training were performed by this reviewer.

## Review ownership and scope

The model agent independently reviewed `data.py`, `overlay.py`, `matching.py`,
`losses.py`, `metrics.py`, `cache.py`, `runtime.py`, `pipeline.py` and
`scripts/run_hiergraph.py`, with their synthetic tests and relevant existing
canonical/graph/export collaborators. The same agent authored `model.py` and
its tests, so its model inspection is not an independent model review.
The supervision agent independently reviewed the model mechanism and identified
the E2 no-cut localization gap; the resulting positive-boundary supervision was
then independently checked by this reviewer. After coordinator-reported native
cache attempt002 failed, this reviewer authored the narrowly scoped validator
geometry exception translation in `cache.py`. That last amendment has explicit
red/green tests; the coordinator was asked to inspect it independently before
freezing the next source snapshot. A subsequent coordinator-requested amendment
adds a separate raw-only 64-track profile; the shared PLL/codec contracts remain
unchanged, and the same independent-review limitation applies to that amendment.

This review checks implementation and launch mechanics. It does not establish
native device admission, actual corpus coverage, run completion, musical quality
or superiority of any constructor. Live native-run state was not inspected. The coordinator subsequently reported
native cache attempt001 failure on a preexisting `work_id` binding and attempt002
failure on validator-rejected cropped bar geometry. This reviewer reproduced
the relevant contracts synthetically and verified the fixes without opening
native artifacts. Native failure/retry accounting remains
in the coordinator-owned experiment records.

## Standards verdict: pass after remediation

- Raw inference consumes only the fixed tensor contract. Targets, analytic views,
  masks, exact source geometry and canonical identities remain outside forward.
- Canonical target ownership and endpoint lookup use exact rational values.
  Floating model coordinates do not replace canonical timing.
- Unknown fields and unmatched predictions do not become implicit negatives.
  Matched positives take precedence over current empty-region geometry.
- Excluded raw tiles/works retain exact core/halo domains and contribute
  unrecovered known H through `missing_reports`; unsupported point events remain
  explicit and owned point counts are aggregated without repeated global counts.
- Source/data/cache/config bindings and immutable checkpoint paths are enforced.
  Raw-cache schema drift and changed E2 capacity on checkpoint replay are refused.
- CLI admission binds mode, config bytes, source/target/cache identities, output
  leaf, wall cap and training condition to the planned record. Positive integer
  wall caps exclude booleans and fractional values that could disable `alarm`.
- Partial training progress is retained per applied/attempted draw. Native gate
  receipts retain each completed sample even if a later sample in that arm fails.

## Specification verdict: pass for the H-constructor preparation slice

E1 uses the supplied raw span bank; E2 creates contiguous bounded segmentation;
E3 predicts anchor-relative endpoints; E4 uses learned query cross-attention;
E5 uses independent sigmoid graph assignments with input-normalized pooling and
separate endpoint prediction. All use the same raw graph encoder, H refiner and
H/F heads. Raw note paths remain available outside E5 pooling.

The approved E2 amendment adds `boundary_positive` BCE with default weight 1 on
known H endpoints that the target-side adapter maps to exact raw-grid indices.
Unknown grid positions receive no negative labels; off-grid endpoints are counted
and excluded. This supplies supervised boundary-confidence gradients when E2
currently emits no interior cut. The model's straight-through endpoint surrogate
and expected-boundary profile alone were not claimed to solve that no-cut case.

Oracle-H uses an explicit no-gradient readout and is reported separately from
predicted-H recovery. Native gate selection uses unique raw tiles before taking
small/median/large sizes; analytic view multiplicity cannot weight that selection.
Each gate condition requires a supervised nonzero-gradient sample.

The implemented hierarchy is raw note/onset/beat/bar/track/song/slice/fragment/
full-work-bar states followed by H construction and iterative H refinement.
Raw states stay fixed during H refinement. H-to-raw semantic feedback, contextual
h/f heads, P processes, addressed p relations and the A/B/C bidirectional comparison
are later phases. This implementation must not be described as the completed
n↔h↔H↔P analyzer.

## Raw-only track-cap follow-up

The inherited `ContextProfile` cap16 belongs to the conditional PLL path:
`pll/contracts.py` constrains ContextProfile/CodecProfile and Event tracks, and
`pll/model.py` requires context capacity not to exceed codec capacity. The direct
`PLLContextEncoder._encode_graph` path does not consume those codec limits.
Raw graph track rows are dynamic, source track index is a continuous feature,
and `models/hierarchy.py` constructs a dynamic song/bar/track token sequence
with sinusoidal ordinal positions rather than a 16-track embedding table.

`cache.RawGraphProfile` therefore supplies seven validated raw geometry fields
with max_tracks64, while shared PLL ContextProfile/CodecProfile/Event remain
unchanged. The cache manifest binds the new limit, so older16-track caches are
not silently reused. Existing fragment `track_index/16` remains a continuous
feature with its prior scale; no existing <=16-track feature values are changed.
Other note/edge/time/fragment limits remain in effect.

Synthetic 17-track and64-track pieces passed real tiling, raw graph construction,
the genuine randomly initialized C3 `_encode_graph` seam, and fragment assembly:
every track/note is retained and C3 outputs finite `[count,128]` states.
Public PLL ContextProfile and CodecProfile still reject max_tracks17. The missing
RawGraphProfile test first failed, then both new tests passed in0.082 seconds.
This establishes dimensional/mechanical support, not pretrained quality or
native resource admission for the newly included real works.

## Findings and disposition

| Finding | Original severity | Verified remediation |
|---|---|---|
| One prediction received positive and negative objectness simultaneously | P1 | Matched positive indices removed from negative loss; regression passes |
| E2 no-cut case had no supervised localization path to boundary confidence | P1, independent model reviewer | Exact positive endpoint BCE; no-cut gradient and unknown-grid tests pass |
| Excluded raw tiles lacked core geometry for known-H denominator accounting | P1 | Exact exclusion metadata plus missing-raw reports; independent tests pass |
| Cache replay admitted changed type schema; final context edges bypassed cap | P2 | Schema/feature/limit checks and final edge cap; independent tests pass |
| Reader sample identity could differ from logged scheduled metadata | P1 | Exact work/core/halo/view/scope check before forward/update; regression passes |
| Same parameter shapes allowed E2 capacity drift on resume | P2 | Behavioral dimensions added to checkpoint signature; regression passes |
| Planned record did not bind the actual launch; partial training updates could be lost | P1, prior integration reviewer | Exact launch admission and per-step state callback; independently checked |
| Native failure discarded already completed samples in the current arm | P1 | Condition registered before its loop and receipt flushed per sample; independent failure test passes |
| Gate median was weighted by analytic-view multiplicity | P2 | Deduplicate work/tile before size ranking; inspected |
| Point-event total was repeated across tiles | P2 | Aggregate `point_events_owned`, omit per-query global total; independent test passes |
| Fractional/bool cap could bypass intended outer deadline semantics | P2 | Strict positive integer admission; independent tests reject 0.5 and True |
| Accepted source binding already contained `work_id`, causing duplicate-keyword construction | P1, reported native attempt001 | Merge as `dict(binding, work_id=wid)`; all cache fixtures now include production `role`/`work_id`, and focused regression passes |
| Cropped irregular bar geometry raised a validator GraphBuildError outside the existing unsupported-geometry path | P1, reported native attempt002 | Narrow prefix-checked translation at graph construction; exact exclusions preserved, other graph/encoder errors stay fatal; red/green tests pass |
| Inherited PLL codec cap16 excluded otherwise representable raw multi-track works | P2, coordinator coverage review | Separate validated RawGraphProfile64; genuine C3 synthetic17/64-track tests pass; shared PLL/codec remain unchanged |

The C3 raw graph builder does not support every irregular source measure after
raw tiling. Only `GraphBuildError` raised directly by `build_tile_graph` with
prefix `build_raw_graph requires a validator-clean CanonicalPiece:` becomes an
explicit raw-geometry exclusion. Its reason is preserved verbatim with exact
core/halo metadata. Other graph safety/framework errors, encoder errors and
nonfinite states remain failures. This does not repair source geometry or invent
C3 states: excluded tiles and their owned global-bar states remain unavailable;
known H in those cores remain in recovery denominators through missing reports.

No unresolved blocker remains in this reviewed implementation scope. Final run
admission must use the frozen source snapshot actually deployed; subsequent code
changes require an affected-scope recheck and new source binding.

## Synthetic verification

Latest affected-scope command after the irregular-geometry and raw-track-profile amendments:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/experiments /private/tmp/musiccritic-hiergraph-venv/bin/python -m unittest test_hiergraph_cache_review test_hiergraph_cache test_hiergraph_pipeline -q
```

Observed result: **17 tests passed, 0 failures, 0 skips, 0.212 seconds**. The
precise validator-error regression first failed with the expected GraphBuildError,
then passed after the narrow translation. Other GraphBuildError, encoder errors
and nonfinite-state fatal behavior remain covered.

Earlier combined command, before that final amendment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tests/experiments /private/tmp/musiccritic-hiergraph-venv/bin/python -m unittest test_hiergraph_cli_review test_hiergraph_cache_review test_hiergraph_runtime test_hiergraph_losses test_hiergraph_model test_hiergraph_pipeline -q
```

Observed preceding-snapshot result: **65 tests passed, 0 failures, 0 skips, 0.251 seconds**.
One dependency `torch.jit.script` deprecation warning appeared during graph imports.
The independent additions now comprise seven CLI/gate/coverage tests and twelve cache tests. The new accepted-binding
regression separately passed in 0.040 seconds before the affected combined run.

Cache tests replace only pretrained checkpoint acquisition with an explicitly
synthetic deterministic `nn.Module`; canonical loading, sanitization, raw tiling,
graph construction, fragment assembly, serialization and cache authentication are
real. Model/gradient/runtime tests use genuine PyTorch, not module stubs.
Earlier broad synthetic HIERGRAPH regression: 75 passed in 1.383 seconds. Its first
sandbox run hit seven `PermissionError: ps` failures in existing exporter memory
checks; rerunning with the required process-inspection access passed.

## Reviewed content identities

SHA-256 values at final review snapshot:

```text
scripts/run_hiergraph.py d53dceb180eedb0c95da6944bde879aaa7e8c2b4b9613cbbd5850e6d4ce4c1e8
cache.py bbf4759f694bddf7723e7aacc69f11f392b4a6fafa2ff1d4074d8c8a4e6ac753
runtime.py 090ebc684a05951cfb5e4152ac98cfad30d846b9e1318e357f1eee284d8c1e27
pipeline.py aa221aca8b4f38ee4a4886a79d111daa1095944086a75d8c6510bfbacba34f2b
losses.py 45682da02a69fb2debeb9cb93a7e6da12405917b385c105dcfb4d4d558d68a6e
model.py 0b63cea7a38755da2946b2b6b118fcd43c4cefb0c92e3d3d58cc73b2e3a01d88
test_hiergraph_cli_review.py 6c0ce7432195df247768b82727687eac4db51d4f360f7abb9ff2856eba4b36b7
test_hiergraph_cache_review.py 86407a68cf798d2d5cde9b71ba0f7e0044c97e124dc60bd68ede425aed9e7c50
```

Module basenames above are under
`src/music_critic/experiments/hierarchical_graph_analyzer/`; test basenames are
under `tests/experiments/`. This compact receipt does not replace a complete
run source manifest or the experiment registry.
