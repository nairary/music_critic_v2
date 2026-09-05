# WS3 result — PDMX pipeline and cache

## Review identity

- Worktree: `/Users/aavalsamakin/.codex/worktrees/06c5/music-critic-v2`.
- Base and current HEAD: `f0e0886161c9f67786011508006db61c803c1752`; initial worktree clean.
- Changes are uncommitted; no branch/base switch, commit, push, PR or merge.
- Status: R1 resume and decimal JSON corrections implemented; synthetic checks passed;
  awaiting coordinator review of the revised diff. Not yet accepted/integrated.
- The final coordinator handoff contains a SHA-256 of the complete tracked and
  untracked review patch. This report does not self-hash its own changing bytes.

## Implemented behavior and files

- `src/music_critic/adapters/pdmx.py`: dependency-free raw metrical MusicRender
  JSON projection, exact ticks, optional metadata, raw meter/tempo/bar/beat
  geometry, explicit defaults, grace handling, input/resource rejection and
  early raw graph categorical meter bounds.
- `src/music_critic/pdmx_pipeline/manifest.py`: versioned explicit source and
  selection metadata, release/license gaps, portable paths/hashes, declared
  filters, transitive exact-hash/supplied-lineage components before filtering,
  and complete split coverage/TEST guard before readers.
- `src/music_critic/pdmx_pipeline/windowing.py`: configurable bounded bar
  windows, global IDs/ranges/exact offsets, original note IDs, explicit
  clipping and start context. Interior right-boundary tempo/meter/key events
  belong to the next window; true source-terminal events remain.
- `src/music_critic/pdmx_pipeline/pipeline.py`: bounded ordered thread queue,
  existing canonical cache composition, sealed per-source shards/journals,
  atomic final publication, corruption/stale-contract rejection, byte-identical
  workers/relocated roots/resume, source revalidation and detailed report.
- `src/music_critic/pdmx_pipeline/loading.py` and `__init__.py`: public
  split/version-guarded lazy windows yielding existing `SSLRawSample` values
  for the unchanged SSL collator; no target-sidecar reader.
- `scripts/preprocess_pdmx.py`, `configs/pdmx/synthetic.json`: explicit
  source-root/manifest/splits/config/output CLI, with no scan on import.
- `tests/adapters/test_pdmx.py`, `tests/pdmx_pipeline/test_pipeline.py`,
  `tests/pdmx_pipeline/test_windowing.py`: 97 initial tiny synthetic cases;
  103 including the R1 additions below.
- `docs/PDMX_PIPELINE_CONTRACT.md`: supported schema, grouping, clipping,
  cache/resume/loader contracts, scientific boundaries and next acceptance.
- `docs/STATUS.md`: separate WS3 task entry.
- `docs/DECISIONS.md`: append-only ADR-WS3-001.
- This report: `docs/workstreams/2026-09-05/WS3_RESULT.md`, with root
  `WS_RESULT.md` pointing to it.

The shared corpus/loading/schema/graph/trainer APIs, dependencies, historical
PDMX EDA replay, experiment ledger/terminal records and frozen C0/C2 profiles
were not edited. Legacy was neither inspected nor reused; no legacy runtime
import exists. Reused logic is current V2 canonical validation/serialization,
cache/index, split validation, graph builder and SSL sample/collator.

## Initial handoff verification (before R1)

From this worktree:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q tests/adapters/test_pdmx.py tests/pdmx_pipeline tests/tasks/test_multisource_dataset.py tests/training/test_split_and_corpus.py tests/adapters/test_pdmx_eda.py tests/adapters/test_midi.py
```

Final result: **270 passed, 2 warnings in 20.37s**. No skipped tests.
Warnings are the existing Torch `torch.jit.script` deprecations.

An earlier sandboxed run on an earlier revision had 259 passed and one failed:
`test_dataloader_num_workers_zero_and_two_have_schedule_parity` could not start
`torch_shm_manager` (`Operation not permitted`). The identical isolated test
outside sandbox passed (1 passed, 2 warnings in 11.99s). The final complete
270-test selection ran outside sandbox through the approved tool execution;
the shared environment was not modified. The failure is retained here rather
than counted as an initial pass.

```text
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/music-critic-v2-integration-venv/bin/python -m compileall -q src/music_critic/adapters/pdmx.py src/music_critic/pdmx_pipeline scripts/preprocess_pdmx.py tests/adapters/test_pdmx.py tests/pdmx_pipeline
git diff --check
```

Both exit 0. No full suite or opt-in real-data integration tests were run.

Checks include exact timing/meter changes; optional metadata; malformed, empty,
oversize inputs; grouping before windows and filtered lineage bridges; clipping,
context, grace and incomplete bars; worker and relocated-root byte equality;
interrupt/resume; partial files; referenced corruption; stale window/projection/
upstream/shared graph contracts; lazy one-window reads; pre-reader TEST denial;
raw graph identity independent of annotations/song_length/title/roles/split/
group; metadata-only future production mode guards; explicit CLI; and bounded
60-item generated queue/64-bar window iterator consumption.

## Schema evidence and limits

Primary format definitions are pinned to
[MusicRender reading/music.py](https://github.com/pnlong/PDMX/blob/5c6a25b613e4497469e855e8bd1423b1d834ff45/reading/music.py)
and
[reading/classes.py](https://github.com/pnlong/PDMX/blob/5c6a25b613e4497469e855e8bd1423b1d834ff45/reading/classes.py)
at upstream commit `5c6a25b613e4497469e855e8bd1423b1d834ff45`, read using the
GitHub connector. The respective Git blob hashes are
`32b03c7ce42c318e595ff8af65935fd3f29c5030` and
`6f91117a5a380bf88579424527aa3a5079018d02`.
No upstream code was executed.

The concrete PDMX release, supplied-source compatibility, release/license
artifact authenticity and cross-corpus lineage remain unknown/unverified.
Source-code pinning does not replace any of them. Production mode requires
explicit declared bindings; the implementation does not authenticate those
claims. Only synthetic generated sources were processed in WS3.

Supported input is uncompressed metrical JSON, not absolute seconds, MXL or
serialized executable objects. Internal irregular shortened bars and
fractional-beat pickups outside the unchanged canonical contract reject
explicitly. Voices are unavailable rather than invented. Notation/lyrics/chord
objects do not generate raw notes, semantic segmentation or theory targets.
Upstream song_length is ignored because it can include semantic annotations.

## Fixture evidence and unresolved acceptance

- The generated four-source determinism fixture has four known components and
  eight windows; one versus two workers and different source roots produce
  byte-identical artifacts.
- Graph-statistics fixtures build four actual graphs from two generated
  sources. Their report denominator is four; no corpus-size extrapolation.
- Decoded payloads are bounded by source budgets and worker queue. Global
  metadata remains O(records + windows). There was no corpus memory/throughput
  stress test and no throughput claim.
- One writer per contract root is required. Corrupt referenced artifacts fail
  closed; automatic in-place repair and concurrent writers are not implemented.
- Explicit VALIDATION overlap requires later aggregation by original source
  piece. Objective-specific SSL eligibility/weighting and cross-corpus split
  coordination remain future caller responsibilities.
- No scientific experiment record was created for ordinary unit/contract tests.
  No real corpus, TEST input/target/loader, trained checkpoint, SSH, GPU,
  training/evaluation run or active C0 process was accessed.

Next engineering step: coordinator actual-diff review and isolated integration.
Next Phase 10 evidence step, after separate authorization: planned experiment,
verified real release/license/schema bindings, explicit source/lineage/split
manifests and budgets, bounded real preprocessing acceptance with retained
artifacts and terminal registration. The historical 1,000-record example is
not a hard-coded limit. Scaled SSL and Phase 11 remain outside this result.

## Review R1-001 — restored source cannot reuse unverified quarantine

The coordinator reviewed the initial complete patch SHA-256
`8034c147541a63f31452ecc71f71fbe3a5b558eed537790b65b51a81ccf12edc`
(157,296 bytes). Its independently reported CPU gate was `269 passed,
1 existing shared-memory worker case deselected, 2 warnings in 8.80s`.
That revision was **not accepted**: an unreadable source becoming available
with the exact expected bytes reused its old `source_hash_observed=false`
quarantine and incorrectly counted a cache hit. Initial test results above
remain the history of that reviewed revision, not evidence that R1 was absent.

The R1-001 correction affects only `pipeline.py` and its regression tests,
plus the PDMX contract and WS3 reports/STATUS. After successful source
verification, an existing result is a hit only when its observed-hash flag is
exactly true and its reason is not one of the pre-read failures. Otherwise
resume raises `pdmx.resume_source_changed` and directs the caller to an explicit
new output root. The prior journal, shard and publication are not relabeled,
repaired or overwritten. The existing artifact fields/versions and shared
APIs are unchanged; adapter/windowing were not edited in R1-001.

Three new parameterized synthetic cases cover unreadable, hash-mismatched and
oversize sources restored to their manifest's expected bytes under unchanged
config/output. Each checks the unchanged pre-read rejection remains a non-hit,
the restored-source failure calls no adapter, and every prior output byte
remains identical. Each also exercises a successful explicit new-root retry,
then an unchanged accepted cache hit without reparsing. At this checkpoint
there were 100 new synthetic cases in WS3.

Targeted command:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q tests/pdmx_pipeline/test_pipeline.py -k 'restored_unverified_source or changed_source_bytes or bad_empty_oversize or resume_is_byte_identical'
```

Result: `6 passed, 28 deselected, 2 warnings in 0.90s`. An independent focused
review found no additional issue and repeated the six cases successfully.

Complete prescribed command after R1-001, before R1-002:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q tests/adapters/test_pdmx.py tests/pdmx_pipeline tests/tasks/test_multisource_dataset.py tests/training/test_split_and_corpus.py tests/adapters/test_pdmx_eda.py tests/adapters/test_midi.py
```

Result: **273 passed, 2 warnings in 20.17s**, no skips/deselections. As in the
initial handoff, this authorized synthetic CPU selection ran outside sandbox
for the existing two-worker shared-memory regression. The shared runtime was
not modified. Compileall and `git diff --check` pass. No real source, TEST,
legacy, checkpoint, training or experiment-record access was introduced.

Base/HEAD remains `f0e0886161c9f67786011508006db61c803c1752`, uncommitted.
The revised full patch and its fingerprint are supplied in the R1 handoff;
the original patch is retained separately. Next step is coordinator review of
the corrected revision, including any separate adapter/window findings.

## Review R1-002 — invalid decimal JSON is a source rejection

The coordinator additionally reproduced an uncaught `decimal.InvalidOperation`
for the tiny JSON numeric token `1e999999999999999999999999999999`. It escaped
`json.loads(parse_float=Decimal)` and could stop a mixed preprocessing pass.
The new pipeline regression reproduced that failure before the correction:
`1 failed, 2 warnings in 0.83s`. The direct adapter regression independently
reproduced the same exception before the fix.

`adapters/pdmx.py` now catches `InvalidOperation` only in its existing JSON
decoding exception boundary and emits `PdmxAdapterError("invalid_json")`.
No broad pipeline catch was added. The direct raw-JSON test and a mixed
bad/good source test prove that the invalid record is quarantined while the
valid record yields its two windows and final publication. The existing
programming-error regression now explicitly covers both `TypeError` and
`RuntimeError`, which still propagate. Windowing and shared APIs remain
unchanged. Together R1-001/R1-002 add six cases, for 103 new WS3 cases total.

Targeted combined command:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q tests/adapters/test_pdmx.py tests/pdmx_pipeline/test_pipeline.py -k 'decimal or programming_exceptions or restored_unverified_source'
```

Result: `9 passed, 73 deselected, 2 warnings in 0.88s`. The complete direct
adapter suite separately passed `46 passed, 2 warnings in 0.70s`.

Final complete prescribed command with both R1 fixes:

```text
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q tests/adapters/test_pdmx.py tests/pdmx_pipeline tests/tasks/test_multisource_dataset.py tests/training/test_split_and_corpus.py tests/adapters/test_pdmx_eda.py tests/adapters/test_midi.py
```

Result: **276 passed, 2 warnings in 20.33s**, no skips/deselections. The warnings
are the unchanged Torch deprecations. This is the same authorized synthetic CPU
gate outside sandbox for the existing shared-memory worker case; no dependency
or shared environment was changed. Compileall and `git diff --check` pass.
The previous 270- and 273-test results are retained as prior-revision history.
No real corpus, TEST, checkpoint, legacy, SSH/GPU or training access occurred.

Both fixes are included in the new complete review patch; its new SHA-256 and
byte size are in the coordinator handoff. The base/HEAD remains unchanged and
no commit was created. Await coordinator re-review; acceptance/integration is
not claimed.
