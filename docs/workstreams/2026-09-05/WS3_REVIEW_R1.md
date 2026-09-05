# WS3 independent review R1 — 2026-09-05

Status: **changes requested; independent review complete**.
Worktree: /Users/aavalsamakin/.codex/worktrees/06c5/music-critic-v2.
Base/HEAD: f0e0886161c9f67786011508006db61c803c1752.
Verified full review patch SHA-256:
8034c147541a63f31452ecc71f71fbe3a5b558eed537790b65b51a81ccf12edc.
Size: 157296 bytes, 16 changed/new files (14 untracked). The coordinator
independently regenerated git diff --binary HEAD plus sorted no-index diffs
for all untracked files and matched the author's /private/tmp patch byte for byte.

## WS3-R1-001 (P2): obsolete pre-read rejection becomes cache hit
pipeline.py returns existing_shard(), True after a newly successful source
verification without checking whether that shard ever observed the expected
source hash. A previous source_unreadable rejection is therefore reused even
after the exact expected file appears, retaining obsolete quarantine and
claiming a cache hit.

Coordinator synthetic runtime reproduction:
1. Manifest binds a tiny metrical JSON fixture's expected SHA, but file is absent.
2. First preprocessing publishes reason pdmx.source_unreadable.
3. Write exactly the expected fixture bytes, preserving manifest/config/output.
4. Resume returns accepted_records=0, same source_unreadable reason and
   cache_hit_records=1 instead of a changed-observation error.

Requested bounded correction: after successful verification, previously
unobserved/pre-read rejection may not be a cache hit. Fail explicitly as
resume_source_changed and preserve previous sealed artifacts; retry requires
a new explicit output. Add unreadable-to-valid and hash-mismatch-to-valid
regressions, other applicable pre-read transitions, unchanged rejection/nohit,
unchanged accepted/hit and artifact non-overwrite assertions.

## WS3-R1-002 (P2): malformed JSON number escapes quarantine
adapters/pdmx.py parses JSON floating values with Decimal but does not catch
decimal.InvalidOperation at the parse boundary. A numeric token such as
1e999999999999999999999999999999 raises this exception before field range
validation. The pipeline deliberately catches only contract errors, so one
malformed record can abort processing instead of entering quarantine.

The coordinator independently reproduced InvalidOperation through the actual
load_pdmx_piece entrypoint on a temporary synthetic JSON with an extreme qpm
token. Requested correction: translate InvalidOperation narrowly at JSON parsing
to PdmxAdapterError("invalid_json"). Add adapter and mixed-source pipeline
regressions; unexpected programming RuntimeError must still propagate.

## Review and tests
Coordinator read actual manifest/grouping/pipeline/cache/loader and source
contract/CLI. Additional independent pipeline review confirmed the same
resume finding. Documentation/provenance review found no blocker. Adapter/
window review found WS3-R1-002 and no further blocking issue.

Independent CPU gate:
```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q -p no:cacheprovider \
tests/adapters/test_pdmx.py tests/pdmx_pipeline tests/tasks/test_multisource_dataset.py \
tests/training/test_split_and_corpus.py tests/adapters/test_pdmx_eda.py \
tests/adapters/test_midi.py \
-k 'not test_dataloader_num_workers_zero_and_two_have_schedule_parity'
```
Result: **269 passed, 1 deselected, 2 warnings in 8.80s**.
One existing shared-memory worker regression was excluded from the coordinator
sandbox run; the author's full 270-test approved run is separately reported,
not claimed as an independent repeat. Warnings are upstream Torch JIT.

The source/filter/split-bound dataset artifact namespace is distinct from the
raw graph fingerprint; binding full preprocessing provenance to this namespace
is intentional. No corpus-wide readiness, source release authenticity, SSL
quality, throughput or actual data acceptance follows. All checks used
synthetic sources. Await remediation of both findings and a new revision.
