# WS1 independent review — 2026-09-05

Status: **accepted for isolated integration**, not integrated into main.
Worktree: /Users/aavalsamakin/.codex/worktrees/ef79/music-critic-v2.
Base and current HEAD: f0e0886161c9f67786011508006db61c803c1752.
Accepted content fingerprint:
a2bb43ea55053aa2bcd6bb7f3b04372ab381ae398c9183caea319250ee7bcf2c.

## Revision binding
The coordinator independently reproduced the supplied fingerprint over nine
changed tracked/untracked files. It hashes compact sorted JSON mapping each
repository-relative file path to SHA-256 of its current bytes, excluding only
WS_RESULT.md and docs/workstreams/2026-09-05/WS1_RESULT.md. This binds code,
tests, contract, STATUS and DECISIONS despite unchanged HEAD. A later content
change requires a new review; the implementation author's prose alone is not
the acceptance evidence.

## Review findings
No blocking findings in the reviewed revision.
- Separate complete transposition recomputes current pitch-derived features
  with builder arithmetic and availability. The independent canonical rebuild
  oracle covers full graph fingerprints. Single validated default-registry
  graph support and absence of throughput evidence are explicitly documented.
- Historical transposition.py, corrected training routing, frozen config/fixture
  paths, experiment records and ledger are unchanged.
- HookTheory chord extent and unused late key annotations do not determine raw
  duration. Clipped/wholly outside target spans retain reason-bearing semantics
  and masks; accepted melody pitch reconstruction remains unchanged.
- Raw projection covers fields currently read by the graph builder.
  HookTheory-only cache glue uses the existing cache_input_sha256 extension.
  Full source and canonical hashes remain available; distinct annotations with
  one raw key have distinct immutable canonical artifacts.
- New adapter/version identity keeps historical artifacts distinguishable.
  No shared graph/cache schema, dependency or existing training protocol changes.

Two additional independent agents statically reviewed transposition and
HookTheory/cache separately and found no blockers. They did not run tests;
their reviews supplement the coordinator's actual-diff inspection and test run.

## Independent execution
From the implementation worktree:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/music-critic-v2-integration-venv/bin/python -m pytest -q \
tests/experiments/test_complete_transposition_inputs.py \
tests/experiments/test_phase9eb5a_transposition.py \
tests/experiments/test_phase9eb5f_transposition_correctness.py \
tests/experiments/test_phase9eb5g_directed_inverse.py \
tests/experiments/test_phase9eb5h_full_orbit_profile.py \
tests/adapters/test_hooktheory_raw_timeline.py \
tests/adapters/test_hooktheory_adapter.py \
tests/training/test_split_and_corpus.py \
tests/test_repository_contract.py
```

Result: **187 passed, 3 warnings in 2.58s**. Two warnings are upstream Torch JIT
deprecations; one is pytest unable to write its optional cache in the separate
worktree under the coordinator sandbox. Test execution itself passed.
Subsequent coordinator runs should disable pytest cache and bytecode writes
when reading another task's worktree.

`git diff --check` passed. The frozen-path diff check passed.
The author's separately reported 310 distinct passing tests are retained in
WS1_RESULT.md; this review does not claim to have independently repeated all
310 or the multiprocessing sandbox escalation.

## Limits and next action
Only synthetic CPU mechanics are accepted. No corpus-wide readiness, historical
C2 impact, augmentation benefit, quality/PLL, GPU performance or general
malformed-source-parser claim follows. No real dataset, trained checkpoint,
TEST input/target/loader, SSH, legacy or scientific experiment was used.

Keep this revision available for integration with WS2/WS3/WS4. Review the
combined result and consolidate documentation before marking it integrated.
Adoption in future training remains a separately registered experiment.
