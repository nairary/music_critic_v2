# Task03 — scientific execution, audit and compatibility retry — 2026-09-13

The bounded scientific data/evaluator/runner/CLI implementation is complete for
CPU-generated mechanics tests. The real Dilemmadata pilot remains unlaunched
here. The [project scientific audit](../../experiments/SCIENTIFIC_AUDIT_20260913.md)
separates existing theory evidence, corrected reporting defects, an unresolved
historical cache issue, and scientific limits. No main checkout, remote process,
real corpus, checkpoint, TEST, legacy repository, commit or merge was touched.

## Implemented behavior and file groups

- `group_masking_scientific_data.py` and `group_masking_scientific_cached_data.py`
  provide sequential native or externally hash-bound cached inputs, complete
  official split/component closure, common structural TRAIN eligibility and
  deterministic replay/recovery schedules. Prepared raw graph/sidecar/batch
  bindings are verified again on load. Cached sizing reads raw plus metadata
  only and can run before validation graph limits/class weights are supplied.
- `group_masking_scientific_eval.py` evaluates complete singleton VALIDATION,
  all 18 heads, VALIDATION-only classes, loss/metric coverage separately, both
  fixed masking policies and exact mergeable anti-collapse statistics. An
  unavailable SSL policy does not remove theory targets. Caller parameters,
  gradients, module modes and RNG remain intact, including interrupted work.
  Unknown TRAIN tuple inventory yields null seen/unseen metrics with a reason;
  supplied empty/populated inventories remain distinct and hash-bound.
- `group_masking_scientific_runner.py` runs baseline → A200 → A endpoint →
  A100 theory recovery → A recovered → B equivalent stages. Each policy has
  the same full C2/fresh SSL start. Recovery retains the full endpoint harness
  and uses fresh Adam. Stage manifests/checkpoints are immutable; default
  checkpoint interval50 avoids saving a full model after every update.
- Genuine interrupted pilot/recovery resume restores nonempty Adam and exact
  confirmed states/rows. Imported completed evaluations are recomputed and
  their receipts checked against actual forward/batch counts. Verification
  replay, confirmed training, unconfirmed outcomes and unknown work after the
  last immutable manifest remain distinct. Startup failures also retain receipts.
- `group_masking_scientific_cli.py`, `scripts/run_group_masking_scientific.py`
  and the deliberately unlaunchable `scientific_pilot.template.json` bind
  plan/record/code/source/request/prepared/resume bytes before payload work.
  `preflight-size`, `size`, `preflight-prepare`, `prepare`, `preflight-run`,
  `run` and `resume` are supported. Size is CPU-only and has its own planned
  engineering record. There is no production tiny-budget/fixture bypass.
- `autocast_runtime.py` and the narrow `group_masking_pilot.py` integration
  observe actual enabled/dtype through modern or legacy CPU/CUDA APIs, reject
  unavailable/malformed queries and retain runtime drift checks. No numerical
  setting, optimizer, masking objective or package dependency declaration changes.
- `tests/experiments/test_registry.py` now checks completed C0-120k with verified
  VALIDATION and unknown historical TEST/attempt counts. The terminal C0 record
  was not changed. All implementation modules have dedicated synthetic tests.

The execution contract and exact CLI examples are in
[SSL_SCIENTIFIC_EXECUTION.md](SSL_SCIENTIFIC_EXECUTION.md). Source fingerprint
for the new scientific entrypoint at acceptance:
`374d5c23bf9e8c0ed6ea418c70770e184319322fe105bca89399653d14221c87`.

## Exact verification and honest failure history

Runtime: Python3.12.14, Torch2.8.0, PyG2.7.0, pytest8.4.2, declared Hydra
available in `/private/tmp/music-critic-pilots-20260913/bin/python`.
Use `PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
No project dependency was added. Earlier CPU environments and historical server
versions are not assumed equivalent to this one.

| Invocation / artifact | Exact outcome | Interpretation |
| --- | --- | --- |
| Initial audit, `outputs/scientific_audit/2026-09-13/pytest.log` | 176 passed, 1 failed, 1 warning,72.82s | Stale registry test expected running C0 |
| Combined expanded suite, `pytest-final.log` | 344 passed, 1 failed, 1 warning,107.17s | Remaining stale ledger assertion in that same registry test; corrected afterward |
| Complete registry suite after correction, `pytest-registry-final.log` | 58 passed,0 failed/skipped,0.28s | Includes corrected lifecycle, unknown access/counters and ledger wording |
| Exact isolated compatibility source, `outputs/ssl_group/cuda-review002-retry002/pytest-standalone.log` | 58 passed,0 failed/skipped,1 warning,19.00s | Old review002 plus compatibility-only delta |

There are **345 distinct tests accepted** in the final current-source selection:
287 non-registry tests from the combined invocation and all 58 registry tests
from the final registry invocation. The isolated-source 58 tests overlap these
and are not added to the distinct total. Warnings are the existing PyG distributed
deprecation. The failed logs remain retained; no claim of a single all-green
combined invocation is made.

Combined arguments after Python:

```text
-m pytest -q -p no:cacheprovider
tests/test_repository_contract.py tests/experiments/test_registry.py
tests/graph/test_leakage.py
tests/ssl/test_group_masking_scientific_data.py
tests/ssl/test_group_masking_scientific_cached_data.py
tests/ssl/test_group_masking_scientific_eval.py
tests/ssl/test_group_masking_scientific_runner.py
tests/ssl/test_group_masking_scientific_cli.py
tests/ssl/test_autocast_runtime.py tests/ssl/test_group_masking_pilot.py
tests/ssl/test_group_masking_accounting.py tests/ssl/test_group_masking_cli.py
```

Independent review found no remaining blockers in the narrow evaluator,
autocast compatibility and sizing admission changes. Unit tests prove mechanics,
not real-data effectiveness, CUDA success, representative eligibility, hard
resource enforcement or original conversion provenance.

## Separate immutable CUDA retry002

`outputs/ssl_group/cuda-review002-retry002/` reconstructs and verifies all752
bootstrap files plus22 review002 files. Original implementation fingerprint
`259cf496e99a4d355bb051d68192107fd86e2b7f357b0d11609ebd6d81d7a7bd` matches.
Its compatibility-only source fingerprint is
`cd3a2fcdae8cbf9412de1ca2f23c7daa994d4adafeb5118cb0ca2dd471e599b0`.
The four-file delta contains pilot/helper/helper-tests and the new planned
`EXP-SSL-GROUP-CUDA-002` record. The new scientific workflow is excluded.

`artifacts.json`: SHA256
`90b76d4f3601845254034734fe3fe3133429c6e1dda8dad63d68051f96e7afcd`,149409bytes,
778 entries. `delta.patch`: SHA256
`4267f0c9f9eb627122e4036b416bd8a87a3d0e5fc3b047150588944e4be44c01`.
Reverse patch check passed.77 artifact references from the earlier review and
review002 were verified unchanged. This new package is sealed separately.

CUDA001 reportedly failed at the coordinator before the first update because
Torch2.2.2 lacks the modern getter signature. The coordinator owns its terminal
evidence. Local CUDA001 bytes are preserved, retry002 is only planned here, and
no native Torch2.2/CUDA success is asserted. This helper does not lower the
repository's declared Torch minimum. The external bootstrap must set and record
actual intra/inter-op counts before model/CUDA/parallel work; OMP=1 does not
prove inter-op=1. The compatibility-only internal guard retains its original
intra-op observation and does not independently bind inter-op.

## Remaining inputs and next phase

The coordinator has supplied compact source/index/class-weight bindings and a
raw JSON sizing receipt. Proposed whole-piece validation limits are explicit,
but actual edge/GPU acceptance and full source execution remain unverified here.
Before the scientific run, the coordinator must retain preparation evidence,
bind its prepared schedules/batches/validation bytes in a matching new planned
execution record, and use the external time/RAM/VRAM/swap wrapper. Existing
review002 planned scientific record is not a substitute for the new CLI contract.

Common TRAIN filtering can bias the admitted population; realised mask density
differs between independent notes and complete beat units. Neither is silently
normalised away. Historical production cache-hit source/identity binding remains
an open audit finding; the new cached provider deliberately uses independent
external bindings. No voice/rhythm/form task, PDMX ingestion, quality claim or
baseline promotion is introduced. The next phase is coordinator-owned CUDA
acceptance and registered bounded real-data execution, followed by interpretation
of its actual endpoint/recovery results.
