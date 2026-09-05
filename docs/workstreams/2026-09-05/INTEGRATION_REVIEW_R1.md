# Isolated integration review R1 — 2026-09-05

Checkout: /private/tmp/music-critic-v2-ws-integration-20260905.
Branch: codex/ws-integration-20260905, base/HEAD f0e0886161c9f67786011508006db61c803c1752.
Status: **changes requested for WS4; integration gate not passed**.

All four individually accepted revisions were reverified and copied; shared
STATUS/DECISIONS additions were combined while preserving base history.
Worker root WS_RESULT.md files are retained in their worktrees; dedicated
WS1/2/3/4_RESULT reports were copied without name conflicts. No main code changed.

The coordinator added two synthetic cross-component regressions, both passing:
1. Generated PDMX source to two raw cached windows, complete +2 transposition
   equal to canonical rebuild, fresh SSL binding, shared-backbone backward/step,
   unchanged source/theory-only weights and original raw graphs.
2. Retained theory state through a newly generated model-only checkpoint into
   the restricted frozen POP909 loader; identical native/metadata_removed
   evaluation views and piece reports, independent frozen state and preserved RNG.

The union of all workstream review gates plus these two tests produced:
**1 failed, 812 passed, 6 CUDA-only skipped, 2 deselected, 2 warnings in 53.72s**.
Selectors: /private/tmp/music-critic-ws-integration-tests.json (29 entries).
Log: /private/tmp/music-critic-ws-integration-tests.log.
The failed revision's 32-file implementation manifest is retained at
/private/tmp/music-critic-ws-integration-r1-code.json; its compact sorted-JSON
SHA-256 is 8a6c0a2b4bee2796938c8451134062688389e6729141bfaf6874b20ad27149d3.
The full review patch is retained at /private/tmp/music-critic-ws-integration-r1.patch
(570697 bytes, SHA-256 8d92fb19ae0eb2cf4d2a822e8d9722d1b4f3e9f55629ece8229033d661596978).
The original component source manifest is retained separately at
/private/tmp/music-critic-ws-integration-r1-sources.json; later remediation does
not overwrite these R1 artifacts.
Deselections: existing shared-memory two-worker case and one existing CUDA
evaluation case. No real-data integration modules were selected. Warnings are
upstream Torch JIT. No skipped/deselected case is counted as passed.

## INT-R1-001 (P2): WS4 smoke violates the existing import boundary
The failing repository-contract test identifies direct import mido in
experiments/pop909_cl_frozen/run.py. This dependency is already installed but
per repository architecture belongs in adapters/exporters/training, not this
experimental module. Individual WS4 focused gates omitted the repository-wide
import check, so this was first exposed by the joint gate.

Returned to WS4: move only the small generated MIDI construction into a new
adapters/pop909_cl_synthetic.py helper, preserving public synthetic_input and
fixture behavior. Coordinator explicitly authorizes this additional WS4-owned
new file. Existing adapters/shared APIs and the import-contract test must not
change; no dynamic-import bypass or new dependency is allowed. Re-run WS4 plus
repository checks, provide a new fingerprint, then repeat the joint gate.

No legacy logic, real data, trained checkpoints, TEST, SSH or C0 state was used.
Fixture-only engineering evidence does not establish model or corpus quality.
