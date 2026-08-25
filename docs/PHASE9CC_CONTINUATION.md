# Phase 9C-C exact continuation to 15,000 and 21,000 updates

The same fail-closed state boundary also supports a sealed second generation
from the verified update-15,000 bundle to update 21,000. That production
profile is restricted to milestones 15,000/18,000/21,000 and the exact parent
manifest/report/checkpoint/state bindings recorded in the implementation. It
uses a new output root, proves schedule-prefix identity, restores full state,
fails on any skipped update, keeps test locked, and makes no automatic plateau
or superiority claim.

## Scope

This continuation answers only whether the verified seed-17 scratch and SSL
MLP cells continue improving after update 9,000. It restores the exact model,
optimizer, scaler, scheduler-null state, RNG and epoch-zero sampler position
from each immutable parent checkpoint and advances to update 15,000. It does
not reload the SSL encoder export, start another epoch, or mutate the parent
bundle.

The production parent is fixed by:

- Git SHA `bff1a405ffb9d8d6de01c4abc3d567dcb02d000b` on branch
  `phase/9cc-mlp-convergence-diagnostic`;
- manifest fingerprint
  `6e64f33e64de9c3d864d75828a6916d95afa9fcbadc75c14359b884cab83ab10`;
- scratch update-9000 SHA-256
  `1b3d6ac9a3d2d6e90687abf1838529c412807b6c41492cc497448e83d150072f`;
- SSL update-9000 SHA-256
  `2ffb2fc03f8901455d8b99696bcc97964b69614370c125cafb6aaf6d073c0239`.

The old plan did not record the path of the JSON file used to construct it.
The continuation therefore requires that path explicitly and compares every
file, hash, cache root, SSL source/export and learning-rate binding against the
verified parent plan. Its `git_head` must bind the continuation implementation
SHA, while the parent SHA remains separately fixed in `parent_binding.json`.
A similarly named config is never substituted.

## Preflight and continuation

Before either optimizer may step, both update-9000 checkpoints are hash- and
state-checked, strictly reconstructed, evaluated again on the same validation
membership, and independently reloaded for the existing CUDA logits replay
comparator. Report metrics use the comparator's existing absolute/relative
tolerances; supports and candidate identities are exact.

The 15,000-update epoch-zero schedule is rebuilt with the same production
dataset view, target sidecars, quota sampler, seed and batch size. Its first
9,000 batches must equal the complete parent schedule. Telemetry covers global
updates 9,100 through 15,000 every 100, and atomic continuation checkpoints are
saved at 10,000 through 15,000 every 1,000. Validation is limited to 9,000,
12,000 and 15,000. Resume advances the same loader before restoring checkpoint
RNG and never appends duplicate committed telemetry.

## RTX 3090 command

Use the exact parent root and the exact config path from the completed run. The
new root must be distinct:

```bash
scripts/run_phase9cc_rtx3090_convergence.sh continue \
  <EXACT_CONTINUATION_SHA> \
  <EXACT_PARENT_OUTPUT_ROOT> \
  <EXACT_PARENT_CONFIG_JSON> \
  outputs/phase9cc-continuation-seed17-<UTC_TIMESTAMP> \
  --start-update 9000 \
  --target-update 15000 \
  --validation-milestones 9000,12000,15000
```

The same command resumes an interrupted unsealed root. It verifies a sealed
completed root without changing it. Completion is printed only after the full
execution log has been copied into the new root, the manifest has been sealed,
and the independent verifier passes.

Independent verification is also available without training:

```bash
.venv/bin/python scripts/verify_phase9cc_continuation.py \
  --bundle <NEW_CONTINUATION_ROOT> \
  --expected-sha <EXACT_CONTINUATION_SHA>
```

The resulting report combines milestones 0, 1,000, 3,000, 6,000, 9,000,
12,000 and 15,000, records the requested deltas, SSL-minus-scratch gaps,
descriptive best milestones, final-minus-best and continuation train-loss
slope. It contains no automatic plateau, superiority or significance verdict,
and test remains locked.
