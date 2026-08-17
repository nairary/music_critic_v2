# Phase 9B.2C executable supervised smoke

Status: implemented; independent RTX 3090 execution is pending.

## Scope and contracts

`DilemmadataSupervisedSmoke@1.2.0` and
`DilemmadataSupervisedSmokeBundle@1.2.0` provide a bounded, fail-closed
mechanics gate for the already accepted Phase 9B.2B model and data contracts.
They do not change the raw projection, canonical cache, target cache, split,
graph, model-input, head, loss, or evaluation contracts. The model contract
advances to `1.1.0` solely to expose the typed post-prediction `supervise` API;
`forward` delegates to the same join/loss implementation.

The run is scratch-only, seed 17, explicit `cuda:0`, float16 autocast with an
enabled GradScaler, AdamW at `3e-4`, and 10--20 attempted updates. Its CE
inventory is exactly AN/DLC chord quality and inversion. Reconstruction has
weight zero; the five PU and 13 open-string tasks have no head or CE loss.

The stable production semantics are pinned to raw index
`c0451976b6b6eab88cb90aa6c47d6afdba1b81ce9b588f0f84daa846154adb0e`,
metadata index
`41e15e1d2edb1c52ad3ca90acf782bec7c26bfb042fea51dc805d6f86b52d0a7`,
719 records, aggregate `TargetBundle` fingerprint
`939ad5b871db28fefd76e47d56243ac2109a8bb01d57c6391f424ae943159072`,
split
`58ac7720f65f7fd3102248fb39d89291a78d65c06fc2ab9a16d78a6ee1666a3e`,
and model contract
`69a1ab3e6f5deb98a8bcfa26af7a3177b345ad157d164a3cf72e0273a0c58c81`.

The target-index fingerprint is an exact physical/run binding, not a universal
scientific fingerprint. Local and RTX builds observed respectively
`76feee8d128cc3c5dd1a5b261599df89ef241baa21d82b3c24202a11218beea4`
and `02fcf7eb03adda2962ade7223924e0fe44483e4900097bd33f50bf93b68d862a`
with the same stable semantic projection. Any self-consistent index is accepted
only after the complete checks below; its exact observed fingerprint is bound
to the report, checkpoint, reload, and evaluation. The cross-host physical
cause is a deferred technical investigation, not an RTX smoke blocker.

## Runtime and evidence boundary

The runner accepts only the raw index/cache, target index/cache, split
manifest, exact Git HEAD, and a dedicated output root. It requires a clean
tree outside that output root and the exact device name `NVIDIA GeForce RTX
3090`; there is no CPU fallback. A deterministic train selection may inspect
train target bundles to obtain complete four-head coverage. Validation
membership is selected without replacement from identity/component data,
without labels or validation target reads. Test samples, targets, inference,
metrics, and unlock all remain closed.

Runtime source conversion and alignment reconstruction are guarded by
fail-closed replacements, and the report requires zero calls. Candidate
prediction runs once for leakage evidence, before either target join. Original
and mutated targets are supervised against the same prediction object. The
gate proves exact candidate identities plus tensor object, storage, layout and
values before and after both joins, while requiring target and supervision/loss
evidence to change. This is independent of CUDA replay.

Independent CUDA+AMP forward replay is
`DilemmadataCudaReplayDiagnostic@1.0.0`: candidate identities are exact, all
logits must be finite, and FP32 comparison records max absolute/relative error
and cosine similarity. Acceptance uses elementwise `atol=0.005`, `rtol=0.005`
and cosine similarity at least `0.9999`. These fixed bounds cover half-precision
ULP and parallel-reduction ordering at unit-scale logits while rejecting
material drift; they are diagnostic evidence, never leakage evidence.
Checkpoint model tensors reload bit-exactly, while its independent logits
replay uses this bounded diagnostic. The remaining evidence covers source-entry
loss reduction, finite losses and gradients, encoder/four-head gradients and
parameter updates, attempted/applied/skipped counters, checkpoint state and
SHA, failure-atomic reload, official validation
metrics, VRAM peaks, and zero retained prediction/CUDA tensors after cleanup.
Before CUDA training, the source-free target-cache preflight verifies the
index self-fingerprint and current cache/adapter/registry contracts, exact raw
and metadata bindings, all 719 index records, every artifact SHA-256, every
decoded `TargetBundle` identity/fingerprint, and the aggregate fingerprint.
Artifact corruption, record mutation, bundle mismatch, or resume under another
observed target index still fails closed. Existing artifacts are read and do
not need rebuilding.

Successful publication atomically renames a new unique run directory. It
contains a sealed evidence directory, a deterministic regular-file tar, and a
SHA-256 sidecar. The independent source-free verifier revalidates semantics,
all artifact hashes, checkpoint bindings/state, memberships, evaluation/test
lock, exact current hardware, and rejects unsafe paths, links, special files,
duplicates, truncation, or unexpected inventory.

## Independent gate

Run only from the exact final reviewed commit and substitute production paths:

```bash
git fetch origin
git switch --detach <EXACT_FINAL_SHA>
scripts/run_phase9b2c_rtx3090_supervised_smoke.sh \
  --expected-head <EXACT_FINAL_SHA> \
  --raw-index <RAW_INDEX_JSON> --raw-cache-root <RAW_CACHE_DIR> \
  --target-index <TARGET_INDEX_JSON> --target-cache-root <TARGET_CACHE_DIR> \
  --split-manifest <SPLIT_MANIFEST_JSON> --output-root <NEW_OUTPUT_DIR>
```

The draft PR must not become ready until the resulting archive and sidecar pass
the committed verifier on the RTX 3090 host. This gate is bounded executable
mechanics evidence, not scratch-versus-SSL, representation-quality,
calibration, significance, or long-training evidence. It does not begin Phase
9C, PDMX, or Phase 10.

The RTX attempt at
`b7254151ef3b4f11eb55b13d33d02b35d114ee3c` successfully passed the semantic
target-index/cache validation and then failed before training at
`dilemmadata.smoke.target_join_changed_raw_predictions`. That SHA compared two
independent CUDA+AMP forwards byte-for-byte and therefore did not establish a
target leak. It is retained as failed hardware evidence; no training success is
claimed from it.

The subsequent RTX attempt at
`809153d311407ae8102731147931cf7bd36b40de` passed production semantic cache
validation and the single-prediction leakage path, then failed before optimizer
updates while fingerprinting scalar total loss evidence. The byte fingerprint
now flattens a detached, single-transfer contiguous CPU tensor before viewing
its bytes, supporting 0-D and empty tensors without converting values through
Python scalars or lists. Shape and dtype remain separate digest inputs, and
existing vector/matrix fingerprints remain bit-exact. This runtime correction
does not change an evidence schema or contract version; hardware training
success remains unclaimed.
