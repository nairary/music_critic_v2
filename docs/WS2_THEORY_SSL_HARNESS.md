# WS2 corrected theory → SSL bridge and retention harness

## Scope and evidence boundary

This is a bounded Phase 9E → Phase 7–8 engineering bridge before Phase 10.
It composes the existing GraphMAE2-inspired representation objectives and
corrected musical tasks over one backbone. Synthetic CPU tests demonstrate
mechanics only. They do not establish forgetting resistance, corpus readiness,
better representations, musical quality, PLL, or an augmentation benefit.
No real-data training, trained checkpoint loading, TEST access, SSH, C0/C2
changes, scheduler, dataset factory, epoch loop, or checkpoint manager is added.

The fixed corrected architecture remains width 128, three local GNN layers,
two Transformer layers with four heads and FFN factor four, dropout 0.1 and
residual connections. The separate one-layer onset BiGRU and all 18 heads
(8 primary, 10 auxiliary) retain their full source weights and configuration.
Vocabulary dimensions come from the existing registry, including 37 spelled
root classes and 17 quality classes. Fixtures reduce graph size only.

## Strict transfer contract

`theory_warmstart.py` introduces `theory-ssl-warmstart@1.0.0`.
`export_theory_warmstart_state(source)` returns an independent CPU clone of the
**complete corrected state**, its tensor fingerprint, a versioned model/raw
feature/graph/runtime-architecture contract and `synthetic_in_memory`
provenance. It is an in-memory fixture envelope, not an encoder-only export or
a claim about checkpoint provenance. It has no file hash or file loader.
The existing `ssl/transfer.py` validator is unchanged and continues to reject
the corrected model as an old-style hierarchical encoder export.

`load_theory_warmstart(destination, export)` accepts a newly created
`TheoryWarmstartSSLModel`, whose architecture is fixed to the corrected model.
Do not reuse a destination already shared with another caller-owned model.
The following mapping is explicit and complete; `k` is each key in the source
group, with its suffix preserved:

| Source group | SSL destination group | Tensor keys | Parameters |
| --- | --- | ---: | ---: |
| `local_baseline.encoder` | `encoder.local_baseline.encoder` | 379 | 1,968,000 |
| `context_encoder.pooling` | `encoder.context_encoder.pooling` | 28 | 298,752 |
| `context_encoder.transformer` | `encoder.context_encoder.transformer` | 27 | 397,184 |
| `context_encoder.fusion` | `encoder.context_encoder.fusion` | 36 | 396,288 |

Every source and destination state key is accounted for. Validation checks
exact contracts and contract fingerprints, complete key sets, shapes, dtypes,
finite dense tensors, source tensor fingerprint, actual module types and
public runtime configuration (including head routes/precision, per-node feature
spec order/normalization, attention heads/dropout and activation), parameter
identity, and the explicit mapping before the first destination mutation.
Each separate input model must have independent parameter storage: distinct
Parameter objects sharing an allocation are rejected, including offset views.
This preflight does not apply across the two intentionally shared output branches.
Temporary reference construction preserves CPU RNG state. All loads use
`strict=True`; there is no filtering of an arbitrary training checkpoint.
The supplied contract must pass its own recomputed fingerprint check, followed
by canonical JSON comparison with the expected contract. Numeric type changes
such as `18 -> 18.0` are rejected even though Python considers the values equal.
The full merged destination is fingerprint-checked after loading. Application
failure restores its original state and gradient flags.

The returned `TheoryWarmstartResult` contains:

- `ssl_model`: the destination with the transferred backbone;
- `theory_model`: an independent corrected branch with retained BiGRU/heads,
  whose four backbone modules are the exact same objects as the SSL branch;
- `report`: source state/contract fingerprints and per-group keys, parameter
  counts and fingerprints for **transferred**, **retained**, **fresh** and
  **inactive** state. Transferred groups include exact destination keys.

Only cloned export tensors are loaded. The original source model is unchanged
and does not share parameters with either branch. SSL `feature_mask_token`,
representation `decoder`, and bar/song projectors/predictors retain their fresh
initializations. The SSL container's inherited
`encoder.local_baseline.task_heads` and `reconstruction_heads` are unchanged,
frozen and excluded from the active optimizer inventory. They remain present
so existing SSL container contracts are preserved.

## Objective API and ownership

```python
from music_critic.ssl.theory_warmstart import (
    TheoryWarmstartSSLModel,
    export_theory_warmstart_state,
    load_theory_warmstart,
)
from music_critic.ssl.theory_retention_harness import TheoryRetentionHarness

# source is a synthetic in-memory CorrectedAnalysisGNNModel in this phase.
warmstart = load_theory_warmstart(
    TheoryWarmstartSSLModel(), export_theory_warmstart_state(source)
)
harness = TheoryRetentionHarness(warmstart)
parameters = harness.optimizer_parameters()  # each active Parameter once

# Batches and binding are prepared independently, before this call.
output = harness(
    lambda_ssl=0.7, lambda_theory=0.3,
    ssl_batch=raw_ssl_batch, prepared_mask_binding=prepared_binding,
    theory_raw_batch=separate_labeled_raw_batch,
    theory_sidecars=sidecars, class_weights=explicit_class_weights,
)
```

The coefficients in the example are synthetic fixture choices, not production
tuning. They must be explicit finite nonnegative numbers. Set the other
coefficient to zero for SSL-only or theory-only; zero skips execution entirely.
SSL batch and binding must be supplied together. Sidecars require a raw batch
and, when active, one sidecar per sample plus explicit `FrozenClassWeights`.
The harness never calls `load_frozen_class_weights` or the historical
`build_source_free_fixture`. Its tests build `SSLRawSample` and collate directly,
and generate uniform weights and labels from the registry.

The theory prediction call receives only the raw graph. Sidecars are aligned
after prediction through `align_target_sidecars_after_prediction`, then
`corrected_supervised_loss` preserves `L_primary + 0.25 * L_auxiliary` and
existing per-head masks. SSL representation reconstruction is before the
separate corrected BiGRU. Phase 7 and Phase 8 hierarchy bindings dispatch to
their existing public `forward` / `forward_hierarchy` methods. No EMA or new
target objective is introduced.

`TheoryRetentionOutput` exposes `ssl_loss`, `theory_loss`, `total`, original
SSL/theory outputs, corrected loss report, target alignment, counts, diagnostics
and the parameter inventory. `total` sums available terms with the explicit
coefficients. Missing batches, missing sidecars, or zero valid target rows have
no corresponding loss term; no available term yields `total=None`. Unavailable
SSL objectives and theory terms are diagnosed, not converted to class zero.
A theory raw batch without sidecars can still return raw predictions.

Raw theory graphs with no note/onset candidates are supported. After prediction
and exact alignment, the harness constructs a temporary loss-only alignment
with zero rows for empty-candidate heads, whose original rows must all be
invalid. The unchanged corrected loss handles these absent terms with its
usual aggregation. The public alignment/diagnostics and complete invalid-row
counts are preserved, and other heads with candidates continue to train.
No substitute logits or class-zero supervision are created. An independent
ordinary SSL batch can still contribute in mixed mode.

The default parameter inventory is:

| Role | Unique parameter tensors | Scalar parameters | Optimizer |
| --- | ---: | ---: | --- |
| Shared backbone | 470 | 3,060,224 | active |
| Fresh SSL branch | 35 | 183,296 | active |
| Retained BiGRU/heads | 100 | 601,712 | active |
| Inherited unused SSL heads | 81 | 324,357 | excluded/frozen |

`optimizer_parameters()` returns 605 unique trainable objects (3,845,232
scalars). Use this inventory instead of concatenating the two branches'
parameters. Use `optimizer.zero_grad(set_to_none=True)` before each new
backward pass; stale gradients from a prior objective must not be applied to
an inactive branch. The harness does not perform optimizer steps itself.

Mode changes should normally be made through the owning harness. If a caller
sets individual child modes, do so after `harness.train()`/`eval()`. The local
SSL subclass snapshots every encoder child mode around the inherited target
pass and restores each flag in `finally`, before online prediction. Targets
remain full-view, detached and computed under `no_grad`.

## Acceptance and next phase

New tests cover exact transfer and retained/fresh state, atomic rejection,
runtime configuration drift invisible to tensor shapes or fingerprints,
source independence, optimizer uniqueness, nonzero finite gradients in all
active groups across three modes, inactive gradients, mixed and raw-only
steps, detached targets, exact mode restoration including exceptions, hierarchy
binding dispatch, and sidecar/mask/removal invariance of raw logits. Graph,
sidecar and synthetic weight fixtures remain unchanged.

Exact executed commands and results are recorded in
`workstreams/2026-09-05/WS2_RESULT.md`. No standalone experiment record is
needed for these unit/contract tests. The next scientific phase is a separately
registered warm-start experiment with actual checkpoint provenance and a
predeclared evaluation of retained theory performance. It is not launched here.
