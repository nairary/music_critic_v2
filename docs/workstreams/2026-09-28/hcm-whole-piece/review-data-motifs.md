# Independent data and motif review

Scope: current untracked implementation and tests against the accepted
whole-piece specification. Reviewer did not author these modules and made no
source edits. Baseline is `main` HEAD
`287bf856643f9e555352e1661a7bf0563772178a`; untracked/WIP inventory is retained in
`outputs/hcm_whole_implementation_20260928/baseline/manifest.json`.
Staged and unstaged Git diffs for the three source paths are empty because the
files are untracked; their complete current contents were read directly.
Existing unrelated changes, datasets, checkpoints and generated experiment
artifacts were excluded. Source collaborator reads included existing manifest,
masking, runtime fingerprint, registry validation and whole-model call sites.

## Standards verdict

No remaining actionable findings in the reviewed scope after the correction
below. Exact-time comparisons, observed-only predictor inputs, immutable
byte-bound preparation, explicit unknown provenance, closed held-out policy,
and bounded detail reads are preserved. No dependencies or legacy runtime
imports were added by these modules.

**Resolved P1: preserve known held-out access in failure records.** The original
registered preparation wrapper converted every observation to null whenever an
unverified read occurred. In an isolated synthetic reproduction, candidate
metadata declared TRAIN while its hash-bound canonical payload declared TEST.
The preparation receipt recorded `test_payloads_read=1` and the parsed TEST
split, but the terminal record was `failed` with `test_inputs_accessed=null`.
This contradicted the known-access/policy-violation requirements in
`docs/EXPERIMENT_LOGGING.md`.

The final code records parsed split and target-array presence, derives known
true observations from each access row, preserves other unknown observations,
and closes such an incident as `invalid` with exact `policy_violations` and
partial primary evidence. New synthetic TEST and VALIDATION regression tests
pass. The reviewed KeyboardInterrupt correction also supplies a nonempty
terminal reason and closes the record as `aborted`.

## Specification verdict

No remaining actionable findings in these modules. Inspection and synthetic
checks cover:

- Stride-one bar candidates at 1/2/4/8/16/32 bars and short beat candidates;
  insufficient support stays explicit, masks are excluded with rational-time
  comparisons, and candidate limits fail rather than drop later scales.
- Keys use pre-global masked states and observed raw content. The relative
  pitch feature is explicitly an onset-bin statistic, not a guessed voice.
  Absolute pitch remains available; learned key invariance is not claimed.
- All-candidate soft attention plus a learned null branch supplies gradients
  outside hard top-k; selected details use positional pooling over their entire
  support. Empty banks, carry-in notes, bounded alignment and diagnostics have
  direct synthetic checks. No gold target enters candidate selection.
- Complete ingestion of a supplied source is separate from full-song coverage.
  Full/excerpt/unknown declarations and supporting evidence remain hash-bound;
  default preparation excludes excerpts/unknown before payload reads.
- Canonical candidates from all four named sources retain exact musical
  boundaries. Metadata preflight checks split/work/source/hash components and
  native held-out hints before reading music. Preparation never joins clips.
  Source sizes/hashes are verified on the same bytes that are parsed.
- Sampler identity includes whole-piece configuration, preserves work-balanced
  source sampling and RNG continuation, and delegates to the full-view splitter.

## Verification and limits

Command:

```text
PYTHONPATH=src:. outputs/critic_authored_curriculum_20260927/test-env/bin/python -m pytest tests/pll/test_motifs.py tests/pll/test_whole_data.py -q
```

Final result: **34 passed, 1 existing torch_geometric deprecation warning**
(1.13 seconds). The earlier 31-test pass did not cover the reported incident;
the independent synthetic reproduction exposed it, and the final test count
includes the added incident/interruption regressions.

No real data, GPU, production record or checkpoint was used in this review.
These checks establish mechanics only. Actual full-song availability,
cross-corpus near-duplicate coverage, learned retrieval quality, 16/32-bar
usefulness and native resource admission remain unverified. Full integration,
resume and the decoder implementation have separate review coverage.

## Reviewed final content identities

| Path | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/music_critic/pll/motifs.py` | 18466 | `30a20225ea6611d92fdb458b613c07d75028f7d7be7b0aab82f4d03f341f66a3` |
| `src/music_critic/pll/whole_data.py` | 16039 | `810f1b9a2a0b88d694f197f0bddf5e91fe938c70dc06d99110a39ce914dac756` |
| `scripts/prepare_pll_whole_sources.py` | 10762 | `a17c6d2f54efefaf233f17fbab08da77493647571cd1755629e590580ffbf247` |
| `tests/pll/test_motifs.py` | 11651 | `9c415727cc7fbb1492fd6d06941ad535853172ecce644ce4c8ac21ae88a47114` |
| `tests/pll/test_whole_data.py` | 15737 | `3bf32c9afcb54ea10ee463e8ae416bc180bcfb38cfb26edd8a9e6204f371d57f` |
| `docs/workstreams/2026-09-28/hcm-whole-piece/spec.md` | 5828 | `275a1753e2912bec595b15a2200e43c5b32c300a12bf7ccae4371f51f4d7147e` |

## Additional independent weight-migration review

Requested after the main review: `whole_warmstart.py` and its tests, plus the
CLI/configuration, model-contract and new-decoder seams that call it. No
remaining Standards or Spec findings in this additional bounded scope.

The loader verifies the exact bytes, checkpoint schema/update, HCM model/codec
and conditional-target contract. It requires equality of the complete shared
encoder/decoder key set and validates every shared tensor's layout, shape,
dtype and finiteness before any write. New memory/retrieval/distant-attention
weights remain untouched. It uses direct tensor copies after prevalidation,
keeps a rollback snapshot, and marks provenance only after successful copying.
Ignored optimizer, RNG, sampler and progress values are never restored.

The CLI checks the warm-start binding, rejects simultaneous C3-pretrained
initialization, and calls migration before moving the model and constructing
the trainer/optimizer. The resulting migration receipt participates in the
whole-model contract and hence strict resume binding. A later resume is still
a separate whole-model checkpoint operation; it must retain the same declared
initialization provenance.

Verification command:

```text
PYTHONPATH=src:. outputs/critic_authored_curriculum_20260927/test-env/bin/python -m pytest tests/pll/test_whole_warmstart.py -q
```

Result: **12 passed, 1 existing dependency warning** (13.05 seconds). Tests cover
exact shared-weight reuse, unchanged new modules and RNG, fresh optimizer and
progress, immutable rejection for ten malformed source cases, explicit ignored
source state, and repeated-migration rejection. The CLI call sequence was
inspected; this additional review did not execute a registered warm-start CLI
training run or simulate an exceptional GPU copy/rollback failure.

| Additional path | Bytes | SHA-256 |
| --- | ---: | --- |
| `src/music_critic/pll/whole_warmstart.py` | 6108 | `96dca534f5a82526fbbcff16a8885cf892343b3009fccd65cbfa944bd3497a71` |
| `tests/pll/test_whole_warmstart.py` | 5357 | `77c031e08c53eae2f8ca422eeabc01883b60498ada96488b1865479966f6868b` |
| `src/music_critic/pll/cli.py` | 20511 | `ae149e40d72220a06db56f23a4652d296e70f296231868810078f04f1084038d` |
| `src/music_critic/pll/whole_model.py` | 7290 | `a13bf421dda4703c4ac58e10b96fac6b7c514aa3e0e4657dc46d2aa1274ac485` |
| `src/music_critic/pll/whole_decoder.py` | 3068 | `d0034bb4b0ba33dc1da8d0b8940f05a877740707080fede740d3ef92e029723f` |
