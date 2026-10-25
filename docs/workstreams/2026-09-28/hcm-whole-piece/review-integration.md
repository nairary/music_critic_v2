# Whole-piece integration review — 2026-09-28

## Scope and source identity

Repository `/Users/aavalsamakin/Projects/music-critic-v2`, branch `main`, HEAD
`287bf856643f9e555352e1661a7bf0563772178a`. Requested current WIP paths below
are untracked in this checkout; Git staged/unstaged diff alone cannot describe
them. The existing shared modules were compared byte-for-byte by textual diff
against `outputs/hcm_whole_implementation_20260928/baseline/src/music_critic/pll/`.
The new integration test was read in full. Baseline manifest is the same path's
`manifest.json`; it is an operational task snapshot, not a new Git commit.

| Reviewed path | SHA-256 |
|---|---|
| `src/music_critic/pll/cli.py` | `049fe4ff4ad9403b3a6eb7bb9cda2bf5894876fa29b38d701ed123770f60e5aa` |
| `src/music_critic/pll/runtime.py` | `caf106fa40eb5ca850a9db0f7c87bf2a6f695648cee329edda11f525c0f2b05b` |
| `src/music_critic/pll/scoring.py` | `a45598e416f448f8457369f0406f5e0c5fef45c7d83c8c356c7a8b91e24356be` |
| `tests/pll/test_whole_runtime.py` | `c3558d9837259b9acec8d8ed94cabcd93128968b8e3f4f70c83abf7e3b4c541d` |

Reviewed callers/contracts: whole model/encoder contract, full-source manifest
and sampler, old HCM model, base event decoder and the currently available
`whole_warmstart.py` API. The latter implementation is being independently
finished/reviewed by its owner; this review covers its CLI call signature,
byte-bound configuration and ordering relative to trainer construction/resume,
not an additional claim of full warm-start implementation approval.

## Standards verdict

No actionable P1/P2 findings. Whole run configuration has an explicit new schema;
old `PLLRunConfig@1` continues through the existing model/manifest/sampler path.
Whole configuration is resolved before launch, complete-source coverage is
required by the CLI, and TEST policy remains closed. Training bindings include
the complete new model contract, data fingerprint and implementation fingerprint.
Changing motif scales is rejected before model mutation by strict checkpoint
binding. Per-example diagnostic receipts are detached JSON values copied into
metrics; they do not retain training autograd graphs.

## Spec verdict

No actionable P1/P2 findings in this integration snapshot. Whole dispatch uses
WholePieceHCM and WholeMaskSampler, fixed-query scoring calls the whole split
method, and unsupported contexts retain a result slot. The default whole profile
rejects an unfreeze schedule for the fully frozen local encoder before launch.
Warm start is explicitly configured with source hash/size/update, invokes the
old-HCM weight migration before creating a fresh optimizer, and is distinct from
restoring a whole-model checkpoint. A resumed warm-started run must preserve its
initialization binding; a resume is not a schema-compatible old-HCM checkpoint.
No silent old-to-whole resume was added.

The source and motif-scale context diagnostics are retained for each example in
an accumulation step; draw receipts remain separately ordered and checkpoint
progress checks enforce sampler/trainer draw agreement.

## Verification

```text
PYTHONPATH=src:. outputs/critic_authored_curriculum_20260927/test-env/bin/python -m pytest -q tests/pll/test_cli.py tests/pll/test_runtime.py tests/pll/test_scoring.py tests/pll/test_whole_runtime.py tests/pll/test_whole_model.py
```

Observed: **30 passed, 1 warning in 9.62s**. The warning is the existing installed
PyG distributed deprecation. This includes legacy paths, new registered synthetic
whole-source train/save/resume versus uninterrupted training, changed motif-bank
binding rejection, per-draw diagnostics, whole fixed-query coverage, frozen-local
schedule rejection and the core budget regression.

Only synthetic CPU fixtures were used. No real data, GPU, cloud, TEST payload,
legacy repository or experiment-quality comparison was accessed. CLI ability is
separate from native admission and complete four-corpus availability. Two concurrent deltas were observed and re-reviewed: resolved motif scales
now use JSON lists in CLI output, and an unused import was removed from the core
regression. The table records the final reviewed bytes. The affected whole runtime
suite plus the core regression was rerun: **7 passed, 1 warning in 5.23s**. Root owns consolidated
STATUS/documentation updates and any further source changes.

## Final integration delta verification

Independently re-read the final whole-only CLI evidence finalizer and three new
synthetic integration cases. Known `observed_split_claim=test` or `validation`
now produces a true input-access observation even when manifest identity checks
reject the payload. Other unverified access fields remain null; a contradiction
with declared policy produces `invalid` consistently in the attempt receipt and
terminal experiment record. Declared permitted validation would remain a failed
TRAIN identity check rather than an invented policy violation. Legacy finalizer
semantics are preserved. No actionable Standards or Spec P1/P2 found in this delta.

The registered whole warm-start smoke loads the explicitly hash/size/update-bound
old HCM checkpoint before constructing the new trainer, completes one synthetic
update, retains migration provenance, and checks frozen encoder weights. It does
not restore source optimizer state or claim that this is strict old-model resume.

Command: `PYTHONPATH=src:. outputs/critic_authored_curriculum_20260927/test-env/bin/python -m pytest -q tests/pll/test_whole_runtime.py tests/pll/test_cli.py`.
Observed: **18 passed, 1 warning in 6.91s**. The three new cases are part of this
run; the warning remains the installed PyG deprecation. The table above has been
updated to the final reviewed CLI and whole-runtime-test source hashes. No source
files were modified by this review, and no real held-out payload was accessed.

## Repository import-guard follow-up

Reviewed the final guard-only diff against
`outputs/hcm_whole_implementation_20260928/baseline/tests/test_repository_contract.py`.
The sole behavioural change permits Torch/PyG inside the explicitly imported
`music_critic.critic` runtime, alongside existing graph/model/PLL runtime
allowances. This matches the existing critic implementation: both its baselines
and C3 adapter directly require Torch, and explicit `music_critic.critic` import
is heavy. It is not represented as a dependency-free public entry point.

The forbidden legacy roots/path scans and all other heavy-dependency restrictions
remain intact. Separate fresh-process tests still require both `music_critic` and
`music_critic.pll` imports to load no Torch/PyG/renderer/legacy modules; data-layer
renderer isolation also remains checked. No actionable P1/P2 Standards or Spec
finding in this narrow change.

Reviewed source identities:

- `tests/test_repository_contract.py`: `e094b72e37ee6fc6e5569c97ae127a46cbf041c647a9e9e85e84cadc59951c34`
- `tests/test_import.py`: `79eedeafaedd06bb19325747abd18904c00b207c00b44172cdee3c96e318ae5d`

Independent verification: `PYTHONPATH=src:. outputs/critic_authored_curriculum_20260927/test-env/bin/python -m pytest -q tests/test_repository_contract.py tests/test_import.py`
returned **10 passed in 0.83s**, with no warnings. Review changed only this note.
