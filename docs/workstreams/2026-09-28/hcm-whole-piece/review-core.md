# Whole-piece HCM core review — 2026-09-28

## Scope and identity

Independent review by the motif-module implementation agent; the review excludes
that agent's own `motifs.py`. Repository `/Users/aavalsamakin/Projects/music-critic-v2`,
branch `main`, HEAD `287bf856643f9e555352e1661a7bf0563772178a`.

Comparison base is HEAD for the requested current WIP, with operational baseline
`outputs/hcm_whole_implementation_20260928/baseline/manifest.json` also inspected.
The eight paths below are untracked; staged and unstaged Git diffs for them are
empty. Their complete current contents, including tests, were therefore reviewed
directly. The operational snapshot itself includes some early concurrent WIP and
is not represented as a pristine pre-implementation commit.

| Reviewed path | SHA-256 |
|---|---|
| `src/music_critic/pll/whole_contracts.py` | `f70119ad590c59940bae920519fb89498017c7967944a5deae5811fadc70059b` |
| `src/music_critic/pll/whole_decoder.py` | `d0034bb4b0ba33dc1da8d0b8940f05a877740707080fede740d3ef92e029723f` |
| `src/music_critic/pll/whole_memory.py` | `0a04da9c74616553a29f3e5d423361f6135870c444cf081ea47bf65cf5735e3a` |
| `src/music_critic/pll/whole_model.py` | `06a15623c56a0604d70f71d9d12256ecd7fa11346f4e4777090f3b518ae6133d` |
| `tests/pll/test_whole_contracts.py` | `b8b4c20e0e3d2d3f8f0c7a049f8ba6d4cd9b81c1be666abfce998de1195db070` |
| `tests/pll/test_whole_decoder.py` | `926e490e9ab2e6ec4f48ce0d711fafdcdfd456e744ffe2d55ceba2152ca38163` |
| `tests/pll/test_whole_memory.py` | `92a5a70b5fbb7b8767636be229904f487f87e9542aeccbd5a5f37a3e2de5e178` |
| `tests/pll/test_whole_model.py` | `2910288da838d2fbbfc5596f31fa455513010447d9927891a0e061eb03e16590` |

Read callers `model.py`, `decoder.py`, `encoder.py`, `masking.py`, contracts and
`tests/pll/whole_fixtures.py` to trace runtime behaviour. Source standard:
AGENTS.md and authoritative exact-time, raw-only, mask-before-graph contracts.
Spec: `docs/workstreams/2026-09-28/hcm-whole-piece/spec.md` and accepted whole-song,
1/2/4/8/16/32-bar motif request. Two distinct Standards and Spec passes were
performed. No additional agent slot was needed for this bounded delegated review.

## Standards verdict

No actionable P1/P2 findings in the reviewed snapshot. Whole masking sanitizes
before encoding; declared source geometry survives, annotations stay excluded,
metric membership uses exact times, and derived float positions do not control
interval decisions. Causal decoder branches read fixed masked context and retain
the event/field grammar. The fully frozen encoder also freezes boundary projection.
Synthetic verification does not establish native GPU readiness or music quality.

## Spec verdict

**P2 — disabled global branch still consumes the decoder memory budget.**
At `src/music_critic/pll/whole_memory.py:190-195`, `global_count` is charged even
when `enable_global=False`. The same count reduces the local-detail quota near
the end of `forward`. `WholePieceHCM.encode` omits the global branch in that mode,
so this can reject an otherwise valid local-only ablation because of tokens the
returned decoder memory will never contain.

Observed reproduction, using only checked-in synthetic fixtures:

```python
model = WholePieceHCM(
    small_config(), codec_profile=small_codec(),
    whole_config=small_whole(enable_motifs=False, enable_global=False),
    context_profile=ContextProfile(max_memory=32),
).eval()
model.encode(whole_example(17).context)
```

Result: `UnsupportedContextError: pll.whole_memory_limit`.
The only requested branch is local, capped at 32 tokens; its summary reservation
is 19 tokens. Twenty disabled global tokens create the false rejection. This
conflicts with the spec's separately configurable global ablation and explicit
resource bounds. Charge only enabled decoder branches, and apply the same choice
to the local-detail quota; keep an independent internal allocation limit if one
is needed. Add a regression where disabled globals fit the actual branch budget
and enabled globals exceed it.

No other actionable P1/P2 Spec findings found. The implementation retains >32
bars, supplies all bar states to the hierarchy, exposes separate decoder memory
paths, carries unknown/rest coverage and supports gradients to distant states.
These are implementation properties, not proven detection of musical form.

## Verification and limitations

Command:

```text
PYTHONPATH=src:. outputs/critic_authored_curriculum_20260927/test-env/bin/python -m pytest -q tests/pll/test_whole_contracts.py tests/pll/test_whole_memory.py tests/pll/test_whole_model.py tests/pll/test_whole_decoder.py
```

Observed result: **20 passed, 1 warning in 2.44s**. Warning is the installed
PyG distributed-module deprecation. A separate tiny CPU reproduction above
confirmed the budget issue without changing implementation or tests.

Excluded: real data preparation/launch, CLI dispatch/resume/warm-start, native
3090 admission, motif implementation, experiment records and scientific quality.
No legacy code, checkpoints, source corpora or TEST data were used. Review edits
are limited to this report; root agent owns fixes and consolidated STATUS updates.

## Fix verification

The root agent resolved the P2 by charging `global_count` only when the global
branch is enabled; both the summary admission and local-detail quota now use
that conditional count. Re-read the changed source and new regression directly.

- `src/music_critic/pll/whole_memory.py`: `e32d77fa3c18210e8ffbce88078e6c68cb4eadddc6ff4bf24a50911f8409b373`
- `tests/pll/test_whole_model.py`: `9d68ca3706a93eb7ee577fda8063988051c3ef862cad72adf7ccc01937deff9a`

`test_disabled_global_branch_does_not_reserve_decoder_tokens` runs the previously
failing 17-bar/32-token case and now passes. It was included in the integration
review's **30 passed, 1 warning** suite. The earlier finding is **resolved** for
this verified snapshot; no outstanding actionable P1/P2 core findings remain.

The fix regression was independently repeated after integration changes: **7 passed, 1 warning in 5.23s** for the whole-runtime suite plus this regression.
