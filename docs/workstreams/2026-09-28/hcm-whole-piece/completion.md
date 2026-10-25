# Whole-piece implementation completion — 2026-09-28

Implemented the user's full-source training context and motif scales
1/2/4/8/16/32 bars. Small masked targets remain 1/2/4 bars; no new real training
was launched. Behaviour and operational boundaries: [HCM_WHOLE](../../../HCM_WHOLE.md).
The pre-existing dirty/untracked checkout was preserved; no staging or commit.

## Files and behaviour

| Files under `src/music_critic/pll/` | Implemented behaviour |
|---|---|
| `whole_contracts.py` | Full observed geometry, explicit limits, bound profiles and mask-before-encoding |
| `whole_data.py` | Complete-source candidates/manifest, coverage provenance, four canonical sources and exact whole sampler resume |
| `whole_memory.py` | Positional track/bar, block and whole-song memory with unique core ownership and bounded local details |
| `motifs.py` | Six bar scales, two beat scales, observed-only keys, soft/null retrieval, overlap suppression and aligned details |
| `whole_decoder.py`, `whole_model.py` | Separate local/global/motif attention, fully frozen local option, generation and loss integration |
| `whole_warmstart.py` | Atomic byte-bound compatible HCM-1 weight transfer with fresh training state |
| `decoder.py` | Extracted overridable main-state seam; original decoder behaviour retained |
| `cli.py`, `runtime.py`, `scoring.py` | Whole config dispatch, per-draw diagnostics, registered attempts, strict checkpoint continuation and full-context scoring |

Added `scripts/prepare_pll_whole_sources.py`,
`configs/pll/hcm1_whole_reference.json`, seven `test_whole_*.py` suites,
`test_motifs.py`, and synthetic `whole_fixtures.py`. The existing repository
import guard now permits the already-used opt-in critic runtime to import
Torch/PyG, matching graph/models/PLL; all other forbidden imports remain checked.
Public `music_critic` and `music_critic.pll` import tests remain dependency-free.

Updated ARCHITECTURE, DATA_CONTRACT, DECISIONS, IMPLEMENTATION_PLAN, ROADMAP,
STATUS, HCM1 and the prior research document; added HCM_WHOLE and this workstream.
Implementation identities and original shared-file snapshots are retained in
ignored `outputs/hcm_whole_implementation_20260928/`.

## Verification and review

Tests ran using the existing environment, without dependency installation:

```text
PYTHONPATH=src:. outputs/critic_authored_curriculum_20260927/test-env/bin/python -m pytest -q tests/pll tests/experiments/test_registry.py tests/test_repository_contract.py tests/test_import.py
```

Initial final-scope run: **332 passed, 1 failed, 1 warning in 38.44s**.
The sole failure was the existing import guard rejecting `torch` in
`critic/baselines.py`, an inherited runtime dependency. After the narrow
allowlist correction, the affected repository/import suites were rerun:

```text
PYTHONPATH=src:. outputs/critic_authored_curriculum_20260927/test-env/bin/python -m pytest -q tests/test_repository_contract.py tests/test_import.py
```

Result: **10 passed in 0.85s**. Therefore all **333 distinct checks** in the
final scope have passed: 246 PLL checks, 77 registry checks and 10 repository/
import checks. The only warning in the broader run was the installed PyG
distributed deprecation. Original failed output was retained in
`final-tests.txt`; correction output is `import-guard-tests.txt`.
Config-only inspection of the whole reference template also succeeded.

The tests cover complete 40-bar input, 16/32 candidates, hidden-answer invariance,
unknown/rest and carry-in, exact rational ownership, positional order, distant
influence, candidate/null gradients, bounded all-interval detail pooling,
generation, strict profile mismatch rejection and uninterrupted-versus-resumed
training with dropout. A registered synthetic warm-start attempt trains with
fresh progress and retains inherited frozen weights exactly. Adversarial split
fixtures exercise invalid-attempt accounting; no real TEST was accessed.

Independent reviews, with inspected file identities:

- [Core](review-core.md): P2 disabled-global token reservation reproduced and
  fixed; regression now passes.
- [Data, motifs and weight transfer](review-data-motifs.md): P1 known heldout
  access erased into unknown observations reproduced, corrected and retested.
- [Runner integration](review-integration.md): old/whole config separation,
  exact continuation, warm start and final contrary-split accounting checked.

No remaining actionable Standards/Spec P1/P2 findings in the reviewed scope.
The final shared runner keeps known contradictory input reads explicit and
marks policy-breaching whole attempts invalid; unknown target access is not guessed.
`git diff --check` passes. No legacy files were inspected, adapted or imported.

## Remaining scientific and runtime work

This completes implementation mechanics. Actual completeness of the four
corpora, cross-source duplicate groups, coverage at the configured ceilings,
3090 peak memory/throughput and checkpoint migration from actual trained weights
still need registered admission. The architecture has not been trained on real
music in this task. Retrieval usefulness, refrain recognition, 16/32-scale gain
and human preference are not established. Auxiliary activity heads, caching and
packed/mixed-precision GPU execution remain separate work.

Next phase: registered full-source preparation and 3090 admission, followed by
a bounded pilot and matched short-bank versus extended-bank comparison. No real
experiment record or ledger mutation was required for the fixture-only tests.
