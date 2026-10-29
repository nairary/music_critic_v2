# Independent implementation reviews

Root: isolated `harmonic-event-pilot/music-critic-v2`.
Base: detached HEAD `287bf856643f9e555352e1661a7bf0563772178a` plus existing WIP.
Scope: six new full modules, CLI, two templates, five new test files and the
accepted spec. All implementation files are untracked WIP, read directly;
ordinary git diff alone would omit them. No staging/commit occurred.
Content identities: [review-scope.json](review-scope.json).
Unrelated WIP, corpora, retained checkpoints/campaigns, legacy and scientific
quality are outside this code-review scope.

## Standards axis

Independent reviewer: fullgraph_data (data author; independently checked
model/runtime/CLI/loss/metrics). Its own data implementation was independently
checked by the Spec reviewer and root integration tests.
Final verdict: no unresolved actionable findings in inspected scope.

Repaired and rechecked:

- Exact E3 core ownership now comes from rational source times; a float
  reconstruction previously lost a true boundary anchor (duration6, halo1..3,
  core_start2). New adversarial regression checks ownership indices.
- Lazy raw input collection stops at max_tiles+1; note bounds are checked before
  any loading. H/P limits reject before local autograd graphs are retained.
- Checkpoint bindings include nested field names/sizes, relation order and all
  FullGraphConfig values. Same-shaped semantic/config changes reject before
  weight/optimizer/RNG mutation. Ordered class vocabulary is bound separately.
- CLI binds the full output location to the declared evidence root, preventing
  a same-basename directory from misregistering artifacts.
- Parsed MIDI source SHA is checked against authenticated input, including a
  fixture mutation between verification and reopening.

Full losses/metrics were traced independently: H/P filtered-row remapping,
physical note/H subcontext matching, explicit-only negatives, original target
coverage, complete endpoint sets and family aggregation. No raw target leakage
was found. A synthetic native projection→adapter→backward→metrics integration
also passed; this is fixture evidence, not a corpus result.

## Spec axis

Independent reviewer: fullgraph_supervision (loss/metric author; independently
checked model/data/runtime/CLI). Its own implementation was reviewed by the
Standards reviewer. Final verdict: no unresolved implementation findings.

Repaired and rechecked:

- MIDI adapter now receives its mandatory dataset identity; a real tiny MIDI
  fixture exercises the actual importer rather than only canonical JSON.
- Equal-sized class vocabulary permutations cannot rename decoded predictions.
- Oracle P capacity checks the supplied P bank, not an unused raw proposal bank.

Acceptance covered: raw-only inference, stable note identities, exact E3 anchor
ownership, individual global H/P context, iterative P→H/h→note feedback, mutable
H/h/P intervals, real masked full-loss gradients into E3 and note states,
unsupported-target coverage, unknown-draw Adam skip, checkpoint replay and
structured canonical/MIDI inference. NBFNet/Query2box remain intentionally absent.

## Final verification

Native CPU command (PYTHONPATH=src; Python
`/private/tmp/musiccritic-hiergraph-venv/bin/python`):

```text
python -m pytest -q -p no:cacheprovider tests/experiments/test_hiergraph_*.py \
  tests/test_hiergraph_full_data.py tests/test_hiergraph_supervision.py \
  tests/tasks/test_hiergraph_projection.py \
  tests/tasks/test_chord_structure_projection.py \
  tests/tasks/test_chord_interval_projection.py --tb=short
```

Final result: **337 passed, 1 skipped (CUDA-only), 118 subtests passed**, 4.54s.
One existing TorchScript deprecation warning. The first sandboxed broad run had
10 old subprocess tests blocked at `ps`; rerunning with process inspection
available passed the same complete suite. No code workaround was introduced.
CLI help and git diff whitespace check passed. Registry check: **541 valid**,
fingerprint `c7794419a74a8c1bdab5da06f29590a96b61253c23d2cf1d3135eba3d718c572`.
No substantial experiment or record created; synthetic engineering tests alone
were executed. Contract/status documentation was synchronized after code review;
no implementation changed after the final reviewed hashes.

Real-work CUDA memory/time, native C3 checkpoint execution, full dataset
admission, training and musical quality remain unmeasured for this full model.
They require a fresh registered native gate and subsequent campaign. Existing
E1–E5 terminal evidence cannot establish these claims.
