# Frozen C0/C2 transfer to POP909-CL

2026-09-08. Interpretation of the predeclared paired evaluation; no additional
training, inference, model selection or corpus access was used for this review.

## Result and claim boundary

C2 performs better than C0 on the two evaluated harmonic tasks in both
predeclared input views. This is evidence of useful transfer from the
Dilemmadata-trained models to this POP909-CL validation set. It is not an
assessment of all 18 tasks, musical quality, PLL/fragility, or independent
statistical replication. A cross-corpus duplicate audit has not been performed.

Both are seed-17 best checkpoints selected using Dilemmadata VALIDATION:
C0 at 95,000 updates; C2 at 115,000. These are not the final-120k pair.
No checkpoint or input view was selected using the observed POP scores.
The 101 pieces are the entire existing POP VALIDATION split and contain
12,833 source chord events. TEST stayed closed; applied training updates: zero.

## Metrics

Macro-F1 is the unweighted mean over classes with observed gold support:
12 root pitch classes and 8 quality classes. It is not the Dilemmadata
eight-head primary_score. Accuracies pool eligible source chord events;
longer/repetitive pieces contribute more events. No confidence interval or
piece-level significance claim is made.

### Native: predeclared primary

| Metric | C0 | C2 | C2 minus C0 |
| --- | ---: | ---: | ---: |
| Root accuracy | 0.895559 | 0.911175 | +0.015616 |
| Root macro-F1 | 0.892292 | 0.908590 | +0.016298 |
| Quality accuracy | 0.737372 | 0.770347 | +0.032975 |
| Quality macro-F1 | 0.373857 | 0.453021 | +0.079165 |
| Root + quality joint accuracy | 0.707735 | 0.749307 | +0.041572 |

### Metadata removed: predeclared secondary

| Metric | C0 | C2 | C2 minus C0 |
| --- | ---: | ---: | ---: |
| Root accuracy | 0.922257 | 0.925195 | +0.002938 |
| Root macro-F1 | 0.920529 | 0.923118 | +0.002589 |
| Quality accuracy | 0.806763 | 0.814398 | +0.007636 |
| Quality macro-F1 | 0.436271 | 0.512451 | +0.076180 |
| Root + quality joint accuracy | 0.788612 | 0.798186 | +0.009574 |

Native joint counts are 8,427/11,907 and 8,922/11,907; secondary counts
are 9,390/11,907 and 9,504/11,907. Root is the chord root pitch class,
not its bass, key or enharmonic spelling.

The secondary view jointly removes note velocity/program/channel and track
program/channel, then rebuilds derived values and availability. Notes and
timing remain. Both models improve under this intervention; it suggests
input-metadata sensitivity, but does not isolate a responsible field or prove
a shortcut, leakage, or a cause of domain mismatch. Native remains primary.

## Coverage and class difficulty

| Target | Eligible events | Coverage | Ambiguous | Unsupported |
| --- | ---: | ---: | ---: | ---: |
| Root | 11,911 | 92.8154% | 826 | 96 |
| Quality | 11,918 | 92.8699% | 815 | 100 |

Gold labels, masks and eligible denominators match between both models and
views. Root and quality masks are independent; joint eligibility is 11,907.
Quality unsupported comprises 96 source labels unavailable to the template
normalizer, three mM7 and one +7 excluded by frozen subset9. There are no
explicit sus2/sus4 labels in these excluded-label counts; ambiguous sonorities
may include them, so this does not establish absence of sus chords.

Alignment failures are zero. Seventeen beats contain multiple events
(34 events); 16 beats have conflicting eligible roots and six have conflicting
qualities. Each source event is retained, using the same prediction within
a beat. Ninety-one no-chord spans and 19 trailing unannotated spans are counted
separately from the 12,833 chord events. No missing-target piece is present.

| Quality | Gold events | C0 native F1 | C2 native F1 |
| --- | ---: | ---: | ---: |
| Major triad | 6,768 | 0.835580 | 0.857143 |
| Minor triad | 3,999 | 0.799090 | 0.832903 |
| Diminished triad | 62 | 0.307692 | 0.377778 |
| Augmented triad | 11 | 0.173913 | 0.333333 |
| Minor seventh | 611 | 0.307886 | 0.367713 |
| Major seventh | 294 | 0.164087 | 0.303136 |
| Dominant seventh | 145 | 0.171026 | 0.177165 |
| Half-diminished seventh | 28 | 0.231579 | 0.375000 |

Major/minor triads comprise 10,767/11,918 = 90.34% of eligible events.
C2 improves native F1 in all eight observed qualities, but uncommon classes
remain weak and estimates for 11 or 28 examples are fragile. High pooled
accuracy therefore does not imply equally reliable recognition of extensions.
Diminished seventh is allowed by subset9 but has zero gold support here.
The model predicts among all 17 classes: predictions outside subset9 remain
errors (C0/C2 native: 2/8; secondary: 3/2). No subset renormalization is used.

## Checkpoint and execution integrity

The earlier [failed input gate](records/EXP-POP-TRANSFER-INPUTS-001.json)
is preserved. Its NumPy RNG envelope incompatibility was resolved by the
separately registered restricted model-only export, not by an unsafe loader.
The exporter validates the bounded model prefix and tensor storage views,
does not execute the optimizer/RNG pickle tail, and strictly reloads its final
artifact. No tensor value or model architecture was changed.

| Model | Original checkpoint SHA-256 | Model-only SHA-256 |
| --- | --- | --- |
| C0 | `7f6c3b15761e7bb18425622591fe1327ca5c74c4b5b3540eea0ed3c9086a40ff` | `19953be08f78274ebf4d6716a79c833f87bedf613daa35f7a17f19bc2174d931` |
| C2 | `fb5247920ca000ed4ca8d6266bbdb24b2dddcef230231d6ad723bd4faba6fc8a` | `a151dec0af22f25ac6ec8ad75927cf2889644284061fd7e35a129c67e836b797` |

Original embedded and exported tensor-state fingerprints agree:

- C0: `02bc65f0a6b380beff5e78c3c02600e529826d9d6c5e1c3af6c26e77ef12b11b`.
- C2: `82ed08304b565a1447062a95112c8a51b47ce75c8f46f07b858443236ba83f39`.

Selection/update provenance is externally bound to the source artifacts;
the exporter does not interpret the training tail to independently recover it.
The original checkpoints are unchanged. Runtime: Python 3.12.14, Torch 2.13.0,
CPU, one thread, eval/inference mode. Core evaluation times are 53.774s/54.367s;
orchestration timing is recorded separately. These are not training benchmarks.

Model inputs include the channel0 score with melody/accompaniment. Channel1
chord annotations are excluded from the raw graph. Both raw-view predictions
precede target extraction for each allowlisted piece. Shared split/index
metadata was read to validate membership; no TEST notes or target values were
read and no TEST metrics or selection were performed.

Independent artifact review recomputed all confusion matrices, pooled
accuracies, observed-class macro-F1 and joint counts. It verified identical
piece/source/target/input/candidate bindings and masks across the pair, report
SHA/fingerprints, planned-record linkage and exported tensor identities.

## Evidence and checks

- [Export record](records/EXP-POP-TRANSFER-EXPORT-001.json): completed.
- [C0 evaluation record](records/EXP-POP-TRANSFER-C0-001.json): completed.
- [C2 evaluation record](records/EXP-POP-TRANSFER-C2-001.json): completed.

Full immutable reports and source snapshots remain under ignored
`outputs/pop909_transfer/20260908/`. Compact records bind their hashes/sizes.
The source checkout was dirty on `codex/develop`, base commit
`da8c9dcd999a4b522f87f9a20bf3506ad26497b7`; source snapshots and file hashes
rather than the commit alone identify the executed implementation.

| Report | Bytes | SHA-256 |
| --- | ---: | --- |
| C0 | 20,359,289 | `74a796b450d1483d0a23ed582724224f5a8f416f1ec33e2de96725727b76f8b4` |
| C2 | 20,359,372 | `933dbba2a196800dd7ef42aa57f69d3bcb4ae269c54389089629188aa5f25b90` |

Focused implementation verification: 56 transfer/core/raw-reader tests
(9.17s), 59 exporter/strict-loader tests (4.52s), six repository-contract
tests (1.00s): **121 passed, zero failures/skips**. Four upstream Torch JIT
deprecation warnings occurred across the first two invocations. Dependencies
were restored in a temporary environment without adding project dependencies.
Legacy code was neither inspected nor reused.

## Next step

Preserve this result as the original frozen baseline. Before training a new
architecture, audit source semantics and support for the
[future modifier objectives](../HARMONIC_MODIFIERS_FUTURE.md), and predeclare
any per-field metadata diagnostic. Modifier supervision should distinguish
base harmony, added/altered/omitted degrees and contextual resolution, with
missing-label masks. Added heads alone do not settle ambiguous POP roots.
Any expanded mM7/+7 projection or new ablation requires a new protocol and
record; do not relabel this subset9 result. PLL/fragility remains a separate
evaluation-first question before adopting a quality-learning objective.
