# Phase 9E-B5E full-training results

## Result boundary

Phase 9E-B5D completed paired C0/C1 training on an NVIDIA GeForce RTX 3090
from commit `aad02835afea8f91cfb9c8601a853c68335a5b47`. Both profiles used seed
17, batch size 2, 10,000 successful updates, 20,000 TRAIN draws, the same
initial model state and record schedule, and identity-only VALIDATION at
`0,500,...,10000`. Only the deterministic B5A-safe transposition schedule
differed. No TEST loader, target, metric, or selection was used.

The source archive SHA-256 is
`a9901c3ab9dd6914415a8ca7205f4247596c4aa261be9abe084d6a9523c7374a`.
The compact comparison fingerprint is
`03971d6568f29131c4cc909fd183f9bf9f6bbb9866a385a12255a3dab54835e9`.

## Final VALIDATION metrics

| Metric | C0 | C1 | C1 - C0 |
|---|---:|---:|---:|
| corrected primary macro score | 0.3548871111 | 0.2715279572 | -0.0833591539 |
| corrected harmonic-event joint accuracy | 0.1143047492 | 0.0140858475 | -0.1002189017 |
| paper-text note joint accuracy | 0.1119960909 | 0.0119716589 | -0.1000244320 |
| seen-tuple joint accuracy | 0.1275353085 | 0.0157162578 | -0.1118190507 |
| unseen-tuple joint accuracy | 0.0000000000 | 0.0000000000 | 0.0000000000 |
| direct Roman-numeral accuracy | 0.5833152463 | 0.4919792427 | -0.0913360036 |

Corrected joint support was 10,507 for both profiles. The unseen slice had
1,090 harmonic events across 187 target tuples absent from TRAIN; neither
profile predicted a complete unseen tuple correctly. Both profiles reached
their best corrected primary score at update 10,000. C0 exceeded C1 on every
one of the eight primary-head observed-class macro-F1 metrics.

## Decision

C0 (`music-critic-v2-corrected-no-transposition-v1`) is the current corrected
AnalysisGNN baseline selected on seed-17 VALIDATION. Its selected model-state
fingerprint is
`37e9dda262ae3db53c548d6d0b228fd4123e08e82b30eb8200b0b4c1327dbee4`.
The external `best-validation.ckpt` is not committed.

C1 is retained as `experimental_deferred`; neither its implementation nor the
B5A semantic transposition audit is invalidated. The paired result rejects a
transposition-benefit claim at this fixed seed/budget, but one seed does not
support a statistical generalization. Any future C1 retry requires a new
declared experiment rather than reinterpretation of this run.

The repository stores only this report, the compact JSON evidence, and their
fingerprints. Checkpoints, full training logs, the result archive, datasets,
caches, rendered audio, and generated MIDI remain outside Git.
