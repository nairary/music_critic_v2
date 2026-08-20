# Post-Phase 9C model proposal

Status: **DISCUSSION DRAFT — NOT AN ACCEPTED ROADMAP OR ARCHITECTURAL DECISION**

Date: 2026-08-20

Intended use: independent review after the one-seed Phase 9C validation pilot

## Executive proposal

Do not conclude from the current Phase 9C result that graph neural networks or
self-supervised learning are unsuitable for the project, and do not respond by
immediately making the encoder deeper. The present experiment does not isolate
encoder capacity from downstream trainability, target imbalance, readout
capacity, augmentation, or SSL-objective alignment.

The next scientific step should be a small, preregistered diagnostic sequence:

1. prove that the supervised path can overfit a deliberately small balanced
   train subset and that predictions react to relevant temporal context;
2. establish a stronger supervised scratch reference on the existing locked
   split by adding train-only transposition augmentation and a temporal
   sequence/readout path alongside the graph hierarchy;
3. add shared multi-task readout/fusion only as a separate ablation;
4. compare scratch with one predeclared SSL variant under the same fixed update
   budget only after the supervised reference passes the trainability gate;
5. revise the SSL objective or scale the encoder only if those experiments
   locate the failure in SSL transfer or encoder capacity.

This sequence preserves the Phase 9C evidence rather than retroactively
changing its protocol. Validation remains the development split and test stays
locked.

## What Phase 9C currently establishes

The immutable unweighted pilot compared scratch, Phase 7A control, Phase 8A
mask-only, and Phase 8B multilevel-equal with seed 17. Every downstream cell
used `last.pt` after exactly 3,000 applied optimizer updates. The primary
comparison used all 71 validation records and did not access test.

The unweighted validation result was weak for every condition. Full fine-tune
macro-F1 was about 0.10--0.11; the scratch full-fine-tune cell had the best
normalized-NLL primary score in that pilot. Frozen probes collapsed to one
argmax class per task despite non-trivial NLL differences.

The opt-in inverse-square-root class-weight diagnostic improved scratch
full-fine-tune macro-F1 from approximately 0.112 to 0.128 and broadened some
class coverage, but it did not make the task well solved. Its SSL
full-fine-tune cells obtained slightly better primary normalized NLL than its
scratch cell, while paired bootstrap intervals still crossed zero. Frozen SSL
probes remained weak. These are one-seed validation diagnostics, not evidence
of generalization superiority.

The active corpus view is small and highly imbalanced:

- 577 train, 71 validation, and 71 locked test records;
- AN chord quality has 64 classes, only 42 with train support, and an observed
  train-count ratio of roughly 6,354:1;
- DLC chord quality has 15 classes and a ratio of roughly 13,334:1;
- inversion targets are less extreme but still imbalanced;
- the strict adapter currently accepts 719 of the 1,633 source records and
  quarantines the remainder for documented alignment failures.

Therefore a macro-F1 near 0.1 is alarming but not yet a clean measurement of
the information stored by the encoder.

## What the current model does and does not do

The present encoder is not a bag of independent event embeddings. It already
contains:

```text
raw note/onset/beat/bar/track/song graph
  -> relation-aware local heterogeneous GNN
  -> deterministic pooling to coarser levels
  -> coarse bar/track Transformer context
  -> top-down fusion to fine nodes
  -> four independent two-layer categorical heads
```

Its important limitation is where the sequence bottleneck occurs. Long-range
context is modeled after fine note/onset information has been pooled into
coarse tokens. The four task heads then consume the resulting representations
independently; they do not explicitly exchange harmonic decisions or model a
sequence of chord analyses.

The existing SSL path also already uses learned mask tokens and masks notes,
onsets, beats, contiguous bars, and track/bar spans. It reconstructs continuous
full-view latent representations at several hierarchy levels. It does not
predict discrete musical tokens or source-native harmonic events. Thus the
question is not simply whether masking exists; it is whether the reconstruction
target forces the encoder to preserve information useful for the downstream
harmony labels.

## Main competing explanations

The next work should discriminate among these explanations rather than assume
one in advance.

| Observation | Plausible explanation | Discriminating check |
| --- | --- | --- |
| A tiny balanced subset cannot be overfit | target alignment, masking, loss, gradient, or readout defect | tiny-subset trainability gate |
| Scratch becomes strong but SSL remains worse | negative transfer or a misaligned SSL objective | paired scratch/SSL run with the same stronger readout |
| Scratch and SSL both remain weak while a faithful external-style baseline works | current readout or input projection is insufficient | exact-split supervised reference baseline |
| A temporal branch improves scratch and makes SSL useful | the old probe could not extract contextual information | temporal-readout ablation |
| More encoder depth helps only after the prior gates | capacity or receptive-field limitation | isolated capacity ablation |

## Proposed experiment sequence

### Gate 0 — preserve the current evidence

- Treat both completed Phase 9C outputs as immutable.
- Do not use test for diagnosis, architecture choice, early stopping, or
  threshold tuning.
- Retain the fixed-budget `last.pt` comparison and group-safe split.
- Do not compare the present macro-F1 directly with published accuracy or
  chord-symbol-recall numbers computed on different units and corpus filters.

### Gate 1 — supervised trainability and label audit

Before another long matrix, run a bounded diagnostic on a deterministic small
train-only subset with supported classes and deliberately balanced examples.
The model must be able to drive training loss close to zero and training
macro-F1 high under a preregistered threshold. Also verify:

- every selected source entry joins the intended raw event and target;
- unavailable labels remain masked rather than becoming negatives;
- gradients reach the encoder and all supported heads;
- changing neighboring musical context changes the relevant logits;
- a trivial majority baseline and a class-balanced linear/MLP baseline are
  reported for calibration.

Failure stops the study and triggers debugging; it is not a reason to increase
model depth.

### Gate 2 — stronger supervised scratch reference

Keep the current raw graph, split, targets, validation metric, checkpoint
policy, and update budget fixed. Add one factor at a time:

- **E0 — immutable baseline:** current hierarchical model and heads;
- **E1 — transposition:** deterministic train-only pitch transpositions with
  exact label-equivariant target transforms and unchanged validation data;
- **E2 — temporal readout:** a small GRU or Transformer path over ordered
  onset/note representations in parallel with the existing graph/hierarchy
  path, fused before prediction;
- **E3 — multi-task fusion:** a shared prediction trunk plus an explicitly
  ablated mechanism for exchanging task logits/features;
- **E4 — combined:** only the components that pass their individual
  ablations.

Balanced source-entry sampling may be tested separately from loss weighting.
Natural, unweighted validation remains the comparison distribution. The
architecture and augmentation specifications, thresholds, seeds, and compute
accounting must be fixed before reading new validation results.

### Gate 3 — controlled SSL transfer

After a scratch configuration passes Gate 2, compare it with one predeclared
SSL initialization, preferably the variant selected from prior validation
evidence before the new run. Both cells must use:

- identical fresh downstream heads;
- identical data order and augmentation schedule;
- identical applied optimizer-update budget;
- the same transfer modes and fixed `last.pt` policy;
- validation-only selection and paired uncertainty reporting.

Only after a directional one-seed result should the comparison be repeated
over multiple seeds. The locked test is opened only for a separately accepted,
preregistered final protocol.

### Gate 4 — revise SSL or encoder capacity only if localized

If scratch is trainable and competitive but SSL transfer consistently hurts,
test SSL-objective alignment before indiscriminate scaling. Candidate
one-factor studies are:

- discrete masked raw-feature or musical-event prediction in addition to
  latent reconstruction;
- semantically meaningful temporal span masking;
- transposition-consistency objectives;
- a target-blind sequence branch shared by SSL and downstream;
- larger hidden width or more layers as an explicit capacity ablation.

PDMX scale remains a later roadmap step. Larger pretraining data cannot repair
a broken downstream join, an untrainable head, or an objective that discards
the required information.

## Why a hybrid model is the leading proposal

Graph message passing is useful for typed musical relations, simultaneity,
metrical ownership, and sparse structural links. Sequence models are useful
for ordered context and long-range dependencies. These strengths are
complementary.

AnalysisGNN is relevant not because it proves that the current project needs a
deeper GNN, but because its successful recipe combines a graph path with a GRU,
task interaction, extensive transposition augmentation, and broader
supervision. Its published ablations report the largest degradation after
removing transposition augmentation, with additional degradation after
removing logit fusion or auxiliary tasks. That points first to the training and
readout protocol, not merely to layer count. RNHybrid similarly combines a
pretrained sequence model with graph structure. Neither published number is a
drop-in baseline for the current 719-record filtered view; a faithful baseline
must be rerun on the exact V2 eligibility and split contract.

Primary references for review:

- [AnalysisGNN paper](https://arxiv.org/abs/2509.06654) and
  [official repository](https://github.com/manoskary/analysisgnn)
- [RNHybrid paper](https://arxiv.org/abs/2607.13587)
- [ChordGNN paper](https://arxiv.org/abs/2307.03544)
- [Dilemmadata paper](https://arxiv.org/abs/2606.31595)

## Questions for independent review

1. Is the proposed ordering sufficient to separate data/target defects,
   readout undercapacity, SSL misalignment, and encoder undercapacity?
2. What exact tiny-subset overfit threshold should be preregistered per task?
3. Should the temporal branch operate on notes, unique onsets, harmonic
   segments inferred from raw timing, or more than one resolution?
4. Should multi-task interaction fuse hidden features, logits, or both without
   leaking gold labels into inference?
5. What transposition group and label transformations are valid for every
   active AN/DLC quality and inversion target?
6. Which external supervised baseline can be reproduced faithfully on the
   exact V2 accepted records and group-safe split?
7. What minimum validation effect and seed count should authorize test access?
8. If SSL remains harmful, which discrete or contrastive target best measures
   musical context without importing theory labels into raw inputs?

## Explicit non-goals of this draft

- no change to an accepted contract, roadmap phase, dataset, split, budget,
  checkpoint, or test-lock policy;
- no claim that GNNs, SSL, or the current encoder have succeeded or failed in
  general;
- no authorization to implement or run the proposed matrix;
- no retrospective replacement of the Phase 9C pilot;
- no automatic increase in model size and no production experiment.

If this draft is accepted after review, the resulting scoped phase must record
its architecture and experimental decisions in `docs/DECISIONS.md` before
implementation.
