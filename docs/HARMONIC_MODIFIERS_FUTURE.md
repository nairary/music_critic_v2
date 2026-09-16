# Future auxiliary learning of chord modifiers

2026-09-08; readiness clarified 2026-09-15. Status: **TRAIN targets prepared;
new prediction head and training not implemented**. The user
wants to learn chord extensions/modifications later and prioritizes evaluating
the existing C0/C2 models on POP909-CL now. This proposal changes no current
model, task vocabulary, loss, checkpoint, split or frozen evaluation metric.

## Motivation and limits

The corrected C0/C2 model learns 18 selected tasks; this is not complete
coverage of Dilemmadata annotations. DLC separates base `chord_type` from
`changes` and `added_tones`. Its derived `a_quality` maps the base type. Some
details can still reach current objectives through derived labels; upstream
also uses `changes` to identify cadential `Cad` labels. Do not describe these
phenomena as wholly absent from training, or claim their prevalence without a
registered TRAIN/VALIDATION audit.

The current quality17 vocabulary lacks sus2/sus4. The frozen POP evaluator
has a separate subset9 eligibility rule; mM7 and +7 already have corrected17
counterparts, but were excluded by the initial evaluator specification.
Expanding that projection is a separately versioned evaluation decision,
not a prerequisite for the existing frozen comparison.

## Proposed stages

1. **Audit source meaning and coverage.** Map source fields to current targets,
   explicit information losses and potential new targets. Keep AN and DLC
   distinct. Retain original strings, parser version, source/derived status,
   confidence and per-field masks. Establish whether an empty value means
   known absence or unavailable annotation before assigning negatives.
2. **Factorized modifier heads.** Keep base harmony prediction and add small
   multi-label heads for additions, omissions, alterations and suspended chord
   forms. Multiple modifiers can coexist; avoid an enumerated class for every
   compound chord label. If a suspended form does not specify a major/minor
   third, do not invent one as a base-quality target. Input remains raw notes;
   any dependency on root/quality at inference uses predictions, not gold.
   Prediction positions come from raw notes/onsets/beats. Target spans join
   after forward and cannot define graph topology or prediction candidates.
3. **Annotated sonority reconstruction.** Compare a compositional 12-bit
   pitch-class-set objective with the existing categorical set head. This is
   the set represented by a harmonic annotation, not automatically every
   simultaneously sounding melody/accompaniment note. Relative intervals need
   an independently supported root. Keep register/degree/spelling separately
   when distinguishing add2/add9 or functional interpretations matters.
4. **Contextual suspension and resolution.** A later note/event objective may
   describe preparation, dissonance and resolution across time. A sus chord
   symbol alone does not establish a classical suspension, its voice or its
   resolution. Automatically derived examples retain weak/derived provenance;
   missing evidence is masked. Gold chord spans, voices or roots cannot be
   required by the deployed model.

## Source feasibility

Update 2026-09-15: the DLC TRAIN census, source-native sidecars and full
conservative raw-onset alignment are complete; see
[DLC_CHANGES_TARGETS.md](DLC_CHANGES_TARGETS.md) and the
[full TRAIN report](experiments/DLC_CHANGES_FULLTRAIN_RU.md).
The selected 1,011 DLC works retain 251,038 usable annotations, including
34,055 nonempty sets. All 99 atomic classes retain positives; 45 have fewer
than ten usable annotations. Bounded transposition checks passed previously;
this is not a full-corpus augmentation equivalence claim. Missing steps are
VALIDATION target preparation with the frozen TRAIN vocabulary, authenticated
binding to the current temporal campaign's raw pieces/slice candidates, new
head/loss/loader integration and trainability/resume checks. The prepared raw
conversions are not proven byte-identical to historical campaign caches.
Temporal slices without supported source annotations must stay masked.
Semantic note-level preparation/suspension/resolution labels and the training
experiments below remain future work.

- HookTheory already has source-native multi-label `adds`, `omits`,
  `alterations` and `suspensions` target families. These were not trained by
  C0/C2. Its raw input is melody-only, so prediction concerns a compatible
  annotated harmonization, not recognition of observed accompaniment.
- DLC offers modification syntax and added/chord-tone fields. The registered
  TRAIN changes export now preserves 99 source-native atomic classes, masks and
  source rows; added/chord-tone objectives and note-resolution semantics remain
  unaudited at full coverage. No VALIDATION coverage is established.
- POP909-CL chord blocks support a set of annotated pitches. They do not
  always establish a unique chord root: {C,D,G} admits Csus2 and Gsus4.
  The existing exact-template normalizer applies independent masks when a
  field's candidate values disagree; the sus example makes both ambiguous.
  Adding output classes does not resolve this ambiguity. Set-valued targets
  or additional independent source evidence may be appropriate; never choose
  the first candidate or assume the lowest note is the root.
- PDMX is unlabeled. Raw-note SSL may complement these objectives; musical
  annotations must not be fabricated as ground truth.

## Acceptance experiment after frozen POP evaluation

Compare equal continuations from one verified C2 parent: existing objectives
versus the same objectives plus the proposed auxiliary targets. Predeclare
the same pieces, seed, update budget, schedule, optimizer initialization and
loss-group weights and source mixture. Normalize within each task/family by
available labels before combining fixed group weights; absent annotation
must not create numerous false negatives or dominate the original tasks.

Report per-modifier support, precision/recall/F1, exact modifier-set accuracy,
coverage and performance of the existing 18 tasks. Report modifier-set scores
separately for examples with positive modifiers, so an all-empty prediction
cannot conceal failure on the intended task. Keep future transfer evaluation
independent of architecture/weight tuning. A theory recognition
gain is not automatically a critic-quality gain: degradation/PLL/fragility
evaluation remains a separate evaluation-first question under
[HARMONIC_SUPERVISION](HARMONIC_SUPERVISION.md).

Relevant implementation references: `tasks/ontology.py`,
`tasks/dilemmadata_registry.py`,
`experiments/analysisgnn/multitask_contract.py`, `adapters/pop909_cl.py`.
DCML semantics: [annotation reference](https://dcmlab.github.io/standards/build/html/reference/reference.html#suspensions-and-retardations).
Legacy inspection/reuse is unnecessary for this proposal.
