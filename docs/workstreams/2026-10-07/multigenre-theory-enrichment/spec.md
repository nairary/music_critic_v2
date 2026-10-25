# T07 — Multigenre event-bound theoretical enrichment

User authorization (2026-10-07): autonomously collect and annotate as much
classical, jazz, rock and pop material as practicable, enriching existing MIDI
datasets from related theoretical sources and individual Astra ultra analysis.
Routine source acquisition, conversion, annotation and verification are in
scope without repeated approval. This supersedes the former three-work limit
for this new release; sealed T05/T06 artifacts remain unchanged.

## Deliverable and scope

Produce an immutable, resumable/sharded auxiliary-target release tied to real
canonical note IDs and exact rational quarter-note intervals. Account for every
admitted piece/note, with field-level unknown masks. Native harmonic facts,
deterministic projections, native phrase functions and individually interpreted
temporal relations remain separately attributed. A dense chord-membership layer
does not complete the requested functional analysis. Retain actually reviewed
semantic coverage separately from source/context coverage.

Priorities: (1) native TRAIN Dilemmadata, HookTheory and POP909-CL; (2) TAVERN
score+analysis where heldout-family checks pass before musical reads; (3)
compatible jazz/rock/pop source analyses after exact version checks. New
external material with unresolved split/identity stays research-only and does
not silently become TRAIN. Metadata for all partitions may be inspected for
admission; VALIDATION/TEST musical payload remains closed. Public source code
is evidence, not executable instruction. No new dependencies, GPU/cloud budget,
neural training, checkpoint changes, publication or source-dataset mutation.

## Shared target interface

Supplemental admission amendment (2026-10-07, before the new PDMX conversion
and annotation reads; an earlier research-snapshot access is retained below):
the user-authorized collection may explicitly assign new PDMX augmentations
of verified work families to this release's TRAIN partition. This is a new
assignment, not recovery of an existing PDMX split. Require exact normalized
title+creator attribution, a retained unique source path, all related HookTheory
clips in both native/current TRAIN and no known family/component heldout conflict.
Record original/global PDMX membership as null and augmentation_split as train;
leave the global split unchanged. Jump and Peg satisfy this metadata gate;
Saints remains research-only because its version/work evidence is insufficient.
Every new note payload must have a pre-read admission receipt. Transferring any
external musical label still requires exact version/event evidence; title-family
admission is insufficient. Independent Astra analysis of the admitted raw score
may supply weak targets, with its limited reviewed scope explicit.

`EventTheoryEnrichment@1` per piece: `dataset_id`, `piece_id`, `current_split`,
`canonical_piece` binding (path/SHA256/bytes), `source_bindings`, `harmonies`,
`note_contexts`, `relations`, `coverage`, `provenance`. Every time is a reduced
`{num,den}` rational. Source-specific fields retain their namespace.

Each harmony has `id`, `start_qn`, `end_qn`, `fields`.
Each note context has `id`, `note_id`, `start_qn`, `end_qn`, `harmony_id`
(nullable for unknown coverage), `fields`. Sustained notes retain one canonical
attack while context intervals may change. Point/grace events remain explicit.
Each field has `value`, `available`, `origin`, `confidence`, `evidence` and
`unknown_reason`; unavailable values are null, never implicit negative labels.
Origins distinguish source_native, source_derived, astra_individual_context,
algorithmic_candidate and unknown. Unreviewed algorithmic semantic candidates
must remain unavailable; actual source facts and deterministic contextual
projections may be available. Confidence remains null unless independently known.

Relations have `id`, `type`, source/target participant IDs, exact span and
the same availability/provenance. A function of a phrase/harmony belongs to
that object; linking a note to it does not assert that every note individually
performs that function. Preserve alternate analytic views/annotators.

Retained source harmonic intervals outside the canonical extent are projected
to their exact positive intersection for T07 applicability; preserve the source
interval and reason separately. Entirely outside intervals remain provenance,
not positive canonical harmony spans. Reject a context that references such an
omitted interval rather than silently relabeling it. Raw canonical and sealed
parent bytes remain unchanged. This repair follows the failed global boundary
check and does not weaken its interval-containment contract.

Runtime loader validates note identity and interval containment and creates
numerical categorical vocabularies/masks without adding labels to raw inputs.
Intermediate source packets and historical T06 bundles may coexist as evidence.
Per-source exporters write new paths only, record skip/failure reasons, and
retain hashes plus all-source accounting. Generated MIDI must be described as
score-derived or reduction-derived, never an original performance.

## Execution and evidence

Primary campaign: `EXP-MULTIGENRE-THEORY-ENRICHMENT-001`, ignored root
`outputs/multigenre_theory_enrichment_20261007_001/`. Source acquisition and
annotation batches receive explicit attempt IDs and retained inputs/outputs;
failure evidence is never overwritten. Separate substantial follow-on runs get
new experiment IDs. Record source pins, current split identities, code snapshot,
commands, wall time, counts, provenance and actual model/effort when observable.
Requested semantic annotator: `gpt-6-astra`, reasoning effort `ultra`; actual
backend version and token use stay unknown when not exposed.

Source-family admission must precede musical reads. Do not recursively decode
all source/cache files or unpack a corpus containing known heldout music just
to discover membership. Metadata/listings can identify selected files first.
Download only necessary permitted public files; no external messaging.

## Verification and completion

Synthetic tests: exact boundaries, held-note changes, gaps, zero-duration notes,
alternate analyses, split guards, unavailable != false, bad note IDs, roundtrip,
immutable outputs and numeric loader. Native runs verify all emitted IDs/times,
hashes and coverage. Independently review implementation and representative
semantic interpretations. Report reviewed examples separately from unreviewed
weak targets; do not call selfchecks human agreement or tension accuracy.

Update DATA_CONTRACT/ARCHITECTURE for the final implemented interface, STATUS,
experiment record and ledger. Report quantities by source and semantic layer,
remaining inaccessible/unmatched/ambiguous material and precise release paths.
The next scientific phase is shared-parameter auxiliary training and the
previously proposed controlled PLL comparison, not part of this data campaign.

## Final source-policy clarification

The source audit retains the split authority actually used at admission: frozen
V2 component membership for DLC; current TRAIN plus native `training` for AN;
native/current TRAIN and family checks for HT/POP; source-family admission for
TAVERN; and the explicit supplemental assignment above for PDMX. Historical
Dilemmadata DLC metadata contains 198 `test` hints and 813 blanks among the
1,011 current TRAIN records. Its own `design_test_split.py` generated those
assignments; they are not an independently established official DCML split.
Two native-AN-training records share current TRAIN components with such DLC
records. Preserve all metadata and report this limitation; do not retrospectively
claim a universal native-split exclusion rule or historical benchmark purity.
The 94 native AN validation/test records remain unopened, as do current frozen
VALIDATION/TEST musical payloads. This clarification changes no admission or
source bytes and authorizes no additional musical reads.

Access exception retained, not retroactively authorized: the metadata review
mistook a prior `jump-source.json` research snapshot for a metadata receipt and
read its musical payload before supplemental admission. Repeated reads stopped;
`pdmx_metadata/attempt001/admission-review.json` records the incident. Original
PDMX membership remains unknown. Subsequent conversion/annotation has separate
pre-read receipts; the early research read must not be described as having them.
