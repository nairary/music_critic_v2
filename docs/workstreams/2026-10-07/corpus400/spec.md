# T08 — 400 complete event-bound musical analyses

User authorization, 2026-10-07: collect 400 complete works, exactly 100 each
classical, jazz, rock and pop, maximizing useful overlap among existing MIDI
and analytical sources. Supplement from public analyses and individual
`gpt-6-astra` agents at `ultra` effort. The user authorizes available account
usage and credits for this work; no numerical token or currency cap was given.
No automatic account-reset redemption, credit purchase, external messaging or
GPU training is included. Routine work proceeds without repeated approvals.

This is the data phase following T07, not a claim that the corpus is complete.
Repository start: `287bf856643f9e555352e1661a7bf0563772178a`, dirty main.
Preserve existing work and immutable releases. Authoritative contracts:
[data](../../../DATA_CONTRACT.md), [logging](../../../EXPERIMENT_LOGGING.md),
[architecture](../../../ARCHITECTURE.md),
[T07](../multigenre-theory-enrichment/spec.md).

## Unit and admission

- One work-family quota slot, one selected complete symbolic version with
  usable pitched events. Alternative arrangements, takes, fragments and
  transpositions do not inflate counts. A complete movement may be an
  explicit unit, but movements from one larger work do not fill several slots.
- Genre is evidenced by source metadata, creator/repertoire context and,
  where needed, public references. Title-only genre guesses are candidates.
  Crossover tags may coexist; one primary quota genre prevents double counting.
- A MIDI export must preserve canonical notes, exact time and event mapping.
  Keep original MIDI where available; mark score-derived MIDI explicitly.
  Single-instrument arrangements remain such; do not invent absent parts.
- Existing frozen V2 VALIDATION/TEST remain closed. New public sources require
  recorded family checks against metadata and a new supplemental TRAIN
  assignment before reading musical payload. Unknown original membership
  remains unknown; do not rewrite the global split. Known family conflicts
  are quarantined. Public catalog metadata may be searched across splits.
- Names/creators retrieve candidates. Theory transfer requires version-specific
  content and timing alignment, including repeats, pickups and transposition.
  Source license and redistribution status remain source-specific.

## Required result

For each selected work: original-source receipt, canonical notes, MIDI plus
note mapping, theory bundle, external evidence links, annotation provenance,
coverage ledger, machine validation and review receipts.

Required analytical layers:

1. Note × harmonic context: membership, key/chord degrees, voice identity,
   contextual roles, explicit preparation/resolution targets when applicable.
2. Chord intervals: complete available sonority/extension/inversion, key,
   Roman interpretation and context-dependent function.
3. Harmonic processes: prolongation, progression/sequence, tonicization,
   modulation, cadence or loop; boundaries and addressed participants.
4. Motive occurrences and relationships: membership, repetition/transformation;
   phrases and phrase functions separately, including supported question/answer.
5. Form: full-work sections and repetition/contrast relationships.
6. Ensemble: parts/voices, texture roles and temporal/vertical interactions;
   explicit applicability for genuinely single-line sources.

These are overlapping structures, not one mandatory containment chain.
Suspension symbols or seventh-to-chord-tone motion alone do not establish
complete tension analysis. Observed sounding extensions are not discarded
when aggregating to larger structures.

## Completeness and provenance

Required states are separate: discovered, family-admitted, raw-materialized,
source-enriched, model-analyzed, structurally-validated, independently-reviewed,
complete-for-declared-scope. Track per-layer/per-field temporal and event
coverage. Every gap has an explicit reason. Empty arrays and missing values
do not mean reviewed absence. Not-applicable needs a musical reason.

Source facts, deterministic derivations, heuristic suggestions, individual
Astra interpretations and reviewer decisions retain separate origins and
versions. Model annotations are weak labels, never human gold. Requested
model/effort and actual returned metadata are recorded separately; unavailable
backend tokens/costs remain null. A second model review is not human agreement.
Alternative readings are retained. Auxiliary theory is not an inference input.

Quota completion requires 100 distinct admitted full-work families per genre
and all required layers accounted for with a declared substantive analysis
scope; unknown required content is not silently counted as complete. Report
each stage separately even when the target quota is not yet satisfied.

## Execution and verification

First register metadata selection/admission under EXP-CORPUS400-CATALOG-001.
This initial record covers metadata joins, file availability and candidate
selection only, not musical payload annotation. Close it before subsequent
materialization/annotation runs, which receive new linked IDs and fresh output
directories. Retain failed attempts. Use outputs/corpus400_20261007/.

Implementation changes need focused tests for family deduplication, missing
metadata, split rejection, exact MIDI roundtrip, layer completeness, masks,
typed participant integrity, source alignment and immutable output. Independent
review must challenge claimed completion and inspect actual target payloads.
No dependency installation is implicit; reuse available runtimes.

Success is the verified dataset and a coverage/provenance report, not a plan,
text-only analyses, a count of clips or classifier training. Publication and
neural-model training remain separate work. Legacy is not read or reused.
