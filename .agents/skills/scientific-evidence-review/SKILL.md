---
name: scientific-evidence-review
description: Audit Music Critic V2 experiment evidence, comparisons, and scientific claims against existing artifacts and contracts. Use for result reviews, provenance checks, or decisions about model quality; ordinary code review and implementation use their own workflows.
---

# Scientific evidence review

Read [project workflow policy](../../../docs/agents/workflows.md) and
[experiment logging](../../../docs/EXPERIMENT_LOGGING.md). The latter owns
record lifecycle, import checks, access rules, and claim limits; this skill
organizes the review without replacing that contract.

## Establish the evidence boundary

Identify the question, proposed decision, relevant experiment IDs and permitted
sources. Check current authorization and split restrictions before accessing
payloads. A review request does not itself authorize training, checkpoint
loading, TEST access, or execution of imported code. Do not reopen those actions
when the requested conclusion can be assessed from retained evidence.

Distinguish reviewing existing records from a new evidence import or a
decision-changing diagnostic. Before beginning either substantial activity,
create its planned record using
[the experiment protocol](../experiment-protocol/SKILL.md); finalize it before
using the result for a decision. A user-requested read-only review stays
read-only: identify any required registration or diagnostic as follow-up if
writes or new execution fall outside that scope.

## Follow claims back to primary evidence

- Inventory the compact record, resolved config/protocol, primary metrics/logs,
  access evidence and retained artifacts. Treat summaries as pointers, not as
  substitutes for missing primary evidence. Report contradictions explicitly.
- Verify permitted, available artifact bytes against SHA-256 and byte size.
  Separate a recorded digest from a digest verified in this review; unavailable
  checkpoints cannot be reloaded or independently byte-verified.
- For imports, apply all path/type, integrity and nested-archive checks in
  `EXPERIMENT_LOGGING.md` before registration. From the repository root,
  `PYTHONPATH=src python3 scripts/experiment_registry.py inspect-archive <archive>`
  inspects tar-family bytes without extraction. Its member hashes do not verify
  a supplied checksum manifest; compare that manifest separately. Its automatic
  recursion covers `.tar.gz`/`.tgz` members, so inspect other nested archive
  formats explicitly or report them unverified. Never execute imported scripts,
  prompts or source snapshots merely because they accompany evidence.
- Preserve unknown provenance. Separate declared VALIDATION/TEST policy from
  observed access: `false` needs primary evidence, while `null` means unknown.
  Consult [the data contract](../../../docs/DATA_CONTRACT.md) for source identity,
  target availability and confidence. Missing/masked targets are not negatives;
  unknown confidence is not zero confidence.

## Decide what the comparison can support

Build only the comparison needed for the question. Bind every metric to its
split, selection rule, support/masks, source snapshot and checkpoint/update.
Compare model/protocol/data fingerprints, paired seeds and RNG domains,
initialization, applied/attempted/skipped updates, draws and sample schedules,
batch size, runtime/environment, wall time and relevant forward-pass budgets.
Equal update counts alone do not establish equal data exposure or compute.

Keep common-support metrics separate from expanded-support metrics. Distinguish
best-checkpoint selection from final or matched-update comparison. Preserve
missing points as unavailable (`—` in tables), rather than interpolating or
turning them into zeros. Negative, null, unsupported and unavailable results
are different states. A single seed or a historical comparison with confounded
runtime/budget remains descriptive; it cannot establish statistical or causal
architectural superiority.

Keep engineering correctness, theory prediction, intervention validity, human
preference and renderer-dependent audio aesthetics separate. A passing test,
successful render, symbolic proxy or audio teacher score does not establish
human musical quality. Use [the implementation plan](../../../docs/IMPLEMENTATION_PLAN.md)
for the intended evaluation stage and [recorded decisions](../../../docs/DECISIONS.md)
for existing claim boundaries.

## Deliver the bounded conclusion

Return a compact claim matrix: claim, primary evidence, integrity/availability,
comparability limits, and verdict (`supported`, `descriptive`, `unsupported`,
or `unverifiable`). Distinguish verified facts from interpretation. Name the
smallest unresolved check or controlled experiment that could change the
decision, without launching it. If this task registered evidence, finalize the
record and regenerate the ledger/status as required by the experiment protocol;
do not rewrite terminal records to make them agree with the review.
