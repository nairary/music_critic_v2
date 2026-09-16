# Coordinator review 001 — structural SSL and voice mapping

Date: 2026-09-08. Disposition: **changes requested before CUDA**.
No real-data or musical-quality result is accepted from fixture mechanics.

## Verified engineering scope

The 19-file own patch matches its before/after hashes, with no hidden inherited
changes. All 34 referenced retained artifacts match SHA-256 and size.
Independent reconstruction from the shared 752-file snapshot plus patch passed
110 focused tests (new pilot/CLI/voice plus existing warmstart): zero failures
or skips, two existing warnings, 25.14 seconds. The author's larger 542-pass /
5-CUDA-skip set was not needlessly repeated.

Existing Phase7A random-note and Phase8A raw-beat masking are reused. The pitch
leakage claim is explicitly limited to audited pitch-derived feature slots;
timing, duration and topology remain visible. Full fresh initial states and
shared optimizer ownership are checked. The new model-only bridge preserves
the old strict loader and atomic state mapping. Scoped voice targets and raw
candidate construction do not equate tracks with voices. A real annotation
importer and corpus readiness remain unverified.

## Required corrections

1. Finite gradients can overflow Adam's second moments: a synthetic finite
   1e30 gradient produces an infinite slot, yet step/save claims success and
   own resume rejects the checkpoint. Check post-update model/optimizer
   finiteness before successful accounting; poison invalid sessions.
2. Runtime flags are bound only at construction. Drift before step/save is
   not rejected. Check actual execution/checkpoint/resume runtime against its
   binding and include applicable CUBLAS workspace configuration.
3. Resume accepts impossible accounting after an internally consistent rehash:
   negative ssl_samples and empty paired_masks pass. Validate strict schema,
   ranges and consistency with the scheduled batch, mode and policy.
4. CLI logs only after step returns. An interruption after optimizer application
   loses attempted forwards and leaves only zero completed receipts. Preserve
   started/stage/forward observations and applied/not-applied/unknown separately
   from fully committed updates.

Short independent synthetic probes reproduced these gaps without changing
repository files. The stated 600-second check is between steps; the coordinator
must supply an external hard timeout/resource supervisor for remote execution.
It is not a measured or enforced in-process hard resource cap.

Corrections were sent to verified task
`01a08069-bfbf-7aa3-8b1d-76ccc75c8b85`; a separate review-002 artifact package
and still-planned record resealing are requested. Existing artifacts are retained.
Scientific readiness additionally needs real source/split/eligibility/schedule/
replay bindings, validation and recovery evaluator; this increment does not
erase those blockers.

Only review/status and ignored coordination evidence changed in the coordinator
checkout. No GPU, SSH, real corpus, TEST, legacy, commit, merge or shared-model
change was performed. Next is acceptance of the narrow corrected package.

