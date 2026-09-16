# Coordinator review 001 — context diagnostics C2

Date: 2026-09-08. Disposition: **narrow corrections requested before CUDA**.

The sealed source package is valid: 16 regular safe archive members, exact
manifest hashes and exact reconstruction from the own patch. Independent
execution of all four new test modules passed **89 tests**, zero failures or
skips, two existing warnings, 1.91 seconds. The reviewed copy is reconstructed
from immutable source/patch rather than from the mutable worktree.

The live STATUS document changed after sealing; the other sealed files match.
The new handoff must bind the latest documentation, keeping the old package.
This is a detected documentation drift, not evidence of hidden model changes.

The local classifier normalizes/fits only eligible TRAIN observations, uses
the same eligible evaluation rows as its reference, keeps unavailable labels
masked and retains exact raw timing. Its small nearest-centroid capacity and
caller-owned provenance checks are explicit limitations. Context interventions
preserve original attack/duration for retained notes and are labelled as
sensitivity probes, not trained local models or correctness tests.

Required corrections:

1. Frozen suite forward progress currently lives only in memory. Failure output
   retains error/type but not attempted/completed forward counts or pair/side/
   view stage. Add bounded progress and failure accounting, with an injected
   failure regression; no extra model-quality target is needed.
2. The GPU packet describes the checkpoint as verified best115k. The independently
   hash-verified export provenance has selection_update_fields_extracted=false.
   Bind the preselected C2 export by SHA; distinguish declared parent selection
   metadata from an independently recovered historical update. Correct only
   unlaunched planning metadata; terminal artifact records remain immutable.
3. Reseal source/protocol and current STATUS in a new package. Operator 600s /
   4GiB ceilings require external supervision and must not be presented as
   already enforced Python guards or observed GPU resource use.

The artifact-only scientific result remains in child-owned
EXP-CTX-C2-ARTIFACT-002, with its failed parent retained. This code review does
not repeat that diagnostic or independently re-estimate its bootstrap interval.
No new real-corpus/checkpoint inference or TEST access occurs here.

Corrections were sent to verified task
`01a08069-a001-71e0-ab5e-894b75b06709`. The real local-vs-C2 experiment still
requires authoritative source/split/component and event-candidate bindings.
The main model and shared evaluator are unchanged; no legacy, GPU/SSH, commit
or merge is performed. Next is review of the corrected bounded suite packet.

