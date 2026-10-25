---
name: mc-implement
description: Implement an approved Music Critic V2 spec or local task through tested behavior slices and independent code review.
---

# Implement the selected task

Read [workflow conventions](../../../docs/agents/workflows.md), the source
spec/task and its blockers. Confirm current repository state and the requested
roadmap phase. Satisfy AGENTS' pre-edit task protocol using the actual files,
tests, legacy need and non-goals. Reuse the user's existing authorization.

Use [mc-tdd](../mc-tdd/SKILL.md) for small behavior slices at the spec's test
boundaries. Keep unrelated working-tree changes intact. Validate the changed
behavior with the relevant tests and repository-required checks; use the
broader suite when scope or unresolved integration risk warrants it.

Use [mc-code-review](../mc-code-review/SKILL.md) to review both project
standards and the originating spec, including WIP files. Fix actionable
findings in scope and rerun the checks affected by the fixes.

Report implementation completion separately from native admission and
scientific quality. Route any required substantial run through
[experiment-protocol](../experiment-protocol/SKILL.md). Do not silently turn a
code task into model selection or a training launch.

Update STATUS and applicable authoritative docs. Stage, commit, push, or
publish only as requested by the user; this workflow does not prescribe an
automatic commit. Finish with behavior, files, exact check results, remaining
limitations and the next phase or decision gate.

Adapted from Matt Pocock's implement, AskMatt 2.0.0, with repository-specific
execution and VCS scope. See [provenance and license](../../../docs/agents/skill-provenance.md).
