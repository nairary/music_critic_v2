---
name: mc-to-spec
description: Turn an agreed Music Critic V2 change into an implementation specification with contracts, test boundaries and explicit scientific claim limits.
---

# Synthesize an implementation spec

Read [workflow conventions](../../../docs/agents/workflows.md). Synthesize the
conversation and verified repository state; do not restart an interview or
reopen accepted decisions. Ask only when a material contradiction blocks a
usable spec; otherwise state a bounded assumption or an open question.

Save alongside the current workstream, or under
`docs/workstreams/<date>/<feature>/spec.md` for new multi-session work. Include:

- The concrete problem, current behavior and requested outcome.
- The roadmap phase, accepted decisions and explicit non-goals.
- Relevant modules, current source revision and contracts to preserve.
- Observable acceptance criteria and the public interfaces tests exercise.
- Failure, unavailable-data and compatibility behavior where relevant.
- Independent completion criteria for implementation, native admission and
  scientific evaluation, including any inputs/budget still unresolved.

Reference exact project contracts rather than copying their schemas. Paths
are navigation aids; bind decision-critical source references to a revision.
Acceptance criteria should expose a failing implementation, not mirror its
internal calculations. Use [mc-codebase-design](../mc-codebase-design/SKILL.md)
only if an interface is genuinely unsettled.

Check that the spec neither grants a new training budget nor treats a proposed
result as measured. Use [experiment-protocol](../experiment-protocol/SKILL.md)
for a substantial experiment's record; a spec cannot replace it. Keep remote
publication within the user's explicit request; a local spec is sufficient
for repository planning. Finish with the spec path and remaining blockers.

Adapted from Matt Pocock's to-spec, AskMatt 2.0.0.
See [provenance and license](../../../docs/agents/skill-provenance.md).
