---
name: mc-to-tickets
description: Split an agreed Music Critic V2 specification into independently verifiable local tasks with explicit blocking dependencies.
---

# Create bounded tasks

Read [workflow conventions](../../../docs/agents/workflows.md) and the source
spec. Work with the user's chosen tracker; absent one, use local files under
the existing workstream or `docs/workstreams/<date>/<feature>/issues/`.

Create one file per task, numbered in dependency order. Include its title,
parent spec, phase, behavior delivered, acceptance criteria, required tests,
scope/non-goals, blockers by stable file link and status. A new agent should
be able to implement one task from these references without this conversation.

Prefer small complete behavior slices. A contract change may need an
expand/migrate/contract sequence to keep callers working. Separate scientific
stages when they have different authorization or evidence requirements:
synthetic implementation, native equivalence/memory gate, authorized experiment,
and evidence review. A gate task is complete only with the specified evidence.

Check that every blocker exists, the dependency graph is acyclic and no task
depends on an unrelated deliverable. Mark unresolved prerequisites explicitly;
do not mark such a task ready. Present the breakdown with what each task
delivers and why it is blocked. Make routine decomposition choices within the
approved scope; ask only about a material choice the spec leaves open.

Local task creation is not permission to create app tasks, publish external
issues, launch runs or close a remote parent. Use external systems only within
the explicit request. Link experimental tasks to
[experiment-protocol](../experiment-protocol/SKILL.md) instead of duplicating
the registry. Work blockers first with [mc-implement](../mc-implement/SKILL.md).

Adapted from Matt Pocock's to-tickets, AskMatt 2.0.0.
See [provenance and license](../../../docs/agents/skill-provenance.md).
