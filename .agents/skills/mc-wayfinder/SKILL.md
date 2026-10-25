---
name: mc-wayfinder
description: Map unresolved decisions for a large Music Critic V2 research or engineering direction that cannot yet be expressed as a buildable specification.
---

# Map the decisions

Read [workflow conventions](../../../docs/agents/workflows.md). Use this only
when uncertainty spans multiple sessions. A bounded bug, ablation or already
agreed feature goes directly to its smaller workflow.

Define the destination and exclusions from the request. Create a local map
in the current workstream: a concise destination, an index of decisions,
unresolved questions and explicit out-of-scope work. Store each decision's
details once in a linked file. A sharp question can become a task even when
blocked; an area not yet understood enough to name a question stays on the map.

Give decision tasks prerequisites and a status. Select an unblocked task,
record who is working on it and avoid duplicating another agent's work.
Resolve factual questions through [mc-research](../mc-research/SKILL.md);
material user choices through [mc-grill-with-docs](../mc-grill-with-docs/SKILL.md).
If an empirical answer is required, describe the smallest useful experiment
through [experiment-protocol](../experiment-protocol/SKILL.md) within the
authorized scope. Do not treat a design discussion as empirical evidence.

Record each resolution with evidence, limits and newly exposed questions.
Preserve superseded reasoning; mark a question out of scope with its reason
rather than reporting it as a solved scientific problem. Reconcile concurrent
edits before updating the map. Link by descriptive title, not bare numbers.

Finish when the path to the requested outcome can be specified. Hand the
decisions to [mc-to-spec](../mc-to-spec/SKILL.md), then
[mc-to-tickets](../mc-to-tickets/SKILL.md), as appropriate. Mapping is planning;
execution and external issue publication follow the user's actual scope.

Adapted from Matt Pocock's wayfinder, AskMatt 2.0.0.
See [provenance and license](../../../docs/agents/skill-provenance.md).
