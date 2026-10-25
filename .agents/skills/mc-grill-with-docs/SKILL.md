---
name: mc-grill-with-docs
description: Stress-test an unsettled Music Critic V2 idea and record the decisions needed before implementation or an experiment protocol.
---

# Resolve the design questions

Read [workflow conventions](../../../docs/agents/workflows.md) and use
[mc-domain-modeling](../mc-domain-modeling/SKILL.md) when terms are ambiguous.

Find facts in code, contracts and existing reports before asking questions.
Map decisions and their dependencies. Ask only the current unresolved choices
whose prerequisites are known, with a recommendation and its tradeoff. Use
available question tools; keep rounds small enough to answer. Continue
independent inspection while waiting, but do not invent the user's answer.

For a scientific idea, distinguish the proposed capability, available labels,
observable inputs, evaluation unit, baseline, success criterion and resources.
Use hypothetical examples only when labelled as hypothetical. A design
discussion neither supplies human preference labels nor measures a hypothesis.

Reuse decisions already made in the conversation. Record accepted terminology
in existing domain documents; put architectural/scientific decisions in
`docs/DECISIONS.md` only when a decision actually changes. Put a bounded design
note alongside the current workstream. Keep rejected choices and open questions
visible where they explain the selected route.

Finish when the requested decision is explicit and its material dependencies
are settled or clearly identified. Produce a spec through
[mc-to-spec](../mc-to-spec/SKILL.md), or an experimental plan through
[experiment-protocol](../experiment-protocol/SKILL.md), if the task calls for it.
An existing authorization to implement remains valid; ask again only for a
material choice outside it.

Adapted from Matt Pocock's grill-with-docs and grilling, AskMatt 2.0.0.
See [provenance and license](../../../docs/agents/skill-provenance.md).
