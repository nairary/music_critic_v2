---
name: mc-ask-matt
description: Choose a repository workflow for Music Critic V2 when the user asks which skill or engineering process fits a task.
---

# Choose a workflow

Read [project workflow conventions](../../../docs/agents/workflows.md).
Select the smallest route that completes the requested work; use the linked
local skills rather than requiring the global AskMatt plugin.

| Situation | Route |
|---|---|
| A code/literature question | [mc-research](../mc-research/SKILL.md) |
| Existing experimental results need a defensible conclusion | [scientific-evidence-review](../scientific-evidence-review/SKILL.md) |
| A new experiment, continuation or result import | [experiment-protocol](../experiment-protocol/SKILL.md) |
| A reproducible bug or performance regression | [mc-diagnosing-bugs](../mc-diagnosing-bugs/SKILL.md) |
| A change needs Standards and Spec review | [mc-code-review](../mc-code-review/SKILL.md) |
| A concrete behavior needs implementation | [mc-tdd](../mc-tdd/SKILL.md) or [mc-implement](../mc-implement/SKILL.md) for a full ticket |
| An idea needs decisions | [mc-grill-with-docs](../mc-grill-with-docs/SKILL.md) |
| Agreed work spans several sessions | [mc-to-spec](../mc-to-spec/SKILL.md), then [mc-to-tickets](../mc-to-tickets/SKILL.md) |
| A large effort still has unresolved directions | [mc-wayfinder](../mc-wayfinder/SKILL.md) |
| Terms or module boundaries are unclear | [mc-domain-modeling](../mc-domain-modeling/SKILL.md) or [mc-codebase-design](../mc-codebase-design/SKILL.md) |
| Work moves to another environment or task | [mc-handoff](../mc-handoff/SKILL.md) |
| Agent instructions need editing | [mc-writing-for-agents](../mc-writing-for-agents/SKILL.md) |

Explain the chosen route briefly and perform the authorized work. A request
for recommendations ends with recommendations; it does not launch every route.
Research, implementation, native admission and model-quality evidence have
different completion criteria; do not turn one into permission for the next.

Adapted from Matt Pocock's Ask Matt, bundled in AskMatt 2.0.0.
See [provenance and license](../../../docs/agents/skill-provenance.md).
