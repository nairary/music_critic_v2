---
name: mc-domain-modeling
description: Clarify Music Critic V2 domain concepts and record resolved contract or scientific terminology. Use when ambiguous terms affect targets, interfaces or interpretation, such as voice versus track or theory accuracy versus human preference.
---

# Music Critic domain modeling

Follow the [project workflow policy](../../../docs/agents/workflows.md).
Use [DATA_CONTRACT](../../../docs/DATA_CONTRACT.md) for data concepts,
[ARCHITECTURE](../../../docs/ARCHITECTURE.md) for module responsibilities and
[DECISIONS](../../../docs/DECISIONS.md) for accepted architectural/scientific
choices. Retain this existing authority structure; do not introduce a competing
`CONTEXT.md` glossary or a second ADR hierarchy.

## Resolve a concept

1. Locate its definition, current implementation and callers. Identify whether
   a disagreement is ambiguous language, implementation drift, a historical
   decision or an actual proposed change.
2. State a precise candidate definition with identity/scope and availability
   where relevant. Check distinctions such as source voice versus track,
   observed relation versus inferred candidate, missing versus negative target,
   auxiliary theory objective versus preference or aesthetic judgment.
3. Test the definition against boundary cases. Label invented examples as
   hypothetical; real corpus claims require traceable source evidence and the
   task's allowed data scope. A convenient example is not a corpus observation.
4. Resolve facts from code and primary evidence yourself. Ask the user only
   for a consequential choice that the current request and accepted decisions
   do not settle. Keep unresolved alternatives explicit rather than silently
   renaming one into the other.
5. When documentation changes are within scope, update the relevant existing
   contract and its direct references. Update DECISIONS when the architectural
   or scientific decision changes, recording reasons and rejected alternatives.
   Do not rewrite historical experiment evidence to fit new terminology.

Conclude with the resolved definitions, behavioral implications, remaining
ambiguities and whether a schema, migration or separate implementation task
is required. A vocabulary clarification does not itself authorize model,
target or data changes. Decision-changing diagnostics follow
[experiment-protocol](../experiment-protocol/SKILL.md).

Source: adapted from Matt Pocock's `domain-modeling` skill and its context/ADR
references in the installed AskMatt 2.0.0 snapshot; existing project contracts
replace generic glossary and ADR file creation.
