---
name: mc-codebase-design
description: Design Music Critic V2 software module interfaces, ownership and test seams. Use to untangle model/runtime/campaign or adapter responsibilities; this is not evidence that a deeper neural graph improves learning.
---

# Music Critic module design

Follow the [project workflow policy](../../../docs/agents/workflows.md) and
the relevant [ARCHITECTURE](../../../docs/ARCHITECTURE.md),
[DATA_CONTRACT](../../../docs/DATA_CONTRACT.md) and
[DECISIONS](../../../docs/DECISIONS.md) sections.

## Working vocabulary

- **Module:** behavior with an interface and implementation, at any useful scale.
- **Interface:** everything callers need to know, including invariants,
  ownership, ordering, configuration, errors and performance expectations.
- **Seam:** the location where behavior can be substituted without changing
  the caller; an **adapter** supplies a concrete implementation there.
- **Depth:** useful behavior behind a small interface. It concentrates changes
  and reduces knowledge callers must carry; it is not a line-count ratio or
  the number of GNN layers.

## Design against actual callers

1. Trace the requested behavior through current call sites, data ownership and
   tests. Show where callers duplicate knowledge or where one change spreads
   across unrelated responsibilities; use file references, not impressions.
2. Define what a proposed interface owns and hides. Make tensor/sidecar
   ownership, exact time, masks/provenance, schema identity and resume state
   explicit whenever this module transports them. Preserve raw-only inference.
3. Keep pure transformations directly testable. Inject unavoidable external
   dependencies or state where a real production/test substitution is needed.
   A new layer must reduce caller complexity or localize a concrete variation.
4. For consequential alternatives, compare distinct interface shapes using a
   representative real call site. Delegate independent designs when useful.
   Compare required caller knowledge, ownership, compatibility and testability;
   give a recommendation with its tradeoff, not a universal architecture rule.
5. Separate design from implementation scope. If implementation is requested,
   migrate a bounded slice with [mc-tdd](../mc-tdd/SKILL.md), retain required
   behavior coverage and document compatibility or migration requirements.
   Record accepted architectural changes in DECISIONS.

The deletion test is useful: if removing a wrapper removes complexity, justify
keeping it; if it distributes hidden obligations among callers, the wrapper may
be valuable. Preserve small adapters that enforce a real contract.

Graph topology, attention and message-passing improvements need
[mc-research](../mc-research/SKILL.md) and suitable experimental evidence.
Software modularity alone cannot establish model-quality or hardware claims.
Finish with the proposed interface, grounded callers, verification plan and
remaining decisions; complete shared-policy documentation updates.

Source: adapted from Matt Pocock's `codebase-design`, `DEEPENING` and
`DESIGN-IT-TWICE` references in the installed AskMatt 2.0.0 snapshot; project
ownership contracts and bounded implementation scope added.
