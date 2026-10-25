---
name: mc-handoff
description: Prepare a portable Music Critic V2 handoff when work moves to another task, checkout, environment or collaborator.
---

# Preserve the continuation boundary

Read [workflow conventions](../../../docs/agents/workflows.md). Capture the
destination's next action and the minimum evidence needed to perform it.
Save to the user-requested location; otherwise use a named OS-temporary
Markdown file and report that it is temporary.

Include the checkout/revision and dirty-state boundary, current phase,
authorized scope, completed versus pending work, exact commands already
validated, failures and blockers. For experiments include IDs/status, source
and data/config identities, evidence locations/availability, last verified
progress and its timestamp, budget and required admission gates. Preserve
unknowns and distinguish declared access policy from observed access.

Link existing specifications, decisions, diffs and records rather than copying
them. Use portable repository-relative paths plus repository identity; include
machine paths only as operational hints, never as the sole artifact identity.
Keep credentials and private payloads out. Name the applicable local skills.

State the first verification the next agent must perform where state may have
changed. A handoff carries existing authorization and constraints, not invented
approvals or evidence. Writing it does not send it to anyone or create a new
app task unless requested.

Adapted from Matt Pocock's handoff and Ask Matt phase-boundary guidance,
AskMatt 2.0.0. See [provenance and license](../../../docs/agents/skill-provenance.md).
