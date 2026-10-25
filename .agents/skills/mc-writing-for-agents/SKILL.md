---
name: mc-writing-for-agents
description: Write or revise Music Critic V2 skills and agent-facing instructions with precise triggers, concise guidance and authoritative project references.
---

# Write maintainable agent instructions

Read [workflow conventions](../../../docs/agents/workflows.md). Identify the
decision this instruction should improve and the agent behavior that would
count as success. Preserve the user's requested scope and invocation mode.

Keep always-loaded descriptions and AGENTS pointers short. A pointer should
state when to read the target, not simply list a file. Keep shared constraints
in their authoritative project documents and link to them. Put conditional
detail behind a relevant reference; avoid duplicating the experiment protocol
or creating a second architecture decision log.

Separate ordered actions from reference material. Give steps observable
completion criteria. Replace generic advice with concrete decision criteria;
remove instructions that do not change the default behavior. Keep hard
scientific safeguards when their omission could alter evidence or permissions.
Prefer capability-specific descriptions over automatic invocation on every task.

For skills, use a directory-matching lowercase hyphenated name and a precise
description in YAML frontmatter. Keep normal discovery unless the user requests
otherwise. Dependencies must resolve within the repository; a global plugin
cache path is provenance, not a portable runtime dependency. Add optional UI
metadata only when useful. Preserve attribution and licenses for adapted text.

Validate frontmatter, names, links and remaining scaffold placeholders using
the available skill validator. Inspect realistic routing and workflow cases;
for a complex scientific skill use a fresh agent with clearly synthetic
fixtures. Tests that assert wording alone cannot establish workflow quality.
Report what was tested and any unresolved behavior without claiming universal
agent reliability.

Adapted from Matt Pocock's writing-for-agents, AskMatt 2.0.0; Codex packaging
follows the current skill-creator guidance.
See [provenance and license](../../../docs/agents/skill-provenance.md).
