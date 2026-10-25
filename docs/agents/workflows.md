# Repository skills

The skills in `.agents/skills/` are portable project adaptations of selected
AskMatt workflows plus two scientific workflows. They work without the global
plugin. `mc-` distinguishes these versions from the plugin's originals.
Normal automatic discovery is retained; an explicit `$skill-name` selects one.
Use [mc-ask-matt](../../.agents/skills/mc-ask-matt/SKILL.md) when choosing a route.
Codex discovers repository skills in this location according to its
[official documentation](https://learn.chatgpt.com/docs/build-skills).

## Shared project conventions

[AGENTS.md](../../AGENTS.md) and its authoritative documents remain the project
rules. Read the relevant current contracts, caller paths and task scope; a
skill is a procedure for applying them, not an additional source of permission.
Reuse existing user authorization and accepted decisions. Ask only for missing
information or authorization that materially blocks the requested action.

Preserve explicit read-only scope. Otherwise follow AGENTS' pre-edit and
completion protocol, update [STATUS](../STATUS.md), and change
[DECISIONS](../DECISIONS.md) only when an architectural/scientific decision
actually changes. Preserve unrelated working-tree changes. VCS publication,
external messages, new dependencies and experiment budgets follow the user's
request rather than an automatic skill step.

Keep engineering correctness, native runtime admission, theory accuracy,
human preference and renderer-dependent audio aesthetics as separate claims.
For substantial experiments, imports and decision-changing diagnostics use
[EXPERIMENT_LOGGING](../EXPERIMENT_LOGGING.md); ordinary synthetic workflow
tests do not become scientific evidence. Only that document owns the experiment
record lifecycle, full-artifact storage, import safety and claim discipline.

## Where work and decisions live

Use the user's supplied spec or tracker when one exists. The local default
needs no account or external publication: reuse the relevant
`docs/workstreams/<date>/` directory. New multi-session efforts can have
`<feature>/spec.md`, `<feature>/issues/NN-slug.md` and a small index there.
Each issue records status and blocking file links. This is work planning;
`docs/experiments/records/` remains the separate experiment registry.

Keep terminology in [DATA_CONTRACT](../DATA_CONTRACT.md) and relevant domain
documents, module structure in [ARCHITECTURE](../ARCHITECTURE.md), and accepted
decisions in [DECISIONS](../DECISIONS.md). Research goes in `docs/research/`;
experimental reports follow `docs/experiments/`. Add links to existing sources
instead of creating a competing CONTEXT.md or ADR hierarchy.

## Available workflows

| Skill | Use |
|---|---|
| [mc-research](../../.agents/skills/mc-research/SKILL.md) | Code and primary-source investigation |
| [mc-diagnosing-bugs](../../.agents/skills/mc-diagnosing-bugs/SKILL.md) | Reproduce, localize and fix a bug or measured performance regression |
| [mc-code-review](../../.agents/skills/mc-code-review/SKILL.md) | Independent Standards/Spec review, including requested WIP |
| [mc-tdd](../../.agents/skills/mc-tdd/SKILL.md) | Implement behavior through meaningful red/green checks |
| [mc-domain-modeling](../../.agents/skills/mc-domain-modeling/SKILL.md) | Clarify terms and domain contracts |
| [mc-codebase-design](../../.agents/skills/mc-codebase-design/SKILL.md) | Design module boundaries and interfaces |
| [mc-grill-with-docs](../../.agents/skills/mc-grill-with-docs/SKILL.md) | Resolve genuinely open design decisions |
| [mc-to-spec](../../.agents/skills/mc-to-spec/SKILL.md) | Synthesize agreed work into an implementation spec |
| [mc-to-tickets](../../.agents/skills/mc-to-tickets/SKILL.md) | Split the spec into independently verifiable tasks |
| [mc-implement](../../.agents/skills/mc-implement/SKILL.md) | Deliver a selected task with tests and review |
| [mc-handoff](../../.agents/skills/mc-handoff/SKILL.md) | Carry evidence and scope to another task or environment |
| [mc-writing-for-agents](../../.agents/skills/mc-writing-for-agents/SKILL.md) | Maintain skills and agent-facing instructions |
| [mc-wayfinder](../../.agents/skills/mc-wayfinder/SKILL.md) | Resolve a large map of uncertain decisions before implementation |
| [scientific-evidence-review](../../.agents/skills/scientific-evidence-review/SKILL.md) | Audit experimental claims, provenance and comparability |
| [experiment-protocol](../../.agents/skills/experiment-protocol/SKILL.md) | Plan and complete authorized experiments/imports through the registry |

The project variants incorporate setup locally. The upstream setup skill is
not required to use these versions; original plugin workflows retain their own
setup assumptions. [Source and adaptations](skill-provenance.md).
