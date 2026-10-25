---
name: mc-code-review
description: Review Music Critic V2 changes against project standards and the task specification as separate axes. Includes committed changes and requested staged, unstaged and untracked work with a recorded comparison base.
---

# Music Critic code review

Follow the [project workflow policy](../../../docs/agents/workflows.md).
Review code and report findings; this skill does not itself authorize repairs,
staging, commits or experiment execution.

## Fix the review scope

1. Record repository root, branch, HEAD, `git status --short` and the requested
   paths. Resolve any user-supplied ref. Honor an exact commit comparison; use a
   merge-base only for a branch comparison and record the resolved commit.
   For current uncommitted work without a supplied base, use HEAD and say so.
   Ask only if multiple plausible bases materially change the requested review.
2. Capture the committed diff when applicable. For WIP, also inspect
   `git diff --cached`, `git diff`, and the final tracked content relative to the
   resolved base. These represent distinct index/worktree states; describe
   whether a finding affects the staged version, current files or both.
3. Inventory untracked files with `git ls-files --others --exclude-standard`.
   Read in-scope source, tests and documentation directly: ordinary `git diff`
   omits them. Keep generated data/checkpoints outside the review unless their
   inspection is separately in scope. State exclusions and unreadable files.
   Do not use `git add -N` or otherwise change the index to obtain a diff.
4. Record the file inventory and relevant content identities, including
   untracked files, so concurrent edits cannot silently change the review.
   If reviewed files change during the review, inspect that delta before
   declaring coverage. An empty committed diff is not an empty WIP review.

## Review two independent axes

Use independent subagents when available; give both the same recorded scope,
revision and file snapshots. Otherwise perform two explicitly separate passes.

- **Standards:** check [AGENTS.md](../../../AGENTS.md), relevant contracts,
  provenance, mask handling, exact timing, raw-only inference, schema/resume
  bindings and required tests. Cite the rule for each violation. Treat naming,
  coupling, duplication and speculative abstraction as justified design
  judgments, not automatic violations; repository requirements prevail.
- **Spec:** use the current user request, approved task/protocol and acceptance
  criteria. Identify missing, incorrect or out-of-scope behavior and cite the
  requirement. Use available local sources without requiring an external issue
  tracker. If no spec can be recovered, report that this axis is limited;
  do not invent requirements.

Trace consequential findings through actual callers and tests. Report concrete
failure conditions with tight file/line references and distinguish observed
failures from plausible risks. Scientific conclusions require
[scientific-evidence-review](../scientific-evidence-review/SKILL.md);
passing code review does not validate model superiority or experiment parity.

## Deliver

Report Standards and Spec separately, including priority, evidence and practical
impact for each actionable finding. State coverage, test commands/results,
unreviewed surfaces and uncertainty; say explicitly if an axis has no findings.
Preserve the two verdicts instead of letting one mask the other. Complete the
shared-policy documentation updates while respecting any read-only scope.

Source: adapted from Matt Pocock's `code-review` skill in the installed
AskMatt 2.0.0 snapshot; WIP coverage and project contracts added, tracker and
automatic mutation assumptions removed.
