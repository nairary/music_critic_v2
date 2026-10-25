---
name: mc-tdd
description: Implement or repair Music Critic V2 behavior test-first through stable interfaces. Use for red-green work on contracts such as masks, timing, batching, resume or MIDI roundtrips; synthetic tests establish mechanics rather than model quality.
---

# Music Critic test-first implementation

Follow the [project workflow policy](../../../docs/agents/workflows.md).
Derive behavior and test interfaces from the current task, approved spec,
existing callers and relevant contracts. State the chosen interface and tests;
existing task authorization is enough to proceed. Ask about a seam only when
its choice changes unresolved product behavior or the requested scope.

## One behavior per cycle

1. Write a small test at the interface that callers use. Run it and verify it
   fails for the missing behavior, not because a dependency or fixture is broken.
2. Implement only enough behavior to satisfy that case. Run the test again.
3. Add the next independent behavior or contract case and repeat. Avoid a large
   batch of speculative tests written before any implementation feedback.
4. After the behavior is covered, make justified cleanup under passing tests
   and run the affected suite. Review substantive changes with
   [mc-code-review](../mc-code-review/SKILL.md) when requested or useful.

## Tests worth keeping

- Assertions come from the contract, independently worked exact examples or
  invariant relationships; do not duplicate the implementation to calculate
  its expected output.
- Prefer real in-process collaborators. Substitute external I/O, time or
  randomness when isolation needs it. Preserve the actual interaction that
  caused a regression rather than mocking it away.
- Use tiny synthetic fixtures for unavailable labels, rational timing,
  overlapping notes, duplicate batch occurrences or continuation boundaries
  when those cases matter to the change. Mark them synthetic.
- Test masks and provenance as observable output contracts, including ambiguity
  and missing data. Missing targets remain unavailable rather than negative.
- Preserve meaningful existing coverage. A refactor may move tests to a better
  interface, but remove old tests only after their required behavior is covered.
- Avoid silent dependencies. Inspect the repository's existing commands and
  environment rather than copying a stale universal test command into a plan.

If the appropriate interface is unclear, consult
[mc-codebase-design](../mc-codebase-design/SKILL.md). A synthetic CPU test is not
evidence of native hardware admission, real-data quality or compute-matched
improvement. Runs beyond ordinary tests follow
[experiment-protocol](../experiment-protocol/SKILL.md).

Report the observed red and green results, affected-suite result and any
untested boundary; complete shared-policy documentation updates. Automatic
commits and training launches are outside this workflow.

Source: adapted from Matt Pocock's `tdd` skill and its testing/mocking
references in the installed AskMatt 2.0.0 snapshot; authorization and scientific
claim boundaries adapted to this project.
