---
name: mc-diagnosing-bugs
description: Diagnose reproducible Music Critic V2 bugs, resume failures and performance regressions, then fix them with a regression test. Use when an observed failure needs causal localization rather than a model-quality hypothesis.
---

# Music Critic diagnosis

Follow the [project workflow policy](../../../docs/agents/workflows.md).
Record the reported symptom, affected revision/environment and expected
behavior from the relevant contract. Existing task authorization determines
whether the result is diagnosis only or diagnosis plus repair.

## Build the feedback loop

Find one command that exercises the actual failure: a focused test, fixture
CLI invocation, bounded replay or small harness. Inspect code as needed to
construct it. Prefer synthetic fixtures and existing retained evidence.
Run it and capture the exact failing assertion or measurable symptom; merely
finishing without a crash is insufficient.

Minimize while preserving the failure, pin relevant RNG/time/configuration and
keep the loop cheap. For intermittent failures, record attempts and failure
rate instead of pretending determinism. For performance, measure a stated
workload and preserve semantic-equivalence checks alongside timing.

If the available environment cannot reproduce the symptom, report attempted
checks and the missing observation. Continue useful static localization, label
hypotheses as unverified, and request only the missing access or evidence needed
for the next dependent step. Do not certify a fix from static inspection alone.

## Localize and repair

1. List plausible causes with falsifiable predictions; test the most informative
   prediction first. Change one relevant factor at a time.
2. Use narrow instrumentation and redact secrets in commands, logs and reports.
   Record unique markers so temporary instrumentation can be removed reliably.
3. Turn the reproducer into a regression test at the interface where the full
   failure is observable. Use [mc-tdd](../mc-tdd/SKILL.md) for a repair.
   A test that mocks away the failing interaction is not a regression test.
4. If no testable interface reaches the failure, document the missing seam and
   consult [mc-codebase-design](../mc-codebase-design/SKILL.md). Keep any
   resulting interface change within the requested scope and test it.
5. Re-run the regression and original scenario after the repair; run affected
   contract tests. Remove temporary instrumentation and retain evidence needed
   to explain the diagnosis in the appropriate ignored evidence location.

Real-data evaluation, costly/native hardware runs and decision-changing
diagnostics follow [experiment-protocol](../experiment-protocol/SKILL.md).
Synthetic reproduction proves mechanics; it does not establish scientific
quality, GPU equivalence or native resume safety.

Finish with the cause supported by observations, exact commands/results,
remaining uncertainty and shared-policy documentation updates. Do not stage,
commit or launch a wider experiment as an automatic cleanup step.

Source: adapted from Matt Pocock's `diagnosing-bugs` skill in the installed
AskMatt 2.0.0 snapshot; project test, evidence and execution boundaries added.
