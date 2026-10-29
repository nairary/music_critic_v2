# HIERGRAPH E1–E5: sequential80000 updates,2026-10-29

User authorizes sequential E1,E2,E3,E4,E5 on existing RTX3090. Seed17,
fresh initialization for each condition. C3 frozen/cache004, current67 targets,
42 H heads, masks, sampler, loss, optimizer and model dimensions remain the
prepared protocol. Source006 changes only update accounting and orchestration.
Main checkout and prior source/artifacts remain unchanged. No commit/push.

## Execution contract

- Exactly80000 **applied optimizer updates** per successful condition. Unknown
  draws remain exposure/skipped counters; no Adam step on unsupported targets.
- max160000 attempted draws protects an all-unknown/inadmissible stream.
- Selection after applied2k,5k,10k,20k,40k,60k,80k; earliest maximum inner-selection
  family-macro known-H recovery; one selected dev pass plus separate oracle-H.
- Same seed17 work→tile→view sequence and named common parameter initialization.
  Compare actual/applied/exposure hashes at matched checkpoints; selected dev
  comparisons may have different selected updates and remain descriptive.
- Worker cap21600s, outer timeout21620s with cleanup10s. Whole campaign cap111600s.
  Caps are safety bounds, not duration predictions. Existing estimated time6–12h.
- One active worker. Entire process group drained before next; inherited GPU
  flock survives supervisor death until worker exits. No implicit retry/resume.
- Worker failure is retained/sealed before the next condition; unlaunched plans
  remain planned on supervisor interruption. Terminal artifacts are immutable.

## Evidence and scope

Parent EXP-HIERGRAPH-80K-CAMPAIGN-001, conditions EXP-HIERGRAPH-80K-E{1..5}-S17-001.
Artifact locators in all these records are relative to remote campaign-80k001
**evidence root**: runs/E*-s17-001/result.json or progress.json for supervisor.
Retain real command/PID, exit code, config/source/data/cache digests, progress,
actual/attempted/skipped counts, checkpoint+optimizer/RNG, exposure and selection.
Source is sibling source006 of the campaign root. Prior planned2000-step pilots
are not executed or overwritten.

Weak source/model labels, oracle annotation interpretations, no human gold;
exploratory dev remains inside globalTRAIN, raw pretraining overlap unknown.
GlobalVALIDATION and TEST closed. First slice H/F only, no augmentation or extra
LLM calls/new paid resources. No quality claim follows from successful execution.

## Verification

Test-first applied-target and drawcap tests observed red, then green. Real
synthetic subprocess tests verify serial order, sealing failures before next,
timeout group cleanup, no implicit resume, GPU mutex and retained counts.
Independent Standards/Spec reviews found/fixed cap/identity/cleanup gaps.
A Linux-only real-data launch follows registered plans and source-byte preflight.
Monitoring: one compact hourly read, at most32 checks; no new runs/agents/corpus
reads. Notify only on failure, completion or required action.
