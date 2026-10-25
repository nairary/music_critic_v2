# Skill provenance and adaptation

Created 2026-09-21 at the user's request to add the recommended workflows to
this repository. The `mc-*` skills are project adaptations, not verbatim
upstream installations. They have no dependency on a personal plugin cache.

Source: [Matt Pocock's skills](https://github.com/mattpocock/skills), as supplied
in the locally installed `engineering-suite-ask-matt` version `2.0.0`.
The plugin manifest identifies the bundle publisher as Saul Martí.
The upstream revision is not recorded by that manifest and remains unknown;
the [source manifest](skill-source-manifest.json) records the actual file hashes
and sizes used for adaptation. The current GitHub branch is not treated as
identical to that local snapshot.

Matt Pocock's [MIT license](licenses/matt-pocock-MIT.txt) is retained.
`scientific-evidence-review` and `experiment-protocol` were newly authored from
this project's AGENTS and EXPERIMENT_LOGGING contracts. OpenAI skill-creator
guidance informed packaging and validation; it is not a runtime dependency.

## Deliberate adaptations

- Unique `mc-` names and normal discovery distinguish project versions from
  installed plugin skills. The local router refers only to included workflows.
- Code review records its actual base and covers requested staged, unstaged
  and untracked files without changing the Git index.
- Tests use already authorized interfaces and task requirements; routine work
  does not introduce a mandatory extra approval round.
- Implementation has no automatic commit, push, external issue publication,
  training launch or baseline promotion.
- Specs and tasks use existing workstreams and local files by default.
  Terminology/decisions use existing project contracts instead of creating a
  competing CONTEXT/ADR tree. Existing tracker choices take precedence.
- Research, code correctness, native admission and scientific outcomes remain
  separate. The two scientific skills reference the existing evidence protocol.
- The upstream HTML/UI `prototype` workflow is not installed as an ML
  experiment workflow. Existing UI tools remain available for UI-specific work.

## Maintenance

Edit these project versions deliberately. A plugin upgrade does not overwrite
them. When incorporating a later upstream change, review it against current
project contracts, preserve the license, and update the source manifest.
The source hashes identify authoring inputs, not an enforced runtime lock.

Validation should include frontmatter/name checks, all local references, and
realistic synthetic behavior cases for consequential workflow changes. Exact
validation performed for this addition is recorded in STATUS.
