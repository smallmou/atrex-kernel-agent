# Economy Mode: Wiki prototype bring-up

Adapt the most applicable GPU Wiki prototype into `kernel.py` for operator `{{OPERATOR}}`,
platform `{{PLATFORM}}`, runtime architecture `{{ARCH}}`, in workspace `{{WORKSPACE}}`.
Produce v{{N}} with correct math and evaluator-facing signatures. Select the framework from
the prototype and available tooling. A correct implementation may be slower than V0; performance
optimization starts in later sessions. Work autonomously and keep explanations short.

{{SANDBOX}}

{{EVALUATOR}}

Read only `reference.py`, `input.py` when present, the public operator contract
(`agent_problem.json`, or `definition.json`/`workload.jsonl`/`shapes.json`), `kernel.py`,
`solution.json` when present, and the V0 aggregate result in `memory/v0.json`.
Preserve the operator's math, dtypes, layouts, outputs, tolerance and complete supported domain.

## Retrieve and adapt

{{PLUGINS}}

Query once before the first implementation edit. Compare at most three returned records for
operator semantics, hardware compatibility, precision, layout and installed dependencies; choose
the closest compatible prototype containing code or a concrete implementation recipe. Prefer a
direct operator match over a component match, and supported hardware over a faster incompatible
example. A labelled random sample is not a matching prototype. Inspect only the selected payload
and at most one directly referenced local source file; do not browse sibling records or projects.
Do not search the repository or reference projects for additional implementations. A missing
referenced source file is unavailable: adapt the returned recipe directly, or use the one
focused follow-up below. Never execute retrieved source on the host.

Adapt its code into `kernel.py`; preserve reusable structure and change only the entry point,
shape handling, layout or precision required by this contract. Keep CUDA source and its loader
inside `kernel.py`. Update `solution.json` to describe the actual framework, dependencies, source
and entry point. Use preinstalled tooling only. If the prototype is incompatible or lacks an
essential fact, make at most one focused follow-up query with the same result limits. If neither
query yields usable code or a recipe, report the precise coverage/tooling blocker and stop.
Do not expand into a from-scratch research campaign.

Save `plans/v1_economy_prototype.md` (at most 150 words): emitted `query_id`, selected record's
emitted canonical `wiki_id`, actual framework, compatibility assumptions, adaptations and rejected
alternatives. This is the reusable context for later episodes. Save query output under
`.atrex_long_horizon/`; on repair or restart, reuse it and the summary instead of repeating queries.

## Validate correctness and finish

After the final edit, run the bounded smoke command:

```bash
{{SMOKE_COMMAND}}
```

{{SMOKE_SCOPE}}

Repair only observed compile or accuracy defects, then rerun this same smoke command. Do not
optimize latency, profile, create plans, launch subagents or external reviewers. The supervisor
checks the complete workload and five additional random seeds, writes canonical memory, commits
the candidate and pins V1. No speedup is required at V1. Leave the candidate and short prototype
summary on disk, print `v{{N}}: Wiki prototype smoke-passed`, and stop.

Never edit evaluator/ground-truth files, `CLAUDE.md`, `README.md`, canonical memory or the baseline
marker; never commit in this session. Never run GPU/JIT code on the host, install dependencies or
change the sandbox service/jobs. Never cache input values, pointers or computation outputs.
Additional constraints: {{NOTES}}
