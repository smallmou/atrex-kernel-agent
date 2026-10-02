# Economy Mode: one small optimization, episode {{EPISODE}}

Improve the accepted Wiki-derived kernel with one focused change and one official evaluator run.
Keep prose and context short. The supervisor owns acceptance, canonical memory and promotion.

- Workspace: `{{WORKSPACE}}`; operator: `{{OPERATOR}}`; platform: `{{PLATFORM}}`
- Version: `v{{VERSION}}`; incumbent: `{{BASE_COMMIT}}`; branch: `{{EPISODE_BRANCH}}`
- Journal: `{{JOURNAL_PATH}}`; handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}

{{RESUME_DIRECTIVE}}

{{MODE_POLICY}}

{{SANDBOX}}

{{EVALUATOR}}

## Reuse evidence and implement

Read `plans/v1_economy_prototype.md`, the current `kernel.py`/`solution.json`, only the needed
operator contract sections, and the latest two canonical memories. Reuse the selected prototype's
framework and algorithm. Pick one small change supported by the saved Wiki recipe or measured
trial results; do not repeat a rejected change. Explain the hypothesis in one sentence in the
journal, without generating a separate plan.

{{PLUGINS}}

Do not profile, invoke `gen-plan`, launch subagents/external reviewers, scan the knowledge base or
projects, or switch frameworks speculatively. Run GPU/JIT code only through the evaluator below;
never install dependencies or mutate the sandbox service/jobs. Preserve evaluator inputs and
tolerances. Commits may contain only `kernel.py` and its matching `solution.json`. Never edit
protected ground truth, `CLAUDE.md`, `README.md`, canonical memory or the saved prototype summary.
Never switch branches, push, merge, rebase or alter refs.

Use `python3 tools/iteration_trace.py phase-start implementation` and `phase-end implementation`
around the edit. Update the manifest if the entry point, languages or dependencies change. Commit
the candidate (`git add -- kernel.py`, also `solution.json` when present, then `git commit`).

## Evaluate once and record

Wrap the single full-workload base-seed evaluator in `benchmark` phase-start/phase-end markers:

```bash
{{FAST_EVALUATOR_COMMAND}}
```

A compile/accuracy failure consumes this episode. Record the defect and let the next episode
repair it; do not rerun the evaluator or add ABBA, multi-seed tests or separate benchmarks here.
Compare the passing `performance_score` with the canonical incumbent; higher is better. A
candidate must pass every workload and strictly improve the score to be proposed for promotion.
The supervisor matches the selected kernel/manifest bytes to the recorded evaluator result.

Wrap recording in `recording` phase-start/phase-end markers. Append one concise experiment:

```bash
{{JOURNAL_COMMAND}} append --path {{JOURNAL_PATH_SHELL}} \
  --experiment-json '{"name":"economy candidate","hypothesis":"...","change":"...","evidence":"official evaluator result","result":"...","evaluation":{"correctness":"pass|fail|unknown","performance":"improved|not_improved|unknown","latency_us":null,"kernel_hash":"<emitted hash>"},"decision":"keep_as_best|reject_and_continue|blocked","wiki_usage_status":"not_queried"}'
```

If Wiki knowledge informed the change, replace the attribution fields with `wiki_usage_status`
`declared`, `wiki_query_ids` and `wiki_usage` rows containing the actual emitted `query_id` and
`wiki_id`, disposition (`applied`, `partially_applied`, `reference_only`, `rejected`), short `use`
and measured `evidence`. Reusing the saved prototype summary counts as reconsidering its query.
Use `no_material_use` with query IDs only when a query was considered without material use.
Use `not_queried` with no IDs/usage only when no new or saved query informed the change.

Finalize as `candidate_ready` only for a committed, passing strict improvement; otherwise use
`pivot`, or `blocked` for infrastructure/tooling failure. Every state requires an experiment and
a non-empty summary. For a candidate:

```bash
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":[],"selected_experiment_index":1}'
```

For `pivot`/`blocked`, omit the candidate commit and selected experiment index. Only after
finalizing, atomically write `{{HANDOFF_PATH}}.tmp` and rename it to `{{HANDOFF_PATH}}`:

```json
{"status":"candidate_ready|pivot|blocked","candidate_commit":"required only for candidate_ready"}
```

Leave the candidate bytes matching the commit for `candidate_ready`. A recorded failure or
non-improvement is a valid `pivot`. Publish the handoff and stop; do not start another trial.
