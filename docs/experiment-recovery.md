# Reliable production experiment recovery

Light keeps candidate outcomes, validation availability and command lifecycle separate.
A command exit code does not establish that an optimized kernel exists.

## Gateway evidence

Acceptance examines private physical-job receipts. A typed candidate compile or correctness
failure is returned to the Episode even if later ABBA measurements were skipped. Missing
shape/stage evidence is not a pass or a candidate rejection. Explicit device-unavailable
evidence remains infrastructure failure. Illegal memory access, device asserts and timeouts
alone do not prove infrastructure failure.

Known pending jobs are recovered by their original IDs. Terminal incomplete evidence gets
three attempts per stage, persisted across restarts. Infrastructure recovery has a 30-minute
window; exhaustion preserves the draft and blocks validation. Queued/running jobs are not
charged to that failure window. An operator must inspect the recorded cause before changing
the recovery policy or evidence; restarting alone does not replenish an exhausted budget.

## Review and coding budgets

Missing, malformed or schema-invalid reviewer output gets at most three invocations per
evidence digest. Valid allow/reject results may be reused. Invalid results are never cached
as candidate verdicts. Both dependency review and numerical-plan generation use durable
budgets. A valid numerical plan written before a CLI timeout remains usable.

Framework-baseline coding repair is separately limited to three invocations. A missing
review or unavailable measurement blocks validation without asking the coding Agent to
change an unjudged candidate. A restored candidate is validated before another coding turn;
restart without a candidate cannot bypass an exhausted repair budget. Code changes rerun the
applicable policy and numerical gates. Loader support is checked against the selected remote
executor's source policy as well as the production dependency policy.

The existing model-service recovery handles provider outages within one Episode. Terminal
report repair permits two additional attempts within that same draft, including a fresh
session for a backend without resume support. Exhaustion sets `paused_protocol`, preserves
the active Episode, and does not advance the version or completed-Episode count. All consumed
tokens remain accounted. Starting the campaign again does not clear a protocol pause.

Completed searches write `.atrex_long_horizon/outcome.json`. `stop_reason`,
`budget_exhausted`, `has_verified_framework_candidate` and `performance_improved` are separate
facts. Old campaigns without this evidence remain unverified rather than being inferred
successful from a scheduler's `Succeeded` state.

## Timeout configuration and small validation runs

Light Workflow configuration accepts `candidate_timeout_seconds` and
`performance_timeout_seconds`, both defaulting to 120 seconds. They apply to the trusted
evaluation executor and participate in measurement identity. Numerical probes use the
configured candidate timeout. Agent environment overrides cannot replace this policy.

Supply these fields in the selected Workflow configuration with the existing launch
`--patch` mechanism, before the startup argument separator `--`. For a workload that
needs a longer performance measurement, use `performance_timeout_seconds: 480`. Preserve
the other selected Workflow configuration fields: a composition patch replaces the whole
module configuration. Configuration changes require a new workspace or explicit migration;
they must not silently change a running campaign's measurement identity.

ABBA retains its explicit `--verify-run-timeout` / `--comparison-run-timeout` setting. Size the
allocation to satisfy `2 * verify_repeats * verify_run_timeout + 30 <= sandbox_timeout`.
For example, increasing an ordinary evaluation's performance timeout to 480 seconds does
not silently increase ABBA's per-run budget. Set both deliberately when that workload needs
them. Do not change global defaults just to accommodate one operator.

`episode_limit: 1` in Workflow configuration stops a validation run after one completed
Episode while retaining the campaign's original `max-iters` ceiling. The default, zero,
imposes no additional per-invocation Episode limit. An `episode-limit` stop reports
`search_finished: false`; it is not budget exhaustion or proof of performance improvement.

## Explicit runtime migration

Use a new workspace for changed benchmark contracts or iteration semantics. For an otherwise
identical, stopped campaign, capture a complete checkpoint context using the target runtime
and the original launch arguments. Preview its changes with:

```bash
python -m aka.light.workflow.runtime_migration /absolute/workspace \
  --context target-checkpoint-context.json --plugin-dir /target/aka/light/plugins
```

The tool acquires the campaign lease, validates the frozen problem identity and target
selection, and requires every launch argument to match. Empty plugin locks may relocate;
nonempty plugin locks must be identical. It prints the old/new hashes and `plan_digest`.
Apply that exact plan by adding `--apply-plan DIGEST`. Inputs changing after preview abort
the operation. Context and plugin-lock originals are retained under the private campaign
`migrations` directory. No candidate, counters, snapshots or episode state are rewritten.

To roll back an applied migration before the context changes again:

```bash
python -m aka.light.workflow.runtime_migration /absolute/workspace --rollback BACKUP_DIRECTORY
```

## Health inspection

```bash
python -m aka.light.workflow.health /absolute/workspace --command-phase Running
```

The report includes phase, completed Episodes, accepted candidates, token usage, last
recorded progress, model-service retries, validation waits, pending physical-job IDs and
the final outcome. `queued`, `waiting_gpu`, `waiting_model_service`, `validation_blocked`,
`paused_protocol`, `budget_exhausted` and `unverified` are distinct. `in_progress_unverified`
requires comparison with a later snapshot; it is not a claim of useful GPU progress.

Sandbox wrappers should pass options before the target, inspect the returned argv, update
existing files with `file write --overwrite`, and verify content after upload. Cancellation
is asynchronous: confirm no experiment commands remain Running/Starting/Queued and inspect
the last dispatched Gateway jobs. Never restart a historical experiment automatically merely
because its wrapper exited without a verified candidate.

## Regression checks

```bash
PYTHONPATH=src python3 tools/test_model_service_recovery.py
PYTHONPATH=src python3 tools/test_experiment_recovery.py
```

The benchmark dataset fixes belong in atrex-bench: explicit supported dtype lookup for
`gdn_qk_l2_norm`, and `mutates_inputs: [initial_state]` for `causal_conv1d`. The generator must
preserve the mutation contract on rebuild. V0 validation must cover every workload and
compare declared post-call state, not merely suppress unexpected-mutation diagnostics.
