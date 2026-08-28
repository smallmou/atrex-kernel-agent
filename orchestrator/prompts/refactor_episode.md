# Refactor-route kernel optimization episode {{EPISODE}}

Continue one persistent architectural refactor route in this isolated Git worktree. This mode exists
because strict monotonic optimization stalled: a correct intermediate checkpoint may be slower than
its parent while it establishes a new dataflow, parallelization, pipeline, or resource-planning
foundation. Complete one coherent route milestone, validate it, and leave the next episode a usable
checkpoint and explicit continuation plan.

The supervisor owns the stable global-best branch, authoritative ABBA verification, canonical
memory, the persistent route ref, and final squash promotion. You own only this episode branch and
its structured route evidence.

## Context

- Workspace: `{{WORKSPACE}}`
- Canonical version produced by the supervisor: `v{{VERSION}}`
- Platform: `{{PLATFORM}}`
- Framework: `{{FRAMEWORK}}`
- Route ID: `{{ROUTE_ID}}`
- Stable global-best commit: `{{ROUTE_BASE_COMMIT}}`
- Current route-parent commit: `{{BASE_COMMIT}}`
- Best measured route checkpoint: `{{ROUTE_BEST_COMMIT}}`
- Route episodes already used: `{{ROUTE_USED_EPISODES}}`
- Route episodes remaining, including this episode: `{{ROUTE_REMAINING_EPISODES}}`
- Episode branch: `{{EPISODE_BRANCH}}`
- Journal: `{{JOURNAL_PATH}}`
- Handoff: `{{HANDOFF_PATH}}`
- Additional constraints: {{NOTES}}
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are linked into the worktree.
{{AGENT_RUNTIME}}

{{RESUME_DIRECTIVE}}

Never switch branches, push, merge, rebase, or alter refs. Private checkpoint commits on the episode
branch are allowed, but every commit must contain only `kernel.py`. Plans, profiles, discussion
transcripts, journals, and handoffs are ignored episode evidence: write them normally but never add
them to Git. Never edit evaluator or ground-truth files, including `test_kernel.py`,
`profile_driver.py`, `definition.json`, `reference.py`, `workload.jsonl`, `input.py`, `shapes.json`,
`agent_problem.json`, `metadata.json`, `roofline.json`, `CLAUDE.md`, or `README.md`. Do not write
canonical `memory/vN.json`; the supervisor creates it after terminal validation.

{{MODE_POLICY}}

{{EVALUATOR}}

{{HARDWARE}}

{{SANDBOX}}

## Non-negotiable execution boundary

- Never run `python test_kernel.py`, `python kernel.py`, or import GPU/JIT kernel packages directly
  on the host. Route every compile, correctness, benchmark, and profiling command through
  `python tools/sandbox.py ... --`.
- Never mutate the shared gateway service, its state, database, logs, or jobs.
- Never install or build dependencies. Use only the immutable campaign environment.
- Static source inspection is allowed. Imports or probes that may initialize GPU/JIT code must run
  through the sandbox.

## Persistent route ledger

The supervisor restored the authoritative route plan and bounded hot progress below. They are
binding context, not suggestions. `history_count` is the total number of recorded route episodes;
`recent_history` contains only the latest checkpoints. Full older evidence remains in the
supervisor-owned per-episode route ledger, so do not attempt to reproduce it in this outcome.

### Route plan

```json
{{ROUTE_PLAN}}
```

### Route progress

```json
{{ROUTE_PROGRESS}}
```

If the plan still has no concrete milestones, first produce a reviewed architectural plan with:

1. the measured bottleneck and why local edits are exhausted;
2. the target architecture and correctness invariants;
3. ordered milestones that can each end in a correctness-passing checkpoint;
4. expected temporary regressions and the later performance-recovery mechanism;
5. rollback and route-exit criteria.

Use the backend-native plan generator and enabled reviewers before committing to that plan. In later
episodes, update the existing plan only when new evidence invalidates it; do not freely switch to an
unrelated optimization direction. Preserve completed milestones and name one concrete next task.

## Engineering loop

`skills/gpu-kernel-episode-loop/SKILL.md` defines the binding evidence loop. Read it now and execute
its profile/research/plan/implement/validate/record cycle. Bind its placeholders as follows:

| Skill placeholder | This episode |
| --- | --- |
| `<PROFILE_DIR>` | `profiles/episode_{{EPISODE}}` |
| `<PLAN_DRAFT>` | `plans/v{{VERSION}}_refactor_draft.md` |
| `<PLAN_FILE>` | `plans/v{{VERSION}}_refactor_plan.md` |
| `<JOURNAL_CLI>` | `{{JOURNAL_COMMAND}}` |
| `<JOURNAL_PATH>` | `{{JOURNAL_PATH_SHELL}}` |

`<PLAN_GENERATOR>` is the backend-native plan generator for this session:

{{PLAN_GENERATOR}}

Correctness, production policy, dependency policy, protected paths, and complete performance
measurement remain hard gates. Performance monotonicity does not: a checkpoint may be slower than
the route parent or stable global best if it completes a justified milestone and gives the next
episode a sound continuation point. Do not claim improvement when the measurement regressed.

## Terminal contract

Reach exactly one evidence-backed terminal state:

1. `candidate_ready`: the milestone checkpoint is committed, the worktree `kernel.py` matches that
   exact commit, protected files are unchanged, full development correctness passes, and a complete
   performance measurement exists. It need not outperform the parent.
2. `pivot`: the current milestone cannot produce a sound checkpoint and the persisted route plan
   must be revised using the recorded evidence.
3. `blocked`: infrastructure or missing authority prevents meaningful progress.

For `candidate_ready`, append the final evidence, commit only `kernel.py`, then finalize the journal.
The complete diff from the route parent must name exactly `kernel.py`:

```bash
git add -- kernel.py
git commit -m "v{{VERSION}}: refactor route checkpoint"
candidate_commit=$(git rev-parse HEAD)
{{JOURNAL_COMMAND}} finalize --path {{JOURNAL_PATH_SHELL}} --state candidate_ready \
  --candidate-commit "$candidate_commit" \
  --outcome-json '{"summary":"...","next_directions":["..."],"route_plan":{"objective":"...","milestones":[],"invariants":[],"next_task":"..."},"route_progress":{"completed_work":[],"current_milestone":"...","next_task":"..."}}'
```

For `pivot` or `blocked`, omit `--candidate-commit` but still persist the updated `route_plan` and
`route_progress` in the outcome. Every terminal journal requires structured experiment evidence and
a non-empty summary.

Only after finalizing, atomically publish the handoff by writing complete JSON to
`{{HANDOFF_PATH}}.tmp` and renaming it to `{{HANDOFF_PATH}}`:

```json
{
  "status": "candidate_ready | pivot | blocked",
  "candidate_commit": "required only for candidate_ready",
  "last_trial_commit": "optional checkpoint for pivot or blocked"
}
```

Chat text is not a handoff. A missing or invalid handoff causes bounded same-session recovery. The
supervisor will advance the route head after checkpoint gates pass, keep the global best untouched
while the route is slower, and run strict global-best ABBA before any formal promotion.
