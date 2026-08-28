# Refactor route mode

## Purpose

Long-running kernel campaigns can reach a local optimum where the next useful change is an
architectural refactor. Such a change may need several correctness-preserving checkpoints before it
recovers the incumbent's performance. The normal supervisor cannot carry that work across episodes
because every candidate must strictly improve on the current incumbent.

Refactor route mode adds a bounded, recoverable non-monotonic track without weakening the stable
promotion contract.

## Entry policy

The supervisor enters one route only when all conditions hold:

- at least `--refactor-after-episodes` effective episodes have completed;
- `--refactor-stall-threshold` consecutive effective episodes did not promote;
- no mandatory framework conversion is pending;
- the current global-best kernel has not already exhausted a route.

An effective episode is a normally completed `pivot`, or a protocol-valid `candidate_ready` that
reaches an authoritative PASS/FAIL measurement. Session failure, timeout, `blocked`, interrupted
work, invalid handoff, protected-path failure, policy failure, and verifier ERROR do not count. This
counter is independent of the legacy stall counter used by `--max-stall` and framework conversion.

Setting any refactor option to zero disables route entry.

## State model

The incumbent and route are separate tracks:

| State | Meaning |
| --- | --- |
| incumbent branch `HEAD` | Stable global-best kernel plus canonical `memory/vN.json` history |
| `route_base_commit` | Frozen global-best revision used by strict route promotion ABBA |
| `route_head_commit` | Latest correctness-passing checkpoint and next episode's base |
| `route_best_commit` | Highest absolute performance-score checkpoint observed on the route |
| `refs/atrex/refactor/<route-id>` | Durable Git reachability for the route head |
| `.atrex_long_horizon/routes/<route-id>/episodes/eNNNN.json` | Full immutable-by-episode route ledger |

`.atrex_long_horizon/state.json` is bounded hot state. It persists `mode`, route identity and
commits, start episode, used and remaining budget, the current plan/progress summary, the latest
eight route checkpoints, and the latest 16 compact attempt indexes. Full attempts remain under
`episodes/eNNNN/`, while each route episode also receives its own ledger file. `active_episode.json`
separately records both the route parent (`base_commit`) and the stable incumbent HEAD
(`incumbent_commit`), because they diverge after the first checkpoint.

## Episode and verification flow

1. Create the episode worktree from `route_head_commit` and inject the durable route plan/progress
   into the independent refactor prompt.
2. Require the ordinary candidate protocol, protected-path checks, production/dependency policy,
   full correctness, and complete measurement.
3. Run candidate-versus-route-parent ABBA with policy `refactor_checkpoint`. Complete valid ABBA
   runs pass even when the measured performance change is negative.
4. Atomically advance `refs/atrex/refactor/<route-id>` and `route_head_commit`; do not alter the
   incumbent kernel.
5. Compare the candidate's absolute evaluator score with the frozen global-best score. Only a
   candidate that can clear `--min-improvement-pct` runs a second, `strict_promotion` ABBA against
   `route_base_commit`.
6. Strict success squash-promotes the accumulated route kernel onto the current incumbent HEAD and
   returns to normal mode. Otherwise only canonical outcome memory is committed on the incumbent
   branch.

The route plan and progress are returned in every terminal journal outcome. The supervisor writes
the complete attempt and raw route update to the per-episode route ledger. Hot state keeps an
idempotent recent window with checkpoint acceptance, candidate-versus-parent change,
candidate-versus-global-best change, summary, and next directions. The next prompt receives only
the latest five checkpoint entries plus the total history count, avoiding linear context growth.

## Budget and exit

`--refactor-max-episodes` is independent of `--max-iters`; an active route may finish its dedicated
budget after the normal version cap is reached. Explicit token and per-invocation episode limits
still pause the campaign safely and preserve route state.

A route exits when:

- strict ABBA beats the global best (`surpassed_global_best`);
- its dedicated budget is exhausted (`budget_exhausted`); or
- a mandatory framework conversion preempts it (`preempted_by_framework_conversion`).

Budget exhaustion leaves the incumbent kernel unchanged and stops the campaign with the route ref,
plan, progress, attempts, measurements, and performance history available for inspection. The same
global-best kernel cannot automatically open another route; a later strict promotion changes the
kernel identity and makes a future route eligible again.

## Recovery invariants

- A route ref update is idempotent. If the supervisor stops after verification but before the
  canonical outcome commit, recovery revalidates the terminal handoff and converges on the same ref.
- Route hot history replaces an existing entry for the same episode, and the corresponding route
  episode ledger is atomically overwritten, preventing duplicate progress after replay.
- Main-branch recovery compares and resets against `incumbent_commit`, while candidate ancestry is
  checked against the independent route parent.
- A canonical commit completed just before interruption is detected from its parent and message;
  route budget is charged exactly once when the attempt is not already in state.
- The global-best kernel remains the final result for every failed, interrupted, preempted, or
  exhausted route.
