# Campaign setup (clean session, run once)

You are the **setup session** for a profile-driven GPU-kernel optimization campaign.
The orchestrator drives the optimization loop after you; your job is to produce the **V0 baseline** and stop.
This is an authorized, non-interactive job. **Never ask the user whether to continue and never stop for
confirmation.** Work autonomously until the required `memory/v0.json` and V0 Git commit both exist, or
report a concrete technical blocker after exhausting the available in-scope fixes.

The workspace already exists at your cwd (`{{WORKSPACE}}`) — it was created by the orchestrator
(`workspace_init.sh` already ran: directory structure, git, and `kernel.py` are in place).
**Do NOT re-run `workspace_init.sh`.**

Environment (resolve all paths against your cwd = the workspace):
- `tools/`, `reference/`, `skills/`, `reference-projects/`, and `gpu-wiki/` are symlinked into the workspace — read/use them by relative path
  (e.g. `python tools/memory_manager.py --workspace .`, `reference/v_iteration.schema.json`).
{{AGENT_RUNTIME}}

{{HARDWARE}}
{{SANDBOX}}
{{EVALUATOR}}

Do the following, in order, but only through baseline:

1. **Step 0 — Hardware specs + Roofline.** Source
   every hardware spec from `gpu-wiki/` (**no fabrication** — every spec value must cite a gpu-wiki path),
   do the Roofline analysis from the public workload contract, compute absolute targets
   (`hardware peak * 90%`), and write `Hardware Spec`, the Roofline analysis, and `Stop Conditions`
   into the workspace `README.md`. If `agent_problem.json` exists, never seek private per-case
   roofline data or exact evaluator shapes.
2. **Write `README.md`** — static config from the parameters below + Step 0 outputs (use `reference/README.md` as the template).
3. **Stage 1 — Baseline.** {{BASELINE_DRIVER}}: implement `kernel.py`, use the evaluator
   route declared above, validate correctness and baseline performance, write `baseline_report.md`, write
   `memory/v0.json` (via `tools/memory_manager.py`), and `git commit` ("V0: baseline kernel").
   If a subagent is used, include the mandatory sandbox block above verbatim in its task. It must run
   `python test_kernel.py --version v0 --no-memory` through `tools/sandbox.py --kind run`, parse the emitted
   `[test_kernel] RESULT_JSON=...`, and write `memory/v0.json` locally. Reject local-GPU measurement and remotely
   written memory. The test must cover the evaluator's complete workload—hidden cases for a generalized
   `agent_problem.json`, or every legacy `shapes.json` entry—and record the aggregate result and complete
   real `latency_us_by_shape` map. Generalized tasks expose only opaque shape ids and their measured
   latency; exact shape inputs and failure details remain private. Canonical V0 must also record
   `latency_us_geomean`, `latency_us_arith_mean`, `measurement_scope=real_evaluator_shapes`,
   `measurement_status=complete`, `measured_shape_count`, `shape_ids_are_opaque`,
   `speedup_vs_ref_geomean=1.0`, `correctness.status=PASS`, and `quality_gate.result=PASS` from the
   authoritative `RESULT_JSON`. Never commit a partial/null aggregate or a map with missing workloads.
   Do not edit an evaluator adapter supplied by the orchestrator. A derived legacy boundary may create
   its harness only before V0, then must commit and preserve it unchanged.

Then **STOP**. Do **NOT** enter Stage 2 / any optimization iteration — the orchestrator spawns those as
separate clean sessions. Exit once `memory/v0.json` exists and the baseline is committed.

## Parameters

- platform: `{{PLATFORM}}`
- framework: `{{FRAMEWORK}}`
- kernel_demo: `{{KERNEL_DEMO}}` (already copied to `kernel.py`)
- additional_notes: `{{NOTES}}`
