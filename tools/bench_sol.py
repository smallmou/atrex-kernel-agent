#!/usr/bin/env python3
"""Evaluate a SOL workload sweep in several gateway jobs and merge the result.

The gateway caps one `agate dev` job at 600s of execution. A few references (paged
attention, MLA, MoE) cannot sweep every workload inside that budget even once, so their
baseline could never be established by a single `test_kernel.py` run. This driver splits
`workload.jsonl` into chunks, evaluates each chunk in its own sandbox job, syncs the
per-chunk traces back, and summarizes all of them together — so the merged numbers are
computed by exactly the same code as an unchunked run.

A chunk that still exceeds the cap is halved and retried, down to a single workload, so a
sweep does not need the right chunk size guessed up front. Only a workload that fails on
its own is a real failure.

Prints one `[test_kernel] RESULT_JSON={...}` line for the whole sweep, and suppresses the
per-chunk ones, so a caller parsing that marker can never mistake a partial chunk for the
complete result.

Usage (from the workspace):
  python tools/bench_sol.py --version v3
  python tools/bench_sol.py --version v3 --chunk 8
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path

CHUNK_MARKER = ".sol_chunk"
CHUNK_DIR = ".bench_chunks"
RESULT_MARKER = "[test_kernel] RESULT_JSON="


def load_harness(workspace: Path):
    """Import the workspace's own test_kernel.py, so merging uses its summarizer."""
    path = workspace / "test_kernel.py"
    if not path.is_file():
        raise SystemExit(f"[bench_sol] no test_kernel.py in {workspace}")
    spec = importlib.util.spec_from_file_location("_sol_harness", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[bench_sol] cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def chunk_size(workspace: Path, override: int) -> int:
    if override > 0:
        return override
    marker = workspace / CHUNK_MARKER
    if marker.is_file():
        try:
            return max(1, int(marker.read_text(encoding="utf-8").strip()))
        except ValueError:
            pass
    return 0  # 0 = one job for the whole sweep


def eval_subset(workspace: Path, harness, lines: list[str], tag: str, seed: int | None = None) -> list[dict] | None:
    """Evaluate one subset in a single sandbox job. None means the job produced no traces."""
    (workspace / CHUNK_DIR / f"wl_{tag}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = f"{CHUNK_DIR}/traces_{tag}.jsonl"
    cmd = [sys.executable, "tools/sandbox.py", "--sync", out, "--",
           "python", "test_kernel.py", "--no-memory",
           "--workload", f"{CHUNK_DIR}/wl_{tag}.jsonl", "--traces-out", out]
    if seed is not None:
        cmd += ["--seed", str(seed)]
    result = subprocess.run(cmd, cwd=str(workspace), capture_output=True, text=True)
    # A chunk's own RESULT_JSON covers only part of the sweep. Drop it so the caller's
    # "last RESULT_JSON wins" parsing cannot read a partial chunk as the whole result.
    for line in result.stdout.splitlines():
        if not line.startswith(RESULT_MARKER):
            print(line, flush=True)
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n",
              file=sys.stderr, flush=True)
    produced = workspace / out
    if result.returncode != 0 or not produced.is_file():
        return None
    return harness._parse_traces(produced)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Chunked SOL-ExecBench evaluation driver.")
    ap.add_argument("--workspace", default=".", help="Workspace dir (default: cwd).")
    ap.add_argument("--version", default=None, help="memory version to record, e.g. v3.")
    ap.add_argument("--chunk", type=int, default=0,
                    help=f"Workloads per gateway job (default: read {CHUNK_MARKER}, else one job).")
    ap.add_argument("--no-memory", action="store_true",
                    help="Print only; do not write memory/ (the orchestrator records it instead).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Forward --seed to each chunk's test_kernel.py run (multi-seed robustness).")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    harness = load_harness(workspace)
    workloads = workspace / "workload.jsonl"
    if not workloads.is_file():
        raise SystemExit(f"[bench_sol] no workload.jsonl in {workspace}")
    lines = [ln for ln in workloads.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise SystemExit("[bench_sol] workload.jsonl is empty")

    size = chunk_size(workspace, args.chunk) or len(lines)
    work = workspace / CHUNK_DIR
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    pending: deque[tuple[list[str], str]] = deque(
        (lines[i:i + size], str(i // size)) for i in range(0, len(lines), size)
    )
    print(f"[bench_sol] {len(lines)} workloads -> {len(pending)} job(s) of <= {size}", flush=True)

    traces: list[dict] = []
    done = 0
    while pending:
        chunk, tag = pending.popleft()
        print(f"[bench_sol] chunk {tag} ({len(chunk)} workloads, {done}/{len(lines)} evaluated)",
              flush=True)
        got = eval_subset(workspace, harness, chunk, tag, seed=args.seed)
        if got is not None:
            traces.extend(got)
            done += len(chunk)
            continue
        if len(chunk) == 1:
            raise SystemExit(
                f"[bench_sol] chunk {tag} holds a single workload and still produced no traces; "
                "it cannot be evaluated within the gateway's per-job limit"
            )
        # Too much work for one job — halve it and retry both halves before anything else.
        middle = len(chunk) // 2
        print(f"[bench_sol] chunk {tag} exceeded its job budget; splitting into "
              f"{middle} + {len(chunk) - middle}", flush=True)
        pending.appendleft((chunk[middle:], f"{tag}b"))
        pending.appendleft((chunk[:middle], f"{tag}a"))

    if len(traces) != len(lines):
        raise SystemExit(
            f"[bench_sol] merged {len(traces)} traces for {len(lines)} workloads; refusing to "
            "report a partial sweep as a complete result"
        )
    summary = harness._summarize(traces)
    print(f"{RESULT_MARKER}{json.dumps(summary, sort_keys=True)}", flush=True)
    print(f"[bench_sol] workloads PASSED {summary['passed']}/{summary['total']} "
          f"geomean {summary['latency_us_geomean']:.1f} us", flush=True)
    if not args.no_memory and args.version:
        harness._record_memory(workspace, args.version, summary)
    return 0 if summary["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
