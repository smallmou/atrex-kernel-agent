#!/usr/bin/env python3
"""Compact status table for the FlashInfer-Bench optimization campaigns.

One row per operator workspace: whether the orchestrator is alive, how far the memory
records have advanced, and whether the geomean latency is actually coming down.

Usage:
  python3 scripts/bench_status.py [--workspace DIR]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def live_ops() -> set[str]:
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    ops = set()
    for line in out.splitlines():
        if "orchestrator/optimize.py" not in line:
            continue
        m = re.search(r"--op-dir\s+(\S+)", line)
        if m:
            ops.add(Path(m.group(1)).name)
    return ops


def version_records(ws: Path) -> list[tuple[int, dict]]:
    records = []
    for path in (ws / "memory").glob("v*.json"):
        m = re.fullmatch(r"v(\d+)", path.stem)
        if not m:
            continue
        try:
            records.append((int(m.group(1)), json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            continue
    return sorted(records)


def latency(record: dict) -> float | None:
    value = ((record.get("performance") or {}).get("latency_us"))
    return value if isinstance(value, (int, float)) and value > 0 else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="/home/liangyan/aka-opt-sol-flashinfer")
    args = ap.parse_args()
    root = Path(args.workspace)

    alive = live_ops()
    now = time.time()
    print(f"{'op':52s} {'state':6s} {'ver':>4s} {'v0_us':>9s} {'best_us':>9s} "
          f"{'gain%':>7s} {'commits':>7s} {'log_age':>8s}")
    down = []
    for ws in sorted(root.glob("kernel_opt_*")):
        op = ws.name[len("kernel_opt_"):]
        state = "RUN" if op in alive else "DEAD"
        records = version_records(ws)
        latest = records[-1][0] if records else -1
        lats = [l for l in (latency(r) for _, r in records) if l]
        base = latency(dict(records[0][1])) if records else None
        best = min(lats) if lats else None
        gain = f"{(base - best) / base * 100:+.1f}" if base and best else "-"
        commits = subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=str(ws),
                                 capture_output=True, text=True).stdout.strip() or "?"
        log = root / "logs" / f"{op}_cuda.log"
        age = f"{(now - log.stat().st_mtime) / 60:.0f}m" if log.exists() else "-"
        print(f"{op:52s} {state:6s} {latest:>4d} "
              f"{base if base else 0:9.2f} {best if best else 0:9.2f} {gain:>7s} "
              f"{commits:>7s} {age:>8s}")
        if state == "DEAD":
            down.append(op)
    print(f"\nalive={len(alive)} workspaces={len(list(root.glob('kernel_opt_*')))}")
    if down:
        print("DOWN: " + " ".join(down))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
