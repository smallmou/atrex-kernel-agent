#!/usr/bin/env python3
"""Adapt the precision-validation policy to the AKA plugin JSON stdin/stdout contract.

Three pure, sub-second tools. Everything that needs a GPU allocation, an agent session
or a queue wait stays on the supervisor side of the boundary, because a plugin tool's
``timeout_seconds`` caps at 3600 while a gateway admission wait alone is budgeted far
higher. What lives here is the *policy*: which probes must run, and whether the
returned evidence and the independent review actually discharge it.

The precision comparison itself belongs to Atrex-Bench, declared as this plugin's
``atrex-bench-runtime`` / ``atrex-bench-runner`` resources and reached inside the
allocation by :mod:`driver`, which the supervisor ships in as ``test_kernel.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import driver

CHECKS = {"input_domain", "precision_and_reductions", "nonlinear_and_quantization",
          "routing_and_boundaries", "distribution_coverage"}


def plan(request: dict) -> dict:
    """Validate the suite and derive the probe schedule for this candidate."""
    suite = driver.validate_suite(request["suite"])
    shapes = request["shapes"]
    mode = request.get("mode", "light")
    schedule = driver.validation_schedule(suite, shapes, request.get("rotation", ""), mode)
    trust_mode = driver.resolve_trust_mode(
        request.get("trust_mode", "untrusted"), request.get("framework_key", "")
    )
    return {
        "schema_version": 1,
        "trust_mode": trust_mode,
        "schedule": [
            {
                "case_id": plan_row["case_id"],
                "shape_ids": plan_row["shape_ids"],
                "seeds": plan_row["seeds"],
                "selection_digest": plan_row["selection_digest"],
                "expected_probes": driver.expected_probes(plan_row, suite, shapes),
            }
            for plan_row in schedule
        ],
        "coverage": {
            "mode": mode,
            "coverage": suite.get("coverage", "compact"),
            "total_shapes": len(shapes),
            "world_size": suite["world_size"],
            "planned_rank_probes": sum(
                len(row["shape_ids"]) * len(row["seeds"]) * suite["world_size"]
                for row in schedule
            ),
            "selected_shapes_per_case": [len(row["shape_ids"]) for row in schedule],
        },
    }


def check_evaluation(request: dict) -> dict:
    """Reject GPU evidence that omits, reorders or under-covers a planned probe."""
    payload = request["evaluation"]
    schedule = request["schedule"]
    world_size = request["world_size"]
    problems: list[str] = []
    if (not isinstance(payload, dict) or payload.get("schema_version") != 1
            or payload.get("all_pass") is not True or payload.get("error")):
        problems.append("numerical distribution evaluation failed")
        return {"schema_version": 1, "accepted": False, "violations": problems}
    rows = payload.get("runs", [])
    if [r.get("case_id") for r in rows] != [p["case_id"] for p in schedule]:
        problems.append("numerical evaluation omitted, duplicated or reordered distribution cases")
        return {"schema_version": 1, "accepted": False, "violations": problems}
    for row, plan_row in zip(rows, schedule):
        expected = plan_row["expected_probes"]
        if (row.get("passed") is not True or row.get("exit_code") != 0
                or not isinstance(row.get("result"), dict) or row["result"].get("all_pass") is not True
                or row.get("expected_probes") != expected or row.get("observed_probes") != expected
                or row.get("shape_count") != len(plan_row["shape_ids"]) or row.get("seeds") != plan_row["seeds"]
                or row.get("selection_digest") != plan_row["selection_digest"]
                or row.get("world_size") != world_size):
            problems.append(f"incomplete numerical evidence for {row.get('case_id')}")
    return {"schema_version": 1, "accepted": not problems, "violations": problems}


def check_review(request: dict) -> dict:
    """Reject a review that is unbound, incomplete, or uncited, and surface its rejections."""
    payload = request["review"]
    digest = request["evidence_digest"]
    supplied = set(request["supplied_files"])
    if (not isinstance(payload, dict) or payload.get("schema_version") != 1
            or payload.get("evidence_digest") != digest):
        return {"schema_version": 1, "accepted": False, "violations": [
            "numerical review is missing or bound to different evidence"]}
    items = payload.get("checks", [])
    if (not isinstance(items, list) or not all(isinstance(i, dict) for i in items)
            or len(items) != len(CHECKS) or {i.get("id") for i in items} != CHECKS):
        return {"schema_version": 1, "accepted": False, "violations": [
            "numerical review omitted or duplicated required checks"]}
    errors: list[str] = []
    for item in items:
        evidence = item.get("evidence", [])
        if (item.get("decision") not in {"allow", "reject"}
                or not isinstance(item.get("reason"), str) or not item["reason"].strip()
                or not isinstance(evidence, list) or not evidence
                or not all(isinstance(e, str) for e in evidence)
                or any(e.split(":", 1)[0] not in supplied or ":" not in e for e in evidence)
                or not any(e.startswith("candidate/kernel.py:") for e in evidence)
                or not any(e.startswith(("trusted/", "evaluation.json:", "numerical_suite.json:"))
                           for e in evidence)):
            return {"schema_version": 1, "accepted": False, "violations": [
                f"numerical check lacks source and contract/evaluation evidence: {item.get('id')}"]}
        if item["decision"] == "reject":
            errors.append(f"numerical safety {item['id']}: {item['reason']}")
    if payload.get("verdict") != ("reject" if errors else "allow"):
        return {"schema_version": 1, "accepted": False, "violations": [
            "numerical review verdict disagrees with checks"]}
    return {"schema_version": 1, "accepted": not errors, "violations": errors}


TOOLS = {"plan": plan, "check-evaluation": check_evaluation, "check-review": check_review}


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in TOOLS:
        print(f"usage: adapter.py {{{'|'.join(sorted(TOOLS))}}}", file=sys.stderr)
        return 2
    try:
        request = json.load(sys.stdin)
        result = TOOLS[sys.argv[1]](request)
    except Exception as exc:
        # A traceback here would be captured into the violation string the supervisor
        # records and shows the candidate, carrying absolute supervisor paths with it.
        # Report the reason only; the schema validator already rejected malformed input,
        # so anything reaching this point is a policy error worth stating plainly.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, allow_nan=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
