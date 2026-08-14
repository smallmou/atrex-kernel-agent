from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

from . import main_adapter
from .git_episode import _git
from .models import VerificationResult, VerificationRun
from .protocol import atomic_write_json
from .store import VERIFY_DIR


ABBA_RESULT_PREFIX = "__ATREX_LONG_HORIZON_ABBA_RESULT__="
DEFAULT_SHAPE_BATCH_SIZE = 4


def _payload_from_stdout(stdout: str) -> dict[str, Any]:
    """Extract the long-horizon ABBA payload from ordinary sandbox stdout."""
    for line in reversed(stdout.splitlines()):
        if not line.startswith(ABBA_RESULT_PREFIX):
            continue
        try:
            payload = json.loads(line[len(ABBA_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            raise ValueError("malformed ABBA result sentinel") from exc
        if not isinstance(payload, dict):
            raise ValueError("ABBA result sentinel must contain a JSON object")
        return payload
    raise ValueError("missing ABBA result sentinel")


def verification_schedule(repeats: int) -> list[dict[str, int | str]]:
    schedule: list[dict[str, int | str]] = []
    for repeat in range(max(1, repeats)):
        revisions = (
            ("incumbent", "candidate")
            if repeat % 2 == 0
            else ("candidate", "incumbent")
        )
        schedule.extend(
            {"revision": revision, "repeat": repeat} for revision in revisions
        )
    return schedule


def _geomean(values: list[float]) -> float | None:
    if not values or any(value <= 0.0 or not math.isfinite(value) for value in values):
        return None
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _shape_sort_key(shape_id: str) -> tuple[int, object]:
    return (0, int(shape_id)) if shape_id.isdigit() else (1, shape_id)


def _verification_shape_batches(
    workspace: Path, private_reference_dir: Path | None, batch_size: int
) -> tuple[list[list[str] | None], list[str]]:
    """Return native Atrex shape batches; SOL workspaces retain one unfiltered job."""
    harness = workspace / "test_kernel.py"
    if not harness.is_file() or "--shape-id" not in harness.read_text(
        encoding="utf-8"
    ):
        # Existing V0 workspaces keep their immutable evaluator. They remain valid but
        # cannot accept the supervisor-only subset selector added with shape batching.
        return [None], []
    reference_dir = private_reference_dir or workspace
    shapes_path = reference_dir / "shapes.json"
    if not shapes_path.is_file():
        return [None], []
    shapes = json.loads(shapes_path.read_text(encoding="utf-8"))
    if not isinstance(shapes, dict) or not shapes:
        raise ValueError("shapes.json must contain a non-empty object")
    shape_ids = sorted((str(value) for value in shapes), key=_shape_sort_key)
    return (
        [
            shape_ids[offset : offset + batch_size]
            for offset in range(0, len(shape_ids), batch_size)
        ],
        shape_ids,
    )


def _merge_optimizer_batch_results(
    results: list[dict[str, Any]], expected_shape_ids: list[str]
) -> dict[str, Any]:
    failures: list[str] = []
    latency_by_shape: dict[str, float] = {}
    eval_ids: list[str] = []
    max_abs = 0.0
    max_rel = 0.0
    for result in results:
        failures.extend(str(value) for value in (result.get("failures") or []))
        if not result.get("all_pass") and not result.get("failures"):
            failures.append("shape batch did not pass")
        by_shape = result.get("latency_us_by_shape")
        by_shape = by_shape if isinstance(by_shape, dict) else {}
        for raw_shape_id, raw_latency in by_shape.items():
            shape_id = str(raw_shape_id)
            if shape_id in latency_by_shape:
                failures.append(f"sid={shape_id}: duplicate batch measurement")
            elif (
                not isinstance(raw_latency, (int, float))
                or not math.isfinite(float(raw_latency))
                or float(raw_latency) <= 0
            ):
                failures.append(f"sid={shape_id}: invalid batch latency")
            else:
                latency_by_shape[shape_id] = float(raw_latency)
        eval_id = result.get("eval_id")
        if eval_id is not None:
            eval_ids.append(str(eval_id))
        for field, current in (("max_abs_err", max_abs), ("max_rel_err", max_rel)):
            value = result.get(field)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                if field == "max_abs_err":
                    max_abs = max(current, float(value))
                else:
                    max_rel = max(current, float(value))

    expected = set(expected_shape_ids)
    measured = set(latency_by_shape)
    failures.extend(
        f"sid={shape_id}: missing batch measurement"
        for shape_id in sorted(expected - measured, key=_shape_sort_key)
    )
    failures.extend(
        f"sid={shape_id}: unexpected batch measurement"
        for shape_id in sorted(measured - expected, key=_shape_sort_key)
    )
    latencies = [
        latency_by_shape[shape_id]
        for shape_id in expected_shape_ids
        if shape_id in latency_by_shape
    ]
    complete = len(latencies) == len(expected_shape_ids) and bool(latencies)
    return {
        "all_pass": complete and not failures,
        "failures": failures,
        "latency_us_geomean": _geomean(latencies) if complete else 0.0,
        "latency_us_arith_mean": sum(latencies) / len(latencies) if complete else 0.0,
        "latency_us_by_shape": latency_by_shape,
        "speedup_vs_ref_geomean": None,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "evaluator": "atrex-bench/run_eval/batched-abba",
        "eval_id": eval_ids[-1] if eval_ids else None,
        "eval_ids": eval_ids,
        "shape_batch_count": len(results),
    }


def _merge_verification_batch_payloads(
    payloads: list[dict[str, Any]],
    *,
    schedule: list[dict[str, int | str]],
    expected_shape_ids: list[str],
) -> dict[str, Any]:
    if len(payloads) == 1:
        return payloads[0]
    errors: list[str] = []
    rows_by_batch: list[list[dict[str, Any]]] = []
    for batch_index, payload in enumerate(payloads):
        if payload.get("schema_version") != 1:
            errors.append(f"batch {batch_index + 1}: unsupported result schema")
        if payload.get("error"):
            errors.append(f"batch {batch_index + 1}: {payload['error']}")
        rows = payload.get("runs")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            errors.append(f"batch {batch_index + 1}: invalid runs")
            rows = []
        actual = [
            {"revision": row.get("revision"), "repeat": row.get("repeat")}
            for row in rows
        ]
        if actual != schedule:
            errors.append(f"batch {batch_index + 1}: ABBA schedule mismatch")
        rows_by_batch.append(rows)

    merged_rows: list[dict[str, Any]] = []
    if not errors:
        for row_index, step in enumerate(schedule):
            batch_rows = [rows[row_index] for rows in rows_by_batch]
            batch_results = [
                row.get("result")
                for row in batch_rows
                if isinstance(row.get("result"), dict)
            ]
            result = (
                _merge_optimizer_batch_results(batch_results, expected_shape_ids)
                if len(batch_results) == len(batch_rows)
                else None
            )
            exit_code = 0
            for row in batch_rows:
                value = row.get("exit_code")
                if not isinstance(value, int) or value != 0:
                    exit_code = int(value) if isinstance(value, int) else -1
                    break
            if result is None or not result.get("all_pass"):
                exit_code = exit_code or 1
            merged_rows.append(
                {
                    "revision": step["revision"],
                    "repeat": step["repeat"],
                    "exit_code": exit_code,
                    "result": result,
                    "stdout_tail": "\n".join(
                        str(row.get("stdout_tail", "")) for row in batch_rows
                    )[-3000:],
                    "stderr_tail": "\n".join(
                        str(row.get("stderr_tail", "")) for row in batch_rows
                    )[-3000:],
                }
            )
    return {
        "schema_version": 1,
        "runs": merged_rows,
        "error": "; ".join(errors) if errors else None,
        "shape_batch_count": len(payloads),
    }


def _result_latency(result: object) -> float | None:
    if not isinstance(result, dict) or not result.get("all_pass"):
        return None
    value = result.get("latency_us_geomean")
    if not isinstance(value, (int, float)):
        performance = result.get("performance")
        value = performance.get("latency_us") if isinstance(performance, dict) else None
    if (
        not isinstance(value, (int, float))
        or value <= 0
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def score_verification_payload(
    payload: object,
    *,
    schedule: list[dict[str, int | str]],
    repeats: int,
    min_improvement_pct: float,
    artifact: str = "",
) -> VerificationResult:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        return VerificationResult(
            "ERROR", None, None, None, error="unsupported result schema"
        )
    rows = payload.get("runs")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        return VerificationResult(
            "ERROR", None, None, None, error="runs must be a list of objects"
        )
    runs = [
        VerificationRun(
            revision=str(row.get("revision", "")),
            repeat=int(row.get("repeat", -1)),
            exit_code=int(row.get("exit_code", -1)),
            result=row.get("result") if isinstance(row.get("result"), dict) else None,
            stdout_tail=str(row.get("stdout_tail", "")),
            stderr_tail=str(row.get("stderr_tail", "")),
        )
        for row in rows
    ]
    if payload.get("error"):
        return VerificationResult(
            "ERROR",
            None,
            None,
            None,
            runs=runs,
            error=str(payload["error"]),
            artifact=artifact,
        )
    actual = [{"revision": run.revision, "repeat": run.repeat} for run in runs]
    if actual != schedule:
        return VerificationResult(
            "ERROR",
            None,
            None,
            None,
            runs=runs,
            error="remote verifier did not execute the exact ABBA schedule",
            artifact=artifact,
        )
    candidate_values = [
        value
        for run in runs
        if run.revision == "candidate" and run.exit_code == 0
        if (value := _result_latency(run.result)) is not None
    ]
    incumbent_values = [
        value
        for run in runs
        if run.revision == "incumbent" and run.exit_code == 0
        if (value := _result_latency(run.result)) is not None
    ]
    candidate = _geomean(candidate_values)
    incumbent = _geomean(incumbent_values)
    if len(candidate_values) != repeats or len(incumbent_values) != repeats:
        return VerificationResult(
            "FAIL",
            candidate,
            incumbent,
            None,
            runs=runs,
            error="not every authoritative ABBA run passed",
            artifact=artifact,
        )
    improvement = (
        ((incumbent - candidate) / incumbent * 100.0)
        if candidate and incumbent
        else None
    )
    if improvement is None or improvement <= min_improvement_pct:
        return VerificationResult(
            "FAIL",
            candidate,
            incumbent,
            improvement,
            runs=runs,
            error=(
                f"candidate improvement {improvement if improvement is not None else 'unknown'} "
                f"did not exceed {min_improvement_pct:.3f}%"
            ),
            artifact=artifact,
        )
    return VerificationResult(
        "PASS", candidate, incumbent, improvement, runs=runs, artifact=artifact
    )


def _git_blob(workspace: Path, revision: str, relative: str) -> bytes | None:
    result = _git(workspace, "show", f"{revision}:{relative}", check=False, binary=True)
    return result.stdout if result.returncode == 0 else None


def _snapshot_revision(
    workspace: Path,
    revision: str,
    changed_paths: list[str],
    destination: Path,
    label: str,
) -> dict[str, str | None]:
    manifest: dict[str, str | None] = {}
    for index, relative in enumerate(changed_paths):
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe candidate path: {relative}")
        content = _git_blob(workspace, revision, relative)
        if content is None:
            manifest[relative] = None
            continue
        snapshot_relative = f"snapshots/{label}/{index:04d}.bin"
        target = destination / snapshot_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest[relative] = snapshot_relative
    return manifest


class GatewayABBAValidator:
    def __init__(
        self,
        *,
        hardware: str,
        profile: str = "",
        url: str = "",
        timeout: int = 600,
        repeats: int = 2,
        per_run_timeout: int = 120,
        min_improvement_pct: float = 0.0,
        queue_wait_grace: int = 14_400,
        private_reference_dir: Path | None = None,
        shape_batch_size: int | None = None,
    ):
        self.hardware = hardware
        self.profile = profile
        self.url = url
        self.timeout = timeout
        self.repeats = max(1, repeats)
        self.per_run_timeout = per_run_timeout
        self.min_improvement_pct = min_improvement_pct
        self.queue_wait_grace = queue_wait_grace
        self.private_reference_dir = private_reference_dir
        configured_batch_size = (
            shape_batch_size
            if shape_batch_size is not None
            else int(
                os.environ.get(
                    "ATREX_EVAL_SHAPE_BATCH_SIZE",
                    str(DEFAULT_SHAPE_BATCH_SIZE),
                )
            )
        )
        if configured_batch_size <= 0:
            raise ValueError("shape_batch_size must be positive")
        self.shape_batch_size = configured_batch_size

    def verify(
        self,
        workspace: Path,
        *,
        base_commit: str,
        candidate_commit: str,
        changed_paths: list[str],
    ) -> VerificationResult:
        schedule = verification_schedule(self.repeats)
        if self.per_run_timeout * len(schedule) + 30 > self.timeout:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error="ABBA schedule cannot fit in one gateway allocation timeout",
            )
        try:
            shape_batches, expected_shape_ids = _verification_shape_batches(
                workspace,
                self.private_reference_dir,
                self.shape_batch_size,
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"cannot prepare ABBA shape batches: {type(exc).__name__}: {exc}",
            )
        verification_id = uuid.uuid4().hex
        relative_dir = f"{VERIFY_DIR}/{verification_id}"
        directory = workspace / relative_dir
        directory.mkdir(parents=True, exist_ok=False)
        # Naming the driver test_kernel.py deliberately selects the sandbox's
        # immutable-evaluator payload route. The runtime bundler includes the verification
        # driver and snapshots from their dedicated artifact directory.
        driver = directory / "test_kernel.py"
        shutil.copy2(Path(__file__).with_name("remote_abba.py"), driver)
        manifests = {
            "incumbent": _snapshot_revision(
                workspace, base_commit, changed_paths, directory, "incumbent"
            ),
            "candidate": _snapshot_revision(
                workspace, candidate_commit, changed_paths, directory, "candidate"
            ),
        }
        batch_specs: list[tuple[int, str, str]] = []
        for batch_index, shape_ids in enumerate(shape_batches):
            suffix = f"-b{batch_index:04d}"
            request_relative = f"{relative_dir}/request{suffix}.json"
            result_relative = f"{relative_dir}/result{suffix}.json"
            command = [
                "python3",
                "test_kernel.py",
                "--version",
                "vlong",
                "--no-memory",
            ]
            if shape_ids is not None:
                for shape_id in shape_ids:
                    command.extend(("--shape-id", shape_id))
            request = {
                "schema_version": 1,
                "schedule": schedule,
                "manifests": manifests,
                "command": command,
                "run_timeout_seconds": self.per_run_timeout,
            }
            atomic_write_json(workspace / request_relative, request)
            batch_specs.append((batch_index, request_relative, result_relative))

        if len(batch_specs) > 1:
            print(
                f"[orchestrator] enqueueing {len(batch_specs)} ABBA shape batches "
                f"(up to {self.shape_batch_size} shapes each)",
                flush=True,
            )

        def run_batch(spec: tuple[int, str, str]) -> dict[str, Any]:
            batch_index, request_relative, result_relative = spec
            try:
                process = main_adapter.run_sandbox(
                    workspace,
                    self.hardware,
                    self.profile,
                    self.url,
                    self.timeout,
                    [
                        "python3",
                        f"{relative_dir}/test_kernel.py",
                        request_relative,
                        result_relative,
                    ],
                    # The result is emitted with a long-horizon-only stdout
                    # sentinel. No artifact is synchronized, avoiding the
                    # gateway's elided-payload/base64 transport failure.
                    sync=(),
                    wall_timeout=self.timeout + self.queue_wait_grace + 120,
                    gateway_kind="dev",
                    private_reference_dir=self.private_reference_dir,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"ABBA shape batch {batch_index + 1} timed out"
                ) from exc
            if process.returncode != 0:
                tail = (process.stdout + "\n" + process.stderr)[-3000:]
                raise RuntimeError(
                    f"ABBA shape batch {batch_index + 1} exited "
                    f"{process.returncode}: {tail}"
                )
            try:
                payload = _payload_from_stdout(process.stdout)
            except ValueError as exc:
                raise RuntimeError(
                    f"ABBA shape batch {batch_index + 1} returned no valid result: "
                    f"{exc}"
                ) from exc
            try:
                atomic_write_json(workspace / result_relative, payload)
            except (OSError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"cannot persist ABBA shape batch {batch_index + 1}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            return payload

        payloads: list[dict[str, Any]] = []
        try:
            with ThreadPoolExecutor(max_workers=len(batch_specs)) as executor:
                futures = [executor.submit(run_batch, spec) for spec in batch_specs]
                payloads = [future.result() for future in futures]
            payload = _merge_verification_batch_payloads(
                payloads,
                schedule=schedule,
                expected_shape_ids=expected_shape_ids,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"gateway ABBA verification failed: {exc}",
            )

        result_path = directory / "result.json"
        try:
            atomic_write_json(result_path, payload)
        except (OSError, TypeError, ValueError) as exc:
            return VerificationResult(
                "ERROR",
                None,
                None,
                None,
                error=f"cannot persist gateway ABBA result: {type(exc).__name__}: {exc}",
            )
        return score_verification_payload(
            payload,
            schedule=schedule,
            repeats=self.repeats,
            min_improvement_pct=self.min_improvement_pct,
            artifact=str(result_path),
        )
