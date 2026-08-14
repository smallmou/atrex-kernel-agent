from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from orchestrator.campaign import Campaign
from long_horizon.verifier import (
    ABBA_RESULT_PREFIX,
    GatewayABBAValidator,
    _merge_verification_batch_payloads,
    verification_schedule,
)
from tools import sandbox


def _eval_payload(shape_ids: list[str], eval_id: str) -> dict[str, object]:
    return {
        "eval_id": eval_id,
        "passed": {
            "compile": {
                shape_id: {"status": "passed"} for shape_id in shape_ids
            },
            "correctness": {
                shape_id: {"status": "passed"} for shape_id in shape_ids
            },
        },
        "correctness": {
            "shapes": {shape_id: {"cases": []} for shape_id in shape_ids}
        },
        "performance": {
            "shapes": {
                shape_id: {"samples": [{"end_to_end_time_ms": int(shape_id) + 1}]}
                for shape_id in shape_ids
            }
        },
    }


class GeneralizedMemoryCoverageTest(unittest.TestCase):
    def _campaign(self, root: Path) -> Campaign:
        operator = root / "operator"
        operator.mkdir()
        reference = operator / "reference.py"
        reference.write_text("def run(*args): pass\n")
        (operator / "shapes.json").write_text(
            json.dumps(
                {
                    "0": {"init_kwargs": None, "input_kwargs": {}},
                    "1": {"init_kwargs": {}, "input_kwargs": {}},
                }
            )
        )
        return Campaign(
            name="coverage",
            kernel_demo=str(reference),
            platform="pro5000",
            framework="cuda",
            work_dir=str(root / "work"),
            atrex_bench_root="enabled",
            optimization_mode="production",
        )

    @staticmethod
    def _valid_memory() -> dict[str, object]:
        return {
            "performance": {
                "latency_us": 12.0,
                "latency_us_geomean": 12.0,
                "latency_us_arith_mean": 12.5,
                "latency_us_by_shape": {"0": 10.0, "1": 15.0},
                "measurement_scope": "real_evaluator_shapes",
                "shape_ids_are_opaque": True,
                "measurement_status": "complete",
                "measured_shape_count": 2,
                "speedup_vs_ref_geomean": 1.0,
            },
            "correctness": {"status": "PASS"},
            "quality_gate": {"result": "PASS"},
        }

    def test_accepts_complete_real_shape_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            campaign = self._campaign(Path(temp_dir))
            self.assertEqual(
                campaign._generalized_memory_coverage_problem(self._valid_memory()),
                "",
            )

    def test_rejects_null_aggregate_metric(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            campaign = self._campaign(Path(temp_dir))
            memory = self._valid_memory()
            performance = memory["performance"]
            assert isinstance(performance, dict)
            performance["latency_us_geomean"] = None
            self.assertIn(
                "latency_us_geomean",
                campaign._generalized_memory_coverage_problem(memory),
            )


class TypedShapeBatchQueueTest(unittest.TestCase):
    def test_submits_every_batch_before_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "kernel.py").write_text("def run(*args): pass\n")
            (workspace / "reference.py").write_text("def run(*args): pass\n")
            (workspace / "input.py").write_text("def generate(*args): pass\n")
            (workspace / "shapes.json").write_text(
                json.dumps({str(index): {"n": index} for index in range(5)})
            )
            args = argparse.Namespace(
                hardware="L20N",
                url="",
                gateway_profile="prod",
                timeout=600,
                env=[],
                profile_level="sol",
                profiler=None,
                profile_counter=[],
                kernel_regex=None,
                top_kernels=None,
                shape_batch_size=2,
                dry_run=False,
            )
            events: list[str] = []
            job_shapes: dict[str, list[str]] = {}

            def submit(command: list[str]) -> subprocess.CompletedProcess[str]:
                job_id = f"ev_{len(job_shapes)}"
                reference_dir = Path(command[command.index("--reference-dir") + 1])
                shapes = json.loads(
                    (reference_dir / "shapes.json").read_text(encoding="utf-8")
                )
                job_shapes[job_id] = list(shapes)
                events.append(f"submit:{job_id}")
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps({"job_id": job_id, "status": "queued"}),
                    stderr="",
                )

            def wait(**kwargs: object) -> subprocess.CompletedProcess[str]:
                job_id = str(kwargs["job_id"])
                events.append(f"wait:{job_id}")
                job = {
                    "job_id": job_id,
                    "status": "succeeded",
                    "result": _eval_payload(job_shapes[job_id], job_id),
                }
                return subprocess.CompletedProcess(
                    ["agate", "get"],
                    0,
                    stdout=json.dumps(job),
                    stderr="",
                )

            with (
                mock.patch.object(sandbox, "_find_agate", return_value="agate"),
                mock.patch.object(
                    sandbox, "_submit_agate_without_wait", side_effect=submit
                ),
                mock.patch.object(sandbox, "_wait_for_agate_job", side_effect=wait),
                redirect_stdout(StringIO()) as stdout,
            ):
                return_code = sandbox._run_typed_gateway(
                    args,
                    workspace,
                    ["python", "test_kernel.py", "--no-memory"],
                    "run",
                    [],
                    0,
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(
                events,
                [
                    "submit:ev_0",
                    "submit:ev_1",
                    "submit:ev_2",
                    "wait:ev_0",
                    "wait:ev_1",
                    "wait:ev_2",
                ],
            )
            result_line = stdout.getvalue().strip().splitlines()[-1]
            result = json.loads(result_line.removeprefix(sandbox.TEST_RESULT_PREFIX))
            self.assertTrue(result["all_pass"])
            self.assertEqual(result["shape_batch_count"], 3)
            self.assertEqual(
                set(result["latency_us_by_shape"]),
                {"0", "1", "2", "3", "4"},
            )

    def test_direct_http_submits_every_batch_before_waiting(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "kernel.py").write_text("def run(*args): pass\n")
            (workspace / "reference.py").write_text("def run(*args): pass\n")
            (workspace / "input.py").write_text("def generate(*args): pass\n")
            (workspace / "shapes.json").write_text(
                json.dumps({str(index): {"n": index} for index in range(5)})
            )
            args = argparse.Namespace(
                hardware="local",
                url="http://127.0.0.1:8000",
                gateway_profile=None,
                timeout=600,
                env=[],
                profile_level="sol",
                profiler=None,
                profile_counter=[],
                kernel_regex=None,
                top_kernels=None,
                shape_batch_size=2,
                dry_run=False,
            )
            events: list[str] = []
            job_shapes: dict[str, list[str]] = {}

            def submit(
                _url: str, _kind: str, request: dict[str, object]
            ) -> dict[str, str]:
                job_id = f"ev_{len(job_shapes)}"
                reference = request["reference"]
                assert isinstance(reference, dict)
                shapes = reference["shapes"]
                assert isinstance(shapes, dict)
                job_shapes[job_id] = list(shapes)
                events.append(f"submit:{job_id}")
                return {"job_id": job_id, "status": "queued"}

            def wait(
                _url: str,
                job_id: str,
                *,
                timeout: int,
                queue_wait_grace: int,
            ) -> dict[str, object]:
                del timeout, queue_wait_grace
                events.append(f"wait:{job_id}")
                return {
                    "job_id": job_id,
                    "status": "succeeded",
                    "result": _eval_payload(job_shapes[job_id], job_id),
                }

            with (
                mock.patch.object(sandbox, "_find_agate", return_value=None),
                mock.patch.object(sandbox, "_submit_direct_job", side_effect=submit),
                mock.patch.object(sandbox, "_wait_direct_job", side_effect=wait),
                redirect_stdout(StringIO()),
            ):
                return_code = sandbox._run_typed_gateway(
                    args,
                    workspace,
                    ["python", "test_kernel.py", "--no-memory"],
                    "run",
                    [],
                    0,
                )

            self.assertEqual(return_code, 0)
            self.assertEqual(
                events,
                [
                    "submit:ev_0",
                    "submit:ev_1",
                    "submit:ev_2",
                    "wait:ev_0",
                    "wait:ev_1",
                    "wait:ev_2",
                ],
            )


class ABBABatchMergeTest(unittest.TestCase):
    def test_merges_each_schedule_row_over_all_shapes(self) -> None:
        schedule = verification_schedule(1)

        def payload(shape_id: str, latency: float) -> dict[str, object]:
            return {
                "schema_version": 1,
                "runs": [
                    {
                        "revision": step["revision"],
                        "repeat": step["repeat"],
                        "exit_code": 0,
                        "result": {
                            "all_pass": True,
                            "failures": [],
                            "latency_us_by_shape": {shape_id: latency},
                            "max_abs_err": 0.0,
                            "max_rel_err": 0.0,
                            "eval_id": f"eval-{shape_id}",
                        },
                    }
                    for step in schedule
                ],
                "error": None,
            }

        merged = _merge_verification_batch_payloads(
            [payload("0", 1.0), payload("1", 4.0)],
            schedule=schedule,
            expected_shape_ids=["0", "1"],
        )
        self.assertIsNone(merged["error"])
        self.assertEqual(merged["shape_batch_count"], 2)
        for row in merged["runs"]:
            self.assertEqual(row["result"]["latency_us_geomean"], 2.0)
            self.assertEqual(
                row["result"]["latency_us_by_shape"], {"0": 1.0, "1": 4.0}
            )

    def test_verifier_enqueues_all_shape_batches_concurrently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "test_kernel.py").write_text("# supports --shape-id\n")
            (workspace / "kernel.py").write_text("candidate = 0\n")
            (workspace / "shapes.json").write_text(
                json.dumps({str(index): {"n": index} for index in range(5)})
            )
            subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=workspace,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=workspace, check=True
            )
            subprocess.run(["git", "add", "."], cwd=workspace, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "base"], cwd=workspace, check=True
            )
            base_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
            ).strip()
            (workspace / "kernel.py").write_text("candidate = 1\n")
            subprocess.run(["git", "add", "kernel.py"], cwd=workspace, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "candidate"], cwd=workspace, check=True
            )
            candidate_commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=workspace, text=True
            ).strip()

            barrier = threading.Barrier(3, timeout=2)
            starts: list[list[str]] = []

            def run_sandbox(
                root: Path,
                _hardware: str,
                _profile: str,
                _url: str,
                _timeout: int,
                command: list[str],
                **_kwargs: object,
            ) -> subprocess.CompletedProcess[str]:
                request = json.loads((root / command[2]).read_text(encoding="utf-8"))
                evaluator_command = request["command"]
                shape_ids = [
                    evaluator_command[index + 1]
                    for index, value in enumerate(evaluator_command)
                    if value == "--shape-id"
                ]
                starts.append(shape_ids)
                barrier.wait()
                runs = []
                for step in request["schedule"]:
                    latency = 1.0 if step["revision"] == "candidate" else 2.0
                    runs.append(
                        {
                            **step,
                            "exit_code": 0,
                            "result": {
                                "all_pass": True,
                                "failures": [],
                                "latency_us_by_shape": {
                                    shape_id: latency for shape_id in shape_ids
                                },
                                "max_abs_err": 0.0,
                                "max_rel_err": 0.0,
                            },
                        }
                    )
                payload = {"schema_version": 1, "runs": runs, "error": None}
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=ABBA_RESULT_PREFIX + json.dumps(payload) + "\n",
                    stderr="",
                )

            validator = GatewayABBAValidator(
                hardware="L20N",
                timeout=100,
                repeats=1,
                per_run_timeout=10,
                shape_batch_size=2,
            )
            with mock.patch(
                "long_horizon.verifier.main_adapter.run_sandbox",
                side_effect=run_sandbox,
            ):
                result = validator.verify(
                    workspace,
                    base_commit=base_commit,
                    candidate_commit=candidate_commit,
                    changed_paths=["kernel.py"],
                )

            self.assertEqual(result.gate, "PASS")
            self.assertEqual(len(starts), 3)
            self.assertEqual({shape_id for batch in starts for shape_id in batch}, {
                "0",
                "1",
                "2",
                "3",
                "4",
            })


if __name__ == "__main__":
    unittest.main()
