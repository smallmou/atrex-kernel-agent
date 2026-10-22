"""Behavioral regressions from the 72-task Pi/Claude experiment.

Run: PYTHONPATH=src python3 tools/test_experiment_recovery.py
"""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aka.light.experiment.records import acceptance_disposition
from aka.light.execution import retry
from aka.light.workflow.episode import LongHorizonCampaign
from aka.light.workflow.model_service import ModelServiceRecovery
from aka.light.workflow.protocol_recovery import ProtocolPaused, check_protocol_paused
from aka.light.workflow.session import LongSessionRunner
from aka.light.workflow.store import CampaignStore
from aka.light.evaluation.policy import measurement_defaults
from aka.light.workflow.health import inspect_health
from aka.light.workflow import runtime_migration as migration
from aka.legacy.workflow.checkpoint import CONTEXT
from aka.legacy.workflow.public_state import campaign_lease
from aka.light.runtime.paths import supervisor_campaign_root
from test_model_service_recovery import FakeAgent


def receipt(job, request=None, phase="terminal"):
    return {"phase": phase, "identity": {"kind": "eval", "payload": request or {
        "reference": {"shapes": {"0": {}}}}}, "process": {
        "returncode": 0, "stdout": json.dumps(job), "stderr": ""}}


def failed_candidate(reason="candidate raised exception"):
    return {"status": "succeeded", "result": {"passed": {
        "compile": {"0": {"status": "passed"}},
        "correctness": {"0": {"status": "failed", "reason": reason}},
        "performance": {"0": {"status": "skipped"}}}}}


class GatewayTests(unittest.TestCase):
    def test_completed_candidate_exception_does_not_wait_for_infrastructure(self):
        self.assertEqual(acceptance_disposition([receipt(failed_candidate())], {}), "candidate_failure")

    def test_failed_job_typed_compilation_rejection(self):
        job = {"status": "failed", "error": {"error_class": "compilation"}}
        self.assertEqual(acceptance_disposition([receipt(job)], {}), "candidate_failure")

    def test_partial_abba_candidate_rejection_is_actionable(self):
        job = {"status": "succeeded", "result": {"abba": {"sdk_results": [{"abba": {"runs": [
            {"revision": "candidate", "result": failed_candidate()["result"]}]}}]}}}
        request = {"reference": {"shapes": {"0": {}}}, "abba": {"repeats": 2}}
        self.assertEqual(acceptance_disposition([receipt(job, request)], {}), "candidate_failure")
        job["result"]["abba"]["sdk_results"][0]["abba"]["runs"][0]["revision"] = "baseline"
        self.assertEqual(acceptance_disposition([receipt(job, request)], {}), "incomplete")

    def test_missing_shape_and_truncated_success_are_not_passes(self):
        job = {"status": "succeeded", "result": {"passed": {"compile": {}}}}
        self.assertEqual(acceptance_disposition([receipt(job)], {}), "incomplete")
        job = {"status": "succeeded", "result": {"passed": {
            "compile": {"0": {"status": "passed"}}, "correctness": {"0": {"status": "passed"}}}}}
        request = {"reference": {"shapes": {"0": {}}}, "mode": "correctness_only"}
        self.assertEqual(acceptance_disposition([receipt(job, request)], {}), "complete")
        self.assertEqual(acceptance_disposition([receipt(job, request)], {"truncated": True}), "incomplete")

    def test_device_unavailable_and_illegal_access_are_different(self):
        for reason, expected in (("no cuda gpus are available", "infra"),
                                 ("CUDA error: an illegal memory access was encountered", "candidate_failure")):
            self.assertEqual(acceptance_disposition([receipt(failed_candidate(reason))], {}), expected)

    def test_unknown_terminal_and_pending_job_never_become_rejections(self):
        for job in ({"status": "cancelled"}, {"status": "failed", "error": {"error_class": "unknown"}}):
            self.assertEqual(acceptance_disposition([receipt(job)], {}), "incomplete")
        self.assertEqual(acceptance_disposition([receipt({}, phase="submitted")], {}), "pending")
        self.assertEqual(acceptance_disposition([], {}), "incomplete")

    def test_malformed_evidence_does_not_crash_or_reject_candidate(self):
        for result in ({"passed": ["invalid"]}, {"passed": {"compile": [False]}},
                       {"abba": {"sdk_results": [None]}}):
            request = {"reference": {"shapes": {"0": {}}}}
            if "abba" in result:
                request["abba"] = {"repeats": 2}
            self.assertEqual(acceptance_disposition([receipt(
                {"status": "succeeded", "result": result}, request)], {}), "incomplete")


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name) / "campaign"

    def test_invalid_review_then_valid_rejection_is_not_overridden(self):
        operation = Mock(side_effect=[retry.InvalidReview("missing JSON"), ["candidate violates policy"]])
        result = retry.retry_invalid_review(self.workspace, "dependency:digest", operation)
        self.assertEqual(result, ["candidate violates policy"])
        cached = Mock(side_effect=AssertionError("review must not rerun"))
        self.assertEqual(retry.retry_invalid_review(self.workspace, "dependency:digest", cached), result)

    def test_three_invalid_reviews_stay_blocked_across_restart(self):
        operation = Mock(side_effect=retry.InvalidReview("schema"))
        for _ in range(2):
            with self.assertRaises(retry.ValidationBlocked):
                retry.retry_invalid_review(self.workspace, "digest", operation)
        self.assertEqual(operation.call_count, 3)
        self.assertEqual(retry.retry_invalid_review(self.workspace, "new-digest", lambda: "valid"), "valid")

    def test_service_interruption_does_not_consume_invalid_review_budget(self):
        with self.assertRaises(retry.InfrastructureUnavailable):
            retry.retry_invalid_review(self.workspace, "digest", Mock(side_effect=retry.InfrastructureUnavailable()))
        operation = Mock(side_effect=[retry.InvalidReview(), retry.InvalidReview(), "valid"])
        self.assertEqual(retry.retry_invalid_review(self.workspace, "digest", operation), "valid")

    def test_incomplete_validation_is_bounded_and_durable(self):
        operation = Mock(side_effect=retry.EvidenceIncomplete("missing shapes"))
        with patch.object(retry, "_wait_until"):
            for _ in range(2):
                with self.assertRaises(retry.ValidationBlocked):
                    retry.retry_infrastructure(self.workspace, "measure:digest", operation)
        self.assertEqual(operation.call_count, 3)

    def test_baseline_repairs_not_replenished_by_restart(self):
        for _ in range(3):
            retry.reserve_baseline_repair(self.workspace, "frozen-reference")
        with self.assertRaises(retry.ValidationBlocked):
            retry.reserve_baseline_repair(self.workspace, "frozen-reference")


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.handoff = self.root / "handoff.json"

    def test_pi_gets_two_fresh_report_repairs_in_same_draft(self):
        active = {}
        agent = FakeAgent(self.handoff, ["", "", "handoff"])
        runner = LongSessionRunner(agent=agent, agent_cli="pi", usage_policy=lambda t: t.terminal_usage.total_tokens)
        result = runner.run(workspace=self.root, prompt="optimize", handoff_path=self.handoff,
            completion_check=lambda h: "", reasoning_effort="high", handoff_resumes=2,
            model_service_recovery=ModelServiceRecovery(active, lambda a: None))
        self.assertIsNotNone(result.handoff)
        self.assertEqual(active["protocol_repair_count"], 2)
        self.assertEqual(len(agent.calls), 3)
        self.assertTrue(all(call[2] == self.root for call in agent.calls))

    def test_missing_handoff_preserves_budget_tokens_and_draft(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        store = CampaignStore(self.root)
        draft = self.root / "draft"
        runtime = draft / ".atrex_long_horizon"
        runtime.mkdir(parents=True)
        (runtime / "journal.json").write_text("{}")
        (draft / "kernel.py").write_text("candidate remains")
        active = {"episode": 1, "memory_version": 2, "phase": "exploring"}
        store.save_active(active)
        worktree = SimpleNamespace(path=draft, episode=1, base_commit="base", branch="episode-1")
        agent = FakeAgent(runtime / "handoff.json", ["", "", ""])
        runner = LongSessionRunner(agent=agent, agent_cli="pi", usage_policy=lambda t: t.terminal_usage.total_tokens)
        base = SimpleNamespace(workspace=self.root, module_options=SimpleNamespace(
            goal_episodes=False, episode_timeout_seconds=100),
            register_runtime_episode=Mock(return_value=draft), agent_environment=lambda: {})
        campaign = LongHorizonCampaign(base, verifier=Mock(), max_version=30, session_runner=runner)
        with ExitStack() as stack:
            for name, value in (("prepare_campaign", None), ("restored_stall", 0),
                                ("conversion_required", False), ("latest_version", 1)):
                stack.enter_context(patch("aka.light.workflow.episode.main_adapter." + name, return_value=value))
            stack.enter_context(patch.object(campaign, "_recover_interrupted", return_value=(worktree, active)))
            stack.enter_context(patch.object(campaign, "_prompt", return_value="optimize"))
            record = stack.enter_context(patch.object(campaign, "_record_terminal_episode"))
            with self.assertRaises(ProtocolPaused):
                campaign.run()
            record.assert_not_called()
        state = store.load_state()
        self.assertEqual((state.episodes, state.tokens), (0, 15))
        self.assertEqual(store.load_active()["memory_version"], 2)
        with self.assertRaises(ProtocolPaused):
            check_protocol_paused(store.load_active())
        self.assertEqual((draft / "kernel.py").read_text(), "candidate remains")


class HealthTests(unittest.TestCase):
    def test_command_success_is_not_optimization_success(self):
        with tempfile.TemporaryDirectory() as root:
            self.assertEqual(inspect_health(root, command_phase="Succeeded")["status"], "unverified")
            self.assertEqual(inspect_health(root, command_phase="Queued")["status"], "queued")

    def test_invocation_limit_is_not_search_completion(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root) / ".atrex_long_horizon"
            directory.mkdir()
            (directory / "outcome.json").write_text(json.dumps({
                "search_finished": False, "budget_exhausted": False}))
            self.assertEqual(inspect_health(root)["status"], "invocation_limited")

    def test_timeout_policy_changes_measurement_identity(self):
        original = measurement_defaults({})
        changed = measurement_defaults({"ATREX_SANDBOX_PERF_TIMEOUT_S": "480"})
        self.assertNotEqual(original, changed)
        self.assertEqual(original["runner_overrides"]["perf_timeout_s"], 120)
        self.assertEqual(changed["runner_overrides"]["perf_timeout_s"], 480)


class BaselineResumeTests(unittest.TestCase):
    def test_preserved_candidate_is_validated_before_another_coding_turn(self):
        from aka.light.workflow import bootstrap as module
        with tempfile.TemporaryDirectory() as directory:
            base = SimpleNamespace(workspace=Path(directory), framework="cuda", generate_memory=False)
            for name in ("_ensure_framework_baseline_solution_manifest", "_link_runtime",
                         "_restore_framework_baseline_candidate", "_sync_framework_baseline_live",
                         "_warn_restored_baseline_paths", "_run_framework_baseline_agent",
                         "_recover_framework_baseline"):
                setattr(base, name, Mock())
            base._framework_baseline_decision = Mock(return_value=("run", "resume"))
            base._v0_baseline_commit = Mock(return_value="baseline")
            base._framework_baseline_problem = Mock(return_value="")
            base._framework_baseline_external_gates = Mock(return_value=({"all_pass": True}, ""))
            base._commit_framework_baseline = Mock(return_value="candidate")
            with ExitStack() as stack:
                for name, value in (("git_head", "baseline"), ("git_path_blob", "V0"),
                                    ("git_worktree_blob", "preserved CUDA"),
                                    ("mark_framework_baseline_accepted", None), ("read_memory", {})):
                    stack.enter_context(patch.object(module, name, return_value=value))
                module.Campaign.ensure_framework_baseline(base)
            base._run_framework_baseline_agent.assert_not_called()
            base._recover_framework_baseline.assert_not_called()
            base._framework_baseline_external_gates.assert_called_once_with(1)

    def test_restart_without_candidate_cannot_bypass_exhausted_repairs(self):
        from aka.light.workflow import bootstrap as module
        with tempfile.TemporaryDirectory() as directory:
            base = SimpleNamespace(workspace=Path(directory))
            for name in ("_ensure_framework_baseline_solution_manifest", "_link_runtime",
                         "_restore_framework_baseline_candidate", "_run_framework_baseline_agent"):
                setattr(base, name, Mock())
            base._framework_baseline_decision = Mock(return_value=("run", "resume"))
            base._v0_baseline_commit = Mock(return_value="baseline")
            base._recover_framework_baseline = lambda *args: module.Campaign._recover_framework_baseline(base, *args)
            for _ in range(3):
                retry.reserve_baseline_repair(base.workspace, "baseline")
            with ExitStack() as stack:
                for name, value in (("git_head", "baseline"), ("git_path_blob", "V0"), ("git_worktree_blob", "V0")):
                    stack.enter_context(patch.object(module, name, return_value=value))
                with self.assertRaises(retry.ValidationBlocked):
                    module.Campaign.ensure_framework_baseline(base)
            base._run_framework_baseline_agent.assert_not_called()


class NumericalReviewTests(unittest.TestCase):
    def setUp(self):
        from aka.light.acceptance import numerical as numerical_policy
        self.module = numerical_policy
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        source = self.root / "kernel.py"
        source.write_text("self-authored candidate")
        self.files = {"candidate/kernel.py": source}
        self.digest = numerical_policy._digest(self.files)
        self.cancel = numerical_policy.NumericalCancellation(SimpleNamespace(closed=False))
        self.campaign = SimpleNamespace(workspace=self.root, production_review_timeout=120,
            agent_cli="pi", _account=Mock(), agent_boundary_environment=lambda label: {})

    def request(self):
        return self.module._request_review(self.campaign, self.root, self.files, self.digest,
            state_root=self.root / "evidence", cancel=self.cancel)

    def result(self, root, *, write=False, timed_out=False, exit_status=0):
        if write:
            (root / "numerical_review.json").write_text(json.dumps({"schema_version": 1,
                "evidence_digest": self.digest, "action": "complete", "summary": "Input contract examined"}))
        return SimpleNamespace(session_id="test", timed_out=timed_out, exit_status=exit_status, stdout_tail="")

    def test_missing_then_malformed_then_valid_artifact(self):
        def invoke(root, prompt, **kwargs):
            count = self.campaign._run_session.call_count
            if count == 2:
                (root / "numerical_review.json").write_text('{"schema_version": 999}')
            return self.result(root, write=count == 3)
        self.campaign._run_session = Mock(side_effect=invoke)
        self.assertEqual(self.request()["action"], "complete")
        self.assertEqual(self.request()["action"], "complete")
        self.assertEqual(self.campaign._run_session.call_count, 3)

    def test_terminal_cli_failure_without_plan_blocks_after_three_invocations(self):
        self.campaign._run_session = Mock(side_effect=lambda root, prompt, **kw:
                                          self.result(root, exit_status=1))
        for _ in range(2):
            with self.assertRaises(retry.ValidationBlocked):
                self.request()
        self.assertEqual(self.campaign._run_session.call_count, 3)
        self.assertEqual(inspect_health(self.root)["status"], "validation_blocked")

    def test_valid_artifact_survives_cli_timeout(self):
        self.campaign._run_session = Mock(side_effect=lambda root, prompt, **kw:
                                          self.result(root, write=True, timed_out=True))
        self.assertEqual(self.request()["action"], "complete")
        self.assertEqual(self.campaign._run_session.call_count, 1)


class MigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.workspace = self.root / "campaign"
        self.context = self.workspace / CONTEXT
        self.context.parent.mkdir(parents=True)
        self.old = {"argv": ["launch"], "problem": {"directory": str(self.root / "problem"), "identity": "frozen"},
                    "selection": "old-runtime"}
        self.context.write_text(json.dumps(self.old))
        self.target = self.root / "target-context.json"
        self.target.write_text(json.dumps(dict(self.old, selection="new-runtime")))
        self.plugins = self.root / "empty-plugins"
        self.plugins.mkdir()
        self.lock = supervisor_campaign_root(self.workspace) / "plugins/.atrex_plugins/lock.json"
        self.lock.parent.mkdir(parents=True)
        self.lock.write_text(json.dumps({"api_version": 1, "plugin_dir": "/old/empty-plugins", "plugins": []}))
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(migration, "restore"))
        self.stack.enter_context(patch.object(migration, "inspect"))
        self.stack.enter_context(patch.object(migration, "decode_launch", return_value=SimpleNamespace(
            request=SimpleNamespace(workspace=self.workspace))))

    def test_preview_apply_and_rollback_preserve_candidate(self):
        candidate = self.workspace / "kernel.py"
        candidate.write_text("preserved candidate")
        plan = migration.migrate(self.workspace, self.target, self.plugins)
        self.assertEqual(json.loads(self.context.read_text()), self.old)
        applied = migration.migrate(self.workspace, self.target, self.plugins, plan["plan_digest"])
        self.assertEqual(json.loads(self.context.read_text())["selection"], "new-runtime")
        migration.rollback(self.workspace, applied["backup"])
        self.assertEqual(json.loads(self.context.read_text()), self.old)
        self.assertEqual(candidate.read_text(), "preserved candidate")

    def test_changed_arguments_and_problem_refused(self):
        for field, value in (("argv", ["different"]), ("problem", {"identity": "different"})):
            self.target.write_text(json.dumps(dict(self.old, **{field: value})))
            with self.assertRaisesRegex(ValueError, "preserve"):
                migration.migrate(self.workspace, self.target, self.plugins)
            self.assertEqual(json.loads(self.context.read_text()), self.old)

    def test_changed_inputs_after_preview_refused(self):
        plan = migration.migrate(self.workspace, self.target, self.plugins)
        self.target.write_text(json.dumps(dict(self.old, selection="newer-runtime")))
        with self.assertRaisesRegex(ValueError, "changed after review"):
            migration.migrate(self.workspace, self.target, self.plugins, plan["plan_digest"])

    def test_active_writer_prevents_migration(self):
        with campaign_lease(self.workspace):
            with self.assertRaisesRegex(ValueError, "active writer"):
                migration.migrate(self.workspace, self.target, self.plugins)

    def test_nonempty_plugin_lock_cannot_be_relocated(self):
        self.lock.write_text(json.dumps({"api_version": 1, "plugin_dir": "/old", "plugins": [{"id": "plugin"}]}))
        with self.assertRaisesRegex(ValueError, "Nonempty"):
            migration.migrate(self.workspace, self.target, self.plugins)


if __name__ == "__main__":
    unittest.main()
