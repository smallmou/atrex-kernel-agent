"""Run with PYTHONPATH=src python3 tools/test_model_service_recovery.py."""
from dataclasses import replace
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from aka.contracts.agent import NativeInvocation, TokenUsage
from aka.legacy.agent.adapter import ClaudeAdapter, PiAdapter
from aka.legacy.agent.model_service import ModelServiceFailure
from aka.light.agent.execution_clock import current_budget
from aka.light.workflow.episode import LongHorizonCampaign
from aka.light.workflow.model_service import ModelServicePaused, ModelServiceRecovery
from aka.light.workflow.session import LongSessionRunner
from aka.light.workflow.store import CampaignStore


def pi_error(message='429 event: error data: {"code":"Throttling"}', **fields):
    return json.dumps({"type": "message_end", "message": {
        "role": "assistant", "stopReason": "error", "errorMessage": message, **fields}})


def claude_error(session="s", **fields):
    return json.dumps({"type": "assistant", "session_id": session,
                       "isApiErrorMessage": True, "message": {"content": [
                           {"type": "text", "text": "Request rejected (429): Throttling"}]}, **fields})


class Clock:
    now = 1000.0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds


class FakeAgent:
    def __init__(self, handoff, outputs):
        self.handoff = handoff
        self.outputs = iter(outputs)
        self.sessions = []
        self.calls = []

    def open_session(self, backend, workspace, session_id, environment):
        agent = self

        class Session:
            supports_resume = backend == "claude"

            def invoke(self, prompt, **kwargs):
                agent.calls.append((session_id, kwargs["resume"], workspace))
                result = next(agent.outputs)
                current_budget().used += 2.0
                if isinstance(result, BaseException):
                    raise result
                if result == "handoff":
                    agent.handoff.write_text(json.dumps({"status": "pivot"}))
                    result = ""
                elif isinstance(result, tuple):
                    agent.handoff.write_text(json.dumps({"status": "pivot"}))
                    result = result[0]
                return NativeInvocation(0, False, session_id, result, "",
                                        replace(TokenUsage.zero(), total_tokens=5), (),
                                        PiAdapter.capabilities, (), False)

            def close(self):
                self.closed = True

        session = Session()
        session.environment = environment or {}
        self.sessions.append(session)
        return session


class AdapterTests(unittest.TestCase):
    def test_pi_exit_zero_error_envelope(self):
        failure = PiAdapter().classify_model_service_failure(pi_error())
        self.assertEqual((failure.kind, failure.code), ("model_rate_limited", "throttling"))

    def test_quotes_and_tool_errors_are_not_provider_errors(self):
        for message in [
            {"role": "assistant", "stopReason": "stop", "content": "429 Throttling"},
            {"role": "assistant", "stopReason": "error", "content": "429 Throttling"},
            {"role": "toolResult", "stopReason": "error", "errorMessage": "429"},
        ]:
            with self.subTest(message=message):
                self.assertIsNone(PiAdapter().classify_model_service_failure(
                    json.dumps({"type": "message_end", "message": message})))

    def test_successful_terminal_message_supersedes_old_error(self):
        stream = pi_error() + '\n' + json.dumps({"type": "message_end", "message": {
            "role": "assistant", "stopReason": "stop"}})
        self.assertIsNone(PiAdapter().classify_model_service_failure(stream))

    def test_usage_count_429_is_not_http_status(self):
        self.assertIsNone(PiAdapter().classify_model_service_failure(
            pi_error("Connection closed", usage={"totalTokens": 429})))
        stream = claude_error(message={"usage": {"input_tokens": 429}, "content": [
            {"type": "text", "text": "Connection closed"}]})
        self.assertIsNone(ClaudeAdapter().classify_model_service_failure(stream, "s"))

    def test_quota_code_and_retry_after(self):
        failure = PiAdapter().classify_model_service_failure(
            pi_error("insufficient_quota", headers={"Retry-After": "3600"}))
        self.assertEqual(failure.kind, "model_quota_exhausted")
        self.assertEqual(failure.retry_after, 3600)
        # A generic provider quota message does not prove permanent quota exhaustion.
        self.assertEqual(PiAdapter().classify_model_service_failure(
            pi_error("429 Throttling: You exceeded your current quota")).kind, "model_rate_limited")

    def test_claude_root_error_only(self):
        adapter = ClaudeAdapter()
        self.assertIsNotNone(adapter.classify_model_service_failure(claude_error(), "s"))
        self.assertIsNone(adapter.classify_model_service_failure(claude_error("other"), "s"))
        self.assertIsNone(adapter.classify_model_service_failure(
            claude_error(parent_tool_use_id="child"), "s"))
        self.assertIsNone(adapter.classify_model_service_failure(
            claude_error(isApiErrorMessage=False), "s"))

    def test_legacy_usage_and_command_contract_unchanged(self):
        adapter = PiAdapter()
        stream = pi_error(usage={"input": 12, "output": 3})
        before = adapter.normalize_stream(stream)
        adapter.classify_model_service_failure(stream)
        self.assertEqual(adapter.normalize_stream(stream), before)
        self.assertEqual(before[1].total_tokens, 15)
        self.assertEqual(adapter.build_command("p", "s", "max", ""),
                         ["pi", "--mode", "json", "--session-id", "s", "--approve", "--thinking", "max", "p"])


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.handoff = self.root / "handoff.json"
        self.path = self.root / "active.json"
        self.clock = Clock()
        self.active = {"episode": 1, "memory_version": 2, "phase": "exploring"}
        self.recovery = self.controller(self.active)

    def controller(self, active):
        return ModelServiceRecovery(active, lambda value: self.path.write_text(json.dumps(value)),
                                    clock=self.clock, wait=self.clock.wait, jitter=lambda a, b: 1.0)

    def run_agent(self, outputs, backend="pi", recovery=None, **kwargs):
        agent = FakeAgent(self.handoff, outputs)
        self.agent = agent
        runner = LongSessionRunner(agent=agent, agent_cli=backend,
                                   usage_policy=lambda turn: turn.terminal_usage.total_tokens)
        result = runner.run(self.root, "Optimize kernel", handoff_path=self.handoff,
                            handoff_resumes=0, completion_check=lambda handoff: "",
                            model_service_recovery=recovery or self.recovery, **kwargs)
        return result

    def test_pi_recovers_same_episode_with_fresh_session_and_counts_tokens(self):
        receipts = []
        draft = self.root / "kernel.py"
        draft.write_text("preserved")
        result = self.run_agent([pi_error(), "handoff"], record_usage=receipts.append)
        self.assertEqual(result.handoff.status, "pivot")
        self.assertEqual(receipts, [5, 10])
        self.assertEqual(result.resume_count, 0)
        self.assertEqual(self.recovery.state["retry_count"], 1)
        self.assertEqual(self.recovery.state["active_seconds"], 4)
        self.assertEqual(self.clock.now, 1030)
        self.assertEqual(draft.read_text(), "preserved")
        self.assertNotEqual(self.agent.calls[0][0], self.agent.calls[1][0])
        self.assertEqual(self.agent.calls[0][2], self.agent.calls[1][2])
        self.assertTrue(all(session.closed for session in self.agent.sessions))

    def test_claude_resumes_without_handoff_resume_allowance(self):
        self.run_agent([claude_error(), "handoff"], backend="claude", session_id="s")
        self.assertEqual([call[:2] for call in self.agent.calls], [("s", False), ("s", True)])

    def test_valid_handoff_wins_over_provider_error(self):
        result = self.run_agent([(pi_error(),)])
        self.assertEqual(result.exit_status, 0)
        self.assertIsNotNone(result.handoff)
        self.assertNotIn("retry_count", self.recovery.state)

    def test_seven_failures_pause_and_restart_cannot_invoke(self):
        receipts = []
        with self.assertRaisesRegex(ModelServicePaused, "6 model-service retries"):
            self.run_agent([pi_error()] * 7, record_usage=receipts.append)
        self.assertEqual(len(self.agent.calls), 7)  # Initial call + six retries.
        self.assertEqual(receipts[-1], 35)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved["phase"], "paused_model_service")
        self.assertEqual(saved["memory_version"], 2)
        with self.assertRaises(ModelServicePaused):
            self.run_agent([], recovery=self.controller(saved))
        self.assertEqual(self.agent.calls, [])

    def test_restart_waits_existing_deadline_and_keeps_active_budget(self):
        self.recovery.checkpoint(session_id="s", active_seconds=8)
        self.recovery.failed(ModelServiceFailure("model_rate_limited", "429", 90))
        restored = self.controller(json.loads(self.path.read_text()))
        self.run_agent(["handoff"], backend="claude", recovery=restored)
        self.assertEqual(self.clock.now, 1090)
        self.assertEqual(self.agent.calls[0][:2], ("s", True))
        self.assertEqual(restored.state["active_seconds"], 10)
        self.assertEqual(restored.state["retry_count"], 1)

    def test_crash_after_reservation_does_not_replay_free_attempt(self):
        self.recovery.failed(ModelServiceFailure("model_rate_limited", "429"))
        self.recovery.before_retry()
        restored = self.controller(json.loads(self.path.read_text()))
        self.run_agent(["handoff"], recovery=restored)
        self.assertEqual(restored.state["retry_count"], 2)
        self.assertEqual(self.clock.now, 1090)

    def test_retry_after_beyond_deadline_pauses_without_call(self):
        with self.assertRaisesRegex(ModelServicePaused, "30-minute"):
            self.run_agent([pi_error(headers={"Retry-After": "3600"})])
        self.assertEqual(len(self.agent.calls), 1)

    def test_expired_restart_window_does_not_call(self):
        self.recovery.failed(ModelServiceFailure("model_rate_limited", "429"))
        self.clock.now += 1801
        with self.assertRaisesRegex(ModelServicePaused, "30-minute"):
            self.run_agent([], recovery=self.controller(json.loads(self.path.read_text())))
        self.assertEqual(self.agent.calls, [])

    def test_known_quota_exhaustion_does_not_retry(self):
        with self.assertRaisesRegex(ModelServicePaused, "quota exhausted"):
            self.run_agent([pi_error("insufficient_quota")])
        self.assertEqual(len(self.agent.calls), 1)

    def test_real_token_and_active_time_limits_remain_enforced(self):
        for limits in ({"remaining_token_budget": 5}, {"episode_timeout_seconds": 1}):
            with self.subTest(limits=limits):
                self.recovery = self.controller({"phase": "exploring"})
                with self.assertRaisesRegex(ModelServicePaused, "budget exhausted"):
                    self.run_agent([pi_error()], **limits)
                self.assertEqual(len(self.agent.calls), 1)

    def test_cancelled_wait_does_not_invoke(self):
        from concurrent.futures import CancelledError
        self.recovery.failed(ModelServiceFailure("model_rate_limited", "429"))
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(CancelledError):
            self.recovery.before_retry(cancel)
        self.assertNotIn("retry_count", self.recovery.state)

    def test_retry_cap_survives_non_service_results(self):
        self.recovery.state["retry_count"] = 6
        self.recovery.state["status"] = "retrying"
        self.recovery.observed_non_service_result()
        with self.assertRaisesRegex(ModelServicePaused, "6 model-service retries"):
            self.recovery.failed(ModelServiceFailure("model_rate_limited", "429"))

    def test_ordinary_missing_handoff_still_returns_invalid_result(self):
        result = self.run_agent([""])
        self.assertIsNone(result.handoff)
        self.assertNotIn("retry_count", self.recovery.state)

    def test_paused_campaign_rejects_restart_before_preparation(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        store = CampaignStore(self.root)
        store.save_active({"phase": "paused_model_service", "model_service": {
            "status": "paused", "reason": "6 model-service retries exhausted", "retry_count": 6}})
        campaign = LongHorizonCampaign(SimpleNamespace(workspace=self.root))
        with patch("aka.light.workflow.episode.main_adapter.prepare_campaign") as prepare:
            with self.assertRaises(ModelServicePaused):
                campaign.run()
            prepare.assert_not_called()
        self.assertEqual(store.load_state().episodes, 0)

    def test_fresh_campaign_does_not_create_store_before_preparation(self):
        workspace = self.root / "not-created"
        campaign = LongHorizonCampaign(SimpleNamespace(workspace=workspace))
        with patch("aka.light.workflow.episode.main_adapter.prepare_campaign",
                   side_effect=RuntimeError("preparation reached")):
            with self.assertRaisesRegex(RuntimeError, "preparation reached"):
                campaign.run()
        self.assertFalse(workspace.exists())

    def test_campaign_pause_never_records_episode_or_advances_version(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        store = CampaignStore(self.root)
        draft = self.root / "draft"
        runtime = draft / ".atrex_long_horizon"
        runtime.mkdir(parents=True)
        (runtime / "journal.json").write_text("{}")
        self.handoff = runtime / "handoff.json"
        (draft / "kernel.py").write_text("candidate remains")
        active = {"episode": 1, "memory_version": 2, "phase": "exploring"}
        store.save_active(active)
        worktree = SimpleNamespace(path=draft, episode=1, base_commit="base", branch="episode-1")
        agent = FakeAgent(self.handoff, [pi_error()] * 7)
        runner = LongSessionRunner(agent=agent, agent_cli="pi",
                                   usage_policy=lambda turn: turn.terminal_usage.total_tokens)

        def run(*args, **kwargs):
            recovery = kwargs["model_service_recovery"]
            recovery.clock, recovery.wait = self.clock, self.clock.wait
            recovery.jitter = lambda a, b: 1.0
            return runner.run(*args, **kwargs)

        base = SimpleNamespace(workspace=self.root, module_options=SimpleNamespace(
            goal_episodes=False, episode_timeout_seconds=100),
            register_runtime_episode=Mock(return_value=draft), agent_environment=lambda: {})
        campaign = LongHorizonCampaign(base, verifier=Mock(), max_version=30,
                                      session_runner=SimpleNamespace(run=run))
        with ExitStack() as stack:
            for name, value in (("prepare_campaign", None), ("restored_stall", 0),
                                ("conversion_required", False), ("latest_version", 1)):
                stack.enter_context(patch("aka.light.workflow.episode.main_adapter." + name,
                                          return_value=value))
            stack.enter_context(patch.object(campaign, "_recover_interrupted", return_value=(worktree, active)))
            stack.enter_context(patch.object(campaign, "_prompt", return_value="optimize"))
            record = stack.enter_context(patch.object(campaign, "_record_terminal_episode"))
            with self.assertRaises(ModelServicePaused):
                campaign.run()
            record.assert_not_called()
        state = store.load_state()
        self.assertEqual((state.episodes, state.protocol_failures, state.tokens), (0, 0, 35))
        self.assertEqual(state.attempts, [])
        self.assertEqual(store.load_active()["memory_version"], 2)
        self.assertEqual(store.load_active()["phase"], "paused_model_service")
        self.assertEqual((draft / "kernel.py").read_text(), "candidate remains")

    def test_existing_worktree_recovery_preserves_wait_checkpoint(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        (self.root / "kernel.py").write_text("baseline")
        subprocess.run(["git", "add", "kernel.py"], cwd=self.root, check=True)
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "baseline"],
                       cwd=self.root, check=True)
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.root, text=True).strip()
        draft = self.root / "draft"
        subprocess.run(["git", "worktree", "add", "-qb", "episode-1", str(draft)],
                       cwd=self.root, check=True)
        (draft / "kernel.py").write_text("unfinished candidate")
        store = CampaignStore(self.root)
        active = {"episode": 1, "memory_version": 2, "phase": "waiting_model_service",
                  "mode": "episode", "base_commit": head, "episode_branch": "episode-1",
                  "worktree": str(draft), "model_service": {"status": "waiting", "retry_count": 3,
                      "first_failure_at": 1000, "next_retry_at": 1240}}
        store.save_active(active)
        handle = SimpleNamespace(path=draft)
        campaign = LongHorizonCampaign(SimpleNamespace(workspace=self.root,
            candidate_workspace=SimpleNamespace(restore_handle=Mock(return_value=handle))))
        with patch.object(campaign, "_recover_completed_handoff", return_value=False):
            recovered, saved = campaign._recover_interrupted(store, store.load_state(), verifier=Mock())
        self.assertIs(recovered, handle)
        self.assertEqual(saved["phase"], "waiting_model_service")
        self.assertEqual(saved["model_service"], active["model_service"])
        self.assertEqual((draft / "kernel.py").read_text(), "unfinished candidate")


if __name__ == "__main__":
    unittest.main()
