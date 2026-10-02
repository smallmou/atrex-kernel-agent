from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import MagicMock, patch

from long_horizon.campaign import LongHorizonCampaign
from long_horizon import journal
from long_horizon.git_episode import EpisodeWorktree
from long_horizon.models import EpisodeHandoff, SessionResult, SupervisorState
from long_horizon.store import CampaignStore
from orchestrator import optimize
from orchestrator.campaign import Campaign
from orchestrator.constants import TEST_RESULT_PREFIX
from orchestrator.hardware import framework_workspace_suffix
from orchestrator.optimization_policy import install_workspace_policy


class EconomyModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def campaign(self, **overrides) -> Campaign:
        values = dict(
            name="rmsnorm", kernel_demo=str(self.root / "source/reference.py"),
            platform="H20", arch="sm_90", framework="", work_dir=str(self.root),
            workspace_suffix="economy_h20", optimization_mode="economy",
        )
        values.update(overrides)
        return Campaign(**values)

    def seed_v0(self, campaign: Campaign) -> str:
        workspace = campaign.workspace
        workspace.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(workspace)], check=True)
        subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, check=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=workspace, check=True)
        for name in ("kernel.py", "reference.py"):
            (workspace / name).write_text("def run(x):\n    return x\n")
        (workspace / "memory").mkdir()
        (workspace / "memory/v0.json").write_text(json.dumps({"version": "v0"}))
        subprocess.run(["git", "add", "."], cwd=workspace, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
             "commit", "-qm", "V0: reference"], cwd=workspace, check=True,
        )
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=workspace, text=True).strip()

    def test_defaults_and_explicit_budget_survive_restart(self) -> None:
        campaign = self.campaign()
        self.assertEqual((campaign.max_iters, campaign.max_stall, campaign.fast_trials), (10, 2, 1))
        self.assertEqual(campaign.convert_after, 0)
        overridden = self.campaign(max_iters=17, max_stall=0, fast_trials=5, fast_episodes=0)
        self.assertEqual((overridden.max_iters, overridden.max_stall, overridden.fast_trials), (17, 0, 1))
        engine = LongHorizonCampaign(base_campaign=overridden, fast_episodes=0)
        stalled = SupervisorState(episodes=60, consecutive_without_promotion=20)
        self.assertEqual(engine._episode_mode(stalled), "fast")
        self.assertEqual(engine._episode_mode(stalled, {"mode": "full"}), "fast")
        self.assertEqual(engine._active_fast_trials({"fast_trials": 5}, episode_mode="fast"), 1)
        self.assertEqual(engine._episode_reasoning_effort(episode_mode="fast"), "medium")

    def test_existing_mode_defaults_and_episode_escalation(self) -> None:
        campaign = self.campaign(optimization_mode="leaderboard", framework="Triton")
        self.assertEqual((campaign.max_iters, campaign.max_stall, campaign.fast_trials), (20, 0, 5))
        engine = LongHorizonCampaign(base_campaign=campaign)
        self.assertEqual(engine._episode_mode(SupervisorState(episodes=2)), "full")
        self.assertEqual(engine._episode_reasoning_effort(episode_mode="full"), "max")

    def test_correctness_first_baseline_runs_once_and_resume_skips_pin(self) -> None:
        campaign = self.campaign()
        commit = self.seed_v0(campaign)
        self.assertEqual(campaign._framework_baseline_decision()[0], "run")
        (campaign.workspace / "framework_baseline.json").write_text(json.dumps({
            "version": "v1", "commit": commit,
            "kernel_blob": subprocess.check_output(
                ["git", "rev-parse", f"{commit}:kernel.py"], cwd=campaign.workspace, text=True,
            ).strip(),
        }))
        subprocess.run(["git", "add", "framework_baseline.json"], cwd=campaign.workspace, check=True)
        subprocess.run(
            ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
             "commit", "-qm", "V1: pin baseline"], cwd=campaign.workspace, check=True,
        )
        self.assertEqual(campaign._framework_baseline_decision()[0], "skip")

    def test_baseline_validates_full_coverage_without_speedup_or_policy_agent(self) -> None:
        campaign = self.campaign()
        campaign.workspace.mkdir(parents=True)
        (campaign.workspace / "memory").mkdir()
        (campaign.workspace / "memory/v0.json").write_text(json.dumps({
            "performance": {"latency_us_by_shape": {"a": 1.0, "b": 1.0}},
        }))
        result = {
            "all_pass": True, "latency_us_geomean": 10.0,
            "performance_score": 0.1, "latency_us_by_shape": {"a": 10.0, "b": 10.0},
        }
        completed = subprocess.CompletedProcess([], 0, TEST_RESULT_PREFIX + json.dumps(result), "")
        with patch("orchestrator.campaign._sandbox_command", return_value=completed) as run, \
             patch.object(campaign, "_production_kernel_violations", side_effect=AssertionError("unexpected reviewer")):
            measured, problem = campaign._framework_baseline_external_gates(1)
        self.assertEqual(problem, "")
        self.assertEqual(measured["performance_score"], 0.1)
        command = run.call_args.args[5]
        self.assertIn("--multi-seed", command)
        self.assertEqual(command[command.index("--multi-seed") + 1], "5")
        result["latency_us_by_shape"] = {"a": 10.0}
        completed.stdout = TEST_RESULT_PREFIX + json.dumps(result)
        with patch("orchestrator.campaign._sandbox_command", return_value=completed):
            self.assertIn("does not cover", campaign._framework_baseline_external_gates(1)[1])

    def test_framework_free_prompts_reuse_wiki_without_planning(self) -> None:
        campaign = self.campaign(fast_episode_ask_codex=True, v1_ask_qoder=True)
        baseline = campaign._framework_baseline_prompt(1)
        self.assertNotIn("{{", baseline)
        self.assertIn("Reuse a prototype for operator rmsnorm.", baseline)
        self.assertIn('"max_records":3,"max_bytes":12000', baseline)
        self.assertNotIn("Supervisor-selected implementation references", baseline)
        self.assertEqual(campaign._episode_plan_reviewers("fast"), ())
        self.assertEqual(campaign._framework_baseline_correctness_reviewers(), ())
        engine = LongHorizonCampaign(base_campaign=campaign)
        worktree = EpisodeWorktree(1, "base", "episode", self.root / "episode")
        prompt = engine._prompt(
            episode=1, version=2, worktree=worktree, journal_path=worktree.path / "journal.json",
            handoff_path=worktree.path / "handoff.json", live_memory_path=self.root / "live.json",
            conversion_pending=False, episode_mode="fast", resumed=True,
        )
        self.assertNotIn("{{", prompt)
        self.assertIn("plans/v1_economy_prototype.md", prompt)
        self.assertIn("no mandatory", prompt)
        self.assertIn("resuming", prompt)
        self.assertNotIn("trial1_plan.md", prompt)
        self.assertNotIn("exactly 5", prompt)

    def test_workspace_policy_is_isolated_from_other_modes(self) -> None:
        self.assertEqual(framework_workspace_suffix("", "H20", "economy"), "economy_h20")
        install_workspace_policy(self.root, "economy", "")
        self.assertIn("Optimization mode: economy", (self.root / "CLAUDE.md").read_text())
        with self.assertRaisesRegex(RuntimeError, "policy mismatch"):
            install_workspace_policy(self.root, "production", "Triton")

    def test_supervisor_stops_on_stall_iteration_and_token_budgets(self) -> None:
        cases = [(10, 2, 0, 2, "stall:"), (3, 0, 0, 3, "budget: max-iters"),
                 (10, 0, 20, 2, "budget: token-budget")]
        for index, (iters, stall, budget, expected_calls, reason) in enumerate(cases):
            with self.subTest(reason=reason):
                campaign = self.campaign(name=f"op{index}", max_iters=iters, max_stall=stall)
                self.seed_v0(campaign)
                CampaignStore.ensure_excluded(campaign.workspace)
                runner = MagicMock()

                def pivot(workspace, prompt, **kwargs):
                    runtime = kwargs["handoff_path"].parent
                    path = runtime / "journal.json"
                    journal.append_experiment(path, {
                        "name": "economy candidate", "result": "no improvement",
                        "wiki_usage_status": "not_queried",
                    })
                    journal.finalize(path, state="pivot", outcome={"summary": "no improvement"})
                    (runtime / "evaluations.jsonl").write_text(json.dumps({
                        "result": {"all_pass": False},
                    }) + "\n")
                    handoff = EpisodeHandoff("pivot")
                    kwargs["handoff_path"].write_text(json.dumps(handoff.as_dict()))
                    self.assertEqual(kwargs["completion_check"](handoff), "")
                    self.assertEqual(kwargs["reasoning_effort"], "medium")
                    self.assertIn("one small optimization", prompt)
                    return SessionResult(0, False, 10, "test-session", 0, handoff)

                runner.run.side_effect = pivot
                engine = LongHorizonCampaign(
                    base_campaign=campaign, max_version=iters, max_stall=stall,
                    token_budget=budget, session_runner=runner,
                )
                with patch("long_horizon.main_adapter.prepare_campaign"), \
                     patch("long_horizon.main_adapter.link_episode_runtime"):
                    self.assertTrue(engine.run().startswith(reason))
                self.assertEqual(runner.run.call_count, expected_calls)
                state = CampaignStore(campaign.workspace).load_state()
                self.assertEqual(state.tokens, 10 * expected_calls)
                self.assertEqual(state.episodes, expected_calls)
                memory = json.loads((campaign.workspace / f"memory/v{expected_calls}.json").read_text())
                self.assertEqual(memory["optimization"]["action_category"], "economy_long_horizon_episode")
                self.assertEqual(memory["profile_evidence"]["tool_used"], "none (economy mode)")

    def run_cli(self, extra: list[str]) -> tuple[dict, MagicMock]:
        fake = MagicMock()
        fake.workspace = self.root / "campaign"
        fake._generalized_memory_coverage_problem.return_value = ""
        with ExitStack() as stack:
            stack.enter_context(patch.object(optimize.shutil, "which", return_value="/bin/true"))
            stack.enter_context(patch.object(optimize, "_resolve_op", return_value={
                "name": "rmsnorm", "reference": str(self.root / "reference.py"),
                "op_dir": str(self.root),
            }))
            submodules = stack.enter_context(patch.object(optimize, "ensure_submodules"))
            for name in ("write_trace_retention_manifest", "configure_recovery",
                         "raise_if_environment_blocked", "check_ssh_environment", "signal_restart_ready"):
                stack.enter_context(patch.object(optimize, name))
            stack.enter_context(patch.object(optimize, "detect_arch", return_value="sm_90"))
            stack.enter_context(patch.object(optimize, "latest_version", return_value=0))
            stack.enter_context(patch.object(optimize, "read_memory", return_value={}))
            constructor = stack.enter_context(patch.object(optimize, "Campaign", return_value=fake))
            dispatch = stack.enter_context(patch.object(optimize, "dispatch_framework_campaigns", return_value=0))
            self.assertEqual(optimize._run_main([
                "--op-dir", str(self.root), "--platform", "H20", "--optimization-mode", "economy",
                "--workspace", str(self.root), *extra,
            ]), 0)
            dispatch.assert_not_called()
            submodules.assert_not_called()
            fake.ensure_framework_baseline.assert_called_once()
            fake.run.assert_called_once()
        return constructor.call_args.kwargs, fake

    def test_cli_without_framework_launches_one_campaign_and_honors_iters(self) -> None:
        values, _ = self.run_cli([])
        self.assertEqual((values["framework"], values["max_iters"], values["workspace_suffix"]), ("", 10, "economy_h20"))
        values, _ = self.run_cli(["--max-iters", "14"])
        self.assertEqual(values["max_iters"], 14)

    def test_ssh_accepts_one_economy_campaign_without_framework(self) -> None:
        values, _ = self.run_cli(["--sandbox-ssh", "gpu-host", "--sandbox-ssh-gpu", "0"])
        self.assertEqual(values["sandbox_ssh"], "gpu-host")
        self.assertEqual(values["sandbox_ssh_gpu"], 0)


if __name__ == "__main__":
    unittest.main()
