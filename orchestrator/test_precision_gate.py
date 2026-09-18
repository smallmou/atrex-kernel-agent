"""Behaviour tests for the precision gate and the policy plugin behind it.

The gate's policy moved out of `orchestrator/numerical_policy.py`,
`orchestrator/numerical_suite.py` and `long_horizon/remote_numerical.py` into
`plugins/precision-validation/`. That move is only safe if it is *identical by
construction*, so what this file pins is the policy itself, not the plumbing:

* the suite schema's accept/reject boundary, including the rule that a suite needs two
  independent construction families rather than two seeds -- the single property that
  distinguishes this gate from re-running the ordinary evaluator;
* the probe schedule's shape and seed selection for both depths, its determinism under
  a rotation key, and the fact that the largest declared workload is never sampled out;
* the evidence checks: a run that omits, reorders, under-covers or silently downgrades
  a planned probe must be rejected, and so must a review that is unbound, incomplete,
  uncited or self-contradicting;
* the `Cuda` guard-profile downgrade, which exists because `--trust-mode untrusted`
  disables the runtime extension loading a CUDA candidate compiles through.

These validators had no coverage at all before the move. Every one of them is pure, so
none of this needs a GPU, an agent session or a workspace.

The plugin lives in a directory whose name contains a dash, so it can never be an
importable package; its modules are loaded by path here, which has the side benefit of
testing the real shipped files rather than a copy. `PluginContractTest` additionally
drives the tools through `PluginRegistry` as real subprocesses, so the JSON schemas are
exercised by the same validator that guards them at runtime.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from orchestrator.campaign import Campaign
from orchestrator.constants import REPO_ROOT

PLUGIN_ROOT = REPO_ROOT / "plugins" / "precision-validation"
CHECK_IDS = ("input_domain", "precision_and_reductions", "nonlinear_and_quantization",
             "routing_and_boundaries", "distribution_coverage")


def _load_module(name: str, path: Path):
    """Load a module by path: the plugin directory's dash makes it unimportable."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(name: str):
    return _load_module(f"_precision_{name}", PLUGIN_ROOT / f"{name}.py")


driver = _load("driver")
adapter = _load("adapter")


def make_suite(*, cases=3, coverage="compact", world_size=1, seeds=(1729, 104729), pinned=None):
    """A minimally valid suite: two independent families, distinct seeds, distinct ids."""
    families = (
        {"generator": "log_uniform", "min_exp": -8, "max_exp": 8},
        {"generator": "sparse", "low": -1.0, "high": 1.0, "density": 0.1},
        {"generator": "uniform", "low": -1.0, "high": 1.0},
        {"generator": "ramp", "low": 0.0, "high": 1.0},
        {"generator": "alternating", "amplitude": 3.0},
    )
    rows = []
    for index in range(cases):
        row = {
            "id": f"c{index}",
            "purpose": "a stated numerical risk",
            "fields": {"a": families[index % len(families)],
                       "b": families[(index + 1) % len(families)]},
        }
        if pinned and index in pinned:
            row["shape_ids"] = pinned[index]
        rows.append(row)
    return {"schema_version": 1, "world_size": world_size, "coverage": coverage,
            "seeds": list(seeds), "cases": rows}


def make_shapes(count):
    return {f"s{i}": {"input_kwargs": {"n": 2 ** i, "m": 3 * i + 1}} for i in range(count)}


class SuiteSchemaTest(unittest.TestCase):
    """The accept/reject boundary of an operator-owned numerical suite."""

    def test_minimal_valid_suite_is_accepted(self):
        suite = make_suite()
        self.assertIs(driver.validate_suite(suite), suite)

    def test_exhaustive_coverage_is_accepted(self):
        driver.validate_suite(make_suite(coverage="exhaustive"))

    def assert_rejected(self, suite, fragment):
        with self.assertRaises(ValueError) as caught:
            driver.validate_suite(suite)
        self.assertIn(fragment, str(caught.exception))

    def test_unsupported_schema_version(self):
        self.assert_rejected({"schema_version": 2}, "unsupported numerical suite schema")

    def test_world_size_must_be_a_positive_int(self):
        for value in (0, -1, "2", 2.0, True):
            self.assert_rejected({**make_suite(), "world_size": value}, "positive world_size")

    def test_seeds_must_be_at_least_two_distinct_ints(self):
        for seeds in ([1729], [1729, 1729], [1729, "104729"], []):
            self.assert_rejected({**make_suite(), "seeds": list(seeds)}, "two distinct integer seeds")

    def test_boolean_seeds_are_not_integers(self):
        # `type(s) is not int` is deliberate: bool is an int subclass and would slip past
        # isinstance, seeding every probe from the same two values.
        self.assert_rejected({**make_suite(), "seeds": [True, False]}, "two distinct integer seeds")

    def test_at_least_three_distinct_cases(self):
        self.assert_rejected(make_suite(cases=2), "three distinct distribution cases")
        duplicated = make_suite(cases=3)
        duplicated["cases"][1]["id"] = duplicated["cases"][0]["id"]
        self.assert_rejected(duplicated, "three distinct distribution cases")

    def test_every_case_needs_a_purpose_and_fields(self):
        for key in ("purpose", "fields"):
            suite = make_suite()
            suite["cases"][1][key] = "" if key == "purpose" else {}
            self.assert_rejected(suite, "purpose and explicit input rules")

    def test_unsupported_generator(self):
        suite = make_suite()
        suite["cases"][0]["fields"]["a"] = {"generator": "gaussian"}
        self.assert_rejected(suite, "unsupported numerical generator: gaussian")

    def test_preserved_inputs_need_a_written_justification(self):
        for reason in ("", "   ", 7):
            suite = make_suite()
            suite["cases"][0]["preserve"] = {"mask": reason}
            self.assert_rejected(suite, "operator-contract justification")

    def test_seeds_alone_are_not_independent_families(self):
        """The property that separates this gate from more seeds of the same generator."""
        degenerate = make_suite()
        for case in degenerate["cases"]:
            case["fields"] = {"a": {"generator": "constant", "value": 1},
                              "b": {"generator": "near_constant", "center": 0.0, "amplitude": 1e-6}}
        self.assert_rejected(degenerate, "independent construction families, not just seeds")

    def test_one_real_family_is_not_enough(self):
        suite = make_suite()
        for case in suite["cases"]:
            case["fields"] = {"a": {"generator": "uniform", "low": 0.0, "high": 1.0},
                              "b": {"generator": "packed_bytes"}}
        self.assert_rejected(suite, "independent construction families, not just seeds")

    def test_unsupported_coverage(self):
        self.assert_rejected({**make_suite(), "coverage": "sampled"},
                             "coverage must be compact or exhaustive")


class ScheduleTest(unittest.TestCase):
    """Shape and seed selection per depth, and its determinism."""

    def test_light_takes_largest_plus_one_rotating_shape(self):
        suite, shapes = make_suite(), make_shapes(6)
        rows = driver.validation_schedule(suite, shapes, "rot", "light")
        self.assertEqual([len(r["shape_ids"]) for r in rows], [2, 2, 2])
        for row in rows:
            self.assertIn("s5", row["shape_ids"], "the largest workload is never sampled out")
            self.assertEqual(len(set(row["shape_ids"])), len(row["shape_ids"]))

    def test_thorough_takes_both_extremes_plus_a_middle(self):
        suite, shapes = make_suite(), make_shapes(6)
        rows = driver.validation_schedule(suite, shapes, "rot", "thorough")
        for row in rows:
            self.assertEqual(len(row["shape_ids"]), 3)
            self.assertIn("s0", row["shape_ids"])
            self.assertIn("s5", row["shape_ids"])

    def test_light_repeats_only_the_first_risk_thorough_repeats_every_risk(self):
        suite, shapes = make_suite(cases=4), make_shapes(4)
        light = driver.validation_schedule(suite, shapes, "rot", "light")
        self.assertEqual([len(r["seeds"]) for r in light], [2, 1, 1, 1])
        thorough = driver.validation_schedule(suite, shapes, "rot", "thorough")
        self.assertEqual([len(r["seeds"]) for r in thorough], [2, 2, 2, 2])

    def test_exhaustive_covers_every_shape_and_every_seed(self):
        suite = make_suite(coverage="exhaustive", seeds=(1, 2, 3))
        shapes = make_shapes(5)
        for mode in ("light", "thorough"):
            for row in driver.validation_schedule(suite, shapes, "rot", mode):
                self.assertEqual(sorted(row["shape_ids"]), sorted(shapes))
                self.assertEqual(row["seeds"], [1, 2, 3])

    def test_a_single_shape_is_used_without_rotation(self):
        rows = driver.validation_schedule(make_suite(), make_shapes(1), "rot", "thorough")
        self.assertEqual([r["shape_ids"] for r in rows], [["s0"]] * 3)

    def test_pinned_regression_shapes_override_sampling(self):
        suite = make_suite(pinned={1: ["s2"]})
        rows = driver.validation_schedule(suite, make_shapes(5), "rot", "thorough")
        self.assertEqual(rows[1]["shape_ids"], ["s2"])

    def test_pinned_shapes_must_exist_and_be_distinct(self):
        for pinned in ({0: ["nope"]}, {0: ["s1", "s1"]}, {0: []}):
            with self.assertRaises(ValueError) as caught:
                driver.validation_schedule(make_suite(pinned=pinned), make_shapes(3), "rot", "light")
            self.assertIn("invalid regression shape selection", str(caught.exception))

    def test_rotation_is_deterministic_and_actually_rotates(self):
        suite, shapes = make_suite(), make_shapes(8)
        first = driver.validation_schedule(suite, shapes, "digest-a", "light")
        self.assertEqual(first, driver.validation_schedule(suite, shapes, "digest-a", "light"))
        selections = {
            tuple(r["shape_ids"]) for key in ("a", "b", "c", "d", "e", "f")
            for r in driver.validation_schedule(suite, shapes, key, "light")
        }
        self.assertGreater(len(selections), 1, "the rotating slot must vary across candidates")

    def test_selection_digest_binds_case_shapes_and_seeds(self):
        shapes = make_shapes(4)
        base = driver.validation_schedule(make_suite(), shapes, "rot", "light")
        other = driver.validation_schedule(make_suite(), shapes, "rot", "thorough")
        self.assertNotEqual([r["selection_digest"] for r in base],
                            [r["selection_digest"] for r in other])

    def test_empty_shapes_and_unknown_mode_are_rejected(self):
        with self.assertRaises(ValueError):
            driver.validation_schedule(make_suite(), {}, "rot", "light")
        with self.assertRaises(ValueError):
            driver.validation_schedule(make_suite(), make_shapes(2), "rot", "medium")

    def test_expected_probes_counts_distinct_signatures_across_ranks(self):
        suite = make_suite(world_size=4)
        shapes = {"a": {"input_kwargs": {"n": 8}}, "b": {"input_kwargs": {"n": 8}},
                  "c": {"input_kwargs": {"n": 16}}}
        row = {"shape_ids": ["a", "b", "c"], "seeds": [1, 2]}
        # a and b share one input signature, so they are one probe, not two.
        self.assertEqual(driver.expected_probes(row, suite, shapes), 2 * 2 * 4)


class TrustModeTest(unittest.TestCase):
    """`untrusted` hardens the evaluator but blocks runtime extension builds."""

    def test_untrusted_is_kept_for_dsl_frameworks(self):
        for key in ("triton", "cutedsl", "tilelang", "flydsl", "gluon", ""):
            self.assertEqual(driver.resolve_trust_mode("untrusted", key), "untrusted")

    def test_cuda_is_downgraded_because_it_jits_native_extensions(self):
        self.assertEqual(driver.resolve_trust_mode("untrusted", "cuda"), "trusted")

    def test_an_explicit_trusted_request_is_never_upgraded(self):
        self.assertEqual(driver.resolve_trust_mode("trusted", "triton"), "trusted")

    def test_unknown_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            driver.resolve_trust_mode("permissive", "triton")

    def test_an_unnormalized_token_is_rejected_rather_than_guessed(self):
        """A raw --framework string must never reach this function."""
        for raw in ("Cuda", " CUDA ", "CUDA C"):
            with self.assertRaises(ValueError):
                driver.resolve_trust_mode("untrusted", raw)

    def test_the_downgrade_agrees_with_the_directive_for_every_spelling(self):
        """The guard profile and the promise made to the agent must not disagree.

        `optimization_policy._framework_key` folds `Cuda`, `CUDA C` and `cuda-c` to one
        token; matching the raw string here instead would run a CUDA candidate under a
        profile that blocks the extension build it needs, while the directive stayed silent.
        """
        from orchestrator.optimization_policy import _framework_key, optimization_mode_directive
        for raw in ("Triton", "CuteDSL", "Cute", "TileLang", "FlyDSL", "Fly", "Gluon",
                    "Cuda", "cuda", "CUDA C", "cuda-c", "CUDA C++"):
            key = _framework_key(raw)
            downgraded = driver.resolve_trust_mode("untrusted", key) == "trusted"
            promised = "anti-tampering guard profile" in optimization_mode_directive("production", raw)
            self.assertEqual(downgraded, not promised, f"{raw!r} -> {key!r}")


class PlanTest(unittest.TestCase):
    def setUp(self):
        self.shapes = make_shapes(5)
        self.request = {"schema_version": 1, "suite": make_suite(world_size=2),
                        "shapes": self.shapes, "mode": "light", "rotation": "rot",
                        "framework_key": "triton", "trust_mode": "untrusted"}

    def test_coverage_receipt_matches_the_schedule(self):
        plan = adapter.plan(self.request)
        coverage = plan["coverage"]
        self.assertEqual(coverage["total_shapes"], 5)
        self.assertEqual(coverage["world_size"], 2)
        self.assertEqual(coverage["mode"], "light")
        self.assertEqual(coverage["coverage"], "compact")
        self.assertEqual(coverage["selected_shapes_per_case"],
                         [len(r["shape_ids"]) for r in plan["schedule"]])
        self.assertEqual(
            coverage["planned_rank_probes"],
            sum(len(r["shape_ids"]) * len(r["seeds"]) * 2 for r in plan["schedule"]),
        )

    def test_plan_rejects_an_invalid_suite(self):
        with self.assertRaises(ValueError):
            adapter.plan({**self.request, "suite": make_suite(cases=2)})

    def test_plan_reports_the_resolved_guard_profile(self):
        self.assertEqual(adapter.plan(self.request)["trust_mode"], "untrusted")
        self.assertEqual(adapter.plan({**self.request, "framework_key": "cuda"})["trust_mode"], "trusted")


class EvaluationEvidenceTest(unittest.TestCase):
    """A run must discharge exactly the plan it was given."""

    def setUp(self):
        self.suite = make_suite(world_size=2)
        self.shapes = make_shapes(4)
        self.plan = adapter.plan({"schema_version": 1, "suite": self.suite, "shapes": self.shapes,
                                  "mode": "light", "rotation": "rot", "framework_key": "triton"})
        self.schedule = self.plan["schedule"]

    def rows(self):
        return [{"case_id": row["case_id"], "passed": True, "exit_code": 0,
                 "expected_probes": row["expected_probes"],
                 "observed_probes": row["expected_probes"],
                 "selection_digest": row["selection_digest"],
                 "shape_count": len(row["shape_ids"]), "seeds": row["seeds"],
                 "world_size": 2, "result": {"all_pass": True}, "numerical_metrics": {}}
                for row in self.schedule]

    def check(self, runs, **overrides):
        payload = {"schema_version": 1, "all_pass": True, "runs": runs, **overrides}
        return adapter.check_evaluation({"schema_version": 1, "evaluation": payload,
                                         "schedule": self.schedule, "world_size": 2})

    def test_a_complete_run_is_accepted(self):
        verdict = self.check(self.rows())
        self.assertTrue(verdict["accepted"])
        self.assertEqual(verdict["violations"], [])

    def test_aggregate_failure_or_error_is_rejected(self):
        self.assertFalse(self.check(self.rows(), all_pass=False)["accepted"])
        self.assertFalse(self.check(self.rows(), error="driver blew up")["accepted"])
        self.assertFalse(self.check(self.rows(), schema_version=2)["accepted"])

    def test_a_non_object_payload_is_rejected(self):
        for payload in (None, [], "ok"):
            verdict = adapter.check_evaluation({"schema_version": 1, "evaluation": payload,
                                               "schedule": self.schedule, "world_size": 2})
            self.assertFalse(verdict["accepted"])

    def test_omitted_reordered_or_duplicated_cases_are_rejected(self):
        for mutate in (lambda r: r.pop(), lambda r: r.reverse(),
                       lambda r: r.append(dict(r[0]))):
            runs = self.rows()
            mutate(runs)
            verdict = self.check(runs)
            self.assertFalse(verdict["accepted"])
            self.assertIn("omitted, duplicated or reordered", verdict["violations"][0])

    def test_every_row_field_is_load_bearing(self):
        mutations = {
            "passed": lambda row: row.update(passed=False),
            "exit_code": lambda row: row.update(exit_code=1),
            "missing result": lambda row: row.update(result=None),
            "result not passing": lambda row: row.update(result={"all_pass": False}),
            "under-covered": lambda row: row.update(observed_probes=row["observed_probes"] - 1),
            "inflated expectation": lambda row: row.update(expected_probes=999),
            "shape count": lambda row: row.update(shape_count=row["shape_count"] + 1),
            "seeds": lambda row: row.update(seeds=[]),
            "selection digest": lambda row: row.update(selection_digest="forged"),
            "world size": lambda row: row.update(world_size=1),
        }
        for label, mutate in mutations.items():
            with self.subTest(field=label):
                runs = self.rows()
                mutate(runs[1])
                verdict = self.check(runs)
                self.assertFalse(verdict["accepted"])
                self.assertIn(f"incomplete numerical evidence for {runs[1]['case_id']}",
                              verdict["violations"])


class ReviewVerdictTest(unittest.TestCase):
    """An independent review must be bound, complete, cited and self-consistent."""

    SUPPLIED = ("candidate/kernel.py", "trusted/reference.py", "evaluation.json",
                "numerical_suite.json")

    def review(self, *, verdict="allow", reject=(), evidence=None, digest="D", ids=CHECK_IDS,
               reason="an evidence-based assessment"):
        cited = ["candidate/kernel.py:12", "trusted/reference.py:40"] if evidence is None else evidence
        return {"schema_version": 1, "evidence_digest": digest, "verdict": verdict,
                "checks": [{"id": name,
                            "decision": "reject" if name in reject else "allow",
                            "reason": reason, "evidence": list(cited)} for name in ids]}

    def check(self, review, digest="D"):
        return adapter.check_review({"schema_version": 1, "review": review,
                                     "evidence_digest": digest,
                                     "supplied_files": list(self.SUPPLIED)})

    def test_a_complete_allow_is_accepted(self):
        verdict = self.check(self.review())
        self.assertTrue(verdict["accepted"])
        self.assertEqual(verdict["violations"], [])

    def test_rejections_become_violations_naming_the_check(self):
        verdict = self.check(self.review(verdict="reject", reject=("input_domain",),
                                         reason="unbounded exp argument"))
        self.assertFalse(verdict["accepted"])
        self.assertEqual(verdict["violations"],
                         ["numerical safety input_domain: unbounded exp argument"])

    def test_every_rejection_is_reported_not_just_the_first(self):
        verdict = self.check(self.review(verdict="reject",
                                         reject=("input_domain", "distribution_coverage")))
        self.assertEqual(len(verdict["violations"]), 2)

    def test_a_review_bound_to_other_evidence_is_rejected(self):
        verdict = self.check(self.review(digest="OTHER"))
        self.assertFalse(verdict["accepted"])
        self.assertIn("bound to different evidence", verdict["violations"][0])

    def test_a_verdict_disagreeing_with_its_checks_is_rejected(self):
        for review in (self.review(verdict="allow", reject=("input_domain",)),
                       self.review(verdict="reject")):
            verdict = self.check(review)
            self.assertFalse(verdict["accepted"])
            self.assertIn("verdict disagrees with checks", verdict["violations"][0])

    def test_missing_or_duplicated_checks_are_rejected(self):
        for ids in (CHECK_IDS[:4], CHECK_IDS + (CHECK_IDS[0],), ()):
            verdict = self.check(self.review(ids=ids))
            self.assertFalse(verdict["accepted"])
            self.assertIn("omitted or duplicated required checks", verdict["violations"][0])

    def test_citations_must_reach_the_candidate_and_the_contract(self):
        for evidence in ([], ["trusted/reference.py:1"], ["candidate/kernel.py:1"],
                         ["candidate/kernel.py:1", "secret/notes.txt:1"],
                         ["candidate/kernel.py", "trusted/reference.py:1"]):
            verdict = self.check(self.review(evidence=evidence))
            self.assertFalse(verdict["accepted"])
            self.assertIn("lacks source and contract/evaluation evidence", verdict["violations"][0])

    def test_evaluation_and_suite_citations_satisfy_the_contract_side(self):
        for contract in ("evaluation.json:runs", "numerical_suite.json:cases"):
            verdict = self.check(self.review(evidence=["candidate/kernel.py:1", contract]))
            self.assertTrue(verdict["accepted"], contract)

    def test_a_blank_reason_or_unknown_decision_is_rejected(self):
        self.assertFalse(self.check(self.review(reason="   "))["accepted"])
        review = self.review()
        review["checks"][0]["decision"] = "maybe"
        self.assertFalse(self.check(review)["accepted"])

    def test_a_non_object_review_is_rejected(self):
        for payload in (None, [], "allow", {"schema_version": 9}):
            self.assertFalse(self.check(payload)["accepted"])


class WorkspacePolicyProseTest(unittest.TestCase):
    """The production directive is injected into every workspace CLAUDE.md, so it must be true."""

    def directive(self, framework):
        from orchestrator.optimization_policy import optimization_mode_directive
        return optimization_mode_directive("production", framework)

    def test_the_gate_still_promises_independent_distribution_families(self):
        text = self.directive("Triton")
        self.assertIn("independent distribution families", text)
        self.assertIn("all required ranks", text)
        self.assertIn("Changing seeds in the ordinary generator", text)

    def test_the_extension_restriction_is_promised_only_where_it_holds(self):
        for framework in ("Triton", "CuteDSL", "TileLang", "FlyDSL"):
            self.assertIn("anti-tampering guard profile", self.directive(framework), framework)
        # A Cuda candidate is evaluated under the trusted profile, so promising it the
        # restriction would be a false statement about the gate.
        self.assertNotIn("anti-tampering guard profile", self.directive("Cuda"))

    def test_leaderboard_mode_makes_no_precision_claim(self):
        from orchestrator.optimization_policy import optimization_mode_directive
        self.assertNotIn("numerical", optimization_mode_directive("leaderboard", "Triton"))


class PluginContractTest(unittest.TestCase):
    """The shipped manifest, schemas and subprocess contract, through the real registry."""

    @classmethod
    def setUpClass(cls):
        from orchestrator.plugins import PluginRegistry
        cls.registry = PluginRegistry()
        cls.plugin = next(p for p in cls.registry.plugins if p.id == "precision-validation")

    def test_the_three_policy_tools_are_enabled(self):
        names = [tool["name"] for tool in self.registry.catalog()]
        for tool in ("plan", "check-evaluation", "check-review"):
            self.assertIn(f"precision-validation.{tool}", names)

    def test_the_evaluator_is_declared_but_never_mounted(self):
        """`atrex-bench` is a reserved workspace mount, and the workspace copy is deliberate."""
        self.assertNotIn("atrex-bench", self.registry.mounts())
        for name in ("atrex-bench-runtime", "atrex-bench-runner"):
            path, mount = self.plugin.resources[name]
            self.assertFalse(mount)
            self.assertTrue(
                path.is_relative_to(REPO_ROOT / "3rdparty" / "atrex-bench"),
                f"{name} must resolve inside the vendored submodule",
            )

    def test_instructions_do_not_teach_the_agent_to_invoke_the_gate(self):
        text = (self.plugin.root / "instructions.md").read_text()
        self.assertNotIn("tools/plugin.py", text)
        self.assertIn("you do not invoke", text)

    def test_plan_round_trips_through_the_schema_validator(self):
        with tempfile.TemporaryDirectory() as workspace:
            plan = self.registry.call(
                "precision-validation.plan",
                {"schema_version": 1, "suite": make_suite(world_size=2),
                 "shapes": make_shapes(4), "mode": "thorough", "rotation": "rot",
                 "framework_key": "triton", "trust_mode": "untrusted"},
                Path(workspace),
            )
        self.assertEqual(plan["schema_version"], 1)
        self.assertEqual(len(plan["schedule"]), 3)
        self.assertEqual(plan["trust_mode"], "untrusted")

    def test_an_input_outside_the_schema_never_reaches_the_tool(self):
        from plugin_runtime.schema import PluginError
        with tempfile.TemporaryDirectory() as workspace:
            for bad in ({"schema_version": 1, "suite": {}},                    # missing shapes
                        {"schema_version": 2, "suite": {}, "shapes": {}},      # unsupported version
                        {"schema_version": 1, "suite": {}, "shapes": {}, "mode": "medium"},
                        {"schema_version": 1, "suite": {}, "shapes": {}, "extra": 1}):
                with self.assertRaises(PluginError) as caught:
                    self.registry.call("precision-validation.plan", bad, Path(workspace))
                self.assertEqual(caught.exception.code, "schema_validation")


class PluginTamperTest(unittest.TestCase):
    """The gate re-fingerprints the policy it is about to run."""

    def test_the_gate_does_not_reuse_the_campaigns_cached_registry(self):
        """`campaign.plugin_registry` is a cached_property fingerprinted at campaign start.

        The accept/reject policy is now a subprocess re-read from disk on every call, and
        `plugins/` is reachable from an agent workspace, so trusting the cached snapshot
        would validate a lock the plugin no longer matches.
        """
        from orchestrator import precision_gate

        class HostileCampaign:
            @property
            def plugin_registry(self):
                raise AssertionError(
                    "the gate must not consult the campaign's cached registry")

        campaign = HostileCampaign()
        with tempfile.TemporaryDirectory() as workspace:
            registry, plugin = precision_gate._plugin(campaign, Path(workspace))
        self.assertEqual(plugin.id, "precision-validation")
        self.assertTrue(plugin.fingerprint)

    def test_a_lock_that_disagrees_with_the_plugin_on_disk_blocks_the_gate(self):
        import json as _json
        from plugin_runtime.registry import STATE_DIR
        from plugin_runtime.schema import PluginError
        from orchestrator import precision_gate
        from orchestrator.plugins import PluginRegistry

        snapshot = PluginRegistry().snapshot()
        for row in snapshot["plugins"]:
            if row["id"] == "precision-validation":
                row["fingerprint"] = "0" * 64
        with tempfile.TemporaryDirectory() as workspace:
            state = Path(workspace) / STATE_DIR
            state.mkdir(parents=True)
            (state / "lock.json").write_text(_json.dumps(snapshot, indent=2) + "\n")
            with self.assertRaises(PluginError) as caught:
                precision_gate._plugin(SimpleNamespace(), Path(workspace))
        self.assertEqual(caught.exception.code, "plugin_changed")

    def test_a_missing_vendored_evaluator_blocks_the_gate(self):
        """The resources are declared optional so discovery survives a shallow clone."""
        import dataclasses
        from orchestrator import precision_gate
        from orchestrator.plugins import PluginRegistry

        real = PluginRegistry()
        plugin = next(p for p in real.plugins if p.id == "precision-validation")
        with tempfile.TemporaryDirectory() as workspace:
            absent = dict(plugin.resources)
            absent["atrex-bench-runner"] = (Path(workspace) / "gone" / "run_eval.py", False)
            stripped = dataclasses.replace(plugin, resources=absent)
            real.plugins = [p for p in real.plugins if p.id != "precision-validation"] + [stripped]
            with mock.patch.object(precision_gate, "PLUGIN_ID", "precision-validation"), \
                 mock.patch("orchestrator.plugins.PluginRegistry", return_value=real):
                with self.assertRaises(precision_gate.PrecisionPluginMissing) as caught:
                    precision_gate._plugin(SimpleNamespace(), Path(workspace))
        self.assertIn("submodule update --init", str(caught.exception))

    def test_an_absent_plugin_blocks_the_gate(self):
        from orchestrator import precision_gate
        from orchestrator.plugins import PluginRegistry

        empty = PluginRegistry()
        empty.plugins = [p for p in empty.plugins if p.id != "precision-validation"]
        with tempfile.TemporaryDirectory() as workspace, \
             mock.patch("orchestrator.plugins.PluginRegistry", return_value=empty):
            with self.assertRaises(precision_gate.PrecisionPluginMissing) as caught:
                precision_gate._plugin(SimpleNamespace(), Path(workspace))
        self.assertIn("has no precision gate", str(caught.exception))


class VerdictReadingTest(unittest.TestCase):
    """Both fields of a plugin verdict are authoritative, in the blocking direction."""

    def problem(self, verdict):
        from orchestrator.precision_gate import verdict_problem
        return verdict_problem(verdict, "precision evidence")

    def test_only_accepted_with_no_violations_continues(self):
        self.assertEqual(self.problem({"accepted": True, "violations": []}), "")

    def test_violations_block_even_when_accepted_is_true(self):
        self.assertEqual(
            self.problem({"accepted": True, "violations": ["probe 2 under-covered"]}),
            "probe 2 under-covered")

    def test_rejection_without_a_reason_still_names_the_subject(self):
        self.assertEqual(self.problem({"accepted": False, "violations": []}),
                         "precision evidence was rejected without a stated reason")

    def test_several_violations_are_joined(self):
        self.assertEqual(self.problem({"accepted": False, "violations": ["a", "b"]}), "a; b")

    def test_a_missing_or_non_boolean_accepted_blocks(self):
        for verdict in ({}, {"violations": []}, {"accepted": "yes", "violations": []},
                        {"accepted": 1, "violations": []}, {"accepted": None}):
            self.assertTrue(self.problem(verdict), verdict)


class WorkspaceSuppliedSuiteTest(unittest.TestCase):
    """A suite found in the agent's own workspace must not be able to replace the comparator."""

    def resolve(self, suite, *, private_reference_dir):
        from orchestrator import precision_gate
        from orchestrator.plugins import PluginRegistry
        import json as _json
        registry = PluginRegistry()
        plugin = next(p for p in registry.plugins if p.id == "precision-validation")
        with tempfile.TemporaryDirectory() as root:
            workspace = Path(root)
            (workspace / "numerical_suite.json").write_text(_json.dumps(suite))
            campaign = SimpleNamespace(private_reference_dir=private_reference_dir,
                                       framework="Triton")
            return precision_gate.resolve_suite(
                campaign, workspace, workspace, registry, plugin, make_shapes(3))

    def test_a_custom_evaluator_from_the_workspace_is_refused(self):
        for field, value in (("evaluator_command", ["/bin/true"]),
                             ("evaluator_files", ["adapter.py"])):
            suite = {**make_suite(), field: value}
            with self.assertRaises(ValueError) as caught:
                self.resolve(suite, private_reference_dir=None)
            self.assertIn("operator-owned private reference directory", str(caught.exception))

    def test_an_ordinary_workspace_suite_is_still_honoured(self):
        path = self.resolve(make_suite(), private_reference_dir=None)
        self.assertEqual(path.name, "numerical_suite.json")

    def test_a_private_directory_may_declare_a_custom_evaluator(self):
        """An operator-owned adapter is a supported feature; only the workspace path is not."""
        suite = {**make_suite(), "evaluator_command": ["/bin/true"]}
        path = self.resolve(suite, private_reference_dir=Path("/somewhere/private"))
        self.assertEqual(path.name, "numerical_suite.json")


class AuthoredSuiteBoundsTest(unittest.TestCase):
    """A cached suite is re-checked against the bounds its authoring run had to satisfy.

    The cache key is a digest of the trusted contract, which a candidate holding that
    contract can compute, and the cache lives in the workspace when there is no private
    reference directory. So a cache hit is not more trusted than a fresh authoring run.
    """

    def check(self, suite):
        from orchestrator.precision_gate import check_authored_suite
        return check_authored_suite(suite)

    def test_a_compact_three_to_six_case_suite_passes(self):
        for cases in (3, 4, 6):
            self.assertIs(self.check(make_suite(cases=cases))["schema_version"], 1)

    def test_too_few_or_too_many_cases_are_refused(self):
        for cases in (2, 7, 12):
            with self.assertRaises(ValueError) as caught:
                self.check(make_suite(cases=cases))
            self.assertIn("3..6 compact risk cases", str(caught.exception))

    def test_exhaustive_coverage_is_not_self_authorable(self):
        with self.assertRaises(ValueError):
            self.check(make_suite(coverage="exhaustive"))

    def test_a_self_authored_custom_evaluator_is_refused(self):
        for field, value in (("evaluator_command", ["/bin/true"]),
                             ("evaluator_files", ["adapter.py"])):
            with self.assertRaises(ValueError) as caught:
                self.check({**make_suite(), field: value})
            self.assertIn("must be operator-supplied", str(caught.exception))

    def test_both_the_cache_and_the_authoring_path_apply_the_bounds(self):
        source = (REPO_ROOT / "orchestrator" / "precision_gate.py").read_text()
        self.assertEqual(source.count("check_authored_suite("), 3,
                         "definition plus the cache-hit and fresh-authoring call sites")


class AdapterErrorPathTest(unittest.TestCase):
    """A policy error must not carry a traceback, and so supervisor paths, to the candidate."""

    def run_tool(self, tool, payload):
        import subprocess as _sp
        import sys as _sys
        return _sp.run([_sys.executable, str(PLUGIN_ROOT / "adapter.py"), tool],
                       input=payload, capture_output=True, text=True)

    def test_a_schema_valid_but_unusable_request_reports_only_a_reason(self):
        import json as _json
        done = self.run_tool("plan", _json.dumps(
            {"schema_version": 1, "suite": {"schema_version": 99}, "shapes": {"s0": {}}}))
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stdout, "")
        self.assertNotIn("Traceback", done.stderr)
        self.assertNotIn(str(REPO_ROOT), done.stderr)
        self.assertEqual(done.stderr.strip(),
                         "ValueError: unsupported numerical suite schema")

    def test_an_unknown_tool_name_is_a_usage_error(self):
        done = self.run_tool("promote", "{}")
        self.assertEqual(done.returncode, 2)
        self.assertIn("usage:", done.stderr)

    def test_a_valid_request_still_writes_json_and_exits_zero(self):
        import json as _json
        done = self.run_tool("plan", _json.dumps(
            {"schema_version": 1, "suite": make_suite(), "shapes": make_shapes(3),
             "mode": "light", "framework_key": "triton"}))
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(_json.loads(done.stdout)["schema_version"], 1)


class GatewayRunnerLookupTest(unittest.TestCase):
    """The vendored pin is a fallback, never a silent substitute for a named root."""

    @classmethod
    def setUpClass(cls):
        cls.gateway = _load_module("_precision_gateway", REPO_ROOT / "tools" / "local_gateway.py")

    def test_no_root_named_falls_back_to_the_vendored_submodule(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop("ATREX_BENCH_ROOT", None)
            runner = self.gateway._find_atrex_bench_runner()
        self.assertEqual(runner, REPO_ROOT / "3rdparty" / "atrex-bench" / "scripts" / "run_eval.py")

    def test_a_wrong_explicit_root_still_fails_loudly(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError) as caught:
                self.gateway._find_atrex_bench_runner(Path(empty))
            self.assertIn(empty, str(caught.exception))
        self.assertIn("run_eval.py", str(caught.exception))

    def test_a_wrong_environment_root_still_fails_loudly(self):
        with tempfile.TemporaryDirectory() as empty:
            with mock.patch.dict("os.environ", {"ATREX_BENCH_ROOT": empty}):
                with self.assertRaises(FileNotFoundError):
                    self.gateway._find_atrex_bench_runner()


class VendoredCorpusContainmentTest(unittest.TestCase):
    """Moving the operator out of the repo is not enough if its twin is still vendored."""

    def vendored_shapes(self):
        candidates = sorted((REPO_ROOT / "3rdparty" / "atrex-bench" / "data").glob("*/shapes.json"))
        if not candidates:
            self.skipTest("3rdparty/atrex-bench is not initialized")
        return candidates[0]

    def test_an_external_copy_of_a_vendored_operator_is_refused(self):
        source = self.vendored_shapes()
        with tempfile.TemporaryDirectory() as outside:
            private = Path(outside) / "rms_norm_copy"
            private.mkdir()
            (private / "shapes.json").write_bytes(source.read_bytes())
            campaign = SimpleNamespace(private_reference_dir=private,
                                       workspace=Path(outside) / "ws",
                                       _generated_agent_problem_digest="")
            (Path(outside) / "ws").mkdir()
            with self.assertRaises(RuntimeError) as caught:
                Campaign._assert_generalized_inputs_are_private(campaign)
        message = str(caught.exception)
        self.assertIn("also vendored in the AKA checkout", message)
        self.assertIn(str(source.relative_to(REPO_ROOT)), message)

    def test_the_same_operator_name_is_refused_even_with_different_shapes(self):
        """A different revision of the same operator is still a near copy of its cases."""
        name = self.vendored_shapes().parent.name
        with tempfile.TemporaryDirectory() as outside:
            private = Path(outside) / name
            private.mkdir()
            (private / "shapes.json").write_text('{"0": {"input_kwargs": {"token_count": 1}}}')
            campaign = SimpleNamespace(private_reference_dir=private,
                                       workspace=Path(outside) / "ws",
                                       _generated_agent_problem_digest="")
            (Path(outside) / "ws").mkdir()
            with self.assertRaises(RuntimeError) as caught:
                Campaign._assert_generalized_inputs_are_private(campaign)
        self.assertIn("also vendored in the AKA checkout", str(caught.exception))

    def test_an_operator_absent_from_the_vendored_corpus_passes_the_guard(self):
        self.vendored_shapes()
        with tempfile.TemporaryDirectory() as outside:
            private = Path(outside) / "private_op"
            private.mkdir()
            (private / "shapes.json").write_text('{"s0": {"input_kwargs": {"n": 987654321}}}')
            campaign = SimpleNamespace(private_reference_dir=private,
                                       workspace=Path(outside) / "ws",
                                       _generated_agent_problem_digest="")
            (Path(outside) / "ws").mkdir()
            with self.assertRaises(RuntimeError) as caught:
                Campaign._assert_generalized_inputs_are_private(campaign)
        # It must fall through to the ordinary public-problem check, not the corpus guard.
        self.assertIn("missing agent_problem.json", str(caught.exception))


class GateCallSiteTest(unittest.TestCase):
    """No call site may read a broken gate as a pass."""

    BROKEN = (None, "ok", {"accepted": True}, 0, [""], ["  "], [1], ["ok", None], (), True)

    def wrapper(self, precision_result):
        from orchestrator import precision_gate
        with mock.patch.object(precision_gate, "precision_violations",
                               return_value=precision_result):
            return precision_gate.blocking_violations(SimpleNamespace(), Path("/tmp/ws"))

    def campaign_call(self, precision_result):
        campaign = SimpleNamespace(workspace=Path("/tmp/ws"), framework="Triton",
                                   _review_production_candidate=lambda *a, **k: [])
        with mock.patch("orchestrator.campaign.production_kernel_violations", return_value=[]), \
             mock.patch("orchestrator.precision_gate.precision_violations",
                        return_value=precision_result):
            return Campaign._production_kernel_violations(campaign)

    def test_an_empty_list_promotes(self):
        self.assertEqual(self.wrapper([]), [])
        self.assertEqual(self.campaign_call([]), [])

    def test_violations_are_passed_through_unchanged(self):
        reasons = ["numerical safety input_domain: bad", "numerical safety coverage: thin"]
        self.assertEqual(self.wrapper(list(reasons)), reasons)
        self.assertEqual(self.campaign_call(list(reasons)), reasons)

    def test_a_malformed_result_fails_closed(self):
        from orchestrator.precision_gate import INVALID_RESULT
        for broken in self.BROKEN:
            with self.subTest(result=broken):
                self.assertEqual(self.wrapper(broken), [INVALID_RESULT])
                self.assertEqual(self.campaign_call(broken), [INVALID_RESULT])

    def test_the_structural_gate_still_short_circuits_the_precision_gate(self):
        """A structural rejection must not pay for a GPU allocation."""
        campaign = SimpleNamespace(workspace=Path("/tmp/ws"), framework="Triton",
                                   _review_production_candidate=lambda *a, **k: [])
        with mock.patch("orchestrator.campaign.production_kernel_violations",
                        return_value=["unsupported production framework"]), \
             mock.patch("orchestrator.precision_gate.precision_violations") as gate:
            self.assertEqual(Campaign._production_kernel_violations(campaign),
                             ["unsupported production framework"])
        gate.assert_not_called()

    def test_both_call_sites_go_through_the_validating_wrapper(self):
        """The repair-admission path must inherit the same guarantee as promotion."""
        for path in (REPO_ROOT / "orchestrator" / "campaign.py",
                     REPO_ROOT / "long_horizon" / "main_adapter.py"):
            text = path.read_text()
            self.assertIn("blocking_violations", text, str(path))
            self.assertNotIn("import precision_violations", text, str(path))


class PrivateReferenceContainmentTest(unittest.TestCase):
    """Vendoring the evaluator must not make exact operator cases reachable."""

    def assert_refused(self, private_dir):
        campaign = SimpleNamespace(private_reference_dir=private_dir,
                                   workspace=Path("/tmp/ws"),
                                   _generated_agent_problem_digest="")
        with self.assertRaises(RuntimeError) as caught:
            Campaign._assert_generalized_inputs_are_private(campaign)
        return str(caught.exception)

    def test_an_operator_inside_the_vendored_submodule_is_refused(self):
        message = self.assert_refused(REPO_ROOT / "3rdparty" / "atrex-bench" / "data" / "rms_norm")
        self.assertIn("must live outside the AKA checkout", message)
        self.assertIn("3rdparty/atrex-bench/data/rms_norm", message)

    def test_any_in_repo_private_directory_is_refused(self):
        for candidate in (REPO_ROOT, REPO_ROOT / "reference", REPO_ROOT / "tools" / "x"):
            self.assertIn("must live outside the AKA checkout", self.assert_refused(candidate))

    def test_a_private_directory_outside_the_repository_is_allowed_through(self):
        """It must reach the ordinary public-problem checks, not the containment guard."""
        with tempfile.TemporaryDirectory() as outside:
            campaign = SimpleNamespace(private_reference_dir=Path(outside),
                                       workspace=Path(outside) / "ws",
                                       _generated_agent_problem_digest="")
            (Path(outside) / "ws").mkdir()
            with self.assertRaises(RuntimeError) as caught:
                Campaign._assert_generalized_inputs_are_private(campaign)
            self.assertIn("missing agent_problem.json", str(caught.exception))

    def test_no_private_reference_directory_is_a_no_op(self):
        campaign = SimpleNamespace(private_reference_dir=None)
        self.assertIsNone(Campaign._assert_generalized_inputs_are_private(campaign))


class HarnessTrustModeTest(unittest.TestCase):
    """The workspace harness must forward the guard profile, and only when asked."""

    @classmethod
    def setUpClass(cls):
        cls.harness = _load_harness()

    def test_trust_mode_defaults_to_absent(self):
        args = self.harness._parser().parse_args(["--correctness-only"])
        self.assertIsNone(args.trust_mode)

    def test_only_the_two_profiles_are_accepted(self):
        for value in ("trusted", "untrusted"):
            self.assertEqual(
                self.harness._parser().parse_args(["--trust-mode", value]).trust_mode, value)
        # argparse writes its own diagnostic to stderr before exiting; keep it out of the
        # suite's output so a green run stays readable.
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.harness._parser().parse_args(["--trust-mode", "permissive"])


def _load_harness():
    spec = importlib.util.spec_from_file_location(
        "_precision_harness", REPO_ROOT / "reference" / "atrex_bench_test_kernel.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    unittest.main()
