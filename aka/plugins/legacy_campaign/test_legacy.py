"""Behaviour tests for the legacy campaign driver plugin.

Stage one's whole job is to be *identical by construction*: the row still builds today's
``orchestrator.campaign.Campaign`` and drives it in the order ``orchestrator/optimize.py`` used
to. So the things worth pinning are the two translation surfaces and the sequence:

* ``CAMPAIGN_FIELDS`` is simultaneously the set of ``Config`` properties, the set of keyword
  arguments ``Campaign.__init__`` accepts, and (through ``aka.cli.FLAG_BINDINGS``) the set of
  fields reachable from the CLI. All three are derived here rather than transcribed -- the
  constructor's fields come from ``dataclasses.fields(Campaign)`` filtered to ``f.init`` -- so
  adding a field to ``Campaign`` without routing it through the plugin fails this file instead
  of silently reverting that field to its dataclass default on every campaign.
* ``Defaults`` must equal ``Campaign``'s own field defaults, since the config layer, not the
  dataclass, is what a booted row actually applies. ``sandbox_ssh_gpu`` is the single encoded
  field: the schema dialect has no null type, so ``-1`` stands for ``None``.
* ``LegacyCampaignDriver.run`` must drive exactly the legacy sequence -- baseline or resume,
  then the generalized-memory coverage gate, then ``on_prepared``, then the framework baseline,
  then the campaign.

No test constructs a real ``Campaign``: that needs an operator directory, a workspace on disk
and a plugin lock, none of which say anything about the translation layer. The driver tests use
a recording fake whose members are checked against the real class's signatures
(``LegacySurfaceTest``), and the two workspace-state helpers ``run`` imports are patched. Those
imports are function-local (``aka/plugins/legacy_campaign/service.py`` line 104), so the patch
target is ``orchestrator.workspace_state``, the module the name is looked up in at call time;
patching an attribute on ``...legacy_campaign.service`` would create a name nothing reads.
"""

from __future__ import annotations

import dataclasses
import inspect
import io
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from pathlib import Path
from typing import Any, Iterator, Mapping
from unittest import mock

from orchestrator.campaign import Campaign

from aka import seams
from aka.cli import CAMPAIGN_ENTRY, bound_fields, config_patch
from aka.core.declaration import declare
from aka.core.errors import DeclarationError
from aka.core.fiber import FiberState, Root
from aka.plugins.legacy_campaign import plugin
from aka.plugins.legacy_campaign.invariant import PACKAGE_NAME
from aka.plugins.legacy_campaign.service import (
    CAMPAIGN_FIELDS,
    UNASSIGNED_GPU,
    LegacyCampaignDriver,
    campaign_kwargs,
)
from aka.seams.driver import DRIVER, STATUSES, CampaignDriver, CampaignRun

import aka.plugins.legacy_campaign as legacy_campaign

#: The four values only the CLI can supply, so they carry no default anywhere.
REQUIRED_FIELDS = ("name", "kernel_demo", "platform", "framework")

#: A minimal but complete row config: the required four, with ``Defaults`` filling the rest.
MINIMAL_CONFIG: dict[str, Any] = {
    "name": "add_rms_norm",
    "kernel_demo": "/operators/add_rms_norm/demo.py",
    "platform": "h20",
    "framework": "triton",
}

#: What ``read_memory(workspace, 0)`` hands the coverage gate.
BASELINE_MEMORY: dict[str, Any] = {
    "performance": {"latency_us_by_shape": {"shape-a": 12.5}}
}

#: The legacy members ``LegacyCampaignDriver`` reaches for, and the parameters it relies on.
LEGACY_MEMBERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("setup_baseline", ("self",)),
    ("_link_runtime", ("self",)),
    ("_generalized_memory_coverage_problem", ("self", "memory")),
    ("ensure_framework_baseline", ("self",)),
    ("run", ("self",)),
)


def campaign_init_fields() -> dict[str, dataclasses.Field]:
    """``Campaign``'s constructor keyword arguments, derived from the dataclass itself."""
    return {
        field.name: field for field in dataclasses.fields(Campaign) if field.init
    }


def campaign_default(field: dataclasses.Field) -> Any:
    """The default ``Campaign`` uses for ``field``, or ``dataclasses.MISSING``."""
    if field.default is not dataclasses.MISSING:
        return field.default
    if field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
        return field.default_factory()  # type: ignore[misc]
    return dataclasses.MISSING


class FakeCampaign:
    """A recording stand-in for ``orchestrator.campaign.Campaign``.

    It records the order of the legacy sequence rather than performing it. Its member names
    and parameters are checked against the real class by :class:`LegacySurfaceTest`.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        coverage_problem: str = "",
        reason: str = "target utilization reached",
    ):
        self.workspace = workspace
        self.campaign_name = "add_rms_norm_triton_h20"
        self.calls: list[str] = []
        self.memory_seen: list[Any] = []
        self._coverage_problem = coverage_problem
        self._reason = reason

    def setup_baseline(self) -> None:
        self.calls.append("setup_baseline")

    def _link_runtime(self) -> None:
        self.calls.append("_link_runtime")

    def _generalized_memory_coverage_problem(self, memory: dict | None) -> str:
        self.calls.append("coverage_gate")
        self.memory_seen.append(memory)
        return self._coverage_problem

    def ensure_framework_baseline(self) -> None:
        self.calls.append("ensure_framework_baseline")

    def run(self) -> str:
        self.calls.append("run")
        return self._reason


@contextmanager
def patched_workspace_state(
    *, version: int, memory: Any = BASELINE_MEMORY
) -> Iterator[tuple[mock.MagicMock, mock.MagicMock]]:
    """Patch the two helpers ``run`` imports, so no workspace on disk is needed."""
    with mock.patch(
        "orchestrator.workspace_state.latest_version", return_value=version
    ) as latest_version, mock.patch(
        "orchestrator.workspace_state.read_memory", return_value=memory
    ) as read_memory:
        yield latest_version, read_memory


class CampaignKwargsTest(unittest.TestCase):
    """``campaign_kwargs`` is a translation, not a policy: only the GPU field is encoded."""

    def test_every_campaign_field_is_forwarded_by_identity(self) -> None:
        values: dict[str, Any] = {
            field: object() for field in CAMPAIGN_FIELDS if field != "sandbox_ssh_gpu"
        }
        values["sandbox_ssh_gpu"] = 7

        kwargs = campaign_kwargs(values)

        self.assertEqual(set(kwargs), set(CAMPAIGN_FIELDS))
        for field, value in values.items():
            with self.subTest(field=field):
                if field == "sandbox_ssh_gpu":
                    continue
                self.assertIs(kwargs[field], value)

    def test_unassigned_gpu_becomes_none_and_a_real_index_survives(self) -> None:
        for supplied, expected in ((UNASSIGNED_GPU, None), (0, 0), (3, 3), (31, 31)):
            with self.subTest(sandbox_ssh_gpu=supplied):
                kwargs = campaign_kwargs({**MINIMAL_CONFIG, "sandbox_ssh_gpu": supplied})
                self.assertEqual(kwargs["sandbox_ssh_gpu"], expected)

    def test_unassigned_gpu_is_minus_one(self) -> None:
        self.assertEqual(UNASSIGNED_GPU, -1)

    def test_an_absent_gpu_field_still_resolves_to_none(self) -> None:
        # -1 stands for None, and so does saying nothing: None is Campaign's own default.
        self.assertIsNone(campaign_kwargs({})["sandbox_ssh_gpu"])
        self.assertIsNone(campaign_default(campaign_init_fields()["sandbox_ssh_gpu"]))

    def test_only_campaign_fields_are_forwarded(self) -> None:
        kwargs = campaign_kwargs(
            {**MINIMAL_CONFIG, "dump_config": True, "environment_poll_interval": 30}
        )

        self.assertEqual(set(kwargs) - {"sandbox_ssh_gpu"}, set(MINIMAL_CONFIG))
        self.assertNotIn("dump_config", kwargs)
        self.assertNotIn("environment_poll_interval", kwargs)

    def test_absent_optional_fields_are_left_to_campaigns_own_default(self) -> None:
        kwargs = campaign_kwargs(MINIMAL_CONFIG)

        self.assertNotIn("max_iters", kwargs)
        self.assertNotIn("notes", kwargs)

    def test_the_supplied_config_is_not_mutated(self) -> None:
        config = {**MINIMAL_CONFIG, "sandbox_ssh_gpu": UNASSIGNED_GPU}
        snapshot = dict(config)

        campaign_kwargs(config)

        self.assertEqual(config, snapshot)


class CampaignFieldSetTest(unittest.TestCase):
    """The field set is the plugin's contract with three neighbours at once."""

    def test_campaign_fields_are_exactly_campaigns_constructor_arguments(self) -> None:
        expected = set(campaign_init_fields())

        self.assertEqual(set(CAMPAIGN_FIELDS), expected)

    def test_campaign_fields_are_exactly_the_config_properties(self) -> None:
        self.assertEqual(set(CAMPAIGN_FIELDS), set(plugin.Config["properties"]))

    def test_campaign_fields_lists_each_field_once(self) -> None:
        self.assertEqual(len(CAMPAIGN_FIELDS), len(set(CAMPAIGN_FIELDS)))

    def test_campaign_internal_state_is_not_a_constructor_argument(self) -> None:
        # tokens_spent and the reviewer caches are init=False campaign state; routing them
        # through the row's config would let a composition preload a campaign's own bookkeeping.
        derived = {
            field.name for field in dataclasses.fields(Campaign) if not field.init
        }

        self.assertTrue(derived, "Campaign has no init=False state; update this test")
        self.assertEqual(derived & set(CAMPAIGN_FIELDS), set())
        self.assertEqual(derived & set(plugin.Config["properties"]), set())

    def test_the_required_config_fields_are_campaigns_mandatory_fields(self) -> None:
        mandatory = {
            name
            for name, field in campaign_init_fields().items()
            if campaign_default(field) is dataclasses.MISSING
        }

        self.assertEqual(mandatory, set(REQUIRED_FIELDS))
        self.assertEqual(set(plugin.Config["required"]), set(REQUIRED_FIELDS))


class DefaultsTest(unittest.TestCase):
    """``Defaults`` is what a booted row actually applies, so it must mirror the dataclass."""

    def test_every_optional_campaign_field_has_a_plugin_default(self) -> None:
        optional = {
            name
            for name, field in campaign_init_fields().items()
            if campaign_default(field) is not dataclasses.MISSING
        }

        self.assertEqual(set(plugin.Defaults), optional)

    def test_each_default_equals_campaigns_own_default(self) -> None:
        for name, field in campaign_init_fields().items():
            expected = campaign_default(field)
            if expected is dataclasses.MISSING:
                continue
            if name == "sandbox_ssh_gpu":
                # The encoded field: -1 in config, None in the constructor.
                expected = UNASSIGNED_GPU
            with self.subTest(field=name):
                self.assertIn(name, plugin.Defaults)
                self.assertEqual(plugin.Defaults[name], expected)
                self.assertIs(type(plugin.Defaults[name]), type(expected))

    def test_defaults_translate_back_to_campaigns_defaults(self) -> None:
        kwargs = campaign_kwargs(plugin.Defaults)

        for name, value in kwargs.items():
            with self.subTest(field=name):
                self.assertEqual(value, campaign_default(campaign_init_fields()[name]))

    def test_required_fields_are_not_pre_answered(self) -> None:
        self.assertEqual(set(plugin.Defaults) & set(REQUIRED_FIELDS), set())


class DeclarationTest(unittest.TestCase):
    """The package module is the plugin; ``declare`` is the gate it must pass."""

    def setUp(self) -> None:
        self.declaration = declare(legacy_campaign, known_seams=seams.keys())

    def test_the_package_declares_the_driver_provider(self) -> None:
        self.assertEqual(self.declaration.name, "legacy-campaign")
        self.assertEqual(self.declaration.module, "aka.plugins.legacy_campaign")
        self.assertEqual(self.declaration.provide, ("driver",))
        self.assertEqual(self.declaration.provide, (DRIVER.name,))
        self.assertEqual(self.declaration.inject, ())
        self.assertEqual(self.declaration.optional_inject, ())
        self.assertEqual(self.declaration.interpolate, ("work_dir",))
        self.assertIs(self.declaration.apply, plugin.apply)

    def test_config_is_an_object_schema_with_the_four_required_fields(self) -> None:
        schema = self.declaration.config_schema

        self.assertIsNotNone(schema)
        self.assertEqual(schema["type"], "object")
        self.assertEqual(tuple(schema["required"]), REQUIRED_FIELDS)
        self.assertFalse(schema["additionalProperties"])
        for field in REQUIRED_FIELDS:
            with self.subTest(field=field):
                self.assertEqual(schema["properties"][field]["minLength"], 1)

    def test_the_declaration_carries_the_plugin_defaults(self) -> None:
        self.assertEqual(self.declaration.defaults, plugin.Defaults)

    def test_the_provided_seam_is_registered(self) -> None:
        self.assertIn("driver", seams.keys())
        self.assertEqual(seams.keys()["driver"].definition, "CampaignDriver")

    def test_declaring_against_an_empty_seam_table_is_a_load_error(self) -> None:
        # The point of known_seams: "driver" only means something because the table says so.
        with self.assertRaises(DeclarationError) as caught:
            declare(legacy_campaign, known_seams={})

        self.assertIn("references unregistered seams: driver", str(caught.exception))


class LegacySurfaceTest(unittest.TestCase):
    """The fake is only evidence if the real campaign still has the members it fakes."""

    def test_campaign_exposes_every_member_the_driver_calls(self) -> None:
        for member, parameters in LEGACY_MEMBERS:
            with self.subTest(member=member):
                attribute = getattr(Campaign, member, None)
                self.assertTrue(
                    callable(attribute), f"Campaign lost {member}(); the driver calls it"
                )
                self.assertEqual(
                    tuple(inspect.signature(attribute).parameters), parameters
                )

    def test_the_fake_matches_those_signatures(self) -> None:
        for member, parameters in LEGACY_MEMBERS:
            with self.subTest(member=member):
                self.assertEqual(
                    tuple(inspect.signature(getattr(FakeCampaign, member)).parameters),
                    parameters,
                )

    def test_workspace_and_campaign_name_are_campaign_properties(self) -> None:
        self.assertIsInstance(Campaign.workspace, property)
        self.assertIsInstance(Campaign.campaign_name, property)

    def test_the_patched_helpers_keep_their_signatures(self) -> None:
        from orchestrator import workspace_state

        self.assertEqual(
            tuple(inspect.signature(workspace_state.latest_version).parameters),
            ("workspace",),
        )
        self.assertEqual(
            tuple(inspect.signature(workspace_state.read_memory).parameters),
            ("workspace", "n"),
        )


class DriverRunTest(unittest.TestCase):
    """``run`` must drive the exact sequence ``orchestrator/optimize.py`` ran inline."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name)
        self.campaign = FakeCampaign(self.workspace)
        self.driver = LegacyCampaignDriver(self.campaign)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def on_prepared(self) -> None:
        """A hook that records itself in the campaign's own call log, to pin its position."""
        self.campaign.calls.append("on_prepared")

    def drive(self, *, version: int, **kwargs: Any) -> tuple[CampaignRun, str]:
        stdout = io.StringIO()
        with patched_workspace_state(version=version) as (latest_version, read_memory):
            with redirect_stdout(stdout):
                result = self.driver.run(**kwargs)
        self.latest_version = latest_version
        self.read_memory = read_memory
        return result, stdout.getvalue()

    def test_a_fresh_workspace_is_established_then_driven(self) -> None:
        result, output = self.drive(version=-1, on_prepared=self.on_prepared)

        self.assertEqual(
            self.campaign.calls,
            [
                "setup_baseline",
                "coverage_gate",
                "on_prepared",
                "ensure_framework_baseline",
                "run",
            ],
        )
        self.assertNotIn("resuming", output)
        self.assertEqual(
            result,
            CampaignRun(
                status="completed", reason="target utilization reached", exit_code=0
            ),
        )
        self.latest_version.assert_called_once_with(self.workspace)

    def test_an_existing_version_resumes_and_relinks_the_runtime(self) -> None:
        result, output = self.drive(version=3, on_prepared=self.on_prepared)

        self.assertEqual(
            self.campaign.calls,
            [
                "_link_runtime",
                "coverage_gate",
                "on_prepared",
                "ensure_framework_baseline",
                "run",
            ],
        )
        self.assertNotIn("setup_baseline", self.campaign.calls)
        self.assertIn("[orchestrator] resuming workspace at v3", output)
        self.assertEqual(result.status, "completed")
        for call in self.latest_version.call_args_list:
            self.assertEqual(call, mock.call(self.workspace))

    def test_version_zero_is_a_resume_not_a_fresh_workspace(self) -> None:
        # The boundary the legacy code encodes as "< 0": v0 exists, so nothing is established.
        self.drive(version=0)

        self.assertEqual(self.campaign.calls[0], "_link_runtime")
        self.assertNotIn("setup_baseline", self.campaign.calls)

    def test_the_coverage_gate_reads_baseline_memory(self) -> None:
        self.drive(version=-1)

        self.read_memory.assert_called_once_with(self.workspace, 0)
        self.assertEqual(len(self.campaign.memory_seen), 1)
        self.assertIs(self.campaign.memory_seen[0], BASELINE_MEMORY)

    def test_a_coverage_problem_aborts_before_the_framework_baseline(self) -> None:
        campaign = FakeCampaign(
            self.workspace, coverage_problem="canonical memory lacks coverage (1/4)"
        )
        driver = LegacyCampaignDriver(campaign)

        with patched_workspace_state(version=-1):
            with self.assertRaises(RuntimeError) as caught:
                driver.run(on_prepared=self.on_prepared)

        # Byte-for-byte the message the inline sequence raised, so an operator's runbook and
        # any log grep still match after the row took the sequence over.
        self.assertEqual(
            str(caught.exception),
            "generalized campaign baseline is incompatible with authoritative per-shape "
            "memory: canonical memory lacks coverage (1/4); start a fresh workspace",
        )
        message = str(caught.exception)
        self.assertIn("start a fresh workspace", message)
        self.assertEqual(campaign.calls, ["setup_baseline", "coverage_gate"])
        self.assertNotIn("ensure_framework_baseline", campaign.calls)
        self.assertNotIn("run", campaign.calls)
        self.assertNotIn("on_prepared", self.campaign.calls)

    def test_an_empty_coverage_problem_is_not_a_problem(self) -> None:
        result, _ = self.drive(version=-1)

        self.assertEqual(result.status, "completed")
        self.assertIn("ensure_framework_baseline", self.campaign.calls)

    def test_on_prepared_runs_exactly_once(self) -> None:
        calls: list[int] = []

        self.drive(version=-1, on_prepared=lambda: calls.append(1))

        self.assertEqual(calls, [1])

    def test_on_prepared_may_be_none_or_omitted(self) -> None:
        for label, kwargs in (("omitted", {}), ("none", {"on_prepared": None})):
            with self.subTest(on_prepared=label):
                self.campaign = FakeCampaign(self.workspace)
                self.driver = LegacyCampaignDriver(self.campaign)

                result, _ = self.drive(version=-1)

                self.assertEqual(result.status, "completed")
                self.assertEqual(
                    self.campaign.calls,
                    [
                        "setup_baseline",
                        "coverage_gate",
                        "ensure_framework_baseline",
                        "run",
                    ],
                )

    def test_the_reason_the_campaign_returned_is_carried_out(self) -> None:
        campaign = FakeCampaign(self.workspace, reason="max_iters exhausted")
        driver = LegacyCampaignDriver(campaign)

        with patched_workspace_state(version=-1):
            result = driver.run()

        self.assertEqual(result.reason, "max_iters exhausted")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.exit_code, 0)

    def test_a_campaign_failure_reaches_the_caller(self) -> None:
        campaign = FakeCampaign(self.workspace)
        campaign.run = mock.Mock(side_effect=RuntimeError("evaluator unreachable"))
        driver = LegacyCampaignDriver(campaign)

        with patched_workspace_state(version=-1):
            with self.assertRaisesRegex(RuntimeError, "evaluator unreachable"):
                driver.run()

    def test_the_driver_writes_nothing_into_the_workspace_itself(self) -> None:
        self.drive(version=-1)

        self.assertEqual(list(self.workspace.iterdir()), [])

    def test_the_driver_delegates_identity_to_the_campaign(self) -> None:
        self.assertIsInstance(self.driver, CampaignDriver)
        self.assertIs(self.driver.campaign, self.campaign)
        self.assertEqual(self.driver.workspace, self.workspace)
        self.assertEqual(self.driver.campaign_name, "add_rms_norm_triton_h20")


class CampaignRunTest(unittest.TestCase):
    """The seam's result type is the only thing the entry point reads back."""

    def test_every_declared_status_is_accepted(self) -> None:
        for status in STATUSES:
            with self.subTest(status=status):
                self.assertEqual(CampaignRun(status=status).status, status)

    def test_an_unsupported_status_is_rejected(self) -> None:
        for status in ("", "success", "done", "Completed", None, 0):
            with self.subTest(status=status):
                with self.assertRaises(ValueError) as caught:
                    CampaignRun(status=status)  # type: ignore[arg-type]
                self.assertIn("unsupported campaign status", str(caught.exception))
                self.assertIn(repr(status), str(caught.exception))

    def test_reason_and_exit_code_default_to_a_clean_run(self) -> None:
        run = CampaignRun(status="completed")

        self.assertEqual(run.reason, "")
        self.assertEqual(run.exit_code, 0)

    def test_a_result_is_frozen(self) -> None:
        run = CampaignRun(status="failed", reason="boom", exit_code=1)

        with self.assertRaises(dataclasses.FrozenInstanceError):
            run.status = "completed"  # type: ignore[misc]


class ApplyTest(unittest.TestCase):
    """``apply`` publishes the driver and reserves the package's invariant name.

    ``build_campaign`` is patched, so the row's config reaches the boundary and stops there.
    The tree is real: ``Root`` merges ``Defaults`` under the row config and validates the whole
    thing against ``Config`` before ``apply`` ever runs.
    """

    def setUp(self) -> None:
        self.stderr = io.StringIO()
        self.root = Root(seams=seams.keys(), stderr=self.stderr)
        self.temporary = tempfile.TemporaryDirectory()
        self.campaign = FakeCampaign(Path(self.temporary.name))
        self.built: list[Mapping[str, Any]] = []

    def tearDown(self) -> None:
        self.root.dispose()
        self.temporary.cleanup()

    def mount(self, config: Mapping[str, Any]) -> Any:
        def build(row_config: Mapping[str, Any]) -> Any:
            self.built.append(dict(row_config))
            return self.campaign

        declaration = declare(legacy_campaign, known_seams=seams.keys())
        with mock.patch.object(plugin, "build_campaign", build):
            fiber = self.root.mount(declaration, config)
            self.root.settle()
        return fiber

    def test_the_row_provides_the_driver_wrapping_the_built_campaign(self) -> None:
        fiber = self.mount(MINIMAL_CONFIG)

        self.assertIs(fiber.state, FiberState.ACTIVE)
        impl = self.root.realm.resolve("driver")
        self.assertIsNotNone(impl)
        self.assertIsInstance(impl.value, LegacyCampaignDriver)
        self.assertIs(impl.value.campaign, self.campaign)

    def test_defaults_are_merged_under_the_row_config_before_apply(self) -> None:
        self.mount({**MINIMAL_CONFIG, "max_iters": 4})

        self.assertEqual(len(self.built), 1)
        config = self.built[0]
        self.assertEqual(set(config), set(CAMPAIGN_FIELDS))
        self.assertEqual(config["max_iters"], 4)
        self.assertEqual(config["notes"], plugin.Defaults["notes"])
        self.assertEqual(config["sandbox_ssh_gpu"], UNASSIGNED_GPU)

    def test_the_package_invariant_name_is_reserved_while_the_row_is_active(self) -> None:
        self.mount(MINIMAL_CONFIG)

        self.assertIn(PACKAGE_NAME, self.root.invariants.reserved)
        self.assertEqual(self.root.invariants.failures, ())

        self.root.dispose()

        self.assertNotIn(PACKAGE_NAME, self.root.invariants.reserved)
        self.assertIsNone(self.root.realm.resolve("driver"))

    def test_a_config_the_schema_rejects_never_reaches_the_campaign(self) -> None:
        fiber = self.mount({**MINIMAL_CONFIG, "max_iters": 0})

        self.assertIs(fiber.state, FiberState.FAILED)
        self.assertEqual(self.built, [])
        self.assertIsNone(self.root.realm.resolve("driver"))

    def test_a_missing_required_field_never_reaches_the_campaign(self) -> None:
        partial = {
            key: value for key, value in MINIMAL_CONFIG.items() if key != "platform"
        }

        fiber = self.mount(partial)

        self.assertIs(fiber.state, FiberState.FAILED)
        self.assertEqual(self.built, [])


class CliBindingTest(unittest.TestCase):
    """Every campaign field must be reachable from the CLI, and encoded on the way in."""

    def test_bound_fields_are_exactly_the_config_properties(self) -> None:
        self.assertEqual(
            set(bound_fields(CAMPAIGN_ENTRY)), set(plugin.Config["properties"])
        )
        self.assertEqual(set(bound_fields(CAMPAIGN_ENTRY)), set(CAMPAIGN_FIELDS))

    def test_config_patch_addresses_the_campaign_row(self) -> None:
        patch = config_patch(CAMPAIGN_ENTRY, {"platform": "h20"})

        self.assertEqual(set(patch), {"id", "config"})
        self.assertEqual(patch["id"], CAMPAIGN_ENTRY)
        self.assertEqual(patch["id"], plugin.name)
        self.assertEqual(patch["config"], {"platform": "h20"})

    def test_an_unassigned_gpu_is_encoded_as_minus_one(self) -> None:
        patch = config_patch(CAMPAIGN_ENTRY, {"sandbox_ssh_gpu": None})

        self.assertEqual(patch["config"]["sandbox_ssh_gpu"], UNASSIGNED_GPU)

    def test_a_real_gpu_index_is_carried_through_as_an_integer(self) -> None:
        for supplied, expected in ((0, 0), (3, 3), ("5", 5)):
            with self.subTest(sandbox_ssh_gpu=supplied):
                patch = config_patch(CAMPAIGN_ENTRY, {"sandbox_ssh_gpu": supplied})
                self.assertEqual(patch["config"]["sandbox_ssh_gpu"], expected)

    def test_the_gpu_encoding_round_trips_back_to_campaigns_argument(self) -> None:
        for supplied in (None, 0, 3):
            with self.subTest(sandbox_ssh_gpu=supplied):
                patch = config_patch(CAMPAIGN_ENTRY, {"sandbox_ssh_gpu": supplied})
                kwargs = campaign_kwargs(patch["config"])
                self.assertEqual(kwargs["sandbox_ssh_gpu"], supplied)

    def test_other_values_are_carried_through_unchanged(self) -> None:
        values = {
            "name": "add_rms_norm",
            "max_iters": 12,
            "target_util": 92.5,
            "full_episode_ask_codex": False,
            "work_dir": "${workspace}",
        }

        self.assertEqual(config_patch(CAMPAIGN_ENTRY, values)["config"], values)

    def test_an_unbound_field_is_named_in_the_error(self) -> None:
        with self.assertRaises(KeyError) as caught:
            config_patch(CAMPAIGN_ENTRY, {"platform": "h20", "tokens_spent": 5})

        message = caught.exception.args[0]
        self.assertIn("tokens_spent", message)
        self.assertIn(CAMPAIGN_ENTRY, message)
        self.assertIn("FLAG_BINDINGS", message)

    def test_every_unbound_field_is_reported_at_once(self) -> None:
        with self.assertRaises(KeyError) as caught:
            config_patch(CAMPAIGN_ENTRY, {"nope": 1, "also_nope": 2})

        self.assertIn("also_nope, nope", caught.exception.args[0])

    def test_an_unknown_entry_binds_nothing(self) -> None:
        self.assertEqual(bound_fields("no-such-row"), frozenset())
        with self.assertRaises(KeyError):
            config_patch("no-such-row", {"platform": "h20"})


if __name__ == "__main__":
    unittest.main()
