"""Behaviour tests for config resolution and validation.

Two pieces of the core decide what a plugin's ``apply`` is ever allowed to see:
:meth:`aka.core.fiber.Fiber._resolve_config` and
:func:`aka.core.composition.apply_interpolation`. The order they run in is the contract, and
these tests pin every step of it:

1. references are substituted, and only inside the fields the plugin allowlisted in
   ``interpolate`` -- anything else is a :class:`CompositionError`;
2. the ``internal/config`` waterfall rewrites the draft, before any schema is consulted;
3. the module's ``Defaults`` are merged *under* the resulting row config;
4. the merged mapping is validated against the module's ``Config`` schema, and only then
   does ``apply`` run, with a complete config.

A row whose config does not survive that pipeline must make its own fiber ``FAILED`` without
ever calling ``apply`` -- a half-configured plugin is worse than an absent one -- and must
leave every other row alone.

Unit-level fibers are built from :class:`PluginDeclaration` directly: the pipeline reads only
the declaration, so a module on disk would add nothing. The end-to-end interpolation tests use
the real ``test/smoke`` profile through ``aka.boot`` so the loader, the JSON layering and
``declare()`` all take part.
"""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable, Mapping

# ``from aka import boot`` recurses forever through ``aka/__init__.py``'s lazy re-export, so
# the submodule is imported directly here (reported as a core bug).
from aka.boot import PROFILES_DIR, boot, compose
from aka.core.composition import COMPOSITION_VARS, apply_interpolation
from aka.core.context import Context
from aka.core.declaration import PluginDeclaration
from aka.core.errors import BootFailure, CompositionError, ConfigError
from aka.core.fiber import FiberState, Root
from aka.core.internal import INTERNAL_CONFIG, ConfigDraft
from aka.seams import DRIVER
from aka.seams import keys as seam_keys

MODULE = "aka.fixtures.config_row"

#: A schema with one required string, a bounded integer, an enum and an array of strings --
#: enough to fail validation in every way ``plugin_runtime.schema`` can report.
CONFIG: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "workspace": {"type": "string"},
        "attempts": {"type": "integer", "minimum": 1, "maximum": 8},
        "mode": {"type": "string", "enum": ["fast", "thorough"]},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["workspace"],
}

#: ``workspace`` is deliberately absent: a required field with no default must come from the
#: composition, exactly like the CLI-supplied fields of the shipped plugin.
DEFAULTS: dict[str, Any] = {"attempts": 2, "mode": "fast"}

#: A Root's variable table. Every name here is one of ``COMPOSITION_VARS``; ``operator`` is
#: left out on purpose so an allowed-but-unfilled variable can be observed.
VARIABLES: dict[str, str] = {
    "repo_root": "/repo",
    "workspace": "/var/aka",
    "campaign_name": "demo",
}

ENV_NAME = "AKA_TEST_CONFIG_WORKSPACE"
UNSET_ENV_NAME = "AKA_TEST_CONFIG_DEFINITELY_UNSET"


class FiberConfigTestCase(unittest.TestCase):
    """A Root with the real seam table, a recording ``apply``, and captured stderr."""

    def setUp(self) -> None:
        self.stderr = io.StringIO()
        self.applied: list[tuple[str, dict[str, Any]]] = []
        self.root = Root(seams=seam_keys(), variables=dict(VARIABLES), stderr=self.stderr)
        self.addCleanup(self.root.dispose)

    # -- fixtures --------------------------------------------------------

    def declaration(self, **overrides: Any) -> PluginDeclaration:
        """A conforming declaration whose ``apply`` records the config it received."""

        def apply(ctx: Context, config: Mapping[str, Any]) -> None:
            self.applied.append((ctx.entry_id, dict(config)))

        fields: dict[str, Any] = {
            "name": "config-row",
            "module": MODULE,
            "apply": apply,
            "config_schema": CONFIG,
            "defaults": dict(DEFAULTS),
            "inject": (),
            "optional_inject": (),
            "provide": (),
            "interpolate": ("workspace", "tags"),
            "doc": "A row that records its config.",
        }
        fields.update(overrides)
        return PluginDeclaration(**fields)

    def providing_declaration(self, **overrides: Any) -> PluginDeclaration:
        """The same row, but it publishes the ``driver`` seam once it is configured."""

        def apply(ctx: Context, config: Mapping[str, Any]) -> None:
            self.applied.append((ctx.entry_id, dict(config)))
            ctx.provide(DRIVER, config["workspace"])

        return self.declaration(apply=apply, provide=("driver",), **overrides)

    def mount(
        self,
        entry_id: str,
        config: Mapping[str, Any] | None = None,
        declaration: PluginDeclaration | None = None,
    ) -> Any:
        fiber = self.root.mount(
            declaration if declaration is not None else self.declaration(),
            config or {},
            entry_id=entry_id,
        )
        self.root.settle()
        return fiber

    def listen(self, listener: Callable[..., Any], *, order: int = 500) -> None:
        self.addCleanup(self.root.bus.on(INTERNAL_CONFIG, listener, order=order))

    # -- assertions ------------------------------------------------------

    def config_seen_by(self, entry_id: str) -> dict[str, Any]:
        seen = [config for row, config in self.applied if row == entry_id]
        self.assertEqual(len(seen), 1, f"apply ran {len(seen)} times for {entry_id}")
        return seen[0]

    def assert_config_rejected(self, fiber: Any, expected: str) -> ConfigError:
        self.assertIs(fiber.state, FiberState.FAILED)
        self.assertEqual(self.applied, [], "apply must never run on an invalid config")
        self.assertIsNone(fiber.config)
        self.assertIsInstance(fiber.error, ConfigError)
        self.assertEqual(fiber.error.entry_id, fiber.entry_id)
        self.assertIn(expected, str(fiber.error))
        return fiber.error


class ConfigValidationTest(FiberConfigTestCase):
    """A row config that the module's ``Config`` rejects must stop before ``apply``."""

    def test_row_config_violating_the_schema_fails_the_fiber_and_never_applies(self) -> None:
        cases = (
            ("missing-required", {"attempts": 3}, "missing workspace"),
            ("wrong-type", {"workspace": 7}, "workspace: expected string"),
            ("out-of-range", {"workspace": "/w", "attempts": 0}, "attempts: number outside allowed range"),
            ("bad-enum", {"workspace": "/w", "mode": "turbo"}, "mode: unsupported value"),
            ("unknown-field", {"workspace": "/w", "retries": 1}, "unknown field retries"),
            ("bad-array-item", {"workspace": "/w", "tags": ["ok", 2]}, "tags[1]: expected string"),
        )
        for entry_id, config, expected in cases:
            with self.subTest(case=entry_id):
                self.applied.clear()
                fiber = self.mount(entry_id, config)
                error = self.assert_config_rejected(fiber, expected)
                # The message has to name the row, not just the field.
                self.assertIn(f'entry "{entry_id}"', str(error))
                self.assertIn(f"{entry_id}.config", str(error))

    def test_a_required_field_with_no_default_cannot_be_left_unset(self) -> None:
        fiber = self.mount("bare", {})

        self.assert_config_rejected(fiber, "missing workspace")

    def test_a_failed_row_leaves_its_neighbour_active_and_its_service_absent(self) -> None:
        broken = self.root.mount(
            self.providing_declaration(), {"attempts": 99}, entry_id="broken"
        )
        healthy = self.root.mount(
            self.declaration(), {"workspace": "/tmp/healthy"}, entry_id="healthy"
        )

        self.root.settle()

        self.assertIs(broken.state, FiberState.FAILED)
        self.assertIs(healthy.state, FiberState.ACTIVE)
        self.assertEqual(self.root.active, ("healthy",))
        self.assertEqual([row for row, _ in self.applied], ["healthy"])
        # Nothing the broken row declared reached the realm, and it owns no effects.
        self.assertIsNone(self.root.realm.resolve("driver"))
        self.assertEqual(broken.provided, set())
        self.assertEqual(len(broken.effects), 0)
        self.assertEqual(self.root.live_effects, 0)
        # The failure is attributable: reported once, naming the row.
        self.assertEqual([entry_id for entry_id, _ in self.root.failed], ["broken"])
        self.assertEqual(len(self.root.errors), 1)
        self.assertIn("broken", self.root.errors[0].source)
        self.assertIn("ConfigError", self.root.errors[0].error)
        self.assertIn("broken", self.stderr.getvalue())

    def test_defaults_are_merged_under_the_row_config(self) -> None:
        fiber = self.mount("merged", {"workspace": "/tmp/ws", "attempts": 5})

        # attempts was set by the row and wins; mode was unset and takes its default.
        self.assertEqual(
            self.config_seen_by("merged"),
            {"workspace": "/tmp/ws", "attempts": 5, "mode": "fast"},
        )
        self.assertEqual(dict(fiber.config), self.config_seen_by("merged"))
        self.assertIs(fiber.state, FiberState.ACTIVE)

    def test_the_config_reaching_apply_is_complete(self) -> None:
        fiber = self.mount("complete", {"workspace": "/tmp/only"})

        self.assertEqual(
            self.config_seen_by("complete"),
            {"workspace": "/tmp/only", "attempts": 2, "mode": "fast"},
        )
        # Merging must not write back into the row or the declaration.
        self.assertEqual(fiber.raw_config, {"workspace": "/tmp/only"})
        self.assertEqual(fiber.declaration.defaults, DEFAULTS)

    def test_a_falsey_row_value_still_beats_its_default(self) -> None:
        declaration = self.declaration(defaults={"attempts": 2, "mode": "thorough"})

        self.mount("falsey", {"workspace": "", "attempts": 1}, declaration)

        # An empty string is a value, not an absence: the default must not resurface.
        self.assertEqual(
            self.config_seen_by("falsey"),
            {"workspace": "", "attempts": 1, "mode": "thorough"},
        )

    def test_config_for_a_plugin_without_a_config_schema_is_a_config_error(self) -> None:
        declaration = self.declaration(config_schema=None, defaults={}, interpolate=())

        fiber = self.mount("schemaless", {"attempts": 1, "workspace": "/w"}, declaration)

        error = self.assert_config_rejected(
            fiber, "row supplies config but aka.fixtures.config_row declares no Config schema"
        )
        # Every offending field is named, sorted, so the operator can delete them.
        self.assertTrue(str(error).endswith("attempts, workspace"), str(error))

    def test_a_plugin_without_a_config_schema_receives_an_empty_config(self) -> None:
        declaration = self.declaration(config_schema=None, defaults={}, interpolate=())

        fiber = self.mount("schemaless-empty", {}, declaration)

        self.assertIs(fiber.state, FiberState.ACTIVE)
        self.assertEqual(self.config_seen_by("schemaless-empty"), {})
        self.assertEqual(dict(fiber.config), {})


class ConfigWaterfallTest(FiberConfigTestCase):
    """``internal/config`` is the pre-validation rewrite hook, so it runs first."""

    def test_the_listener_sees_the_interpolated_row_config_before_defaults(self) -> None:
        drafts: list[ConfigDraft] = []

        def observe(draft: ConfigDraft, forward: Callable[..., Any]) -> Any:
            drafts.append(draft)
            return forward(draft)

        self.listen(observe)

        fiber = self.mount(
            "observed", {"workspace": "${aka:workspace}/runs", "attempts": 99}
        )

        self.assertEqual(len(drafts), 1)
        self.assertEqual(drafts[0].entry_id, "observed")
        self.assertEqual(drafts[0].module, MODULE)
        # Interpolation already happened; defaults have not been merged yet, and the
        # out-of-range value is still there because validation comes last.
        self.assertEqual(
            dict(drafts[0].config), {"workspace": "/var/aka/runs", "attempts": 99}
        )
        self.assert_config_rejected(fiber, "attempts: number outside allowed range")

    def test_a_listener_can_repair_an_invalid_config_before_validation(self) -> None:
        def repair(draft: ConfigDraft, forward: Callable[..., Any]) -> Any:
            return forward(draft.replace({"workspace": "/repaired", "attempts": 3}))

        self.listen(repair)

        # Invalid twice over on arrival: no workspace, and attempts above the maximum.
        fiber = self.mount("repaired", {"attempts": 99})

        self.assertIs(fiber.state, FiberState.ACTIVE)
        self.assertEqual(self.root.failed, ())
        # Defaults are still merged under whatever the listener produced.
        self.assertEqual(
            self.config_seen_by("repaired"),
            {"workspace": "/repaired", "attempts": 3, "mode": "fast"},
        )

    def test_a_listener_that_breaks_a_valid_config_fails_the_fiber(self) -> None:
        def sabotage(draft: ConfigDraft, forward: Callable[..., Any]) -> Any:
            return forward(draft.replace({**draft.config, "mode": "turbo"}))

        self.listen(sabotage)

        fiber = self.mount("sabotaged", {"workspace": "/ok"})

        self.assert_config_rejected(fiber, "mode: unsupported value")
        self.assertIn("sabotaged", self.stderr.getvalue())

    def test_listeners_run_in_order_and_each_sees_the_previous_rewrite(self) -> None:
        def stamp(suffix: str) -> Callable[..., Any]:
            def listener(draft: ConfigDraft, forward: Callable[..., Any]) -> Any:
                config = {**draft.config, "workspace": draft.config["workspace"] + suffix}
                return forward(draft.replace(config))

            return listener

        self.listen(stamp("-outer"), order=100)
        self.listen(stamp("-inner"), order=900)

        self.mount("chained", {"workspace": "/base"})

        self.assertEqual(self.config_seen_by("chained")["workspace"], "/base-outer-inner")
        self.assertEqual(self.root.bus.short_circuits, ())

    def test_the_waterfall_runs_before_the_no_schema_check(self) -> None:
        def inject_config(draft: ConfigDraft, forward: Callable[..., Any]) -> Any:
            return forward(draft.replace({"surprise": True}))

        self.listen(inject_config)
        declaration = self.declaration(config_schema=None, defaults={}, interpolate=())

        fiber = self.mount("schemaless-rewritten", {}, declaration)

        # The row itself supplied nothing, so only a listener can have added this.
        self.assert_config_rejected(fiber, "declares no Config schema: surprise")

    def test_a_throwing_config_listener_never_reaches_apply(self) -> None:
        """A listener bug must not be able to smuggle an unvalidated config into apply.

        ``Fiber._load`` contains a failing ``apply`` (``except Exception`` -> ``_fail``) but
        contains only ``CoreError`` around ``_resolve_config``, so a listener raising anything
        else escapes ``Root.settle()`` and leaves the fiber in ``LOADING`` for good: its epoch
        is already current, so ``refresh()`` never retries it, and it appears in none of
        ``Root.active``, ``Root.failed`` or ``Root.pending``.
        """

        def boom(draft: ConfigDraft, forward: Callable[..., Any]) -> Any:
            raise KeyError("listener bug")

        self.listen(boom)
        fiber = self.root.mount(self.declaration(), {"workspace": "/ok"}, entry_id="thrown")

        self.root.settle()

        # A throwing listener is contained exactly like a failing apply: the row is FAILED and
        # visible in every diagnostic, rather than stranded in LOADING with a live epoch that
        # refresh() will never revisit.
        self.assertEqual(self.applied, [])
        self.assertIsNone(fiber.config)
        self.assertIs(fiber.state, FiberState.FAILED)
        self.assertEqual([entry_id for entry_id, _ in self.root.failed], ["thrown"])
        self.assertEqual(len(self.root.errors), 1)


class FiberInterpolationTest(FiberConfigTestCase):
    """Interpolation as a fiber sees it: allowlisted fields only, before validation."""

    def setUp(self) -> None:
        super().setUp()
        os.environ[ENV_NAME] = "/from/env"
        self.addCleanup(os.environ.pop, ENV_NAME, None)
        os.environ.pop(UNSET_ENV_NAME, None)

    def test_allowlisted_scalar_and_list_fields_are_substituted(self) -> None:
        fiber = self.mount(
            "interpolated",
            {
                "workspace": "${aka:workspace}/runs/${env:" + ENV_NAME + "}",
                "tags": ["${aka:campaign_name}", "static", "${env:" + UNSET_ENV_NAME + "}"],
            },
        )

        self.assertEqual(
            self.config_seen_by("interpolated"),
            {
                "workspace": "/var/aka/runs//from/env",
                "tags": ["demo", "static", ""],
                "attempts": 2,
                "mode": "fast",
            },
        )
        # The row keeps the literal reference; substitution happens per load.
        self.assertEqual(
            fiber.raw_config["workspace"],
            "${aka:workspace}/runs/${env:" + ENV_NAME + "}",
        )

    def test_a_reference_in_a_non_allowlisted_field_fails_the_row(self) -> None:
        fiber = self.mount("leaked", {"workspace": "/w", "mode": "${aka:campaign_name}"})

        self.assertIs(fiber.state, FiberState.FAILED)
        self.assertEqual(self.applied, [])
        self.assertIsInstance(fiber.error, CompositionError)
        self.assertEqual(fiber.error.source, "entry:leaked")
        self.assertIn(
            'field "mode" contains "${" but is not listed in the plugin\'s interpolate '
            "allowlist",
            str(fiber.error),
        )

    def test_an_unknown_variable_in_an_allowlisted_field_fails_the_row(self) -> None:
        fiber = self.mount("unknown-var", {"workspace": "${aka:kernel_name}"})

        self.assertIs(fiber.state, FiberState.FAILED)
        self.assertEqual(self.applied, [])
        self.assertIsInstance(fiber.error, CompositionError)
        self.assertIn('unknown composition variable "kernel_name"', str(fiber.error))


class ApplyInterpolationTest(unittest.TestCase):
    """``apply_interpolation`` on its own: two literal forms, one field allowlist."""

    def setUp(self) -> None:
        self.variables = dict(VARIABLES)
        os.environ[ENV_NAME] = "/from/env"
        self.addCleanup(os.environ.pop, ENV_NAME, None)
        os.environ.pop(UNSET_ENV_NAME, None)

    def interpolate(
        self, config: Mapping[str, Any], allowlist: tuple[str, ...] = ()
    ) -> Mapping[str, Any]:
        return apply_interpolation(
            config, allowlist=allowlist, variables=self.variables, source="entry:row"
        )

    def assert_rejected(
        self, config: Mapping[str, Any], allowlist: tuple[str, ...], expected: str
    ) -> CompositionError:
        with self.assertRaises(CompositionError) as caught:
            self.interpolate(config, allowlist)
        self.assertIn(expected, str(caught.exception))
        self.assertEqual(caught.exception.source, "entry:row")
        return caught.exception

    def test_aka_reference_is_substituted_from_the_variable_table(self) -> None:
        self.assertEqual(
            self.interpolate({"work_dir": "${aka:workspace}"}, ("work_dir",)),
            {"work_dir": "/var/aka"},
        )

    def test_every_supported_variable_is_substitutable(self) -> None:
        self.variables = {name: f"value-of-{name}" for name in COMPOSITION_VARS}
        for name in COMPOSITION_VARS:
            with self.subTest(variable=name):
                self.assertEqual(
                    self.interpolate({"work_dir": "${aka:%s}" % name}, ("work_dir",)),
                    {"work_dir": f"value-of-{name}"},
                )

    def test_a_supported_variable_missing_from_the_table_becomes_empty(self) -> None:
        # "operator" is a legal composition variable this table never filled in.
        self.assertNotIn("operator", self.variables)
        self.assertEqual(
            self.interpolate({"work_dir": "[${aka:operator}]"}, ("work_dir",)),
            {"work_dir": "[]"},
        )

    def test_env_reference_reads_the_process_environment(self) -> None:
        self.assertEqual(
            self.interpolate({"work_dir": "${env:%s}/x" % ENV_NAME}, ("work_dir",)),
            {"work_dir": "/from/env/x"},
        )

    def test_an_unset_env_reference_becomes_an_empty_string(self) -> None:
        self.assertEqual(
            self.interpolate({"work_dir": "a${env:%s}b" % UNSET_ENV_NAME}, ("work_dir",)),
            {"work_dir": "ab"},
        )

    def test_env_is_not_restricted_to_the_composition_variables(self) -> None:
        os.environ["repo_root"] = "/env-repo"
        self.addCleanup(os.environ.pop, "repo_root", None)

        self.assertEqual(
            self.interpolate({"work_dir": "${env:repo_root}"}, ("work_dir",)),
            {"work_dir": "/env-repo"},
        )

    def test_several_references_in_one_string_are_all_substituted(self) -> None:
        self.assertEqual(
            self.interpolate(
                {"work_dir": "${aka:repo_root}/${aka:campaign_name}/${env:%s}" % ENV_NAME},
                ("work_dir",),
            ),
            {"work_dir": "/repo/demo//from/env"},
        )

    def test_a_reference_in_a_non_allowlisted_field_is_rejected(self) -> None:
        for label, value in (
            ("aka", "${aka:workspace}"),
            ("env", "${env:%s}" % ENV_NAME),
            ("unrecognised form", "${workspace}"),
            ("embedded", "prefix ${aka:workspace} suffix"),
        ):
            with self.subTest(form=label):
                self.assert_rejected(
                    {"work_dir": value},
                    ("other",),
                    'field "work_dir" contains "${" but is not listed in the plugin\'s '
                    "interpolate allowlist",
                )

    def test_an_unknown_aka_variable_is_rejected_even_when_allowlisted(self) -> None:
        error = self.assert_rejected(
            {"work_dir": "${aka:kernel_name}"},
            ("work_dir",),
            'unknown composition variable "kernel_name"',
        )
        # The message must list what is actually available.
        for name in COMPOSITION_VARS:
            self.assertIn(name, str(error))

    def test_a_reference_inside_a_list_is_substituted_item_by_item(self) -> None:
        self.assertEqual(
            self.interpolate(
                {"tags": ["${aka:campaign_name}", "plain", "${env:%s}" % ENV_NAME, 3]},
                ("tags",),
            ),
            {"tags": ["demo", "plain", "/from/env", 3]},
        )

    def test_a_reference_inside_a_non_allowlisted_list_is_rejected(self) -> None:
        self.assert_rejected(
            {"tags": ["fine", "${aka:workspace}"]},
            ("work_dir",),
            'field "tags" contains "${"',
        )

    def test_the_allowlist_matches_the_full_dotted_path(self) -> None:
        config = {"outer": {"inner": "${aka:workspace}", "other": "plain"}}

        self.assertEqual(
            self.interpolate(config, ("outer.inner",)),
            {"outer": {"inner": "/var/aka", "other": "plain"}},
        )
        # Allowlisting the parent object is not allowlisting its leaves.
        self.assert_rejected(config, ("outer",), 'field "outer.inner" contains "${"')

    def test_the_input_mapping_is_never_mutated(self) -> None:
        nested = {"inner": "${aka:workspace}"}
        config = {"outer": nested, "tags": ["${aka:campaign_name}"]}

        result = self.interpolate(config, ("outer.inner", "tags"))

        self.assertEqual(result["outer"], {"inner": "/var/aka"})
        self.assertEqual(config, {"outer": {"inner": "${aka:workspace}"}, "tags": ["${aka:campaign_name}"]})
        self.assertEqual(nested, {"inner": "${aka:workspace}"})

    def test_values_without_references_pass_through_unchanged(self) -> None:
        config = {
            "attempts": 3,
            "enabled": True,
            "ratio": 0.5,
            "nothing": None,
            "tags": ["a", "b"],
            "outer": {"inner": "plain $ text {not a reference}"},
        }

        self.assertEqual(self.interpolate(config, ("outer.inner", "tags")), config)

    def test_a_reference_inside_an_object_in_a_list_is_rejected(self) -> None:
        """A reference must never survive as literal text just because it sits in an object.

        ``apply_interpolation`` promises to "substitute references in allowlisted fields; reject
        them anywhere else", and the module docstring ties that promise to postmortem 0002, where
        an unnoticed expression became silently truthy data. An array-of-objects field is legal in
        the supported schema dialect, so ``_contains_reference`` recurses into dicts as well as
        lists: unlisted, the field is rejected; listed, the nested reference is substituted. The
        one outcome that must never happen is the reference reaching ``apply`` as literal text.
        """
        config = {"mounts": [{"path": "${aka:workspace}"}]}

        # The shape is a legal Config, so the hole is reachable from a real plugin.
        from plugin_runtime.schema import check_schema

        check_schema(
            {
                "type": "array",
                "items": {"type": "object", "properties": {"path": {"type": "string"}}},
            }
        )
        self.assert_rejected(config, (), 'field "mounts" contains "${"')
        # Allowlisted, the reference inside the nested object is substituted rather than reaching
        # the plugin as literal text.
        allowed = self.interpolate(config, ("mounts",))
        self.assertEqual(allowed["mounts"], [{"path": VARIABLES["workspace"]}])


class SmokeProfileConfigTest(unittest.TestCase):
    """The same pipeline through the real loader: JSON layering plus ``declare()``."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = str(Path(self.temporary.name) / "runs")
        self.stderr = io.StringIO()
        self.profiles = PROFILES_DIR / "test"
        self.addCleanup(self.temporary.cleanup)
        os.environ[ENV_NAME] = self.workspace
        self.addCleanup(os.environ.pop, ENV_NAME, None)

    def compose(self, **kwargs: Any) -> Any:
        return compose("smoke", profiles_dir=self.profiles, **kwargs)

    def test_the_profile_reference_is_resolved_at_load_not_at_compose(self) -> None:
        composition = self.compose(variables={"workspace": self.workspace})

        # Composition keeps the literal reference, so ``aka compose --dump`` stays honest
        # about which layer set the field.
        row = composition.row("stub-driver")
        self.assertEqual(row.config["workspace"], "${aka:workspace}")
        self.assertEqual(row.config_source, "profile:smoke")

        report = boot(composition, stderr=self.stderr)
        self.addCleanup(report.dispose)

        # Composition rows only: an invariant installer mounts a child fiber of its own,
        # and invariants default on under unittest.
        self.assertEqual(
            tuple(entry_id for entry_id in report.active if "/" not in entry_id),
            ("driver-observer", "stub-driver"),
        )
        self.assertEqual(report.failed, ())
        self.assertEqual(report.warnings, ())
        # The provider was constructed from the substituted config.
        self.assertEqual(report.service("driver").workspace, Path(self.workspace))
        self.assertEqual(report.service("driver").campaign_name, "smoke")
        # The row's own Defaults filled everything the profile left unset.
        self.assertEqual(
            dict(report.root.fibers["stub-driver"].config),
            {
                "workspace": self.workspace,
                "campaign_name": "smoke",
                "status": "completed",
                "reason": "stub",
            },
        )

    def test_an_env_reference_from_a_patch_reaches_the_provider(self) -> None:
        composition = self.compose(
            patches=[
                {
                    "id": "stub-driver",
                    "config": {
                        "campaign_name": "smoke",
                        "workspace": "${env:%s}/patched" % ENV_NAME,
                    },
                }
            ],
        )

        report = boot(composition, stderr=self.stderr)
        self.addCleanup(report.dispose)

        self.assertEqual(
            report.service("driver").workspace, Path(self.workspace) / "patched"
        )

    def test_a_reference_in_a_non_allowlisted_field_fails_a_required_row(self) -> None:
        # stub_driver allowlists "workspace" only, so campaign_name may not hold one.
        composition = self.compose(
            patches=[
                {
                    "id": "stub-driver",
                    "config": {
                        "workspace": self.workspace,
                        "campaign_name": "${aka:campaign_name}",
                    },
                }
            ],
            variables={"campaign_name": "demo"},
        )

        with self.assertRaises(BootFailure) as caught:
            boot(composition, stderr=self.stderr)

        self.assertEqual(
            [entry_id for entry_id, _ in caught.exception.failures], ["stub-driver"]
        )
        self.assertIn(
            'field "campaign_name" contains "${" but is not listed', str(caught.exception)
        )

    def test_a_row_config_the_stub_schema_rejects_fails_the_required_row(self) -> None:
        composition = self.compose(
            patches=[
                {
                    "id": "stub-driver",
                    "config": {"workspace": self.workspace, "status": "exploded"},
                }
            ],
        )

        with self.assertRaises(BootFailure) as caught:
            boot(composition, stderr=self.stderr)

        self.assertIn("stub-driver.config.status: unsupported value", str(caught.exception))
        self.assertIn("ConfigError", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
