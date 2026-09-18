"""Behaviour tests for :mod:`aka.core.declaration`.

``declare()`` is the gate every plugin module passes before it can be mounted, so these
tests pin each rule its docstrings state: no ``apply`` at all, an alternative plugin export
alongside ``apply`` (while the harmless package-re-exports-its-submodule shape stays legal),
an ``apply`` whose arity cannot receive ``(ctx, config)``, a ``name`` that does not match the
module's own package, malformed ``inject``/``optional_inject``/``provide``/``interpolate``
sequences, services that are both required and provided, names that are not registered seams,
a ``Config`` that is not a valid object schema, ``Defaults`` that a ``Config`` would reject,
and ``interpolate`` pointing at a field that does not exist.

The modules under test are throwaway :class:`types.ModuleType` objects -- ``declare()`` reads
only ``__name__``, ``__doc__`` and the declaration attributes, so a fixture on disk would add
nothing. The one real module checked here is ``aka.plugins.legacy_campaign``, the only shipped
plugin at stage one, which must keep declaring the ``driver`` seam.
"""

from __future__ import annotations

import unittest
from types import ModuleType
from typing import Any

from aka import seams
from aka.core.declaration import PluginDeclaration, declare, is_plugin
from aka.core.errors import DeclarationError

MODULE = "aka.fixtures.demo_row"

#: A schema that exercises both a constrained scalar and a nested object property.
NESTED_CONFIG: dict[str, Any] = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "minimum": 1},
        "label": {"type": "string"},
        "outer": {"type": "object", "properties": {"inner": {"type": "string"}}},
    },
}

SEQUENCE_EXPORTS = ("inject", "optional_inject", "provide", "interpolate")


def conforming_apply(ctx: Any, config: Any) -> Any:
    """The shape ``declare()`` requires: a plain function taking exactly (ctx, config)."""
    return (ctx, config)


class ApplyClass:
    """Callable, but a class rather than a plain function."""

    def __init__(self, ctx: Any, config: Any):
        self.ctx = ctx


class ApplyObject:
    """Callable instance: ``callable()`` says yes, ``inspect.isfunction`` says no."""

    def __call__(self, ctx: Any, config: Any) -> None:
        return None


def make_module(
    module_name: str = MODULE,
    *,
    doc: str | None = None,
    without_apply: bool = False,
    **attributes: Any,
) -> ModuleType:
    """A throwaway plugin module: conforming by default, broken per keyword argument."""
    module = ModuleType(module_name, doc)
    module.name = "demo-row"
    if not without_apply:
        module.apply = conforming_apply
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class DeclarationTestCase(unittest.TestCase):
    """Shared assertion helpers: a rejection must name the module and the failed rule."""

    def assert_rejected(
        self, module: ModuleType, expected: str, **kwargs: Any
    ) -> DeclarationError:
        with self.assertRaises(DeclarationError) as caught:
            declare(module, **kwargs)
        self.assertIn(expected, str(caught.exception))
        self.assertEqual(caught.exception.module, module.__name__)
        self.assertIn(module.__name__, str(caught.exception))
        return caught.exception


class ApplyExportTest(DeclarationTestCase):
    def test_module_without_a_callable_apply_is_rejected(self) -> None:
        for label, module in (
            ("absent", make_module(without_apply=True)),
            ("not callable", make_module(apply="orchestrator.optimize:main")),
            ("none", make_module(apply=None)),
        ):
            with self.subTest(apply=label):
                self.assert_rejected(module, "module exports no callable apply")

    def test_class_apply_is_rejected(self) -> None:
        self.assert_rejected(
            make_module(apply=ApplyClass),
            "apply must be a plain function taking (ctx, config)",
        )

    def test_callable_object_apply_is_rejected(self) -> None:
        self.assert_rejected(
            make_module(apply=ApplyObject()),
            "apply must be a plain function taking (ctx, config)",
        )

    def test_apply_taking_one_positional_argument_is_rejected(self) -> None:
        def apply(ctx: Any) -> None:
            return None

        self.assert_rejected(
            make_module(apply=apply), "apply takes 1 positional argument(s)"
        )

    def test_apply_taking_three_positional_arguments_is_rejected(self) -> None:
        def apply(ctx: Any, config: Any, extra: Any) -> None:
            return None

        self.assert_rejected(
            make_module(apply=apply), "apply takes 3 positional argument(s)"
        )

    def test_variadic_apply_is_rejected(self) -> None:
        def star_args(ctx: Any, config: Any, *rest: Any) -> None:
            return None

        def star_kwargs(ctx: Any, config: Any, **rest: Any) -> None:
            return None

        def only_variadic(*args: Any) -> None:
            return None

        for label, apply in (
            ("*rest", star_args),
            ("**rest", star_kwargs),
            ("*args only", only_variadic),
        ):
            with self.subTest(shape=label):
                self.assert_rejected(
                    make_module(apply=apply), "apply must not be variadic"
                )

    def test_required_keyword_only_apply_is_rejected(self) -> None:
        def apply(ctx: Any, config: Any, *, workspace: Any) -> None:
            return None

        self.assert_rejected(
            make_module(apply=apply),
            "apply must not require keyword-only arguments",
        )

    def test_conforming_apply_shapes_are_accepted_and_returned_unchanged(self) -> None:
        calls: list[tuple[Any, Any]] = []

        def two_positional(ctx: Any, config: Any) -> None:
            calls.append((ctx, config))

        def positional_only(ctx: Any, config: Any, /) -> None:
            calls.append((ctx, config))

        def optional_keyword(ctx: Any, config: Any, *, verbose: bool = False) -> None:
            calls.append((ctx, config))

        for label, apply in (
            ("(ctx, config)", two_positional),
            ("(ctx, config, /)", positional_only),
            ("keyword-only default", optional_keyword),
        ):
            with self.subTest(shape=label):
                calls.clear()
                declaration = declare(make_module(apply=apply))
                self.assertIs(declaration.apply, apply)
                declaration.apply("ctx-sentinel", {"count": 2})
                self.assertEqual(calls, [("ctx-sentinel", {"count": 2})])

    def test_is_plugin_only_asks_for_a_callable_apply(self) -> None:
        self.assertTrue(is_plugin(make_module()))
        self.assertTrue(is_plugin(make_module(apply=ApplyObject())))
        self.assertFalse(is_plugin(make_module(without_apply=True)))
        self.assertFalse(is_plugin(make_module(apply="not-callable")))


class RivalExportTest(DeclarationTestCase):
    def test_non_module_plugin_export_alongside_apply_is_rejected(self) -> None:
        for export, value in (
            ("PLUGIN", {"name": "demo-row"}),
            ("Plugin", ApplyClass),
            ("plugin", ApplyObject()),
        ):
            with self.subTest(export=export):
                error = self.assert_rejected(
                    make_module(**{export: value}),
                    f"exports apply and also {export}",
                )
                self.assertIn("declares a plugin one way only", str(error))

    def test_every_rival_export_is_named_in_one_error(self) -> None:
        module = make_module(PLUGIN=1, Plugin=2, plugin=3)
        self.assert_rejected(module, "exports apply and also PLUGIN, Plugin, plugin")

    def test_module_valued_plugin_attribute_is_accepted(self) -> None:
        # The normal package shape: ``aka.plugins.x/__init__.py`` re-exports from
        # ``aka.plugins.x.plugin``, which binds the submodule as an attribute.
        package = make_module("aka.fixtures.demo_row")
        package.plugin = make_module("aka.fixtures.demo_row.plugin")

        declaration = declare(package)

        self.assertEqual(declaration.name, "demo-row")
        self.assertEqual(declaration.module, "aka.fixtures.demo_row")


class NameTest(DeclarationTestCase):
    def test_name_must_be_a_lowercase_stable_identifier(self) -> None:
        for value in ("Demo-Row", "1demo", "demo row", "", "-demo"):
            with self.subTest(name=value):
                self.assert_rejected(
                    make_module(name=value),
                    "name must be a lowercase stable identifier",
                )

    def test_missing_or_non_string_name_is_rejected(self) -> None:
        module = make_module()
        del module.name
        self.assert_rejected(module, "name must be a lowercase stable identifier")
        self.assert_rejected(
            make_module(name=b"demo-row"), "name must be a lowercase stable identifier"
        )

    def test_name_must_equal_the_module_last_segment(self) -> None:
        error = self.assert_rejected(
            make_module("aka.fixtures.demo_row", name="other-row"),
            'name must be "demo-row" to match its package, got "other-row"',
        )
        self.assertIsInstance(error, DeclarationError)

    def test_module_segment_underscores_become_dashes(self) -> None:
        self.assertEqual(
            declare(make_module("aka.fixtures.demo_row", name="demo-row")).name,
            "demo-row",
        )
        # The regex allows an underscore, but the package-matching rule does not.
        self.assert_rejected(
            make_module("aka.fixtures.demo_row", name="demo_row"),
            'name must be "demo-row" to match its package, got "demo_row"',
        )

    def test_plugin_submodule_takes_its_package_name(self) -> None:
        declaration = declare(make_module("aka.fixtures.demo_row.plugin"))

        self.assertEqual(declaration.name, "demo-row")
        self.assertEqual(declaration.module, "aka.fixtures.demo_row.plugin")
        self.assert_rejected(
            make_module("aka.fixtures.demo_row.plugin", name="plugin"),
            'name must be "demo-row" to match its package, got "plugin"',
        )

    def test_doc_is_the_stripped_module_docstring(self) -> None:
        self.assertEqual(
            declare(make_module(doc="  Drives one campaign.\n\nMore prose.\n")).doc,
            "Drives one campaign.\n\nMore prose.",
        )
        self.assertEqual(declare(make_module()).doc, "")


class SequenceExportTest(DeclarationTestCase):
    def test_bare_string_is_rejected_for_every_sequence_export(self) -> None:
        for export in SEQUENCE_EXPORTS:
            with self.subTest(export=export):
                self.assert_rejected(
                    make_module(**{export: "driver"}),
                    f"{export} must be a tuple of strings, got str",
                )

    def test_non_sequence_value_is_rejected(self) -> None:
        self.assert_rejected(
            make_module(inject=None), "inject must be a tuple of strings, got NoneType"
        )
        self.assert_rejected(
            make_module(provide={"driver"}),
            "provide must be a tuple of strings, got set",
        )

    def test_repeated_entry_is_rejected_for_every_sequence_export(self) -> None:
        for export in SEQUENCE_EXPORTS:
            with self.subTest(export=export):
                self.assert_rejected(
                    make_module(**{export: ("driver", "driver")}),
                    f"{export} repeats driver",
                )

    def test_empty_or_non_string_entry_is_rejected(self) -> None:
        self.assert_rejected(
            make_module(inject=("driver", "")), "inject contains a non-string entry"
        )
        self.assert_rejected(
            make_module(optional_inject=("driver", 7)),
            "optional_inject contains a non-string entry",
        )

    def test_lists_are_normalised_to_tuples_and_defaults_are_empty(self) -> None:
        declaration = declare(make_module(inject=["driver", "telemetry"]))

        self.assertEqual(declaration.inject, ("driver", "telemetry"))
        self.assertEqual(declaration.required_services, ("driver", "telemetry"))
        self.assertEqual(declaration.optional_inject, ())
        self.assertEqual(declaration.provide, ())
        self.assertEqual(declaration.interpolate, ())
        self.assertEqual(declaration.defaults, {})
        self.assertIsNone(declaration.config_schema)

    def test_inject_and_provide_may_not_overlap(self) -> None:
        self.assert_rejected(
            make_module(inject=("driver", "telemetry"), provide=("telemetry",)),
            "cannot inject and provide the same service: telemetry",
        )

    def test_inject_and_optional_inject_may_not_overlap(self) -> None:
        self.assert_rejected(
            make_module(inject=("driver",), optional_inject=("driver", "telemetry")),
            "service is both required and optional: driver",
        )


class KnownSeamsTest(DeclarationTestCase):
    def test_unregistered_seam_is_rejected(self) -> None:
        self.assert_rejected(
            make_module(inject=("gpu",)),
            "references unregistered seams: gpu",
            known_seams=seams.keys(),
        )

    def test_optional_inject_and_provide_are_checked_too(self) -> None:
        self.assert_rejected(
            make_module(optional_inject=("telemetry",), provide=("bogus",)),
            "references unregistered seams: bogus, telemetry",
            known_seams=seams.keys(),
        )

    def test_registered_seam_is_accepted(self) -> None:
        declaration = declare(
            make_module(inject=("driver",)), known_seams=seams.keys()
        )

        self.assertEqual(declaration.inject, ("driver",))

    def test_names_are_unchecked_when_no_seam_table_is_supplied(self) -> None:
        declaration = declare(make_module(inject=("gpu",), provide=("bogus",)))

        self.assertEqual(declaration.inject, ("gpu",))
        self.assertEqual(declaration.provide, ("bogus",))


class ConfigTest(DeclarationTestCase):
    def test_config_must_be_a_dict(self) -> None:
        self.assert_rejected(
            make_module(Config="an object with count"), "Config must be a schema dict"
        )

    def test_config_must_pass_check_schema(self) -> None:
        for label, schema in (
            ("required without property", {"type": "object", "required": ["count"]}),
            ("misapplied keyword", {"type": "object", "minLength": 2}),
            ("unknown keyword", {"type": "object", "patternProperties": {}}),
            ("no type", {"properties": {}}),
            (
                "invalid child",
                {"type": "object", "properties": {"count": {"type": "int"}}},
            ),
        ):
            with self.subTest(schema=label):
                self.assert_rejected(make_module(Config=schema), "Config is invalid:")

    def test_config_must_have_object_type(self) -> None:
        self.assert_rejected(
            make_module(Config={"type": "string"}), 'Config must have type "object"'
        )

    def test_valid_object_config_is_carried_on_the_declaration(self) -> None:
        declaration = declare(make_module(Config=NESTED_CONFIG))

        self.assertEqual(declaration.config_schema, NESTED_CONFIG)


class DefaultsTest(DeclarationTestCase):
    def test_defaults_must_be_a_dict(self) -> None:
        self.assert_rejected(
            make_module(Config=NESTED_CONFIG, Defaults=[("count", 2)]),
            "Defaults must be a dict",
        )

    def test_defaults_may_not_set_unknown_fields(self) -> None:
        self.assert_rejected(
            make_module(Config=NESTED_CONFIG, Defaults={"count": 2, "nope": 1}),
            "Defaults sets unknown fields: nope",
        )

    def test_each_default_validates_against_its_own_property_schema(self) -> None:
        self.assert_rejected(
            make_module(Config=NESTED_CONFIG, Defaults={"count": 0}),
            "Defaults.count: number outside allowed range",
        )
        self.assert_rejected(
            make_module(Config=NESTED_CONFIG, Defaults={"label": 3}),
            "Defaults.label: expected string",
        )
        self.assert_rejected(
            make_module(Config=NESTED_CONFIG, Defaults={"outer": {"inner": 1}}),
            "Defaults.outer.inner: expected string",
        )

    def test_defaults_without_a_config_is_rejected(self) -> None:
        self.assert_rejected(
            make_module(Defaults={"count": 2}),
            "Defaults declared without a Config schema",
        )

    def test_empty_defaults_without_a_config_is_accepted(self) -> None:
        self.assertEqual(declare(make_module(Defaults={})).defaults, {})

    def test_valid_defaults_are_copied_onto_the_declaration(self) -> None:
        module = make_module(Config=NESTED_CONFIG, Defaults={"count": 4, "label": "x"})

        declaration = declare(module)
        module.Defaults["count"] = 99

        self.assertEqual(declaration.defaults, {"count": 4, "label": "x"})


class InterpolateTest(DeclarationTestCase):
    def test_interpolate_must_name_a_config_property(self) -> None:
        self.assert_rejected(
            make_module(Config=NESTED_CONFIG, interpolate=("work_dir",)),
            'interpolate names "work_dir", which is not a Config property',
        )

    def test_interpolate_without_a_config_is_rejected(self) -> None:
        self.assert_rejected(
            make_module(interpolate=("count",)),
            'interpolate names "count", which is not a Config property',
        )

    def test_nested_property_path_is_accepted(self) -> None:
        declaration = declare(
            make_module(Config=NESTED_CONFIG, interpolate=("label", "outer.inner"))
        )

        self.assertEqual(declaration.interpolate, ("label", "outer.inner"))

    def test_missing_nested_leaf_is_rejected(self) -> None:
        self.assert_rejected(
            make_module(Config=NESTED_CONFIG, interpolate=("outer.missing",)),
            'interpolate names "outer.missing", which is not a Config property',
        )
        self.assert_rejected(
            make_module(Config=NESTED_CONFIG, interpolate=("label.inner",)),
            'interpolate names "label.inner", which is not a Config property',
        )


class WithExtraInjectTest(unittest.TestCase):
    def setUp(self) -> None:
        self.declaration = PluginDeclaration(
            name="demo-row",
            module=MODULE,
            apply=conforming_apply,
            config_schema=NESTED_CONFIG,
            defaults={"count": 4},
            inject=("beta",),
            optional_inject=("alpha", "gamma"),
            provide=("driver",),
            interpolate=("label",),
            doc="Demo row.",
        )

    def test_row_injections_merge_into_inject(self) -> None:
        merged = self.declaration.with_extra_inject(["delta"])

        self.assertEqual(merged.inject, ("beta", "delta"))
        self.assertEqual(merged.required_services, ("beta", "delta"))

    def test_row_injection_is_dropped_from_optional_inject(self) -> None:
        merged = self.declaration.with_extra_inject(["alpha"])

        self.assertEqual(merged.inject, ("alpha", "beta"))
        self.assertEqual(merged.optional_inject, ("gamma",))

    def test_empty_names_are_ignored(self) -> None:
        merged = self.declaration.with_extra_inject(["", "delta"])

        self.assertEqual(merged.inject, ("beta", "delta"))

    def test_nothing_new_returns_the_same_declaration(self) -> None:
        self.assertIs(self.declaration.with_extra_inject([]), self.declaration)
        self.assertIs(self.declaration.with_extra_inject(["beta", ""]), self.declaration)

    def test_the_original_declaration_is_untouched(self) -> None:
        self.declaration.with_extra_inject(["alpha", "delta"])

        self.assertEqual(self.declaration.inject, ("beta",))
        self.assertEqual(self.declaration.optional_inject, ("alpha", "gamma"))

    def test_every_other_field_is_preserved(self) -> None:
        merged = self.declaration.with_extra_inject(["delta"])

        self.assertEqual(merged.name, "demo-row")
        self.assertEqual(merged.module, MODULE)
        self.assertIs(merged.apply, conforming_apply)
        self.assertEqual(merged.config_schema, NESTED_CONFIG)
        self.assertEqual(merged.defaults, {"count": 4})
        self.assertEqual(merged.provide, ("driver",))
        self.assertEqual(merged.interpolate, ("label",))
        self.assertEqual(merged.doc, "Demo row.")

    def test_row_injection_may_not_shadow_a_provided_service(self) -> None:
        """A row must not be able to reintroduce the inject/provide overlap.

        ``declare()`` rejects a module whose ``inject`` and ``provide`` name the same service,
        because such a row can never become active. A composition row's own ``inject`` list
        reaches the declaration through ``with_extra_inject``, which must enforce the same rule --
        otherwise an operator typo produces a row that waits on itself forever instead of a loud
        failure.
        """
        with self.assertRaises(DeclarationError) as caught:
            self.declaration.with_extra_inject(["driver"])

        self.assertIn("driver", str(caught.exception))
        self.assertIn("provides", str(caught.exception))
        # An unrelated row injection is still merged.
        merged = self.declaration.with_extra_inject(["gamma"])
        self.assertEqual(merged.inject, ("beta", "gamma"))
        self.assertEqual(set(merged.inject) & set(merged.provide), set())


class RealPluginTest(unittest.TestCase):
    """The one shipped plugin must satisfy the contract with the real seam table."""

    def test_legacy_campaign_package_declares_the_driver_provider(self) -> None:
        import aka.plugins.legacy_campaign as legacy_campaign

        declaration = declare(legacy_campaign, known_seams=seams.keys())

        self.assertEqual(declaration.name, "legacy-campaign")
        self.assertEqual(declaration.module, "aka.plugins.legacy_campaign")
        self.assertEqual(declaration.provide, ("driver",))
        self.assertEqual(declaration.inject, ())
        self.assertEqual(declaration.optional_inject, ())
        self.assertEqual(declaration.interpolate, ("work_dir",))
        self.assertIs(declaration.apply, legacy_campaign.apply)

    def test_legacy_campaign_config_and_defaults_agree(self) -> None:
        import aka.plugins.legacy_campaign as legacy_campaign

        declaration = declare(legacy_campaign, known_seams=seams.keys())
        properties = declaration.config_schema["properties"]

        self.assertEqual(declaration.config_schema["type"], "object")
        self.assertIn("work_dir", properties)
        self.assertTrue(set(declaration.defaults) <= set(properties))
        # Required fields come from the CLI, so they must not be pre-answered here.
        self.assertEqual(
            set(declaration.defaults) & set(declaration.config_schema["required"]),
            set(),
        )

    def test_legacy_campaign_submodule_declares_the_same_plugin(self) -> None:
        from aka.plugins.legacy_campaign import plugin

        declaration = declare(plugin, known_seams=seams.keys())

        self.assertEqual(declaration.name, "legacy-campaign")
        self.assertEqual(declaration.module, "aka.plugins.legacy_campaign.plugin")
        self.assertEqual(declaration.provide, ("driver",))


if __name__ == "__main__":
    unittest.main()
