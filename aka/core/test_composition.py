"""Behaviour tests for JSON composition layering (:mod:`aka.core.composition`).

``resolve()`` is the only thing that turns files on disk into the row list the loader mounts,
so everything it decides is a contract:

1. a profile stacks its bundles in the order it lists them, and two bundles claiming the same
   row id is a hard failure rather than a silent last-one-wins;
2. layers apply bundle -> profile patches -> ``--patch`` files -> programmatic patches, the last
   writer wins, and the surviving row names the layer that wrote it (``source``) and the layer
   that set its config (``config_source``);
3. a patch replaces a row's **whole** config -- never a deep merge, because a merge makes an
   override's effective value depend on a bundle the override cannot see;
4. row order carries no semantics: permuting the entries of every bundle must produce the same
   resolved rows, since ordering comes from ``inject``;
5. an unknown field, a wrong ``api_version``, or a patch aimed at a row nobody declared is a
   :class:`CompositionError` that names what it rejected.

The layers are written into a tempdir shaped like ``aka/profiles/`` (``<profile>.json`` plus
``bundles/<bundle>.json``) so the real file-reading path runs; nothing here imports a plugin
module, because ``resolve()`` deliberately does not.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import fields
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from aka.core.composition import (
    API_VERSION,
    COMPOSITION_VARS,
    ResolvedComposition,
    Row,
    dump,
    resolve,
)
from aka.core.errors import CompositionError


def rows_by_id(composition: ResolvedComposition) -> dict[str, Row]:
    """The resolved rows keyed by id, i.e. with the list order thrown away."""
    return {row.id: row for row in composition.rows}


class CompositionFixture(unittest.TestCase):
    """A tempdir shaped like ``aka/profiles/`` plus writers for each layer."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.profiles = self.root / "profiles"
        (self.profiles / "bundles").mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    # -- layer writers ---------------------------------------------------

    def write(self, path: Path, document: Any) -> Path:
        """Write a document verbatim, so a test can omit or corrupt any field."""
        path.write_text(json.dumps(document, indent=2), encoding="utf-8")
        return path

    def write_profile(
        self,
        name: str = "main",
        *,
        bundles: Sequence[str] = ("base",),
        patches: Sequence[Mapping[str, Any]] | None = None,
        **extra: Any,
    ) -> Path:
        document: dict[str, Any] = {
            "api_version": API_VERSION,
            "bundles": list(bundles),
        }
        if patches is not None:
            document["patches"] = list(patches)
        document.update(extra)
        return self.write(self.profiles / f"{name}.json", document)

    def write_bundle(self, name: str, entries: Sequence[Any], **extra: Any) -> Path:
        document: dict[str, Any] = {"api_version": API_VERSION, "entries": list(entries)}
        document.update(extra)
        return self.write(self.profiles / "bundles" / f"{name}.json", document)

    def write_patch_file(
        self, name: str, patches: Sequence[Mapping[str, Any]], **extra: Any
    ) -> Path:
        document: dict[str, Any] = {"api_version": API_VERSION, "patches": list(patches)}
        document.update(extra)
        return self.write(self.root / f"{name}.json", document)

    # -- driver ----------------------------------------------------------

    def compose(self, profile: str = "main", **kwargs: Any) -> ResolvedComposition:
        return resolve(self.profiles, profile, **kwargs)

    def failure(self, profile: str = "main", **kwargs: Any) -> CompositionError:
        with self.assertRaises(CompositionError) as caught:
            self.compose(profile, **kwargs)
        return caught.exception


class BundleStackingTest(CompositionFixture):
    """A profile is an ordered stack of bundles, and ids may not collide across them."""

    def test_bundles_stack_in_the_order_the_profile_lists_them(self) -> None:
        self.write_bundle("first", [{"id": "one", "name": "m.one"}])
        self.write_bundle(
            "second",
            [{"id": "two", "name": "m.two"}, {"id": "three", "name": "m.three"}],
        )
        self.write_profile("forward", bundles=("first", "second"))
        self.write_profile("backward", bundles=("second", "first"))

        forward = self.compose("forward")
        backward = self.compose("backward")

        self.assertEqual([row.id for row in forward.rows], ["one", "two", "three"])
        self.assertEqual([row.id for row in backward.rows], ["two", "three", "one"])
        self.assertEqual(forward.layers, ("bundle:first", "bundle:second"))
        self.assertEqual(backward.layers, ("bundle:second", "bundle:first"))
        self.assertEqual(forward.profile, "forward")
        # Every row is attributed to the bundle it came from, not to the profile.
        self.assertEqual(
            {row.id: row.source for row in forward.rows},
            {
                "one": "bundle:first",
                "two": "bundle:second",
                "three": "bundle:second",
            },
        )

    def test_two_bundles_declaring_the_same_row_id_is_a_composition_error(self) -> None:
        self.write_bundle(
            "first",
            [{"id": "shared", "name": "m.one", "config": {"from": "first"}}],
        )
        self.write_bundle(
            "second",
            [{"id": "shared", "name": "m.two", "config": {"from": "second"}}],
        )
        self.write_profile(bundles=("first", "second"))

        error = self.failure()

        # The second bundle is the one that cannot be honoured, so it is named.
        self.assertEqual(error.source, "bundle:second")
        self.assertIn('duplicate row id "shared"', str(error))

    def test_a_row_id_repeated_inside_one_bundle_is_a_composition_error(self) -> None:
        self.write_bundle(
            "base",
            [
                {"id": "one", "name": "m.one"},
                {"id": "one", "name": "m.one.again"},
            ],
        )
        self.write_profile()

        error = self.failure()

        self.assertEqual(error.source, "bundle:base")
        self.assertIn('duplicate row id "one"', str(error))

    def test_stacking_the_same_bundle_twice_is_a_composition_error(self) -> None:
        """Stacking is not idempotent: a bundle listed twice collides with itself."""
        self.write_bundle("base", [{"id": "one", "name": "m.one"}])
        self.write_profile(bundles=("base", "base"))

        error = self.failure()

        self.assertEqual(error.source, "bundle:base")
        self.assertIn('duplicate row id "one"', str(error))

    def test_a_profile_that_stacks_no_bundles_is_a_composition_error(self) -> None:
        self.write_profile(bundles=())

        error = self.failure()

        self.assertEqual(error.source, str(self.profiles / "main.json"))
        self.assertIn("lists no bundles", str(error))

    def test_an_unreadable_layer_names_the_file_it_could_not_use(self) -> None:
        self.write_profile(bundles=("missing",))
        missing = self.failure()
        self.assertEqual(missing.source, str(self.profiles / "bundles" / "missing.json"))
        self.assertIn("cannot read", str(missing))

        (self.profiles / "bundles" / "broken.json").write_text("{not json", encoding="utf-8")
        self.write_profile("broken_profile", bundles=("broken",))
        broken = self.failure("broken_profile")
        self.assertEqual(broken.source, str(self.profiles / "bundles" / "broken.json"))
        self.assertIn("invalid JSON", str(broken))


class WholeConfigReplacementTest(CompositionFixture):
    """A patch restates the config it wants; it never merges into the bundle's."""

    def setUp(self) -> None:
        super().setUp()
        self.write_bundle(
            "base",
            [
                {
                    "id": "one",
                    "name": "m.one",
                    "config": {
                        "attempts": 2,
                        "bundle_only": "gone after any patch",
                        "nested": {"keep": "a", "drop": "b"},
                    },
                }
            ],
        )

    def test_a_profile_patch_replaces_the_targeted_rows_whole_config(self) -> None:
        self.write_profile(patches=[{"id": "one", "config": {"attempts": 7}}])

        row = self.compose().row("one")

        assert row is not None
        self.assertEqual(row.config, {"attempts": 7})
        # The field only the bundle set is gone: replacement, not a deep merge.
        self.assertNotIn("bundle_only", row.config)
        self.assertNotIn("nested", row.config)
        self.assertEqual(row.config_source, "profile:main")
        self.assertEqual(row.source, "profile:main")

    def test_a_nested_object_is_replaced_whole_rather_than_merged_key_by_key(self) -> None:
        self.write_profile(patches=[{"id": "one", "config": {"nested": {"added": "c"}}}])

        row = self.compose().row("one")

        assert row is not None
        self.assertEqual(row.config, {"nested": {"added": "c"}})

    def test_an_empty_config_in_a_patch_clears_the_bundles_config(self) -> None:
        self.write_profile(patches=[{"id": "one", "config": {}}])

        row = self.compose().row("one")

        assert row is not None
        self.assertEqual(row.config, {})
        self.assertEqual(row.config_source, "profile:main")

    def test_a_patch_without_config_keeps_the_config_and_its_provenance(self) -> None:
        """Only ``config`` moves ``config_source``; ``source`` follows the last writer."""
        self.write_profile(patches=[{"id": "one", "required": True}])

        row = self.compose().row("one")

        assert row is not None
        self.assertEqual(
            row.config,
            {
                "attempts": 2,
                "bundle_only": "gone after any patch",
                "nested": {"keep": "a", "drop": "b"},
            },
        )
        self.assertEqual(row.config_source, "bundle:base")
        self.assertEqual(row.source, "profile:main")
        self.assertTrue(row.required)


class LayerOrderTest(CompositionFixture):
    """bundle -> profile patches -> patch files -> programmatic patches, last writer wins."""

    def setUp(self) -> None:
        super().setUp()
        self.write_bundle(
            "base",
            [
                {"id": "one", "name": "m.one", "config": {"owner": "bundle"}},
                {"id": "two", "name": "m.two", "config": {"owner": "bundle"}},
                {"id": "three", "name": "m.three", "config": {"owner": "bundle"}},
                {"id": "four", "name": "m.four", "config": {"owner": "bundle"}},
            ],
        )

    def test_the_last_layer_to_write_a_config_owns_it(self) -> None:
        self.write_profile(
            patches=[
                {"id": "one", "config": {"owner": "profile"}},
                {"id": "two", "config": {"owner": "profile"}},
                {"id": "three", "config": {"owner": "profile"}},
            ]
        )
        first = self.write_patch_file(
            "overlay",
            [
                {"id": "two", "config": {"owner": "patch-file"}},
                {"id": "three", "config": {"owner": "patch-file"}},
            ],
        )

        composition = self.compose(
            patch_files=[first],
            patches=[{"id": "three", "config": {"owner": "cli"}}],
        )

        rows = rows_by_id(composition)
        self.assertEqual(
            {entry_id: row.config["owner"] for entry_id, row in rows.items()},
            {
                "one": "profile",
                "two": "patch-file",
                "three": "cli",
                "four": "bundle",
            },
        )
        # The winning layer is recorded on the row, so provenance is readable per row.
        self.assertEqual(
            {entry_id: row.config_source for entry_id, row in rows.items()},
            {
                "one": "profile:main",
                "two": f"patch:{first}",
                "three": "cli",
                "four": "bundle:base",
            },
        )
        self.assertEqual(
            composition.layers,
            ("bundle:base", "profile:main", f"patch:{first}", "cli"),
        )

    def test_patch_files_apply_in_the_order_they_were_given(self) -> None:
        self.write_profile(patches=[])
        early = self.write_patch_file("early", [{"id": "one", "config": {"owner": "early"}}])
        late = self.write_patch_file("late", [{"id": "one", "config": {"owner": "late"}}])

        forward = self.compose(patch_files=[early, late])
        backward = self.compose(patch_files=[late, early])

        forward_row = forward.row("one")
        backward_row = backward.row("one")
        assert forward_row is not None and backward_row is not None
        self.assertEqual(forward_row.config, {"owner": "late"})
        self.assertEqual(forward_row.config_source, f"patch:{late}")
        self.assertEqual(backward_row.config, {"owner": "early"})
        self.assertEqual(backward_row.config_source, f"patch:{early}")
        self.assertEqual(forward.layers, ("bundle:base", f"patch:{early}", f"patch:{late}"))

    def test_a_layer_that_writes_nothing_does_not_appear_in_layers(self) -> None:
        self.write_profile(patches=[])

        composition = self.compose()

        self.assertEqual(composition.layers, ("bundle:base",))

    def test_every_patchable_attribute_can_be_overridden_by_a_later_layer(self) -> None:
        self.write_profile(
            patches=[
                {
                    "id": "one",
                    "required": True,
                    "inject": ["driver"],
                    "isolate": ["driver"],
                    "doc": "set by the profile",
                }
            ]
        )
        overlay = self.write_patch_file(
            "overlay",
            [{"id": "one", "required": False, "inject": [], "doc": "set by the patch file"}],
        )

        row = self.compose(patch_files=[overlay]).row("one")

        assert row is not None
        self.assertFalse(row.required)
        self.assertEqual(row.inject, ())
        self.assertEqual(row.isolate, ("driver",))
        self.assertEqual(row.doc, "set by the patch file")
        self.assertEqual(row.source, f"patch:{overlay}")


class InsertRemoveDisableTest(CompositionFixture):
    """What a patch may do to the row set: create, delete, or park a row."""

    def setUp(self) -> None:
        super().setUp()
        self.write_bundle(
            "base",
            [
                {"id": "one", "name": "m.one", "config": {"owner": "bundle"}},
                {"id": "two", "name": "m.two"},
            ],
        )

    def test_a_patch_for_an_unknown_id_without_insert_is_a_composition_error(self) -> None:
        self.write_profile(patches=[{"id": "ghost", "config": {"owner": "profile"}}])

        error = self.failure()

        self.assertEqual(error.source, "profile:main")
        self.assertIn('patch targets unknown row "ghost"', str(error))
        self.assertIn("insert", str(error))

    def test_a_patch_with_insert_appends_the_row(self) -> None:
        self.write_profile(
            patches=[
                {
                    "id": "added",
                    "insert": {
                        "name": "m.added",
                        "config": {"owner": "profile"},
                        "required": True,
                        "inject": ["driver"],
                        "isolate": ["driver"],
                        "doc": "inserted by the profile",
                    },
                }
            ]
        )

        composition = self.compose()

        self.assertEqual([row.id for row in composition.rows], ["one", "two", "added"])
        row = composition.row("added")
        assert row is not None
        self.assertEqual(row.name, "m.added")
        self.assertEqual(row.config, {"owner": "profile"})
        self.assertTrue(row.required)
        self.assertEqual(row.inject, ("driver",))
        self.assertEqual(row.isolate, ("driver",))
        self.assertEqual(row.doc, "inserted by the profile")
        self.assertEqual(row.source, "profile:main")
        self.assertEqual(row.config_source, "profile:main")

    def test_an_inserted_row_still_needs_a_module_path(self) -> None:
        self.write_profile(patches=[{"id": "added", "insert": {"config": {"owner": "x"}}}])

        error = self.failure()

        self.assertEqual(error.source, "profile:main")
        self.assertIn('entry "added" needs a module path in "name"', str(error))

    def test_insert_cannot_replace_a_row_that_already_exists(self) -> None:
        self.write_profile(
            patches=[{"id": "one", "insert": {"name": "m.other"}}]
        )

        error = self.failure()

        self.assertEqual(error.source, "profile:main")
        self.assertIn('patch "one" cannot insert a row that already exists', str(error))

    def test_remove_deletes_the_row_entirely(self) -> None:
        self.write_profile(patches=[{"id": "one", "remove": True}])

        composition = self.compose()

        self.assertEqual([row.id for row in composition.rows], ["two"])
        self.assertIsNone(composition.row("one"))

    def test_remove_false_keeps_the_row(self) -> None:
        self.write_profile(patches=[{"id": "one", "remove": False, "required": True}])

        row = self.compose().row("one")

        assert row is not None
        self.assertTrue(row.required)

    def test_removing_a_row_a_later_layer_patches_is_a_composition_error(self) -> None:
        """A removed row is gone, so a later patch aimed at it has no target."""
        self.write_profile(patches=[{"id": "one", "remove": True}])
        overlay = self.write_patch_file("overlay", [{"id": "one", "config": {"owner": "late"}}])

        error = self.failure(patch_files=[overlay])

        self.assertEqual(error.source, f"patch:{overlay}")
        self.assertIn('patch targets unknown row "one"', str(error))

    def test_disabled_keeps_the_row_in_rows_but_out_of_enabled(self) -> None:
        self.write_profile(patches=[{"id": "one", "disabled": True}])

        composition = self.compose()

        row = composition.row("one")
        assert row is not None
        self.assertTrue(row.disabled)
        self.assertIn("one", [entry.id for entry in composition.rows])
        self.assertEqual([entry.id for entry in composition.enabled], ["two"])

    def test_a_later_layer_can_re_enable_a_disabled_row(self) -> None:
        self.write_bundle(
            "base",
            [
                {"id": "one", "name": "m.one", "disabled": True},
                {"id": "two", "name": "m.two"},
            ],
        )
        self.write_profile(patches=[{"id": "one", "disabled": False}])

        composition = self.compose()

        self.assertEqual([row.id for row in composition.enabled], ["one", "two"])


class RowOrderInvarianceTest(CompositionFixture):
    """Row order is for readers only: permuting entries cannot change the outcome."""

    ENTRIES_A: tuple[Mapping[str, Any], ...] = (
        {"id": "one", "name": "m.one", "config": {"owner": "bundle-a"}, "inject": ["driver"]},
        {"id": "two", "name": "m.two", "required": True},
        {
            "id": "realm",
            "isolate": ["driver"],
            "entries": [
                {"id": "three", "name": "m.three", "config": {"owner": "bundle-a"}},
                {"id": "four", "name": "m.four", "isolate": ["telemetry"]},
            ],
        },
        {"id": "five", "name": "m.five", "disabled": True},
    )
    ENTRIES_B: tuple[Mapping[str, Any], ...] = (
        {"id": "six", "name": "m.six"},
        {"id": "seven", "name": "m.seven", "config": {"owner": "bundle-b"}},
        {"id": "eight", "name": "m.eight", "doc": "last in the second bundle"},
    )

    def write_layers(
        self,
        entries_a: Iterable[Mapping[str, Any]],
        entries_b: Iterable[Mapping[str, Any]],
    ) -> ResolvedComposition:
        self.write_bundle("a", list(entries_a))
        self.write_bundle("b", list(entries_b))
        self.write_profile(
            bundles=("a", "b"),
            patches=[
                {"id": "seven", "config": {"owner": "profile"}},
                {"id": "two", "required": False},
                {"id": "nine", "insert": {"name": "m.nine", "config": {"owner": "profile"}}},
            ],
        )
        overlay = self.write_patch_file("overlay", [{"id": "one", "disabled": True}])
        return self.compose(
            patch_files=[overlay],
            patches=[{"id": "six", "config": {"owner": "cli"}}],
        )

    def test_permuting_the_entries_of_every_bundle_resolves_identically(self) -> None:
        baseline = rows_by_id(self.write_layers(self.ENTRIES_A, self.ENTRIES_B))
        self.assertEqual(
            set(baseline),
            {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine"},
        )

        for permutation_a in permutations(self.ENTRIES_A):
            for permutation_b in permutations(self.ENTRIES_B):
                with self.subTest(a=[e["id"] for e in permutation_a], b=[e["id"] for e in permutation_b]):
                    shuffled = rows_by_id(self.write_layers(permutation_a, permutation_b))
                    # Identical row set, and identical rows: config, flags, group, isolation
                    # and provenance all come from the layers, never from row position.
                    self.assertEqual(shuffled, baseline)

    def test_permuting_a_groups_children_resolves_identically(self) -> None:
        group = dict(self.ENTRIES_A[2])
        children = list(group["entries"])
        baseline = rows_by_id(self.write_layers(self.ENTRIES_A, self.ENTRIES_B))

        for permutation in permutations(children):
            entries = list(self.ENTRIES_A)
            entries[2] = {**group, "entries": list(permutation)}
            with self.subTest(children=[child["id"] for child in permutation]):
                self.assertEqual(rows_by_id(self.write_layers(entries, self.ENTRIES_B)), baseline)


class RejectedDocumentTest(CompositionFixture):
    """Unknown fields and wrong api_versions are refused by name, at every level."""

    def test_an_unknown_profile_field_is_rejected_by_name(self) -> None:
        self.write_bundle("base", [{"id": "one", "name": "m.one"}])
        self.write_profile(entries=[], plugins=[])

        error = self.failure()

        self.assertEqual(error.source, str(self.profiles / "main.json"))
        self.assertIn("unsupported field(s): entries, plugins", str(error))

    def test_an_unknown_bundle_field_is_rejected_by_name(self) -> None:
        self.write_bundle("base", [{"id": "one", "name": "m.one"}], patches=[])
        self.write_profile()

        error = self.failure()

        self.assertEqual(error.source, str(self.profiles / "bundles" / "base.json"))
        self.assertIn("unsupported field(s): patches", str(error))

    def test_an_unknown_row_field_is_rejected_by_name(self) -> None:
        self.write_bundle(
            "base", [{"id": "one", "name": "m.one", "settings": {"attempts": 2}}]
        )
        self.write_profile()

        error = self.failure()

        self.assertEqual(error.source, "bundle:base")
        self.assertIn("unsupported field(s): settings", str(error))

    def test_an_unknown_group_field_is_rejected_by_name(self) -> None:
        self.write_bundle(
            "base",
            [{"id": "realm", "children": [], "entries": [{"id": "one", "name": "m.one"}]}],
        )
        self.write_profile()

        error = self.failure()

        self.assertEqual(error.source, "bundle:base")
        self.assertIn("unsupported field(s): children", str(error))

    def test_an_unknown_patch_field_is_rejected_by_name(self) -> None:
        self.write_bundle("base", [{"id": "one", "name": "m.one"}])
        self.write_profile(patches=[{"id": "one", "name": "m.replacement"}])

        error = self.failure()

        self.assertEqual(error.source, "profile:main")
        # "name" is a row field but not a patchable one: a patch retargets config, not module.
        self.assertIn("unsupported field(s): name", str(error))

    def test_an_unknown_patch_file_field_is_rejected_by_name(self) -> None:
        self.write_bundle("base", [{"id": "one", "name": "m.one"}])
        self.write_profile()
        overlay = self.write_patch_file("overlay", [{"id": "one", "disabled": True}], bundles=[])

        error = self.failure(patch_files=[overlay])

        self.assertEqual(error.source, f"patch:{overlay}")
        self.assertIn("unsupported field(s): bundles", str(error))

    def test_api_version_must_be_exactly_one(self) -> None:
        self.write_bundle("base", [{"id": "one", "name": "m.one"}])
        self.write_profile()
        self.assertEqual(self.compose().layers, ("bundle:base",))

        # A profile from a future or a past schema is refused, not guessed at.
        self.write_profile(api_version=2)
        self.assertIn("api_version must be 1, got 2", str(self.failure()))

        # A missing api_version is as unusable as a wrong one.
        self.write(self.profiles / "main.json", {"bundles": ["base"]})
        self.assertIn("api_version must be 1, got None", str(self.failure()))

        # The version is a number, not a string that happens to read as one.
        self.write_profile(api_version="1")
        self.assertIn("api_version must be 1, got '1'", str(self.failure()))

        # Every layer is checked, not just the profile.
        self.write_profile()
        self.write_bundle("base", [{"id": "one", "name": "m.one"}], api_version=0)
        bundle_error = self.failure()
        self.assertEqual(bundle_error.source, str(self.profiles / "bundles" / "base.json"))
        self.assertIn("api_version must be 1, got 0", str(bundle_error))

        self.write_bundle("base", [{"id": "one", "name": "m.one"}])
        overlay = self.write_patch_file("overlay", [{"id": "one", "disabled": True}])
        self.write(self.root / "overlay.json", {"api_version": 2, "patches": []})
        patch_error = self.failure(patch_files=[overlay])
        self.assertEqual(patch_error.source, str(overlay))
        self.assertIn("api_version must be 1, got 2", str(patch_error))

        # Python makes True == 1 and 1.0 == 1, so an equality check alone would let a document
        # from some future schema load as version 1.
        self.write_profile(api_version=True)
        self.assertIn("api_version must be 1, got True", str(self.failure()))
        self.write_profile(api_version=1.0)
        self.assertIn("api_version must be 1, got 1.0", str(self.failure()))

    def test_a_row_needs_a_string_id_and_a_module_path(self) -> None:
        self.write_profile()
        self.write_bundle("base", [{"name": "m.one"}])
        self.assertIn("entry needs a string id", str(self.failure()))

        self.write_bundle("base", [{"id": "one"}])
        self.assertIn('entry "one" needs a module path in "name"', str(self.failure()))

        self.write_bundle("base", [{"id": "one", "name": "m.one", "config": []}])
        self.assertIn('entry "one" config must be an object', str(self.failure()))

        self.write_bundle("base", [{"id": "one", "name": "m.one", "inject": "driver"}])
        self.assertIn('entry "one" inject must be an array of strings', str(self.failure()))


class GroupTest(CompositionFixture):
    """A group is a readability and isolation device, not a mount level."""

    def test_a_group_flattens_its_children_and_records_the_group_id(self) -> None:
        self.write_bundle(
            "base",
            [
                {"id": "solo", "name": "m.solo"},
                {
                    "id": "realm",
                    "doc": "two rows that share one isolation realm",
                    "entries": [
                        {"id": "child-a", "name": "m.a", "config": {"owner": "bundle"}},
                        {"id": "child-b", "name": "m.b"},
                    ],
                },
            ],
        )
        self.write_profile()

        composition = self.compose()

        # The group itself never becomes a row; only its children do.
        self.assertEqual([row.id for row in composition.rows], ["solo", "child-a", "child-b"])
        self.assertIsNone(composition.row("realm"))
        self.assertEqual(
            {row.id: row.group for row in composition.rows},
            {"solo": "", "child-a": "realm", "child-b": "realm"},
        )
        child = composition.row("child-a")
        assert child is not None
        self.assertEqual(child.source, "bundle:base")
        self.assertEqual(child.config, {"owner": "bundle"})

    def test_a_group_propagates_its_isolation_and_merges_it_with_the_childs(self) -> None:
        self.write_bundle(
            "base",
            [
                {
                    "id": "realm",
                    "isolate": ["driver"],
                    "entries": [
                        {"id": "plain", "name": "m.plain"},
                        {"id": "extra", "name": "m.extra", "isolate": ["telemetry"]},
                        {"id": "same", "name": "m.same", "isolate": ["driver"]},
                    ],
                }
            ],
        )
        self.write_profile()

        rows = rows_by_id(self.compose())

        self.assertEqual(rows["plain"].isolate, ("driver",))
        self.assertEqual(rows["extra"].isolate, ("driver", "telemetry"))
        # A child that repeats the group's realm is not isolated twice.
        self.assertEqual(rows["same"].isolate, ("driver",))

    def test_a_disabled_group_disables_every_child(self) -> None:
        self.write_bundle(
            "base",
            [
                {
                    "id": "realm",
                    "disabled": True,
                    "entries": [
                        {"id": "child-a", "name": "m.a"},
                        {"id": "child-b", "name": "m.b", "required": True},
                    ],
                },
                {"id": "solo", "name": "m.solo"},
            ],
        )
        self.write_profile()

        composition = self.compose()

        self.assertTrue(all(row.disabled for row in composition.rows if row.group == "realm"))
        self.assertEqual([row.id for row in composition.enabled], ["solo"])
        # A disabled group's required child cannot block boot, because it is not enabled.
        self.assertEqual(composition.required, frozenset())

    def test_a_patch_can_re_enable_one_child_of_a_disabled_group(self) -> None:
        self.write_bundle(
            "base",
            [
                {
                    "id": "realm",
                    "disabled": True,
                    "entries": [
                        {"id": "child-a", "name": "m.a"},
                        {"id": "child-b", "name": "m.b"},
                    ],
                }
            ],
        )
        self.write_profile(patches=[{"id": "child-b", "disabled": False}])

        composition = self.compose()

        self.assertEqual([row.id for row in composition.enabled], ["child-b"])

    def test_a_group_is_not_a_patch_target(self) -> None:
        self.write_bundle(
            "base",
            [{"id": "realm", "entries": [{"id": "child-a", "name": "m.a"}]}],
        )
        self.write_profile(patches=[{"id": "realm", "disabled": True}])

        error = self.failure()

        self.assertIn('patch targets unknown row "realm"', str(error))

    def test_nested_groups_accumulate_isolation_and_keep_the_outermost_label(self) -> None:
        self.write_bundle(
            "base",
            [
                {
                    "id": "outer",
                    "isolate": ["driver"],
                    "entries": [
                        {
                            "id": "inner",
                            "isolate": ["telemetry"],
                            "entries": [{"id": "leaf", "name": "m.leaf"}],
                        }
                    ],
                }
            ],
        )
        self.write_profile()

        leaf = self.compose().row("leaf")

        assert leaf is not None
        self.assertEqual(leaf.isolate, ("driver", "telemetry"))
        self.assertEqual(leaf.group, "outer")

    def test_a_group_cannot_carry_the_row_fields_it_cannot_honour(self) -> None:
        """Only ``id``, ``isolate``, ``disabled``, ``doc`` and ``entries`` mean anything on a group.

        A group is a readability and isolation device, not a mount level: it owns no module and no
        config. ``required``, ``config``, ``inject`` and ``name`` written on one would have nowhere
        to go, so they are refused by name rather than dropped without a word -- the silent-data
        class this module's docstring sets out to remove.
        """
        self.write_bundle(
            "base",
            [
                {
                    "id": "realm",
                    "name": "m.realm",
                    "required": True,
                    "config": {"shared": 1},
                    "inject": ["driver"],
                    "entries": [{"id": "kid", "name": "m.kid"}],
                }
            ],
        )
        self.write_profile()

        error = self.failure()
        self.assertIn("unsupported field(s): config, inject, name, required", str(error))

        # Stripped back to what a group can honour, it resolves and records itself on its child.
        self.write_bundle(
            "base",
            [
                {
                    "id": "realm",
                    "isolate": ["driver"],
                    "entries": [{"id": "kid", "name": "m.kid"}],
                }
            ],
        )
        child = self.compose().row("kid")
        assert child is not None
        self.assertEqual(child.group, "realm")
        self.assertEqual(child.isolate, ("driver",))

    def test_a_group_needs_a_string_id_and_an_array_of_entries(self) -> None:
        self.write_profile()
        self.write_bundle("base", [{"entries": [{"id": "one", "name": "m.one"}]}])
        self.assertIn("group needs a string id", str(self.failure()))

        self.write_bundle("base", [{"id": "realm", "entries": {"id": "one"}}])
        self.assertIn("entries must be an array", str(self.failure()))


class RequiredSetTest(CompositionFixture):
    """``.required`` is what the loader turns into a boot-or-die set."""

    def test_required_is_exactly_the_enabled_rows_marked_required(self) -> None:
        self.write_bundle(
            "base",
            [
                {"id": "needed", "name": "m.needed", "required": True},
                {"id": "optional", "name": "m.optional"},
                {
                    "id": "parked",
                    "name": "m.parked",
                    "required": True,
                    "disabled": True,
                },
                {"id": "off", "name": "m.off", "disabled": True},
            ],
        )
        self.write_profile()

        composition = self.compose()

        self.assertEqual(composition.required, frozenset({"needed"}))
        self.assertEqual(
            composition.required,
            frozenset(row.id for row in composition.enabled if row.required),
        )
        # A disabled row keeps its required flag; it simply is not part of the set.
        parked = composition.row("parked")
        assert parked is not None
        self.assertTrue(parked.required)

    def test_a_patch_can_add_to_and_drop_from_the_required_set(self) -> None:
        self.write_bundle(
            "base",
            [
                {"id": "one", "name": "m.one", "required": True},
                {"id": "two", "name": "m.two"},
            ],
        )
        self.write_profile(
            patches=[{"id": "one", "required": False}, {"id": "two", "required": True}]
        )

        self.assertEqual(self.compose().required, frozenset({"two"}))

    def test_disabling_a_required_row_removes_it_from_the_required_set(self) -> None:
        self.write_bundle("base", [{"id": "one", "name": "m.one", "required": True}])
        self.write_profile(patches=[{"id": "one", "disabled": True}])

        composition = self.compose()

        self.assertEqual(composition.required, frozenset())
        self.assertEqual(composition.enabled, ())


class VariableTableTest(CompositionFixture):
    """``resolve()`` publishes a closed variable table; interpolation happens later."""

    def setUp(self) -> None:
        super().setUp()
        self.write_bundle("base", [{"id": "one", "name": "m.one"}])
        self.write_profile()

    def test_the_variable_table_is_the_closed_set_with_blanks_for_the_unset(self) -> None:
        composition = self.compose(variables={"workspace": "/var/aka"})

        self.assertEqual(set(composition.variables), set(COMPOSITION_VARS))
        self.assertEqual(composition.variables["workspace"], "/var/aka")
        self.assertEqual(composition.variables["operator"], "")

    def test_an_unsupported_variable_name_is_rejected_by_name(self) -> None:
        error = self.failure(variables={"kernel_name": "gemm"})

        self.assertEqual(error.source, "variables")
        self.assertIn('unsupported composition variable(s): kernel_name', str(error))

    def test_a_reference_in_a_config_survives_resolve_untouched(self) -> None:
        """Substitution is the fiber's job, against the plugin's interpolate allowlist."""
        self.write_bundle(
            "base",
            [{"id": "one", "name": "m.one", "config": {"workspace": "${aka:workspace}"}}],
        )

        row = self.compose(variables={"workspace": "/var/aka"}).row("one")

        assert row is not None
        self.assertEqual(row.config, {"workspace": "${aka:workspace}"})


class DumpTest(CompositionFixture):
    """``dump`` is how an operator reads a composition before booting it."""

    def setUp(self) -> None:
        super().setUp()
        self.write_bundle(
            "base",
            [
                {
                    "id": "one",
                    "name": "m.one",
                    "config": {"attempts": 2, "mode": "fast"},
                    "required": True,
                    "inject": ["driver"],
                    "doc": "the row a reader is looking for",
                },
                {
                    "id": "realm",
                    "isolate": ["driver"],
                    "entries": [{"id": "two", "name": "m.two"}],
                },
            ],
        )
        self.write_profile(patches=[{"id": "one", "config": {"attempts": 9}}])
        self.composition = self.compose(variables={"workspace": "/var/aka"})

    def test_json_round_trips_every_row_field(self) -> None:
        payload = json.loads(dump(self.composition, fmt="json"))

        self.assertEqual(payload["api_version"], API_VERSION)
        self.assertEqual(payload["profile"], "main")
        self.assertEqual(payload["layers"], list(self.composition.layers))
        self.assertEqual(payload["variables"], dict(self.composition.variables))
        # No row field is dropped, so the dump can be read back into Rows.
        self.assertEqual(
            [set(entry) for entry in payload["entries"]],
            [{item.name for item in fields(Row)}] * len(self.composition.rows),
        )
        rebuilt = tuple(
            Row(
                **{
                    **entry,
                    "inject": tuple(entry["inject"]),
                    "isolate": tuple(entry["isolate"]),
                }
            )
            for entry in payload["entries"]
        )
        self.assertEqual(rebuilt, self.composition.rows)

    def test_text_reports_each_rows_doc_and_config_provenance(self) -> None:
        lines = dump(self.composition, fmt="text").splitlines()

        self.assertIn("profile: main", lines)
        self.assertIn("layers: bundle:base -> profile:main", lines)
        self.assertIn("[one] m.one  (required)", lines)
        self.assertIn("  doc: the row a reader is looking for", lines)
        self.assertIn("  from: profile:main", lines)
        self.assertIn("  inject: driver", lines)
        # The config's provenance is the layer that wrote the config, not the row's last writer.
        self.assertIn("  config (set by profile:main):", lines)
        self.assertIn("    attempts = 9", lines)
        # Replacement, not a merge: the bundle's other key is not in the dump either.
        self.assertNotIn('    mode = "fast"', lines)
        self.assertIn("[two] m.two", lines)
        self.assertIn("  group: realm", lines)
        self.assertIn("  isolate: driver", lines)
        self.assertIn("  config: module Defaults only", lines)

    def test_text_marks_a_disabled_row(self) -> None:
        composition = self.compose(patches=[{"id": "two", "disabled": True}])

        self.assertIn("[two] m.two  (disabled)", dump(composition, fmt="text").splitlines())

    def test_an_unsupported_format_is_a_composition_error(self) -> None:
        with self.assertRaises(CompositionError) as caught:
            dump(self.composition, fmt="yaml")

        self.assertEqual(caught.exception.source, "dump")
        self.assertIn("unsupported format: yaml", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
