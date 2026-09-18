"""Behaviour tests for the composition lock (:mod:`aka.core.lock`).

The lock decides whether a campaign workspace may be resumed by a plugin tree that is not
byte-identical to the one that created it, so every rule it enforces is a contract:

1. a workspace with no ``.atrex_plugins/composition.json`` is **adopted silently** -- written,
   no diffs -- because every workspace that predates the plugin tree has no lock to compare;
2. an identical composition reports nothing;
3. a row that only *tunes* a campaign (``--numerical-gate``, ``--verify-repeats``) reports a
   diff and keeps going, because the CLI legally changes those flags on resume;
4. a row in :data:`RESUME_SENSITIVE` -- the rows that decide what a resumed campaign *is* --
   fails closed with a :class:`CompositionError` that names the lock path;
5. :func:`differences` distinguishes added, removed, module-changed and config-changed rows;
6. a corrupt lock names the file it could not use, rather than surfacing a decode error;
7. :func:`snapshot` is a pure function of the enabled rows: stable across calls, different the
   moment a row's config changes.

The last test guards the *other* lock. ``plugin_runtime.registry.PluginRegistry.snapshot()``
is compared against the on-disk ``.atrex_plugins/lock.json`` by ``check_lock``, which is why
the in-process tree records itself in a sibling file instead of extending that one.

Compositions are built by constructing :class:`~aka.core.composition.Row` values directly: the
lock reads ids, module paths, configs and package directories, and never loads a plugin.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from aka.core.composition import ResolvedComposition, Row
from aka.core.errors import CompositionError
from aka.core.lock import (
    LOCK_NAME,
    LOCK_VERSION,
    RESUME_SENSITIVE,
    differences,
    lock_path,
    package_dir,
    read,
    reconcile,
    snapshot,
    write,
)
from plugin_runtime.registry import STATE_DIR, PluginRegistry
from plugin_runtime.schema import PluginError

#: A row whose change redefines a resumed campaign, and one that merely tunes it.
SENSITIVE_ID = "journal"
TUNABLE_ID = "verification"

#: Real, importable test-only plugin modules, so package/module digests are exercised.
DRIVER_MODULE = "aka.testing.stub_driver"
OBSERVER_MODULE = "aka.testing.driver_observer"


def composition(*rows: Row, profile: str = "default") -> ResolvedComposition:
    return ResolvedComposition(profile=profile, rows=rows, layers=("bundle:core",))


class LockFixture(unittest.TestCase):
    """A tempdir campaign workspace plus the two-row composition that created it."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name) / "campaign"
        self.workspace.mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    # -- compositions ----------------------------------------------------

    def baseline(self, **overrides: Any) -> ResolvedComposition:
        """The composition the workspace was created by."""
        return self.variant(**overrides)

    def variant(
        self,
        *,
        journal_config: Mapping[str, Any] | None = None,
        verify_config: Mapping[str, Any] | None = None,
        journal_module: str = DRIVER_MODULE,
        journal_disabled: bool = False,
        drop_journal: bool = False,
    ) -> ResolvedComposition:
        rows = []
        if not drop_journal:
            rows.append(
                Row(
                    id=SENSITIVE_ID,
                    name=journal_module,
                    config=dict(journal_config or {"path": "journal.json"}),
                    required=True,
                    disabled=journal_disabled,
                )
            )
        rows.append(
            Row(
                id=TUNABLE_ID,
                name=OBSERVER_MODULE,
                config=dict(verify_config or {"numerical_gate": "strict", "repeats": 3}),
            )
        )
        return composition(*rows)

    # -- workspace helpers -----------------------------------------------

    def adopt(self) -> bytes:
        """Run the first reconcile and return the bytes it locked in."""
        self.assertEqual(reconcile(self.workspace, self.baseline()), ())
        return lock_path(self.workspace).read_bytes()

    def stored(self) -> Mapping[str, Any]:
        document = read(self.workspace)
        assert document is not None  # narrows the Optional for the type checker
        return document


class ResumeSensitiveSetTest(unittest.TestCase):
    def test_sensitive_set_is_the_documented_row_list(self) -> None:
        """Pinned on purpose: adding a row here makes older workspaces unresumable."""
        self.assertEqual(
            RESUME_SENSITIVE,
            frozenset(
                {
                    "workspace-git",
                    "journal",
                    "memory",
                    "campaign-driver",
                    "episode-engine",
                    "framework-baseline",
                    "numerics",
                    "legacy-campaign",
                }
            ),
        )
        self.assertIn(SENSITIVE_ID, RESUME_SENSITIVE)
        self.assertNotIn(TUNABLE_ID, RESUME_SENSITIVE)


class AdoptionTest(LockFixture):
    def test_workspace_without_a_lock_is_adopted_silently(self) -> None:
        current = snapshot(self.baseline())

        diffs = reconcile(self.workspace, self.baseline())

        self.assertEqual(diffs, ())
        path = lock_path(self.workspace)
        self.assertEqual(path, self.workspace / STATE_DIR / LOCK_NAME)
        self.assertTrue(path.is_file())
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), current)
        self.assertEqual(self.stored()["lock_version"], LOCK_VERSION)
        self.assertEqual(self.stored()["profile"], "default")

    def test_adoption_leaves_the_external_plugin_lock_untouched(self) -> None:
        """The in-process lock is a sibling of ``lock.json``, never an extension of it."""
        external = self.workspace / STATE_DIR / "lock.json"
        external.parent.mkdir(parents=True)
        external.write_text('{"api_version": 1, "plugin_dir": "/x", "plugins": []}\n')
        before = external.read_bytes()

        reconcile(self.workspace, self.baseline())

        self.assertEqual(external.read_bytes(), before)
        self.assertTrue(lock_path(self.workspace).is_file())
        self.assertNotEqual(lock_path(self.workspace), external)

    def test_identical_composition_reports_no_diffs_and_does_not_rewrite(self) -> None:
        locked = self.adopt()

        self.assertEqual(reconcile(self.workspace, self.baseline()), ())
        self.assertEqual(lock_path(self.workspace).read_bytes(), locked)


class ResumeDecisionTest(LockFixture):
    def test_tunable_row_config_change_is_reported_not_rejected(self) -> None:
        locked = self.adopt()

        diffs = reconcile(
            self.workspace,
            self.variant(verify_config={"numerical_gate": "off", "repeats": 7}),
        )

        self.assertEqual(diffs, (f"{TUNABLE_ID}: configuration changed",))
        # Adoption happens once: a tuning change must not quietly relock the workspace,
        # or the composition it was created by would be lost after one resume.
        self.assertEqual(lock_path(self.workspace).read_bytes(), locked)

    def test_sensitive_row_config_change_fails_closed(self) -> None:
        locked = self.adopt()

        with self.assertRaises(CompositionError) as caught:
            reconcile(self.workspace, self.variant(journal_config={"path": "other.json"}))

        message = str(caught.exception)
        self.assertIn("this workspace was created by a different plugin composition", message)
        self.assertIn(f"{SENSITIVE_ID}: configuration changed", message)
        self.assertIn(str(lock_path(self.workspace)), message)
        self.assertEqual(caught.exception.source, str(lock_path(self.workspace)))
        self.assertEqual(lock_path(self.workspace).read_bytes(), locked)

    def test_sensitive_row_module_swap_fails_closed(self) -> None:
        self.adopt()

        with self.assertRaises(CompositionError) as caught:
            reconcile(self.workspace, self.variant(journal_module=OBSERVER_MODULE))

        self.assertIn(
            f"{SENSITIVE_ID}: module {DRIVER_MODULE} -> {OBSERVER_MODULE}",
            str(caught.exception),
        )

    def test_dropping_or_disabling_a_sensitive_row_fails_closed(self) -> None:
        self.adopt()

        for label, changed in (
            ("dropped", self.variant(drop_journal=True)),
            ("disabled", self.variant(journal_disabled=True)),
        ):
            with self.subTest(label):
                with self.assertRaises(CompositionError) as caught:
                    reconcile(self.workspace, changed)
                self.assertIn(f"{SENSITIVE_ID}: removed", str(caught.exception))

    def test_sensitivity_is_taken_from_the_caller_when_given(self) -> None:
        """``resume_sensitive`` is a parameter so a caller can narrow the fail-closed set."""
        self.adopt()
        changed = self.variant(verify_config={"repeats": 9})

        self.assertEqual(
            reconcile(self.workspace, changed, resume_sensitive=frozenset()),
            (f"{TUNABLE_ID}: configuration changed",),
        )
        with self.assertRaises(CompositionError):
            reconcile(self.workspace, changed, resume_sensitive=frozenset({TUNABLE_ID}))


class DifferencesTest(unittest.TestCase):
    def test_each_kind_of_change_is_reported_distinctly(self) -> None:
        was = composition(
            Row(id="kept", name=DRIVER_MODULE, config={"a": 1}),
            Row(id="gone", name=DRIVER_MODULE),
            Row(id="moved", name=DRIVER_MODULE),
            Row(id="tuned", name=OBSERVER_MODULE, config={"repeats": 3}),
        )
        now = composition(
            Row(id="kept", name=DRIVER_MODULE, config={"a": 1}),
            Row(id="fresh", name=DRIVER_MODULE),
            Row(id="moved", name=OBSERVER_MODULE),
            Row(id="tuned", name=OBSERVER_MODULE, config={"repeats": 7}),
        )

        diffs = differences(snapshot(was), snapshot(now))

        self.assertEqual(
            diffs,
            (
                "fresh: added",
                "gone: removed",
                f"moved: module {DRIVER_MODULE} -> {OBSERVER_MODULE}",
                "tuned: configuration changed",
            ),
        )
        self.assertNotIn("kept", " ".join(diffs))

    def test_a_module_rename_hides_its_config_change(self) -> None:
        """One line per row: the module swap is the actionable fact, not the config below it."""
        was = composition(Row(id="row", name=DRIVER_MODULE, config={"a": 1}))
        now = composition(Row(id="row", name=OBSERVER_MODULE, config={"a": 2}))

        self.assertEqual(
            differences(snapshot(was), snapshot(now)),
            (f"row: module {DRIVER_MODULE} -> {OBSERVER_MODULE}",),
        )

    def test_required_and_isolation_are_part_of_a_row_fingerprint(self) -> None:
        plain = composition(Row(id="row", name=DRIVER_MODULE))
        for label, changed in (
            ("required", Row(id="row", name=DRIVER_MODULE, required=True)),
            ("isolate", Row(id="row", name=DRIVER_MODULE, isolate=("driver",))),
            ("group", Row(id="row", name=DRIVER_MODULE, group="numerics")),
        ):
            with self.subTest(label):
                self.assertEqual(
                    differences(snapshot(plain), snapshot(composition(changed))),
                    ("row: configuration changed",),
                )

    def test_a_module_directory_the_lock_never_recorded_is_reported(self) -> None:
        """Module digests are compared as a union, not an intersection.

        Comparing only directories present in both snapshots would hide the two cases that matter
        most: a plugin package that moved to a different install path, and one that vanished. Both
        mean a resumed campaign is about to run different code than the one that wrote the lock.
        """
        current = snapshot(composition(Row(id="row", name=DRIVER_MODULE)))
        self.assertTrue(current["modules"], "expected a package digest for aka.testing")
        directory = next(iter(current["modules"]))

        self.assertEqual(
            differences(dict(current, modules={}), current),
            (f"{directory}: plugin code added",),
        )
        self.assertEqual(
            differences(current, dict(current, modules={})),
            (f"{directory}: plugin code no longer present",),
        )


class ModuleDigestTest(LockFixture):
    """Plugin code digests come from the package on disk, not from the composition."""

    def install_package(self, name: str, body: str) -> Path:
        site = Path(self.temporary.name) / "site"
        package = site / name
        package.mkdir(parents=True, exist_ok=True)
        (package / "__init__.py").write_text(body, encoding="utf-8")
        if str(site) not in sys.path:
            sys.path.insert(0, str(site))
            self.addCleanup(sys.path.remove, str(site))
        self.addCleanup(sys.modules.pop, name, None)
        self.addCleanup(importlib.invalidate_caches)
        importlib.invalidate_caches()
        return package

    def test_editing_plugin_code_is_reported_and_keyed_by_directory(self) -> None:
        package = self.install_package("aka_lock_probe_plugin", "VALUE = 1\n")
        rows = composition(Row(id=SENSITIVE_ID, name="aka_lock_probe_plugin"))
        self.assertEqual(
            list(snapshot(rows)["modules"]),
            [str(package)],
            "package_dir must find the freshly installed package",
        )
        self.adopt_composition(rows)

        (package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")

        # The row itself is untouched, so the only diff is the directory digest. It is keyed by
        # path, but the row that loads from that path is resume-sensitive, so changed code there
        # is exactly as decisive as changed config: a resumed campaign must not silently run a
        # different implementation of what it is.
        with self.assertRaises(CompositionError) as caught:
            reconcile(self.workspace, rows)
        self.assertIn("plugin code changed", str(caught.exception))
        self.assertIn("different plugin composition", str(caught.exception))

    def test_editing_plugin_code_under_a_tunable_row_is_only_reported(self) -> None:
        package = self.install_package("aka_lock_probe_tunable", "VALUE = 1\n")
        rows = composition(Row(id="measure-abba", name="aka_lock_probe_tunable"))
        self.adopt_composition(rows)

        (package / "__init__.py").write_text("VALUE = 2\n", encoding="utf-8")

        self.assertEqual(
            reconcile(self.workspace, rows), (f"{package}: plugin code changed",)
        )

    def test_module_outside_the_import_path_is_skipped(self) -> None:
        rows = composition(Row(id="row", name="aka_lock_probe_absent.plugin"))
        self.assertEqual(snapshot(rows)["modules"], {})
        self.assertEqual(reconcile(self.workspace, rows), ())

    def adopt_composition(self, resolved: ResolvedComposition) -> None:
        self.assertEqual(reconcile(self.workspace, resolved), ())


class CorruptLockTest(LockFixture):
    def test_undecodable_lock_names_the_path(self) -> None:
        self.adopt()
        path = lock_path(self.workspace)
        path.write_text('{"entries": [', encoding="utf-8")

        with self.assertRaises(CompositionError) as caught:
            reconcile(self.workspace, self.baseline())

        self.assertIn(str(path), str(caught.exception))
        self.assertIn("unreadable composition lock", str(caught.exception))
        self.assertEqual(caught.exception.source, str(path))

    def test_lock_that_is_not_an_object_names_the_path(self) -> None:
        self.adopt()
        path = lock_path(self.workspace)
        path.write_text("[]", encoding="utf-8")

        with self.assertRaises(CompositionError) as caught:
            read(self.workspace)

        self.assertIn(str(path), str(caught.exception))
        self.assertIn("must be a JSON object", str(caught.exception))

    def test_missing_lock_reads_as_absent_rather_than_failing(self) -> None:
        self.assertIsNone(read(self.workspace))

    def test_lock_entry_without_an_id_should_name_the_path(self) -> None:
        write(
            self.workspace,
            {
                "lock_version": LOCK_VERSION,
                "profile": "default",
                "entries": [{"name": DRIVER_MODULE, "fingerprint": "0" * 64}],
                "modules": {},
            },
        )
        # Every corrupt-lock path fails the same way: a CompositionError naming the file, not a
        # KeyError from deep inside the comparison.
        for call in (lambda: read(self.workspace), lambda: reconcile(self.workspace, self.baseline())):
            with self.assertRaises(CompositionError) as caught:
                call()
            self.assertIn(str(lock_path(self.workspace)), str(caught.exception))
            self.assertIn("string id", str(caught.exception))


class PackageDirTest(unittest.TestCase):
    def test_package_resolves_to_its_directory(self) -> None:
        directory = package_dir("aka.testing")

        self.assertIsNotNone(directory)
        assert directory is not None
        self.assertTrue(directory.is_dir())
        self.assertEqual(directory.name, "testing")
        self.assertTrue((directory / "stub_driver.py").is_file())
        self.assertEqual(directory, Path(importlib.import_module("aka.testing").__file__).parent)

    def test_plain_module_resolves_to_its_containing_directory(self) -> None:
        self.assertEqual(package_dir(DRIVER_MODULE), package_dir("aka.testing"))

    def test_missing_modules_resolve_to_none(self) -> None:
        for name in (
            "aka.testing.no_such_plugin_module",
            "aka_lock_probe_absent_top_level",
            "aka_lock_probe_absent_top_level.child",
            "",
        ):
            with self.subTest(name):
                self.assertIsNone(package_dir(name))


class SnapshotTest(unittest.TestCase):
    def test_snapshot_is_stable_across_calls(self) -> None:
        resolved = composition(
            Row(id=SENSITIVE_ID, name=DRIVER_MODULE, config={"path": "journal.json"}),
            Row(id=TUNABLE_ID, name=OBSERVER_MODULE, config={"repeats": 3}),
        )

        first, second = snapshot(resolved), snapshot(resolved)

        self.assertEqual(first, second)
        self.assertEqual(
            json.dumps(first, sort_keys=True), json.dumps(second, sort_keys=True)
        )
        self.assertEqual(
            [entry["id"] for entry in first["entries"]], [SENSITIVE_ID, TUNABLE_ID]
        )

    def test_snapshot_changes_only_the_row_whose_config_changed(self) -> None:
        before = snapshot(
            composition(
                Row(id=SENSITIVE_ID, name=DRIVER_MODULE, config={"path": "journal.json"}),
                Row(id=TUNABLE_ID, name=OBSERVER_MODULE, config={"repeats": 3}),
            )
        )
        after = snapshot(
            composition(
                Row(id=SENSITIVE_ID, name=DRIVER_MODULE, config={"path": "journal.json"}),
                Row(id=TUNABLE_ID, name=OBSERVER_MODULE, config={"repeats": 7}),
            )
        )

        self.assertNotEqual(before, after)
        prints = {
            entry["id"]: entry["fingerprint"]
            for snap in (before, after)
            for entry in snap["entries"]
        }
        self.assertEqual(len(prints), 2)
        self.assertEqual(
            before["entries"][0]["fingerprint"], after["entries"][0]["fingerprint"]
        )
        self.assertNotEqual(
            before["entries"][1]["fingerprint"], after["entries"][1]["fingerprint"]
        )
        self.assertEqual(before["modules"], after["modules"])

    def test_disabled_rows_are_not_in_the_snapshot(self) -> None:
        enabled = composition(Row(id="row", name=DRIVER_MODULE))
        disabled = composition(Row(id="row", name=DRIVER_MODULE, disabled=True))

        self.assertEqual(snapshot(disabled)["entries"], [])
        # Disabling the only row removes its module digest too, so both facts are reported.
        self.assertEqual(
            differences(snapshot(enabled), snapshot(disabled))[0], "row: removed"
        )
        self.assertIn(
            "plugin code no longer present",
            differences(snapshot(enabled), snapshot(disabled))[1],
        )


class ExternalPluginLockShapeTest(unittest.TestCase):
    """The frozen shape of ``.atrex_plugins/lock.json``.

    ``PluginRegistry.check_lock`` compares the stored lock against a fresh
    ``PluginRegistry.snapshot()`` byte-for-byte, so *any* change to these keys -- adding a
    field, renaming one, reordering the plugin entry's contents -- makes every existing
    campaign workspace unresumable: the workspace's locked file can no longer equal the
    snapshot the current code produces. That is precisely why ``aka.core.lock`` records the
    in-process plugin tree in the sibling ``composition.json`` instead of extending this file.
    """

    SNAPSHOT_KEYS = frozenset({"api_version", "plugin_dir", "plugins"})
    PLUGIN_KEYS = frozenset({"id", "version", "root", "fingerprint", "commands"})

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.plugin_dir = Path(self.temporary.name) / "plugins"
        root = self.plugin_dir / "demo"
        (root / "schemas").mkdir(parents=True)
        for name in ("in", "out"):
            (root / "schemas" / f"{name}.json").write_text(
                json.dumps({"type": "object"}), encoding="utf-8"
            )
        (root / "plugin.json").write_text(
            json.dumps(
                {
                    "id": "demo",
                    "version": "0.1.0",
                    "api_version": 1,
                    "tools": {
                        "ping": {
                            "description": "probe",
                            "command": ["{python}", "-c", "pass"],
                            "input_schema": "schemas/in.json",
                            "output_schema": "schemas/out.json",
                            "timeout_seconds": 5,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        self.registry = PluginRegistry(self.plugin_dir)
        self.workspace = Path(self.temporary.name) / "campaign"
        (self.workspace / STATE_DIR).mkdir(parents=True)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_snapshot_keys_are_exactly_the_locked_set(self) -> None:
        current = self.registry.snapshot()

        self.assertEqual(frozenset(current), self.SNAPSHOT_KEYS)
        self.assertEqual(current["api_version"], 1)
        self.assertEqual(current["plugin_dir"], str(self.plugin_dir.resolve()))
        self.assertEqual(len(current["plugins"]), 1)
        entry = current["plugins"][0]
        self.assertEqual(frozenset(entry), self.PLUGIN_KEYS)
        self.assertEqual(entry["id"], "demo")
        self.assertEqual(entry["version"], "0.1.0")
        self.assertEqual(entry["root"], str((self.plugin_dir / "demo").resolve()))
        self.assertEqual(list(entry["commands"]), ["ping"])
        self.assertEqual(entry["commands"]["ping"][1:], ["-c", "pass"])
        self.assertEqual(len(entry["fingerprint"]), 64)

    def test_check_lock_compares_the_snapshot_whole(self) -> None:
        lock = self.workspace / STATE_DIR / "lock.json"
        lock.write_text(json.dumps(self.registry.snapshot(), indent=2) + "\n", encoding="utf-8")

        self.registry.check_lock(self.workspace)  # unchanged inputs resume cleanly

        extended = dict(self.registry.snapshot(), composition_lock_version=1)
        lock.write_text(json.dumps(extended, indent=2) + "\n", encoding="utf-8")

        with self.assertRaises(PluginError) as caught:
            self.registry.check_lock(self.workspace)
        self.assertEqual(caught.exception.code, "plugin_changed")

    def test_in_process_lock_is_a_different_file(self) -> None:
        self.assertEqual(LOCK_NAME, "composition.json")
        self.assertNotEqual(
            lock_path(self.workspace), self.workspace / STATE_DIR / "lock.json"
        )


if __name__ == "__main__":
    unittest.main()
