"""Boot the shipped test composition for real, through the real loader path.

``aka/profiles/test/smoke.json`` exists so that one test can answer the question no unit test
can: does the whole path -- ``resolve()`` layering a profile over a bundle, ``importlib``
finding each row's module, ``declare()`` validating it, the fiber epoch deciding load order,
interpolation filling a config field, the invariant registry reserving a package name, the
scope collecting a contribution, and ``dispose()`` unwinding all of it -- actually work on the
files this repository ships? A hand-built ``Root`` with hand-built ``PluginDeclaration``
objects is not a substitute: it skips the JSON, the patch layer, the module import, and
``declare()``, which is exactly where a real composition breaks. So every assertion here is
made against a tree booted from ``aka.boot.compose`` plus ``aka.boot.boot``.

What is pinned:

* every enabled row of the profile reaches ``ACTIVE``, and ``report.failed``,
  ``report.pending`` and ``report.warnings`` are empty -- a clean boot is silent;
* the shipped bundle lists the *consumer* before its *provider*, and the tree still loads,
  with the provider going active first: row order carries no load semantics, ``inject`` does;
* every seam an enabled row provides reads back through ``report.service()`` and is an
  instance of the Definition ABC its ``ServiceKey`` names, looked up through
  ``aka.seams.SEAMS`` rather than hardcoded;
* ``${aka:workspace}`` in the profile's patch reached the stub driver as a real path, while
  the composition still holds the literal reference -- interpolation is a load-time step on an
  allowlisted field, and a reference in a field nobody allowlisted fails the row instead;
* every plugin package that publishes an invariant companion has its ``PACKAGE_NAME``
  reserved in ``root.invariants.reserved``;
* the observer's scoped contribution is visible from the root scope, with its owner and order;
* after ``report.dispose()`` nothing survives: no live effects, a closed bus, every fiber
  ``DISPOSED``, and every contribution, service and reservation gone.

A second test case composes the *default* profile and runs ``declare()`` on its row, proving
the shipped campaign composition resolves and that ``aka.plugins.legacy_campaign`` satisfies
the declaration contract. It deliberately stops there: booting that row constructs a real
``Campaign``, which needs an operator directory and a workspace, so the row's identity fields
come from the CLI's config patch (see ``aka/cli.py``) and are absent from the bundle.
"""

from __future__ import annotations

import importlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path

from plugin_runtime.schema import PluginError, validate_schema

# ``from aka import boot`` re-exports the *function* through ``aka/__init__.py``'s lazy
# ``__getattr__``, so the submodule is imported directly, as the core's own tests do.
from aka.boot import PROFILES_DIR, boot, compose
from aka.core.declaration import declare
from aka.core.errors import BootFailure
from aka.core.fiber import FiberState
from aka.core.loader import BootReport, import_plugin
from aka.core.internal import INTERNAL_FIBER, FiberTransition
from aka.seams import SEAMS
from aka.seams import keys as seam_keys
from aka.seams.driver import CampaignDriver
from aka.testing import driver_observer

#: The shipped test composition. Both files are read from disk by ``resolve()``.
TEST_PROFILES = PROFILES_DIR / "test"
SMOKE_PROFILE = "smoke"
SMOKE_PATH = TEST_PROFILES / f"{SMOKE_PROFILE}.json"
STUBS_BUNDLE = TEST_PROFILES / "bundles" / "stubs.json"

OBSERVER_ROW = "driver-observer"
DRIVER_ROW = "stub-driver"
#: The child fiber an invariant installer is mounted into, named by the registry.
OBSERVER_INVARIANT = f"{OBSERVER_ROW}/invariant"

#: The shipped campaign composition, composed but never booted here.
CAMPAIGN_PROFILE = "default"
CAMPAIGN_ROW = "legacy-campaign"
CAMPAIGN_MODULE = "aka.plugins.legacy_campaign"


def invariant_package_names(module_path: str) -> tuple[str, ...]:
    """The invariant package names one row publishes, discovered from the filesystem.

    ``aka/scripts/check_declarations.py`` defines publication as an ``invariant.py`` companion
    exporting ``PACKAGE_NAME`` and ``install``; a single-module plugin such as
    ``aka.testing.driver_observer`` publishes the same two names from the module itself. Both
    shapes are discovered here so the reservation assertion cannot pass vacuously just because
    a package moved its companion.
    """
    module = importlib.import_module(module_path)
    names: list[str] = []
    spec = importlib.util.find_spec(module_path)
    locations = list(getattr(spec, "submodule_search_locations", None) or ())
    if locations and (Path(locations[0]) / "invariant.py").is_file():
        companion = importlib.import_module(f"{module_path}.invariant")
        published = getattr(companion, "PACKAGE_NAME", None)
        if isinstance(published, str) and published:
            names.append(published)
    inline = getattr(module, "PACKAGE_NAME", None)
    if isinstance(inline, str) and inline:
        names.append(inline)
    return tuple(dict.fromkeys(names))


def definition_of(name: str) -> type:
    """The Definition ABC a seam's ``ServiceKey`` names, resolved through the token."""
    key = SEAMS[name]
    return getattr(importlib.import_module(key.module), key.definition)


class SmokeProfileTest(unittest.TestCase):
    """The shipped ``test/smoke`` profile, booted through ``compose`` and ``boot``."""

    def setUp(self) -> None:
        # ``driver_observer`` records what it observed in module-level lists that outlive any
        # one tree, and only ``OBSERVED`` is unwound by its disposer, so both are reset here
        # and again afterwards: neither test may read another test's observations.
        driver_observer.OBSERVED.clear()
        driver_observer.TRANSITIONS.clear()
        self.addCleanup(driver_observer.TRANSITIONS.clear)
        self.addCleanup(driver_observer.OBSERVED.clear)

        self.workspace_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.workspace_dir.cleanup)
        self.workspace = Path(self.workspace_dir.name)

        self.stderr = io.StringIO()
        self.composition = compose(
            SMOKE_PROFILE,
            profiles_dir=TEST_PROFILES,
            variables={"workspace": str(self.workspace)},
        )
        self.report = boot(self.composition, stderr=self.stderr)
        self.addCleanup(self.report.dispose)
        self.root = self.report.root

    # -- helpers ---------------------------------------------------------

    @property
    def enabled_ids(self) -> tuple[str, ...]:
        return tuple(row.id for row in self.composition.enabled)

    def expected_active(self) -> tuple[str, ...]:
        """Every enabled row, plus the invariant child fiber when checks are enabled."""
        expected = list(self.enabled_ids)
        if self.root.invariants.enabled:
            expected.append(OBSERVER_INVARIANT)
        return tuple(sorted(expected))

    def provided_seams(self) -> tuple[str, ...]:
        """Seam names the booted rows declared they provide, read from the loader's own
        declarations rather than re-derived."""
        names: set[str] = set()
        for fiber in self.root.fibers.values():
            names.update(fiber.declaration.provide)
        return tuple(sorted(names))

    def first_transition(self, entry_id: str, state: str) -> int:
        for index, transition in enumerate(self.root.transitions):
            if transition.entry_id == entry_id and transition.current == state:
                return index
        self.fail(f"{entry_id} never transitioned to {state}")

    # -- the profile itself ----------------------------------------------

    def test_the_booted_composition_came_from_the_shipped_profile_and_bundle_files(
        self,
    ) -> None:
        """The rows under test are the ones on disk, not a fixture built in this test."""
        self.assertTrue(SMOKE_PATH.is_file(), SMOKE_PATH)
        self.assertTrue(STUBS_BUNDLE.is_file(), STUBS_BUNDLE)

        self.assertEqual(self.composition.profile, SMOKE_PROFILE)
        # The bundle is one layer, the profile's own patch is the next.
        self.assertEqual(self.composition.layers, ("bundle:stubs", "profile:smoke"))
        self.assertEqual(self.composition.row(OBSERVER_ROW).source, "bundle:stubs")
        self.assertEqual(self.composition.row(DRIVER_ROW).source, "profile:smoke")
        self.assertEqual(self.composition.required, frozenset({DRIVER_ROW}))

        # Each fiber holds the declaration read from the real module by ``declare()``.
        for entry_id, module_path in (
            (OBSERVER_ROW, "aka.testing.driver_observer"),
            (DRIVER_ROW, "aka.testing.stub_driver"),
        ):
            declaration = self.root.fibers[entry_id].declaration
            self.assertEqual(declaration.module, module_path)
            self.assertIs(declaration.apply, importlib.import_module(module_path).apply)

    def test_every_row_reaches_active_and_nothing_failed_or_waited(self) -> None:
        self.assertIsInstance(self.report, BootReport)
        self.assertEqual(self.report.active, self.expected_active())
        self.assertEqual(self.report.failed, ())
        self.assertEqual(self.report.pending, ())
        # A clean boot is silent: no warnings collected, and nothing printed for an operator.
        self.assertEqual(self.report.warnings, ())
        self.assertEqual(self.stderr.getvalue(), "")
        self.assertEqual(self.root.errors, [])

        for entry_id in self.enabled_ids:
            self.assertIs(self.root.state_of(entry_id), FiberState.ACTIVE, entry_id)
        # Every enabled row became a fiber, and no row was silently skipped.
        self.assertEqual(
            sorted(self.enabled_ids),
            sorted(entry_id for entry_id in self.enabled_ids if entry_id in self.root.fibers),
        )
        self.assertEqual(self.root.invariants.failures, ())

    def test_the_consumer_is_listed_before_its_provider_and_still_loads(self) -> None:
        """Row order carries no load semantics; the epoch behind ``inject`` decides."""
        self.assertEqual([row.id for row in self.composition.rows], [OBSERVER_ROW, DRIVER_ROW])

        observer = self.root.fibers[OBSERVER_ROW].declaration
        provider = self.root.fibers[DRIVER_ROW].declaration
        self.assertEqual(observer.inject, ("driver",))
        self.assertEqual(provider.provide, ("driver",))

        # Both are active even though the consumer was mounted first -- mount order does follow
        # row order, which is what makes this composition a real test of the epoch.
        self.assertIs(self.root.state_of(OBSERVER_ROW), FiberState.ACTIVE)
        self.assertIs(self.root.state_of(DRIVER_ROW), FiberState.ACTIVE)
        self.assertLess(
            self.root.fibers[OBSERVER_ROW].uid, self.root.fibers[DRIVER_ROW].uid
        )
        # The consumer waited in its initial PENDING state and then loaded exactly once: no
        # churn, and no load against a provider that was not there yet.
        self.assertEqual(
            [
                (transition.previous, transition.current)
                for transition in self.root.transitions
                if transition.entry_id == OBSERVER_ROW
            ],
            [("pending", "loading"), ("loading", "active")],
        )
        # ... and the provider went active before the consumer started loading, which is the
        # reverse of the order the bundle lists them in.
        self.assertLess(
            self.first_transition(DRIVER_ROW, "active"),
            self.first_transition(OBSERVER_ROW, "loading"),
        )
        # The consumer really read the live provider rather than being handed a placeholder.
        self.assertEqual(driver_observer.OBSERVED, ["smoke"])
        self.assertEqual(
            self.report.service("driver").campaign_name, driver_observer.OBSERVED[0]
        )

    # -- seams -----------------------------------------------------------

    def test_every_provided_seam_resolves_as_an_instance_of_its_definition(self) -> None:
        provided = self.provided_seams()
        self.assertEqual(provided, ("driver",))

        for name in provided:
            key = SEAMS[name]
            definition = definition_of(name)
            value = self.report.service(name)
            self.assertIsNotNone(value, f"seam {name} does not read back")
            self.assertIsInstance(value, definition)
            self.assertEqual(key.cardinality, "single")
            # The seam token names the class, so a Definition rename cannot slip past.
            self.assertEqual(definition.__name__, key.definition)
            # ``service()`` hands back the very object the row published.
            self.assertIs(value, self.root.realm.resolve(name).value)
            self.assertEqual(self.root.realm.resolve(name).entry_id, DRIVER_ROW)

        # The driver seam's Definition is the ABC this test imported directly.
        self.assertIs(definition_of("driver"), CampaignDriver)
        self.assertIsInstance(self.report.service("driver"), CampaignDriver)
        # A single-implementation seam means exactly one live entry for it.
        self.assertEqual(sorted(self.root.realm.entries()), list(provided))
        # A name nobody provided reads as absent instead of raising.
        self.assertIsNone(self.report.service("sandbox"))

    # -- interpolation ---------------------------------------------------

    def test_the_workspace_variable_in_the_patch_reached_the_driver(self) -> None:
        driver = self.report.service("driver")

        self.assertEqual(driver.workspace, self.workspace)
        self.assertTrue(driver.workspace.is_absolute())
        self.assertEqual(driver.campaign_name, "smoke")

        # The composition still holds the literal reference: interpolation happens when the
        # row loads, against the allowlisted field, and never rewrites the composition.
        row = self.composition.row(DRIVER_ROW)
        self.assertEqual(row.config["workspace"], "${aka:workspace}")
        self.assertEqual(row.config_source, "profile:smoke")
        self.assertEqual(self.composition.variables["workspace"], str(self.workspace))

        fiber = self.root.fibers[DRIVER_ROW]
        self.assertEqual(fiber.declaration.interpolate, ("workspace",))
        self.assertEqual(fiber.config["workspace"], str(self.workspace))
        # A field the patch did not set still comes from the module's own Defaults.
        self.assertEqual(fiber.config["status"], "completed")

    def test_a_reference_in_a_field_outside_the_allowlist_fails_the_row(self) -> None:
        """The closed allowlist is what keeps an unsubstituted reference from becoming data.

        ``campaign_name`` is a real Config property of the stub driver but is not listed in its
        ``interpolate`` allowlist, so a reference there is a composition error rather than a
        literal ``${aka:campaign_name}`` campaign name. The row is required, so the whole tree
        goes down loudly.
        """
        composition = compose(
            SMOKE_PROFILE,
            profiles_dir=TEST_PROFILES,
            variables={"workspace": str(self.workspace)},
            patches=[
                {
                    "id": DRIVER_ROW,
                    "config": {
                        "campaign_name": "${aka:campaign_name}",
                        "workspace": "${aka:workspace}",
                    },
                }
            ],
        )
        self.assertEqual(composition.layers[-1], "cli")

        # ``setUp``'s healthy tree already recorded its own observation, so the second tree is
        # judged on what it adds.
        observed_before = list(driver_observer.OBSERVED)
        stderr = io.StringIO()
        with self.assertRaises(BootFailure) as caught:
            boot(composition, stderr=stderr)

        self.assertEqual([entry_id for entry_id, _ in caught.exception.failures], [DRIVER_ROW])
        reason = dict(caught.exception.failures)[DRIVER_ROW]
        self.assertIn("CompositionError", reason)
        self.assertIn("campaign_name", reason)
        self.assertIn("interpolate allowlist", reason)
        self.assertIn("interpolate allowlist", stderr.getvalue())
        # The consumer of the failed tree never saw a half-configured provider.
        self.assertEqual(driver_observer.OBSERVED, observed_before)

    # -- invariants ------------------------------------------------------

    def test_every_published_invariant_package_name_is_reserved(self) -> None:
        published: set[str] = set()
        for row in self.composition.enabled:
            published.update(invariant_package_names(row.name))

        # The observer publishes one; the stub driver publishes none, so the reserved set is
        # exactly what the booted rows publish rather than everything that exists.
        self.assertEqual(published, {"aka.testing.driver-observer"})
        self.assertEqual(invariant_package_names("aka.testing.stub_driver"), ())
        self.assertEqual(set(self.root.invariants.reserved), published)

        # The reservation is an effect owned by the row that made it, so it is released when
        # that row unloads and cannot be claimed twice.
        observer = self.root.fibers[OBSERVER_ROW]
        labels = [
            label
            for label in observer.effects.live
            if driver_observer.PACKAGE_NAME in label
        ]
        self.assertEqual(len(labels), 1, observer.effects.live)
        self.assertIn("invariant", labels[0])

        if self.root.invariants.enabled:
            self.assertEqual(labels, [f"invariant {driver_observer.PACKAGE_NAME}"])
            # An enabled installer runs in a child fiber of its owner, and that fiber is a row
            # of the tree like any other.
            child = self.root.fibers[OBSERVER_INVARIANT]
            self.assertIs(child.state, FiberState.ACTIVE)
            self.assertIs(child.parent, observer)
            self.assertIn(OBSERVER_INVARIANT, self.report.active)
        else:  # pragma: no cover - only when AKA_INVARIANTS disables checks
            self.assertNotIn(OBSERVER_INVARIANT, self.root.fibers)

    # -- scope -----------------------------------------------------------

    def test_the_observers_scoped_contribution_is_visible_from_the_root_scope(self) -> None:
        entries = self.root.scope.resolve("observers")

        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry.name, OBSERVER_ROW)
        self.assertEqual(entry.value, "smoke")
        # Attributed to the row that contributed it, at the order the module asked for.
        self.assertEqual(entry.owner, OBSERVER_ROW)
        self.assertEqual(entry.order, 100)

        self.assertEqual(self.root.scope.kinds(), ("observers",))
        self.assertEqual(dict(self.root.scope.merged("observers")), {OBSERVER_ROW: entry})
        # It landed in the root layer itself, since no row asked for a nested scope.
        self.assertEqual(self.root.scope.local("observers"), (entry,))
        self.assertEqual(self.root.scope.resolve("tools"), ())

    def test_the_observer_watches_the_event_bus_it_subscribed_to(self) -> None:
        """The subscription is a real listener on the shared bus, not a stored callback."""
        self.assertIn(OBSERVER_ROW, driver_observer.TRANSITIONS)
        owners = {
            listener.owner for listener in self.root.bus.listeners(INTERNAL_FIBER)
        }
        self.assertIn(OBSERVER_ROW, owners)
        # Every recorded id belongs to a row of this tree.
        self.assertLessEqual(
            set(driver_observer.TRANSITIONS), set(self.expected_active())
        )
        self.assertIsInstance(self.root.transitions[0], FiberTransition)

    # -- teardown --------------------------------------------------------

    def test_dispose_leaves_no_effects_no_bus_and_only_disposed_fibers(self) -> None:
        # A disposed tree forgets its fibers, so they are captured while it is still up.
        fibers = dict(self.root.fibers)
        self.assertEqual(sorted(fibers), sorted(self.expected_active()))
        self.assertGreater(self.root.live_effects, 0)

        self.report.dispose()

        self.assertEqual(self.root.live_effects, 0)
        self.assertTrue(self.root.bus.closed)
        for entry_id, fiber in fibers.items():
            self.assertIs(fiber.state, FiberState.DISPOSED, entry_id)
            # ``live_effects`` sums over the fibers the root still knows, and it knows none
            # after disposal, so each stack is checked directly too.
            self.assertEqual(len(fiber.effects), 0, entry_id)
            self.assertIsNone(fiber.config, entry_id)
        self.assertEqual(self.root.fibers, {})
        self.assertIsNone(self.root.state_of(DRIVER_ROW))

        # Everything the rows contributed went with them.
        self.assertIsNone(self.report.service("driver"))
        self.assertEqual(self.root.realm.entries(), {})
        self.assertEqual(self.root.scope.resolve("observers"), ())
        self.assertEqual(self.root.invariants.reserved, ())
        self.assertEqual(driver_observer.OBSERVED, [])
        self.assertFalse(self.root.bus.has_listeners(INTERNAL_FIBER))
        self.assertEqual(self.root.errors, [])


class DefaultProfileCompositionTest(unittest.TestCase):
    """The shipped campaign composition resolves and declares -- without being booted.

    Booting ``legacy-campaign`` constructs ``orchestrator.campaign.Campaign``, which needs an
    operator directory, a workspace, and eventually a GPU. ``compose()`` plus ``declare()``
    covers everything that can go wrong in the composition and the declaration, which is what
    stage one of the refactor put at risk.
    """

    def setUp(self) -> None:
        self.composition = compose(CAMPAIGN_PROFILE)
        self.row = self.composition.row(CAMPAIGN_ROW)
        self.assertIsNotNone(self.row, "the default profile lost its campaign row")

    def test_the_default_profile_composes_into_one_required_campaign_row(self) -> None:
        self.assertTrue((PROFILES_DIR / f"{CAMPAIGN_PROFILE}.json").is_file())
        self.assertEqual(self.composition.profile, CAMPAIGN_PROFILE)
        # The profile carries no patches of its own, so the bundle is the only layer.
        self.assertEqual(self.composition.layers, ("bundle:core",))
        self.assertEqual([row.id for row in self.composition.enabled], [CAMPAIGN_ROW])
        self.assertEqual(self.row.name, CAMPAIGN_MODULE)
        self.assertTrue(self.row.required)
        self.assertEqual(self.composition.required, frozenset({CAMPAIGN_ROW}))
        # The repo root is always available to interpolation, whoever composed.
        self.assertEqual(
            self.composition.variables["repo_root"],
            str(Path(__file__).resolve().parent.parent),
        )

    def test_declare_accepts_the_campaign_row_module(self) -> None:
        module = import_plugin(self.row.name)
        declaration = declare(module, known_seams=seam_keys()).with_extra_inject(
            self.row.inject
        )

        self.assertEqual(declaration.name, CAMPAIGN_ROW)
        self.assertEqual(declaration.module, CAMPAIGN_MODULE)
        self.assertIs(declaration.apply, module.apply)
        # It is the provider half of the driver seam, exactly like the test stub, which is what
        # makes the smoke profile a row swap of the real composition rather than a fork.
        self.assertEqual(declaration.provide, ("driver",))
        self.assertIn("driver", seam_keys())
        self.assertEqual(declaration.inject, ())
        self.assertEqual(declaration.optional_inject, ())
        self.assertEqual(declaration.interpolate, ("work_dir",))
        # Its Config is the whole legacy campaign configuration, and every Default is one of
        # its own properties.
        self.assertEqual(declaration.config_schema["type"], "object")
        properties = declaration.config_schema["properties"]
        self.assertEqual(sorted(set(declaration.defaults) - set(properties)), [])
        for field in declaration.interpolate:
            self.assertIn(field, properties)

    def test_the_campaign_package_publishes_an_invariant_companion(self) -> None:
        self.assertEqual(
            invariant_package_names(CAMPAIGN_MODULE), ("aka.plugins.legacy-campaign",)
        )
        companion = importlib.import_module(f"{CAMPAIGN_MODULE}.invariant")
        self.assertTrue(callable(companion.install))

    def test_the_shipped_row_alone_cannot_be_booted_without_operator_config(self) -> None:
        """Why this test composes and declares but never boots.

        The bundle row deliberately carries no config: the campaign's identity comes from the
        CLI's whole-config patch (``aka/cli.py`` binds ``--op-dir`` and friends to this row).
        Module ``Defaults`` alone therefore cannot satisfy the row's own required fields, so
        the shipped default profile is not a bootable tree until an operator supplies them --
        and even then it would build a real ``Campaign``.
        """
        self.assertEqual(dict(self.row.config), {})
        self.assertEqual(self.row.config_source, "")

        module = import_plugin(self.row.name)
        declaration = declare(module, known_seams=seam_keys())
        merged = {**declaration.defaults, **dict(self.row.config)}
        missing = [
            field
            for field in declaration.config_schema["required"]
            if field not in merged
        ]
        self.assertEqual(missing, ["name", "kernel_demo", "platform", "framework"])
        with self.assertRaises(PluginError) as caught:
            validate_schema(declaration.config_schema, merged, f"{CAMPAIGN_ROW}.config")
        self.assertIn("name", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
