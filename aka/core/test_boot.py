"""Behavioural tests for :func:`aka.core.loader.load` and its required set.

Boot is best-effort *with a required set*. The loader's contract, quoted from its own module
docstring, is:

    A required row that cannot become active disposes the whole tree and fails loud; any
    other row that fails or stays waiting is a warning, and its service simply reads as
    absent, which is exactly how a plugin that was never listed behaves.

Every claim in that sentence has an observable consequence, and these tests pin those rather
than the mechanism:

* an *optional* row that cannot be imported, that raises inside ``apply``, or that waits on a
  service nobody provides, leaves boot successful -- it lands in ``report.failed`` or
  ``report.pending``, is warned about on stderr, and its service reads as ``None``;
* a *required* row in either of those states disposes the whole tree -- every other row's
  effects are unwound, the realm is empty and the bus is closed -- and raises
  :class:`BootFailure` naming the row and the reason;
* the required set is the enabled rows marked ``required`` unless the caller passes one, so a
  disabled ``required`` row cannot block boot and a caller-supplied set is authoritative;
* ``report.active``, ``report.service``, and ``describe()`` report the settled tree, including
  child fibers mounted during ``apply``;
* a disabled row is never mounted at all.

The rows are real: every test composes a real JSON profile and bundle in a temporary directory
and boots it through ``aka.boot.compose``/``boot``, so ``resolve()``, the bundle/patch layering,
``declare()`` and ``importlib`` all take part. Most rows are fixture plugin modules written to a
second temporary directory that ``setUp`` puts on ``sys.path``; each one records what it did --
and, crucially, when its effects were unwound -- in a shared ``aka_probe_log`` module, because a
disposed tree has forgotten its fibers and can no longer be asked. Two tests use the shipped
test-only plugins in ``aka.testing`` instead, to keep the real modules on this path too.
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping, Sequence

# ``from aka import boot`` recurses forever through ``aka/__init__.py``'s lazy re-export, so
# the submodule is imported directly (already reported as a core bug by test_config.py).
from aka.boot import boot, compose
from aka.core.errors import BootFailure, CompositionError
from aka.core.fiber import FiberState
from aka.core.loader import BootReport, describe, import_plugin

# -- fixture plugin modules ----------------------------------------------
#
# Written to a tempdir in setUp. A plugin's ``name`` must match its module path, so the module
# file names and the ``name`` exports below have to stay in step.

LOG = "aka_probe_log"
PROVIDER = "aka_probe_provider"
CONSUMER = "aka_probe_consumer"
IDLE = "aka_probe_idle"
BOOM = "aka_probe_boom"
BREAK = "aka_probe_break"
IMPORT_BOOM = "aka_probe_import_boom"
#: A module path that is deliberately absent from sys.path, to exercise the unresolvable row.
GHOST = "aka_probe_nowhere"

FIXTURES: dict[str, str] = {
    LOG: '''
"""Shared in-memory record of what the boot fixtures did.

A disposed tree forgets its fibers, so "this row's effects were unwound" can only be observed
from outside the tree. Every fixture files an effect that appends here.
"""

from __future__ import annotations

from typing import Any

EVENTS: list[str] = []
#: The ``Root`` each fixture was mounted into, so a test can inspect a tree boot never returned.
ROOTS: list[Any] = []


def record(event: str) -> None:
    EVENTS.append(event)
''',
    PROVIDER: '''
"""A fixture provider: publishes ctx.driver, and records its load and its unload."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

import aka_probe_log as log
from aka.core.context import Context
from aka.seams.driver import DRIVER, CampaignDriver, CampaignRun

name = "aka-probe-provider"
provide: tuple[str, ...] = ("driver",)

Config: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"label": {"type": "string"}},
}
Defaults: dict[str, Any] = {"label": "probe"}


class ProbeDriver(CampaignDriver):
    def __init__(self, label: str) -> None:
        self.label = label

    @property
    def workspace(self) -> Path:
        return Path("/tmp/aka-probe") / self.label

    @property
    def campaign_name(self) -> str:
        return self.label

    def run(self, *, on_prepared: Callable[[], None] | None = None) -> CampaignRun:
        if on_prepared is not None:
            on_prepared()
        return CampaignRun(status="completed", reason="probe")


def apply(ctx: Context, config: Mapping[str, Any]) -> None:
    log.ROOTS.append(ctx.fiber.root)
    log.record(f"apply:{ctx.entry_id}:{config['label']}")
    ctx.provide(DRIVER, ProbeDriver(config["label"]))
    ctx.effect(lambda: log.record(f"dispose:{ctx.entry_id}"), label="probe provider")
''',
    CONSUMER: '''
"""A fixture consumer: requires ctx.driver, contributes, and records what it read."""

from __future__ import annotations

from typing import Any, Mapping

import aka_probe_log as log
from aka.core.context import Context

name = "aka-probe-consumer"
inject: tuple[str, ...] = ("driver",)


def apply(ctx: Context, config: Mapping[str, Any]) -> None:
    log.ROOTS.append(ctx.fiber.root)
    observed = ctx.driver.campaign_name
    log.record(f"apply:{ctx.entry_id}:{observed}")
    ctx.contribute("observers", ctx.entry_id, observed)
    ctx.effect(lambda: log.record(f"dispose:{ctx.entry_id}"), label="probe consumer")
''',
    IDLE: '''
"""A fixture row that injects nothing and provides nothing.

Its ``apply`` returns its disposer instead of filing one, so the ``callable(result)`` branch of
the fiber load path is exercised by the loader tests too.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

import aka_probe_log as log
from aka.core.context import Context

name = "aka-probe-idle"


def apply(ctx: Context, config: Mapping[str, Any]) -> Callable[[], None]:
    log.ROOTS.append(ctx.fiber.root)
    log.record(f"apply:{ctx.entry_id}")
    return lambda: log.record(f"dispose:{ctx.entry_id}")
''',
    BOOM: '''
"""A fixture row that publishes ctx.driver and then raises out of ``apply``."""

from __future__ import annotations

from typing import Any, Mapping

import aka_probe_log as log
from aka.core.context import Context
from aka.seams.driver import DRIVER

name = "aka-probe-boom"
provide: tuple[str, ...] = ("driver",)

#: What a leaked half-load would leave readable through ``report.service("driver")``.
LEAKED = "boom-driver"


def apply(ctx: Context, config: Mapping[str, Any]) -> None:
    log.ROOTS.append(ctx.fiber.root)
    log.record(f"apply:{ctx.entry_id}")
    ctx.provide(DRIVER, LEAKED)
    ctx.effect(lambda: log.record(f"dispose:{ctx.entry_id}"), label="probe boom")
    raise RuntimeError("probe apply exploded")
''',
    BREAK: '''
"""A fixture row whose ``apply`` raises before contributing anything."""

from __future__ import annotations

from typing import Any, Mapping

import aka_probe_log as log
from aka.core.context import Context

name = "aka-probe-break"


def apply(ctx: Context, config: Mapping[str, Any]) -> None:
    log.record(f"apply:{ctx.entry_id}")
    raise RuntimeError("probe row broke")
''',
    IMPORT_BOOM: '''
"""A fixture module that fails while being imported, before any declaration is read."""

raise RuntimeError("probe import exploded")
''',
}

#: Reason strings the fixtures produce, as ``BootReport.failed`` formats them.
APPLY_FAILED = "RuntimeError: probe apply exploded"
BREAK_FAILED = "RuntimeError: probe row broke"

STUB_DRIVER = "aka.testing.stub_driver"
DRIVER_OBSERVER = "aka.testing.driver_observer"


def row(entry_id: str, module: str, **extra: Any) -> dict[str, Any]:
    """One composition entry, as it appears in a bundle's JSON."""
    return {"id": entry_id, "name": module, **extra}


class BootTestCase(unittest.TestCase):
    """A tempdir of fixture plugin modules plus a tempdir of real JSON profiles."""

    def setUp(self) -> None:
        self.stderr = io.StringIO()

        self.sources = tempfile.TemporaryDirectory()
        self.addCleanup(self.sources.cleanup)
        source_dir = Path(self.sources.name)
        for module, code in FIXTURES.items():
            (source_dir / f"{module}.py").write_text(code.lstrip(), encoding="utf-8")

        sys.path.insert(0, str(source_dir))
        self.addCleanup(self._restore_path, str(source_dir))
        importlib.invalidate_caches()
        self.log = importlib.import_module(LOG)

        self.profile_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.profile_dir.cleanup)
        self.profiles = Path(self.profile_dir.name)
        (self.profiles / "bundles").mkdir(parents=True, exist_ok=True)

    def _restore_path(self, source_dir: str) -> None:
        while source_dir in sys.path:
            sys.path.remove(source_dir)
        for module in list(sys.modules):
            if module.startswith("aka_probe"):
                del sys.modules[module]
        importlib.invalidate_caches()

    # -- composition -----------------------------------------------------

    def compose_rows(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        patches: Sequence[Mapping[str, Any]] = (),
        profile: str = "probe",
    ) -> Any:
        """Write a real bundle and profile, then resolve them through ``compose``."""
        (self.profiles / "bundles" / f"{profile}.json").write_text(
            json.dumps(
                {
                    "api_version": 1,
                    "doc": f"rows for {self.id()}",
                    "entries": list(entries),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (self.profiles / f"{profile}.json").write_text(
            json.dumps(
                {
                    "api_version": 1,
                    "doc": f"profile for {self.id()}",
                    "bundles": [profile],
                    "patches": list(patches),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return compose(profile, profiles_dir=self.profiles)

    def boot_rows(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        patches: Sequence[Mapping[str, Any]] = (),
        required: Sequence[str] | None = None,
    ) -> BootReport:
        """Boot a composition that is expected to succeed, and dispose it afterwards."""
        composition = self.compose_rows(entries, patches=patches)
        report = boot(composition, required=required, stderr=self.stderr)
        self.assertIsInstance(report, BootReport)
        self.addCleanup(report.dispose)
        return report

    def boot_failure(
        self,
        entries: Sequence[Mapping[str, Any]],
        *,
        patches: Sequence[Mapping[str, Any]] = (),
        required: Sequence[str] | None = None,
    ) -> BootFailure:
        composition = self.compose_rows(entries, patches=patches)
        with self.assertRaises(BootFailure) as caught:
            boot(composition, required=required, stderr=self.stderr)
        return caught.exception

    # -- observation -----------------------------------------------------

    @property
    def events(self) -> list[str]:
        return list(self.log.EVENTS)

    def only_root(self) -> Any:
        """The single ``Root`` the fixtures were mounted into.

        Boot raises before handing back a report, so a fixture's own record of its root is the
        only way to inspect a tree that failed.
        """
        roots = {id(root): root for root in self.log.ROOTS}
        self.assertEqual(len(roots), 1, "fixtures were mounted into more than one tree")
        return next(iter(roots.values()))

    def assert_disposed(self, root: Any) -> None:
        """Nothing of the tree survives: no fibers, no effects, no services, no bus."""
        self.assertEqual(root.fibers, {})
        self.assertEqual(root.live_effects, 0)
        self.assertIsNone(root.realm.resolve("driver"))
        self.assertEqual(root.realm.entries(), {})
        self.assertTrue(root.bus.closed)

    def stderr_text(self) -> str:
        return self.stderr.getvalue()


class ImportPluginTest(BootTestCase):
    """``import_plugin`` is the loader's only door onto the filesystem."""

    def test_an_importable_module_is_returned_as_itself(self) -> None:
        module = import_plugin(STUB_DRIVER)

        self.assertIs(module, sys.modules[STUB_DRIVER])
        self.assertEqual(module.name, "stub-driver")

    def test_an_absent_module_becomes_a_composition_error_naming_the_module(self) -> None:
        with self.assertRaises(CompositionError) as caught:
            import_plugin(GHOST)

        self.assertEqual(caught.exception.source, GHOST)
        self.assertIn("cannot import", str(caught.exception))
        self.assertIn(GHOST, str(caught.exception))
        # The original ImportError stays attached, so the reason is never lost.
        self.assertIsInstance(caught.exception.__cause__, ImportError)

    def test_a_module_that_raises_at_import_is_reported_the_same_way(self) -> None:
        with self.assertRaises(CompositionError) as caught:
            import_plugin(IMPORT_BOOM)

        self.assertIn("probe import exploded", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)


class OptionalUnresolvableRowTest(BootTestCase):
    """A row whose module cannot even be imported is a warning, not a boot failure."""

    def test_an_absent_module_is_failed_and_warned_and_boot_still_succeeds(self) -> None:
        report = self.boot_rows(
            [row("ghost", GHOST), row("provider", PROVIDER, config={"label": "live"})]
        )

        self.assertEqual([entry_id for entry_id, _ in report.failed], ["ghost"])
        reason = dict(report.failed)["ghost"]
        self.assertIn("CompositionError", reason)
        self.assertIn("cannot import", reason)
        self.assertIn(GHOST, reason)
        # Reported as a warning, and printed where an operator will see it.
        self.assertEqual(len(report.warnings), 1)
        self.assertIn('optional row "ghost" failed', report.warnings[0])
        self.assertIn(report.warnings[0], self.stderr_text())
        # An unresolvable row never becomes a fiber at all.
        self.assertNotIn("ghost", report.root.fibers)
        self.assertIsNone(report.root.state_of("ghost"))
        self.assertEqual(report.pending, ())
        # ... and the rest of the tree booted normally.
        self.assertEqual(report.active, ("provider",))
        self.assertEqual(report.service("driver").campaign_name, "live")

    def test_a_module_that_raises_at_import_never_reaches_declare(self) -> None:
        report = self.boot_rows([row("exploding", IMPORT_BOOM), row("idle", IDLE)])

        self.assertIn("probe import exploded", dict(report.failed)["exploding"])
        self.assertEqual(report.active, ("idle",))
        # Nothing of the failed module ran: only the healthy row recorded anything.
        self.assertEqual(self.events, ["apply:idle"])

    def test_unresolvable_and_failing_rows_are_merged_into_one_sorted_list(self) -> None:
        report = self.boot_rows([row("zed-ghost", GHOST), row("alpha-break", BREAK)])

        self.assertEqual(
            [entry_id for entry_id, _ in report.failed], ["alpha-break", "zed-ghost"]
        )
        self.assertEqual(dict(report.failed)["alpha-break"], BREAK_FAILED)
        self.assertEqual(report.active, ())
        self.assertEqual(len(report.warnings), 2)


class OptionalFailingApplyTest(BootTestCase):
    """A row that raises inside ``apply`` fails alone, and publishes nothing."""

    def test_a_failing_apply_is_failed_and_its_service_reads_as_absent(self) -> None:
        report = self.boot_rows([row("boom", BOOM)])

        self.assertEqual(report.failed, (("boom", APPLY_FAILED),))
        self.assertIs(report.root.state_of("boom"), FiberState.FAILED)
        # The service the row published before raising was rolled back, so reading it is
        # indistinguishable from a row that was never listed.
        self.assertIsNone(report.service("driver"))
        self.assertEqual(report.root.realm.entries(), {})
        self.assertEqual(report.active, ())
        self.assertEqual(report.root.live_effects, 0)
        # Its own effect was unwound as part of failing.
        self.assertEqual(self.events, ["apply:boom", "dispose:boom"])
        self.assertIn('optional row "boom" failed: %s' % APPLY_FAILED, report.warnings)

    def test_a_consumer_of_a_failed_provider_waits_instead_of_failing(self) -> None:
        report = self.boot_rows([row("consumer", CONSUMER), row("boom", BOOM)])

        self.assertEqual([entry_id for entry_id, _ in report.failed], ["boom"])
        self.assertEqual(report.pending, (("consumer", ("driver",)),))
        self.assertEqual(report.active, ())
        # The consumer never saw the half-published provider.
        self.assertEqual(self.events, ["apply:boom", "dispose:boom"])
        self.assertEqual(len(report.warnings), 2)

    def test_a_failing_row_leaves_its_neighbours_active(self) -> None:
        report = self.boot_rows(
            [
                row("provider", PROVIDER, config={"label": "live"}),
                row("break", BREAK),
                row("consumer", CONSUMER),
            ]
        )

        self.assertEqual(report.active, ("consumer", "provider"))
        self.assertEqual(report.failed, (("break", BREAK_FAILED),))
        self.assertEqual(report.service("driver").campaign_name, "live")
        # Rows load in mount order once their epoch is satisfied; the failure interrupts none
        # of it.
        self.assertEqual(
            self.events, ["apply:provider:live", "apply:break", "apply:consumer:live"]
        )


class OptionalPendingRowTest(BootTestCase):
    """Waiting for a service nobody provides is a steady state, and it is reported."""

    def test_a_row_waiting_on_an_absent_service_is_pending_with_the_missing_names(
        self,
    ) -> None:
        report = self.boot_rows([row("consumer", CONSUMER), row("idle", IDLE)])

        self.assertEqual(report.pending, (("consumer", ("driver",)),))
        self.assertIs(report.root.state_of("consumer"), FiberState.PENDING)
        self.assertEqual(report.active, ("idle",))
        self.assertEqual(report.failed, ())
        # A waiting row is a warning naming the service it waits for, not an error.
        self.assertEqual(
            report.warnings, ('optional row "consumer" is waiting for driver',)
        )
        self.assertIn(report.warnings[0], self.stderr_text())
        self.assertEqual(report.root.errors, [])
        # It never ran, so it contributed nothing.
        self.assertEqual(self.events, ["apply:idle"])
        self.assertEqual(report.root.fibers["consumer"].config, None)

    def test_a_row_level_inject_can_put_an_otherwise_ready_row_in_pending(self) -> None:
        """A row's own ``inject`` list reaches the declaration through the loader."""
        report = self.boot_rows([row("idle", IDLE, inject=["driver"])])

        self.assertEqual(report.pending, (("idle", ("driver",)),))
        self.assertEqual(report.active, ())
        self.assertEqual(self.events, [])
        self.assertEqual(report.root.fibers["idle"].declaration.inject, ("driver",))

    def test_a_pending_row_activates_once_its_provider_is_listed(self) -> None:
        """Row order carries no load semantics: the consumer is listed first."""
        report = self.boot_rows(
            [
                row("consumer", CONSUMER),
                row("provider", PROVIDER, config={"label": "late"}),
            ]
        )

        self.assertEqual(report.pending, ())
        self.assertEqual(report.active, ("consumer", "provider"))
        self.assertEqual(report.warnings, ())
        self.assertEqual(
            self.events, ["apply:provider:late", "apply:consumer:late"]
        )


class RequiredRowTest(BootTestCase):
    """A required row that cannot become active takes the whole tree down with it."""

    def test_a_required_failure_disposes_every_other_row_and_names_the_row(self) -> None:
        failure = self.boot_failure(
            [
                row("provider", PROVIDER, config={"label": "live"}),
                row("consumer", CONSUMER),
                row("break", BREAK, required=True),
            ]
        )

        # The message names the row and its reason, so the failure is attributable.
        self.assertEqual(failure.failures, (("break", BREAK_FAILED),))
        self.assertIn("break", str(failure))
        self.assertIn(BREAK_FAILED, str(failure))
        self.assertIn("required plugin rows failed to load", str(failure))

        # Every other row was unwound, in reverse mount order.
        self.assertEqual(
            self.events,
            [
                "apply:provider:live",
                "apply:consumer:live",
                "apply:break",
                "dispose:consumer",
                "dispose:provider",
            ],
        )
        self.assert_disposed(self.only_root())

    def test_a_required_row_that_stays_pending_is_a_boot_failure(self) -> None:
        failure = self.boot_failure(
            [row("idle", IDLE), row("consumer", CONSUMER, required=True)]
        )

        self.assertEqual(failure.failures, (("consumer", "waiting for driver"),))
        self.assertIn("consumer", str(failure))
        self.assertIn("waiting for driver", str(failure))
        # The optional row that did load was disposed with the rest of the tree.
        self.assertEqual(self.events, ["apply:idle", "dispose:idle"])
        self.assert_disposed(self.only_root())

    def test_a_required_row_whose_module_is_absent_is_a_boot_failure(self) -> None:
        failure = self.boot_failure([row("idle", IDLE), row("ghost", GHOST, required=True)])

        self.assertEqual([entry_id for entry_id, _ in failure.failures], ["ghost"])
        self.assertIn("cannot import", str(failure))
        self.assertEqual(self.events, ["apply:idle", "dispose:idle"])
        self.assert_disposed(self.only_root())

    def test_several_required_failures_are_reported_together_and_sorted(self) -> None:
        failure = self.boot_failure(
            [
                row("zed-break", BREAK, required=True),
                row("alpha-consumer", CONSUMER, required=True),
                row("idle", IDLE),
            ]
        )

        self.assertEqual(
            failure.failures,
            (
                ("alpha-consumer", "waiting for driver"),
                ("zed-break", BREAK_FAILED),
            ),
        )
        self.assert_disposed(self.only_root())

    def test_the_callers_required_set_can_make_an_unmarked_row_blocking(self) -> None:
        entries = [row("idle", IDLE), row("break", BREAK)]

        # As composed, the failure is only a warning.
        report = self.boot_rows(entries)
        self.assertEqual(report.failed, (("break", BREAK_FAILED),))
        self.assertEqual(report.active, ("idle",))

        # The same composition, with the caller declaring what it cannot do without.
        failure = self.boot_failure(entries, required=["break"])
        self.assertEqual(failure.failures, (("break", BREAK_FAILED),))

    def test_the_callers_required_set_replaces_the_compositions(self) -> None:
        """``required=`` is authoritative: the composition's own flags do not union in.

        ``load`` reads ``frozenset(required) if required is not None else
        composition.required``, so a caller that names its own required rows takes full
        responsibility for the set -- that is what makes a caller-supplied set usable to
        boot a partially available tree on purpose.
        """
        report = self.boot_rows(
            [row("idle", IDLE), row("break", BREAK, required=True)],
            required=["idle"],
        )

        self.assertEqual(report.failed, (("break", BREAK_FAILED),))
        self.assertEqual(report.active, ("idle",))
        self.assertIn('optional row "break" failed', report.warnings[0])

    def test_a_required_name_that_matches_no_row_is_not_silently_accepted(self) -> None:
        """A required row that is not in the tree at all can never become active.

        ``load``'s contract is that "a required row that cannot become active disposes the
        whole tree and fails loud". A caller-supplied name that matches no composition row is
        the strongest case of that: the row is not merely broken, it is missing. The rest of
        the core treats a name that resolves to nothing as a load-time error on purpose --
        ``declare()`` rejects an unregistered seam, ``TokenTable`` rejects a duplicate, and a
        patch aimed at an unknown row is a ``CompositionError`` that tells the operator to add
        ``insert`` -- because the alternative is a typo that only surfaces hours into a campaign.
        So ``load`` checks that every required id is actually ACTIVE, rather than intersecting the
        required set with the failed and waiting ones: a required name that is disabled, removed,
        or misspelled is a ``BootFailure``, not a silently dropped guarantee.
        """
        composition = self.compose_rows([row("idle", IDLE)])

        with self.assertRaises(BootFailure) as caught:
            boot(composition, required=["idle", "typo-not-a-row"], stderr=self.stderr)

        self.assertIn("typo-not-a-row", str(caught.exception))
        self.assertNotIn("idle", str(caught.exception))

    def test_an_empty_required_set_is_not_the_same_as_no_required_set(self) -> None:
        entries = [row("break", BREAK, required=True)]

        # No required set: the row's own flag decides, and boot fails.
        self.assertEqual(self.boot_failure(entries).failures, (("break", BREAK_FAILED),))
        # An explicitly empty one: nothing is required, so the same failure is a warning.
        report = self.boot_rows(entries, required=[])
        self.assertEqual(report.failed, (("break", BREAK_FAILED),))
        self.assertIn('optional row "break" failed', report.warnings[0])


class ReportShapeTest(BootTestCase):
    """What the report says about a settled tree."""

    def test_active_lists_exactly_the_rows_that_reached_active_sorted(self) -> None:
        report = self.boot_rows(
            [
                row("zeta-idle", IDLE),
                row("mid-break", BREAK),
                row("alpha-idle", IDLE),
                row("beta-consumer", CONSUMER),
            ]
        )

        self.assertEqual(report.active, ("alpha-idle", "zeta-idle"))
        self.assertEqual(report.active, tuple(sorted(report.active)))
        # Mount order was not sorted order, so the sort is the report's own doing.
        self.assertEqual(
            self.events, ["apply:zeta-idle", "apply:mid-break", "apply:alpha-idle"]
        )
        # Active, failed and waiting partition the enabled rows exactly.
        failed = [entry_id for entry_id, _ in report.failed]
        pending = [entry_id for entry_id, _ in report.pending]
        self.assertEqual(failed, ["mid-break"])
        self.assertEqual(pending, ["beta-consumer"])
        self.assertEqual(
            sorted(list(report.active) + failed + pending),
            sorted(row.id for row in report.composition.enabled),
        )
        for entry_id in report.active:
            self.assertIs(report.root.state_of(entry_id), FiberState.ACTIVE)

    def test_service_returns_the_provided_value_and_none_for_an_unknown_name(self) -> None:
        report = self.boot_rows([row("provider", PROVIDER, config={"label": "served"})])

        driver = report.service("driver")
        self.assertEqual(driver.campaign_name, "served")
        self.assertEqual(driver.workspace, Path("/tmp/aka-probe/served"))
        # The very object the row published, not a copy.
        self.assertIs(driver, report.root.realm.resolve("driver").value)
        # Names nobody provided, and names that are not seams at all, read as absent.
        self.assertIsNone(report.service("sandbox"))
        self.assertIsNone(report.service("driver-observer"))
        self.assertIsNone(report.service(""))
        # The report reads the live tree, so disposal is visible through it.
        report.dispose()
        self.assertIsNone(report.service("driver"))

    def test_describe_mentions_each_active_failed_and_waiting_row(self) -> None:
        report = self.boot_rows(
            [
                row("keeper", IDLE),
                row("breaker", BREAK),
                row("waiter", CONSUMER),
            ]
        )

        text = describe(report)

        self.assertIn("profile probe: 1 active, 1 failed, 1 waiting", text)
        self.assertIn("  active   keeper", text)
        self.assertIn(f"  failed   breaker: {BREAK_FAILED}", text)
        self.assertIn("  waiting  waiter: driver", text)
        # One line per row plus the header, and every row named exactly once.
        self.assertEqual(len(text.splitlines()), 4)
        for entry_id in ("keeper", "breaker", "waiter"):
            self.assertEqual(text.count(entry_id), 1)

    def test_a_clean_boot_reports_no_failures_no_waiting_and_no_warnings(self) -> None:
        report = self.boot_rows(
            [
                row("provider", PROVIDER, config={"label": "clean"}),
                row("consumer", CONSUMER),
            ]
        )

        self.assertEqual(report.failed, ())
        self.assertEqual(report.pending, ())
        self.assertEqual(report.warnings, ())
        self.assertEqual(self.stderr_text(), "")
        self.assertEqual(report.composition.profile, "probe")
        self.assertEqual(describe(report), "\n".join(
            [
                "profile probe: 2 active, 0 failed, 0 waiting",
                "  active   consumer",
                "  active   provider",
            ]
        ))


class DisabledRowTest(BootTestCase):
    """A disabled row is not a mounted row that happens to do nothing."""

    def test_a_disabled_row_is_never_mounted(self) -> None:
        report = self.boot_rows(
            [
                row("live-provider", PROVIDER, config={"label": "live"}),
                row(
                    "shadow-provider",
                    PROVIDER,
                    config={"label": "shadow"},
                    disabled=True,
                ),
            ]
        )

        # Had it been mounted it would have collided on the single-implementation seam, so an
        # empty ``failed`` is itself evidence.
        self.assertEqual(report.failed, ())
        self.assertEqual(report.pending, ())
        self.assertEqual(report.active, ("live-provider",))
        self.assertNotIn("shadow-provider", report.root.fibers)
        self.assertIsNone(report.root.state_of("shadow-provider"))
        self.assertEqual(self.events, ["apply:live-provider:live"])
        self.assertEqual(report.service("driver").campaign_name, "live")
        # The row is still in the composition -- disabling is not deletion.
        self.assertTrue(report.composition.row("shadow-provider").disabled)
        self.assertNotIn(
            "shadow-provider", [entry.id for entry in report.composition.enabled]
        )

    def test_a_row_disabled_by_a_patch_is_never_mounted_either(self) -> None:
        report = self.boot_rows(
            [row("idle", IDLE), row("break", BREAK)],
            patches=[{"id": "break", "disabled": True}],
        )

        self.assertEqual(report.active, ("idle",))
        self.assertEqual(report.failed, ())
        self.assertEqual(report.warnings, ())
        self.assertEqual(self.events, ["apply:idle"])

    def test_a_disabled_required_row_cannot_block_boot(self) -> None:
        """The required set is drawn from the *enabled* rows only."""
        composition = self.compose_rows(
            [row("idle", IDLE), row("break", BREAK, required=True, disabled=True)]
        )

        self.assertEqual(composition.required, frozenset())
        self.assertTrue(composition.row("break").required)

        report = boot(composition, stderr=self.stderr)
        self.addCleanup(report.dispose)

        self.assertEqual(report.active, ("idle",))
        self.assertEqual(report.failed, ())


class RealTestPluginBootTest(BootTestCase):
    """The same loader path, driven with the shipped test-only plugin modules."""

    def setUp(self) -> None:
        super().setUp()
        from aka.testing import driver_observer

        self.observer = driver_observer
        driver_observer.OBSERVED.clear()
        driver_observer.TRANSITIONS.clear()
        self.addCleanup(driver_observer.TRANSITIONS.clear)
        self.addCleanup(driver_observer.OBSERVED.clear)

    def test_the_shipped_stubs_boot_and_the_observer_reads_its_provider(self) -> None:
        report = self.boot_rows(
            [
                row("driver-observer", DRIVER_OBSERVER),
                row("stub-driver", STUB_DRIVER, config={"campaign_name": "probe-smoke"}),
            ]
        )

        expected = ["driver-observer", "stub-driver"]
        if report.root.invariants.enabled:
            # An invariant installer mounts a child fiber, which is a row of the tree too.
            expected.append("driver-observer/invariant")
        self.assertEqual(report.active, tuple(sorted(expected)))
        self.assertEqual(report.failed, ())
        self.assertEqual(report.pending, ())
        self.assertEqual(report.warnings, ())
        self.assertEqual(report.service("driver").campaign_name, "probe-smoke")
        self.assertEqual(self.observer.OBSERVED, ["probe-smoke"])

        report.dispose()

        # The observer's contribution was an effect, so disposal removed it again.
        self.assertEqual(self.observer.OBSERVED, [])
        self.assertIsNone(report.service("driver"))

    def test_a_required_shipped_observer_without_its_provider_fails_the_boot(self) -> None:
        failure = self.boot_failure([row("driver-observer", DRIVER_OBSERVER, required=True)])

        self.assertEqual(failure.failures, (("driver-observer", "waiting for driver"),))
        self.assertEqual(self.observer.OBSERVED, [])


if __name__ == "__main__":
    unittest.main()
