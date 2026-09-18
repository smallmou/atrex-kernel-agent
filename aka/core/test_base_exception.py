"""Behavioural tests for ``BaseException`` propagation through the plugin core.

``orchestrator/environment_recovery.py`` defines ``EnvironmentUnavailable(BaseException)``
so that a confirmed remote-environment failure escapes every ``except Exception`` between
the failing call and the entry point, which turns it into ``ENVIRONMENT_TEMPFAIL = 75``.
The whole protocol rests on one property: nothing in between may catch it, contain it,
report it, or convert it into something else. A plugin container is exactly the kind of
code that breaks that property, because containment is its job -- a throwing observer must
not starve its peers, a failing row must not stop the tree, a bad disposer must not abort
teardown. Every one of those containment points is written as ``except Exception`` on
purpose, and these tests pin that choice down.

What is asserted here, per the core's own docstrings:

* the five dispatch modes contain ``Exception`` but let ``BaseException`` through, and it
  never reaches the error sink (``events.py``);
* a plugin whose ``apply`` aborts is left *inert*, not failed: ``PENDING``, epoch
  ``INACTIVE``, partial effects unwound, providing nothing, nothing reported -- and the
  abort continues out of :meth:`Root.settle` (``fiber.py``);
* a disposer that aborts propagates out of :meth:`EffectStack.unwind` rather than being
  collected into ``DisposalErrors`` (``effects.py``);
* :meth:`Root.dispose` called from a ``finally`` while an abort is in flight still runs
  every disposer and lets the original exception continue;
* an ordinary ``Exception`` from ``apply``, for contrast, ends in ``FAILED`` and does not
  propagate at all.

The stand-in for ``EnvironmentUnavailable`` is the local :class:`_Abort`; the core is not
allowed to know about the orchestrator, so neither is this test.
"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping

from aka.core.composition import ResolvedComposition, Row
from aka.core.declaration import PluginDeclaration
from aka.core.effects import EffectStack
from aka.core.errors import CoreError, DisposalErrors
from aka.core.events import EventBus
from aka.core.fiber import INACTIVE, FiberState, Root
from aka.core.internal import INTERNAL_CONFIG
from aka.core.keys import MODES, Event, ServiceKey
from aka.core.loader import load


class _Abort(BaseException):
    """Stand-in for ``orchestrator.environment_recovery.EnvironmentUnavailable``.

    A ``BaseException`` and deliberately *not* an ``Exception``, so the containment points
    in the core cannot see it. Defined locally: importing the orchestrator into a core test
    would be the very layering leak the seams exist to prevent.
    """


class _Contained(Exception):
    """An ordinary plugin failure, the kind the core is supposed to absorb."""


# -- events under test ---------------------------------------------------------

ABORT_EMIT = Event(name="base-exception/emit", mode="emit", payload=str)
ABORT_PARALLEL = Event(name="base-exception/parallel", mode="parallel", payload=str)
ABORT_SERIAL = Event(name="base-exception/serial", mode="serial", payload=str, result=str)
ABORT_BAIL = Event(name="base-exception/bail", mode="bail", payload=str, result=str)
ABORT_WATERFALL = Event(
    name="base-exception/waterfall", mode="waterfall", payload=tuple, result=tuple
)

BY_MODE: dict[str, Event] = {
    "emit": ABORT_EMIT,
    "parallel": ABORT_PARALLEL,
    "serial": ABORT_SERIAL,
    "bail": ABORT_BAIL,
    "waterfall": ABORT_WATERFALL,
}

# Seams used by the fiber tests. A fiber provides a ``ServiceKey`` token directly, so these
# deliberately are not registered in ``aka.seams.SEAMS``.
ALPHA = ServiceKey(name="alpha_svc", definition="Alpha")


def _declaration(
    name: str,
    apply: Callable[..., Any],
    *,
    inject: tuple[str, ...] = (),
    provide: tuple[str, ...] = (),
) -> PluginDeclaration:
    """One validated declaration, assembled without a module on disk."""
    return PluginDeclaration(
        name=name,
        module=f"test.base_exception.{name.replace('-', '_')}",
        apply=apply,
        config_schema=None,
        defaults={},
        inject=inject,
        optional_inject=(),
        provide=provide,
        interpolate=(),
    )


# -- the premise ---------------------------------------------------------------


class AbortShapeTest(unittest.TestCase):
    """The tests below are only meaningful if the stand-in has the real class's shape."""

    def test_an_abort_escapes_except_exception(self) -> None:
        caught: list[str] = []
        try:
            try:
                raise _Abort("environment gone")
            except Exception:  # noqa: BLE001 - the containment the core writes everywhere
                caught.append("exception")
        except _Abort:
            caught.append("abort")

        self.assertEqual(caught, ["abort"])
        self.assertNotIsInstance(_Abort("x"), Exception)

    def test_environment_unavailable_still_subclasses_base_exception(self) -> None:
        """Read as text, not imported: the core must not depend on the orchestrator.

        If this ever fails, every ``except Exception`` in the core silently becomes a trap
        for the exit-75 protocol and the tests in this file stop describing reality.
        """
        source = Path(__file__).resolve().parents[2] / "orchestrator"
        source = source / "environment_recovery.py"
        if not source.is_file():  # pragma: no cover - the core is usable standalone
            self.skipTest("orchestrator/environment_recovery.py is not present")
        text = source.read_text(encoding="utf-8")
        self.assertIn("class EnvironmentUnavailable(BaseException):", text)
        self.assertIn("ENVIRONMENT_TEMPFAIL = 75", text)


# -- listeners -----------------------------------------------------------------


class ListenerAbortTest(unittest.TestCase):
    """One assertion set per dispatch mode: the abort escapes, the sink stays empty."""

    def setUp(self) -> None:
        self.reported: list[tuple[str, BaseException]] = []
        self.bus = EventBus(error_sink=self.sink, workers=2)

    def tearDown(self) -> None:
        self.bus.close()

    def sink(self, source: str, error: BaseException) -> None:
        self.reported.append((source, error))

    @staticmethod
    def raising(error: BaseException) -> Callable[..., Any]:
        """A listener that fails. Accepts the extra ``next`` a waterfall listener gets."""

        def listener(payload: Any, *rest: Any) -> Any:
            raise error

        return listener

    def test_every_dispatch_mode_is_covered(self) -> None:
        self.assertEqual(set(BY_MODE), set(MODES))

    def test_emit_listener_abort_propagates_instead_of_being_reported(self) -> None:
        abort = _Abort("environment gone")
        calls: list[str] = []
        self.bus.on(ABORT_EMIT, self.raising(abort), order=100, label="aborting")
        self.bus.on(ABORT_EMIT, lambda payload: calls.append("after"), order=200)

        with self.assertRaises(_Abort) as caught:
            self.bus.emit(ABORT_EMIT, "go")

        self.assertIs(caught.exception, abort)
        # emit is the most forgiving mode there is -- it contains every Exception -- and
        # even it stops dead here rather than carrying on with the next observer.
        self.assertEqual(calls, [])
        self.assertEqual(self.reported, [])

    def test_parallel_listener_abort_propagates_instead_of_being_reported(self) -> None:
        abort = _Abort("environment gone")
        self.bus.on(ABORT_PARALLEL, self.raising(abort), label="aborting")

        with self.assertRaises(_Abort) as caught:
            self.bus.parallel(ABORT_PARALLEL, "go")

        # The abort crossed a worker-thread boundary through ``future.result()`` and came
        # back as the same object, not as a wrapper.
        self.assertIs(caught.exception, abort)
        self.assertEqual(self.reported, [])

    def test_serial_listener_abort_propagates_instead_of_being_reported(self) -> None:
        abort = _Abort("environment gone")
        calls: list[str] = []
        self.bus.on(ABORT_SERIAL, self.raising(abort), order=100, label="aborting")
        self.bus.on(
            ABORT_SERIAL, lambda payload: calls.append("after") or "answer", order=200
        )

        with self.assertRaises(_Abort) as caught:
            self.bus.serial(ABORT_SERIAL, "q")

        self.assertIs(caught.exception, abort)
        self.assertEqual(calls, [])
        self.assertEqual(self.reported, [])

    def test_bail_listener_abort_propagates_instead_of_being_reported(self) -> None:
        abort = _Abort("environment gone")
        calls: list[str] = []
        self.bus.on(ABORT_BAIL, self.raising(abort), order=100, label="aborting")
        self.bus.on(
            ABORT_BAIL, lambda payload: calls.append("after") or "veto", order=200
        )

        with self.assertRaises(_Abort) as caught:
            self.bus.bail(ABORT_BAIL, "q")

        self.assertIs(caught.exception, abort)
        self.assertEqual(calls, [])
        self.assertEqual(self.reported, [])

    def test_waterfall_listener_abort_propagates_instead_of_being_reported(self) -> None:
        abort = _Abort("environment gone")
        calls: list[str] = []

        def outer(value: tuple, forward: Callable[..., Any]) -> tuple:
            calls.append("enter:outer")
            inner = forward(value)
            calls.append("exit:outer")
            return inner

        self.bus.on(ABORT_WATERFALL, outer, order=100, label="outer")
        self.bus.on(ABORT_WATERFALL, self.raising(abort), order=200, label="aborting")

        with self.assertRaises(_Abort) as caught:
            self.bus.waterfall(
                ABORT_WATERFALL,
                ("seed",),
                lambda value: calls.append("terminal") or value,
            )

        self.assertIs(caught.exception, abort)
        # The abort unwound straight through the wrapper: no post-delegation work, no
        # terminal, and no short-circuit record for the listener that never returned.
        self.assertEqual(calls, ["enter:outer"])
        self.assertEqual(self.bus.short_circuits, ())
        self.assertEqual(self.reported, [])

    def test_a_contained_failure_is_reported_while_an_abort_beside_it_is_not(self) -> None:
        self.bus.on(
            ABORT_EMIT, self.raising(_Contained("boom")), order=100, label="contained"
        )
        self.bus.on(
            ABORT_EMIT, self.raising(_Abort("environment gone")), order=200, label="abort"
        )

        with self.assertRaises(_Abort):
            self.bus.emit(ABORT_EMIT, "go")

        # The sink is wired up and working -- it simply never sees the abort.
        self.assertEqual(
            [source for source, _ in self.reported],
            ["base-exception/emit listener contained"],
        )
        self.assertIsInstance(self.reported[0][1], _Contained)


# -- apply ---------------------------------------------------------------------


class ApplyAbortTest(unittest.TestCase):
    """A row that aborts mid-load is left inert, and the abort keeps going."""

    def setUp(self) -> None:
        self.stderr = io.StringIO()
        self.receipt: list[str] = []
        self.root = Root(stderr=self.stderr)

    def tearDown(self) -> None:
        self.root.dispose()

    # -- row builders ----------------------------------------------------

    def row(
        self,
        name: str,
        *,
        error: BaseException | None = None,
        provides: tuple[ServiceKey, Any] | None = None,
        inject: tuple[str, ...] = (),
        fail_once: bool = False,
    ) -> PluginDeclaration:
        """A row that files an effect, optionally publishes, then optionally fails.

        The effect is filed *before* the failure so every test can see whether a partial
        load was unwound.
        """

        def apply(ctx: Any, config: Mapping[str, Any]) -> None:
            self.receipt.append(f"apply:{name}")
            ctx.effect(
                lambda: self.receipt.append(f"teardown:{name}"), label="partial"
            )
            if provides is not None:
                ctx.provide(provides[0], provides[1])
            if error is not None and not (
                fail_once and self.receipt.count(f"apply:{name}") > 1
            ):
                raise error

        return _declaration(
            name,
            apply,
            inject=inject,
            provide=() if provides is None else (provides[0].name,),
        )

    def transitions_for(self, entry_id: str) -> list[str]:
        return [
            f"{transition.previous}->{transition.current}"
            for transition in self.root.transitions
            if transition.entry_id == entry_id
        ]

    # -- the abort path --------------------------------------------------

    def test_abort_in_apply_propagates_out_of_settle(self) -> None:
        abort = _Abort("environment gone")
        self.root.mount(self.row("row", error=abort), entry_id="row")

        with self.assertRaises(_Abort) as caught:
            self.root.settle()

        self.assertIs(caught.exception, abort)
        self.assertNotIsInstance(caught.exception, Exception)
        self.assertNotIsInstance(caught.exception, CoreError)

    def test_aborting_row_ends_pending_and_is_never_marked_failed(self) -> None:
        fiber = self.root.mount(
            self.row("row", error=_Abort("environment gone")), entry_id="row"
        )

        with self.assertRaises(_Abort):
            self.root.settle()

        # PENDING, the container's legitimate resting state: the row is inert and could be
        # loaded again, which is exactly what FAILED would forbid.
        self.assertIs(fiber.state, FiberState.PENDING)
        self.assertIs(fiber.epoch, INACTIVE)
        self.assertIsNone(fiber.error)
        self.assertEqual(self.root.failed, ())
        self.assertEqual(self.root.active, ())
        self.assertEqual(self.root.pending, (("row", ()),))
        self.assertEqual(
            self.transitions_for("row"), ["pending->loading", "loading->pending"]
        )
        # Nothing was reported: an abort is the caller's business, not a contained failure.
        self.assertEqual(self.root.errors, [])
        self.assertEqual(self.stderr.getvalue(), "")

    def test_partial_effects_are_unwound_and_the_row_provides_nothing(self) -> None:
        fiber = self.root.mount(
            self.row("row", error=_Abort("gone"), provides=(ALPHA, "alpha-1")),
            entry_id="row",
        )

        with self.assertRaises(_Abort):
            self.root.settle()

        self.assertEqual(self.receipt, ["apply:row", "teardown:row"])
        self.assertEqual(len(fiber.effects), 0)
        self.assertEqual(fiber.effects.live, ())
        self.assertEqual(self.root.live_effects, 0)
        # The service it had already published is gone with the effect that carried it.
        self.assertEqual(fiber.provided, set())
        self.assertIsNone(self.root.realm.resolve("alpha_svc"))

    def test_a_consumer_of_an_aborting_provider_keeps_waiting(self) -> None:
        consumer = self.root.mount(
            self.row("consumer", inject=("alpha_svc",)), entry_id="consumer"
        )
        self.root.mount(
            self.row("provider", error=_Abort("gone"), provides=(ALPHA, "alpha-1")),
            entry_id="provider",
        )

        with self.assertRaises(_Abort):
            self.root.settle()

        # The provider withdrew what it had published, so the consumer is still waiting on
        # an absent service rather than holding a value from a half-loaded neighbour.
        self.assertIs(consumer.state, FiberState.PENDING)
        self.assertEqual(consumer.unsatisfied, ("alpha_svc",))
        self.assertEqual(self.receipt, ["apply:provider", "teardown:provider"])
        self.assertEqual(self.root.errors, [])

    def test_pending_after_an_abort_is_retryable(self) -> None:
        """Ending inert rather than failed is what makes a retry possible at all."""
        fiber = self.root.mount(
            self.row("row", error=_Abort("gone"), fail_once=True), entry_id="row"
        )

        with self.assertRaises(_Abort):
            self.root.settle()
        self.assertIs(fiber.state, FiberState.PENDING)

        fiber.restart()
        self.root.settle()

        self.assertIs(fiber.state, FiberState.ACTIVE)
        self.assertEqual(
            self.receipt, ["apply:row", "teardown:row", "apply:row"]
        )
        self.assertEqual(fiber.effects.live, ("partial",))

    def test_a_listener_abort_crossing_a_plugin_boundary_leaves_the_emitter_inert(
        self,
    ) -> None:
        """The real shape of the protocol: row A aborts because row B's listener did.

        Nothing in the path from the listener, through ``ctx.emit``, through ``apply``, to
        ``settle`` may absorb it -- and the emitting row still has to be cleaned up.
        """
        abort = _Abort("environment gone")

        def observer_apply(ctx: Any, config: Mapping[str, Any]) -> None:
            self.receipt.append("apply:observer")
            ctx.on(ABORT_EMIT, ListenerAbortTest.raising(abort), label="aborting")

        def producer_apply(ctx: Any, config: Mapping[str, Any]) -> None:
            self.receipt.append("apply:producer")
            ctx.effect(lambda: self.receipt.append("teardown:producer"), label="partial")
            ctx.emit(ABORT_EMIT, "go")
            self.receipt.append("after-emit:producer")

        observer = self.root.mount(
            _declaration("observer", observer_apply), entry_id="observer"
        )
        producer = self.root.mount(
            _declaration("producer", producer_apply), entry_id="producer"
        )

        with self.assertRaises(_Abort) as caught:
            self.root.settle()

        self.assertIs(caught.exception, abort)
        self.assertIs(observer.state, FiberState.ACTIVE)
        self.assertIs(producer.state, FiberState.PENDING)
        self.assertIsNone(producer.error)
        self.assertEqual(len(producer.effects), 0)
        self.assertEqual(
            self.receipt,
            ["apply:observer", "apply:producer", "teardown:producer"],
        )
        self.assertEqual(self.root.errors, [])

    # -- contrast: an ordinary failure is contained -----------------------

    def test_plain_exception_in_apply_ends_failed_and_does_not_propagate(self) -> None:
        boom = _Contained("plugin is broken")
        fiber = self.root.mount(self.row("row", error=boom), entry_id="row")

        self.root.settle()  # no raise: a failing row is a contained failure

        self.assertIs(fiber.state, FiberState.FAILED)
        self.assertIs(fiber.error, boom)
        self.assertIs(fiber.epoch, INACTIVE)
        self.assertEqual(self.root.failed, (("row", "_Contained: plugin is broken"),))
        self.assertEqual(self.root.pending, ())
        # Same cleanup as the abort path, but reported instead of raised.
        self.assertEqual(self.receipt, ["apply:row", "teardown:row"])
        self.assertEqual(self.root.live_effects, 0)
        self.assertEqual([report.source for report in self.root.errors], ["entry row"])
        self.assertIn("plugin is broken", self.stderr.getvalue())

    def test_config_stage_abort_propagates_but_strands_the_row(self) -> None:
        """An abort raised before ``apply`` -- here from an ``internal/config`` listener.

        ``_load`` guards the config stage with ``except CoreError`` only, so the abort does
        reach the caller -- that part is the contract -- and the row must still be left inert, so
        the config stage gets the same containment as the apply stage.
        """
        abort = _Abort("environment gone")
        self.root.bus.on(
            INTERNAL_CONFIG, ListenerAbortTest.raising(abort), label="aborting"
        )
        fiber = self.root.mount(self.row("row"), entry_id="row")

        with self.assertRaises(_Abort) as caught:
            self.root.settle()

        self.assertIs(caught.exception, abort)
        self.assertEqual(self.receipt, [])  # apply never ran
        self.assertIsNone(fiber.error)
        self.assertEqual(len(fiber.effects), 0)
        self.assertEqual(self.root.active, ())
        self.assertEqual(self.root.failed, ())
        self.assertEqual(self.root.errors, [])

        # The abort reaches the caller, and the row is left inert and retryable in PENDING -- the
        # same resting state the apply path documents -- rather than stranded in LOADING with a
        # live epoch, invisible to every Root diagnostic.
        self.assertIs(fiber.state, FiberState.PENDING)
        self.assertIs(fiber.epoch, INACTIVE)
        self.assertEqual(self.root.pending, (("row", ()),))


# -- disposers -----------------------------------------------------------------


class DisposerAbortTest(unittest.TestCase):
    """A disposer that aborts is not a disposal failure: it is the end of the process."""

    def test_abort_from_a_disposer_propagates_out_of_unwind(self) -> None:
        abort = _Abort("environment gone")
        order: list[str] = []
        stack = EffectStack("fiber-a")
        stack.add(lambda: order.append("alpha"), label="alpha")
        stack.add(self.failing(order, "beta", abort), label="beta")
        stack.add(lambda: order.append("gamma"), label="gamma")

        with self.assertRaises(_Abort) as caught:
            stack.unwind()

        self.assertIs(caught.exception, abort)
        self.assertNotIsInstance(caught.exception, DisposalErrors)
        # The abort reaches the caller untouched -- exit 75 depends on that -- but the stack is
        # still drained: an effect left installed behind an abort has nothing left to remove it.
        self.assertEqual(order, ["gamma", "beta", "alpha"])
        self.assertEqual(stack.live, ())

    def test_an_abort_is_not_folded_into_a_collected_disposal_failure(self) -> None:
        stack = EffectStack("fiber-a")
        order: list[str] = []
        stack.add(self.failing(order, "alpha", _Abort("environment gone")), label="alpha")
        stack.add(self.failing(order, "beta", _Contained("boom")), label="beta")

        # "beta" fails first and is collected; then "alpha" aborts. The abort wins over the
        # aggregate rather than being wrapped in it.
        with self.assertRaises(_Abort):
            stack.unwind()

        self.assertEqual(order, ["beta", "alpha"])
        self.assertEqual(stack.live, ())

    def test_abort_from_a_row_teardown_reaches_the_caller_unreported(self) -> None:
        stderr = io.StringIO()
        root = Root(stderr=stderr)
        self.addCleanup(root.dispose)
        abort = _Abort("environment gone")
        receipt: list[str] = []

        def apply(ctx: Any, config: Mapping[str, Any]) -> None:
            ctx.effect(lambda: receipt.append("kept"), label="kept")
            ctx.effect(self.failing(receipt, "aborting", abort), label="aborting")

        fiber = root.mount(_declaration("row", apply), entry_id="row")
        root.settle()
        self.assertIs(fiber.state, FiberState.ACTIVE)

        with self.assertRaises(_Abort) as caught:
            fiber.dispose()

        self.assertIs(caught.exception, abort)
        # ``Fiber._discard_effects`` converts DisposalErrors into a report; an abort must be
        # neither converted nor logged, because the caller has to see it to exit 75. The rest of
        # the row's teardown still runs, so the abort costs nothing in leaked registrations.
        self.assertEqual(root.errors, [])
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(receipt, ["aborting", "kept"])

    @staticmethod
    def failing(log: list[str], name: str, error: BaseException) -> Callable[[], None]:
        def dispose() -> None:
            log.append(name)
            raise error

        return dispose


# -- teardown while an abort is in flight --------------------------------------


class DisposeDuringAbortTest(unittest.TestCase):
    """The orchestrator's own shape: ``try: ... finally: root.dispose()``."""

    def setUp(self) -> None:
        self.stderr = io.StringIO()
        self.receipt: list[str] = []
        self.root = Root(stderr=self.stderr)

    def tearDown(self) -> None:
        self.root.dispose()

    def row(self, name: str, *, error: BaseException | None = None) -> PluginDeclaration:
        def apply(ctx: Any, config: Mapping[str, Any]) -> Callable[[], None]:
            self.receipt.append(f"apply:{name}")

            def teardown() -> None:
                self.receipt.append(f"teardown:{name}")
                if error is not None:
                    raise error

            return teardown

        return _declaration(name, apply)

    def test_dispose_in_a_finally_runs_every_disposer_and_the_abort_continues(
        self,
    ) -> None:
        for name in ("a", "b", "c"):
            self.root.mount(self.row(name), entry_id=name)
        self.root.settle()
        abort = _Abort("environment gone")

        with self.assertRaises(_Abort) as caught:
            try:
                raise abort
            finally:
                self.root.dispose()

        # The teardown completed in reverse mount order, and the exception that was already
        # in flight is the one that came out -- dispose() neither swallowed nor replaced it.
        self.assertIs(caught.exception, abort)
        self.assertEqual(
            self.receipt[-3:], ["teardown:c", "teardown:b", "teardown:a"]
        )
        self.assertEqual(self.root.live_effects, 0)
        self.assertEqual(self.root.fibers, {})
        self.assertTrue(self.root.bus.closed)
        self.assertEqual(self.root.errors, [])

    def test_a_failing_disposer_during_the_abort_neither_starves_the_rest_nor_wins(
        self,
    ) -> None:
        self.root.mount(self.row("a"), entry_id="a")
        self.root.mount(self.row("b", error=_Contained("teardown is broken")), entry_id="b")
        self.root.mount(self.row("c"), entry_id="c")
        self.root.settle()
        abort = _Abort("environment gone")

        with self.assertRaises(_Abort) as caught:
            try:
                raise abort
            finally:
                self.root.dispose()

        self.assertIs(caught.exception, abort)
        # Every row was still torn down, including the one after the broken disposer...
        self.assertEqual(
            self.receipt[-3:], ["teardown:c", "teardown:b", "teardown:a"]
        )
        self.assertEqual(self.root.fibers, {})
        self.assertTrue(self.root.bus.closed)
        # ...and the contained failure was reported rather than raised over the abort.
        self.assertEqual([report.source for report in self.root.errors], ["b teardown"])
        self.assertIn("DisposalErrors", self.root.errors[0].error)
        self.assertIn("teardown is broken", self.root.errors[0].error)


# -- the host boundary ---------------------------------------------------------


class HostBoundaryTest(unittest.TestCase):
    """``load()`` is where the tree meets the process, so the abort has to arrive here."""

    def setUp(self) -> None:
        self.stderr = io.StringIO()
        self.receipt: list[str] = []

    def module(self, suffix: str, apply: Callable[..., Any]) -> str:
        """Register a throwaway plugin module and return its import path.

        ``loader.import_plugin`` goes through ``importlib``, which answers from
        ``sys.modules`` first, so a plugin fixture needs no file on disk.
        """
        module_name = f"aka.core.test_base_exception_{suffix}"
        module = ModuleType(module_name)
        module.name = module_name.rsplit(".", 1)[-1].replace("_", "-")
        module.apply = apply
        sys.modules[module_name] = module
        self.addCleanup(sys.modules.pop, module_name, None)
        return module_name

    def test_abort_reaches_the_host_through_load_and_the_tree_is_disposed(self) -> None:
        abort = _Abort("environment gone")

        def good_apply(ctx: Any, config: Mapping[str, Any]) -> Callable[[], None]:
            self.receipt.append("apply:good")
            return lambda: self.receipt.append("teardown:good")

        def aborting_apply(ctx: Any, config: Mapping[str, Any]) -> None:
            self.receipt.append("apply:aborting")
            ctx.effect(lambda: self.receipt.append("teardown:aborting"), label="partial")
            raise abort

        composition = ResolvedComposition(
            profile="test/base-exception",
            rows=(
                Row(id="good", name=self.module("host_good", good_apply)),
                Row(id="aborting", name=self.module("host_abort", aborting_apply)),
            ),
            layers=("in-test",),
        )

        with self.assertRaises(_Abort) as caught:
            load(composition, stderr=self.stderr)

        # Not a BootFailure, not a warning: the same object the plugin raised.
        self.assertIs(caught.exception, abort)
        self.assertNotIsInstance(caught.exception, CoreError)
        # ``load`` unwinds the whole tree on its way out, so the row that had already
        # loaded is torn down before the abort continues to the entry point.
        self.assertEqual(
            self.receipt,
            ["apply:good", "apply:aborting", "teardown:aborting", "teardown:good"],
        )


if __name__ == "__main__":
    unittest.main()
