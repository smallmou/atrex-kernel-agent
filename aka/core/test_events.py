"""Behavioural tests for the typed event bus.

The bus is the only extension point plugins get, so its contract has to hold exactly:
the declared dispatch mode is enforced at dispatch, an observer that throws is contained
rather than allowed to starve its peers, a decision mode stops at the first answer, and a
waterfall's delegation discipline (one ``next()`` per listener, monotonic growth when the
event asks for it) is checked rather than trusted.
"""

from __future__ import annotations

import threading
import unittest
from typing import Any, Callable

from aka.core.errors import (
    EventContractError,
    MonotonicViolation,
    WaterfallReentry,
)
from aka.core.events import EventBus
from aka.core.keys import MODES, Event


# -- events under test ---------------------------------------------------------

EMIT = Event(name="test/emit", mode="emit", payload=str)
PARALLEL = Event(name="test/parallel", mode="parallel", payload=str)
SERIAL = Event(name="test/serial", mode="serial", payload=str, result=str)
BAIL = Event(name="test/bail", mode="bail", payload=str, result=str)
WATERFALL = Event(name="test/waterfall", mode="waterfall", payload=tuple, result=tuple)


def _dropped(outer: Any, inner: Any) -> str | None:
    """Shrink check for a tuple waterfall: nothing the inner chain produced may vanish."""
    if not isinstance(inner, tuple) or not isinstance(outer, tuple):
        return None
    missing = [item for item in inner if item not in outer]
    if missing:
        return "dropped " + ", ".join(str(item) for item in missing)
    return None


MONOTONIC = Event(
    name="test/monotonic",
    mode="waterfall",
    payload=tuple,
    result=tuple,
    monotonic=True,
    shrink_check=_dropped,
)

BY_MODE: dict[str, Event] = {
    "emit": EMIT,
    "parallel": PARALLEL,
    "serial": SERIAL,
    "bail": BAIL,
    "waterfall": WATERFALL,
}


class Boom(Exception):
    """A listener failure that the bus is expected to contain."""


class Fatal(BaseException):
    """Stands in for ``EnvironmentUnavailable``: must never be contained."""


class EventBusTestCase(unittest.TestCase):
    """Shared bus with a recording error sink."""

    def setUp(self) -> None:
        self.reported: list[tuple[str, BaseException]] = []
        self.lock = threading.Lock()
        self.bus = EventBus(error_sink=self.sink, workers=4)

    def tearDown(self) -> None:
        self.bus.close()

    def sink(self, source: str, error: BaseException) -> None:
        with self.lock:
            self.reported.append((source, error))

    def dispatch(self, mode: str, event: Event) -> Any:
        """Call the bus method for ``mode`` with a payload the event would accept."""
        if mode == "emit":
            return self.bus.emit(event, self.payload_for(event))
        if mode == "parallel":
            return self.bus.parallel(event, self.payload_for(event))
        if mode == "serial":
            return self.bus.serial(event, self.payload_for(event))
        if mode == "bail":
            return self.bus.bail(event, self.payload_for(event))
        if mode == "waterfall":
            return self.bus.waterfall(event, self.payload_for(event), lambda value: value)
        raise AssertionError(f"unhandled mode {mode!r}")

    @staticmethod
    def payload_for(event: Event) -> Any:
        return () if event.payload is tuple else "payload"


# -- mode enforcement ----------------------------------------------------------


class ModeEnforcementTest(EventBusTestCase):
    def test_every_wrong_mode_pairing_raises_event_contract_error(self) -> None:
        self.assertEqual(set(BY_MODE), set(MODES))
        checked = 0
        for declared, event in BY_MODE.items():
            for dispatched in sorted(MODES):
                if dispatched == declared:
                    continue
                checked += 1
                with self.subTest(declared=declared, dispatched=dispatched):
                    with self.assertRaises(EventContractError) as caught:
                        self.dispatch(dispatched, event)
                    message = str(caught.exception)
                    self.assertIn(event.name, message)
                    self.assertIn(declared, message)
                    self.assertIn(dispatched, message)
                    self.assertEqual(caught.exception.event, event.name)
        self.assertEqual(checked, 20)

    def test_waterfall_event_cannot_be_emitted(self) -> None:
        seen: list[Any] = []
        self.bus.on(WATERFALL, lambda value, forward: forward(value))
        with self.assertRaises(EventContractError):
            self.bus.emit(WATERFALL, ("a",))
        self.assertEqual(seen, [])

    def test_mode_is_enforced_before_the_closed_guard(self) -> None:
        self.bus.close()
        with self.assertRaises(EventContractError):
            self.bus.emit(SERIAL, "payload")

    def test_registering_a_non_callable_raises(self) -> None:
        with self.assertRaises(EventContractError):
            self.bus.on(EMIT, "not a function")  # type: ignore[arg-type]
        self.assertFalse(self.bus.has_listeners(EMIT))


# -- emit ----------------------------------------------------------------------


class EmitTest(EventBusTestCase):
    def test_listeners_run_by_order_then_registration_sequence(self) -> None:
        calls: list[str] = []

        def record(name: str) -> Callable[[Any], None]:
            return lambda payload: calls.append(f"{name}:{payload}")

        self.bus.on(EMIT, record("late"), order=900)
        self.bus.on(EMIT, record("mid-first"), order=500)
        self.bus.on(EMIT, record("mid-second"), order=500)
        self.bus.on(EMIT, record("early"), order=100)

        self.bus.emit(EMIT, "go")

        self.assertEqual(
            calls,
            ["early:go", "mid-first:go", "mid-second:go", "late:go"],
        )
        self.assertEqual(
            [listener.order for listener in self.bus.listeners(EMIT)],
            [100, 500, 500, 900],
        )

    def test_default_order_places_listener_in_the_middle_band(self) -> None:
        calls: list[str] = []
        self.bus.on(EMIT, lambda payload: calls.append("default"))
        self.bus.on(EMIT, lambda payload: calls.append("first"), order=1)
        self.bus.on(EMIT, lambda payload: calls.append("last"), order=999)
        self.bus.emit(EMIT, "go")
        self.assertEqual(calls, ["first", "default", "last"])

    def test_throwing_listener_is_reported_and_later_listeners_still_run(self) -> None:
        calls: list[str] = []

        def explode(payload: str) -> None:
            calls.append("explode")
            raise Boom("observer failed")

        self.bus.on(EMIT, lambda payload: calls.append("before"), order=100)
        self.bus.on(EMIT, explode, order=200, owner="pkg", label="bad-observer")
        self.bus.on(EMIT, lambda payload: calls.append("after"), order=300)

        self.bus.emit(EMIT, "go")

        self.assertEqual(calls, ["before", "explode", "after"])
        self.assertEqual(len(self.reported), 1)
        source, error = self.reported[0]
        self.assertIn("test/emit", source)
        self.assertIn("bad-observer", source)
        self.assertIsInstance(error, Boom)

    def test_unlabelled_listener_is_described_by_owner_and_sequence(self) -> None:
        def explode(payload: str) -> None:
            raise Boom("nope")

        self.bus.on(EMIT, explode, owner="pkg-a")
        self.bus.emit(EMIT, "go")

        self.assertEqual(len(self.reported), 1)
        self.assertIn("pkg-a#", self.reported[0][0])

    def test_every_throwing_listener_is_reported(self) -> None:
        for index in range(3):
            self.bus.on(
                EMIT,
                lambda payload: (_ for _ in ()).throw(Boom("x")),
                label=f"bad-{index}",
            )
        self.bus.emit(EMIT, "go")
        self.assertEqual(len(self.reported), 3)
        self.assertEqual(
            sorted(source.rsplit(" ", 1)[-1] for source, _ in self.reported),
            ["bad-0", "bad-1", "bad-2"],
        )

    def test_emit_without_error_sink_still_contains_the_failure(self) -> None:
        bus = EventBus()
        self.addCleanup(bus.close)
        calls: list[str] = []
        bus.on(EMIT, lambda payload: (_ for _ in ()).throw(Boom("x")), order=100)
        bus.on(EMIT, lambda payload: calls.append("after"), order=200)
        bus.emit(EMIT, "go")
        self.assertEqual(calls, ["after"])

    def test_base_exception_is_not_contained(self) -> None:
        calls: list[str] = []

        def fatal(payload: str) -> None:
            raise Fatal("environment gone")

        self.bus.on(EMIT, fatal, order=100)
        self.bus.on(EMIT, lambda payload: calls.append("after"), order=200)

        with self.assertRaises(Fatal):
            self.bus.emit(EMIT, "go")
        self.assertEqual(calls, [])
        self.assertEqual(self.reported, [])

    def test_listeners_snapshot_is_taken_before_dispatch(self) -> None:
        calls: list[str] = []

        def grow(payload: str) -> None:
            calls.append("grow")
            self.bus.on(EMIT, lambda later: calls.append("added"), order=900)

        self.bus.on(EMIT, grow, order=100)
        self.bus.emit(EMIT, "go")
        self.assertEqual(calls, ["grow"])
        self.bus.emit(EMIT, "again")
        self.assertEqual(calls, ["grow", "grow", "added"])

    def test_disposer_removes_only_its_own_listener(self) -> None:
        calls: list[str] = []
        first = self.bus.on(EMIT, lambda payload: calls.append("first"), order=100)
        self.bus.on(EMIT, lambda payload: calls.append("second"), order=200)

        first()
        self.bus.emit(EMIT, "go")
        self.assertEqual(calls, ["second"])
        self.assertEqual(len(self.bus.listeners(EMIT)), 1)

        first()  # idempotent
        self.assertEqual(len(self.bus.listeners(EMIT)), 1)


# -- parallel ------------------------------------------------------------------


class ParallelTest(EventBusTestCase):
    def test_every_listener_runs(self) -> None:
        seen: list[str] = []
        started = threading.Barrier(3, timeout=5)

        def waiter(name: str) -> Callable[[Any], None]:
            def listener(payload: str) -> None:
                started.wait()
                with self.lock:
                    seen.append(f"{name}:{payload}")

            return listener

        for name in ("a", "b", "c"):
            self.bus.on(PARALLEL, waiter(name))

        self.bus.parallel(PARALLEL, "go")

        self.assertEqual(sorted(seen), ["a:go", "b:go", "c:go"])

    def test_all_exceptions_are_reported_and_none_escapes(self) -> None:
        seen: list[str] = []

        def listener(name: str, fail: bool) -> Callable[[Any], None]:
            def run(payload: str) -> None:
                with self.lock:
                    seen.append(name)
                if fail:
                    raise Boom(name)

            return run

        self.bus.on(PARALLEL, listener("ok-1", False), label="ok-1")
        self.bus.on(PARALLEL, listener("bad-1", True), label="bad-1")
        self.bus.on(PARALLEL, listener("bad-2", True), label="bad-2")
        self.bus.on(PARALLEL, listener("ok-2", False), label="ok-2")

        self.assertIsNone(self.bus.parallel(PARALLEL, "go"))

        self.assertEqual(sorted(seen), ["bad-1", "bad-2", "ok-1", "ok-2"])
        self.assertEqual(
            sorted(source.rsplit(" ", 1)[-1] for source, _ in self.reported),
            ["bad-1", "bad-2"],
        )
        for source, error in self.reported:
            self.assertIn("test/parallel", source)
            self.assertIsInstance(error, Boom)

    def test_parallel_with_no_listeners_is_a_noop(self) -> None:
        self.assertIsNone(self.bus.parallel(PARALLEL, "go"))
        self.assertEqual(self.reported, [])

    def test_more_listeners_than_pool_workers_all_run(self) -> None:
        bus = EventBus(error_sink=self.sink, workers=1)
        self.addCleanup(bus.close)
        seen: list[str] = []
        for index in range(5):
            bus.on(
                PARALLEL,
                lambda payload, index=index: seen.append(str(index)),
                order=index,
            )
        bus.parallel(PARALLEL, "go")
        self.assertEqual(sorted(seen), ["0", "1", "2", "3", "4"])


# -- serial --------------------------------------------------------------------


class SerialTest(EventBusTestCase):
    def test_first_non_none_result_wins_and_stops_the_chain(self) -> None:
        calls: list[str] = []

        def abstain(payload: str) -> None:
            calls.append("abstain")
            return None

        def answer(payload: str) -> str:
            calls.append("answer")
            return f"answered:{payload}"

        def never(payload: str) -> str:
            calls.append("never")
            return "too late"

        self.bus.on(SERIAL, abstain, order=100)
        self.bus.on(SERIAL, answer, order=200)
        self.bus.on(SERIAL, never, order=300)

        result = self.bus.serial(SERIAL, "q")

        self.assertEqual(result, "answered:q")
        self.assertEqual(calls, ["abstain", "answer"])

    def test_all_abstaining_returns_none(self) -> None:
        calls: list[str] = []
        for index in range(3):
            self.bus.on(
                SERIAL,
                lambda payload, index=index: calls.append(str(index)),
                order=index,
            )
        self.assertIsNone(self.bus.serial(SERIAL, "q"))
        self.assertEqual(calls, ["0", "1", "2"])

    def test_no_listeners_returns_none(self) -> None:
        self.assertIsNone(self.bus.serial(SERIAL, "q"))

    def test_empty_string_is_a_valid_serial_answer(self) -> None:
        calls: list[str] = []
        self.bus.on(SERIAL, lambda payload: "", order=100)
        self.bus.on(SERIAL, lambda payload: calls.append("never") or "x", order=200)
        self.assertEqual(self.bus.serial(SERIAL, "q"), "")
        self.assertEqual(calls, [])

    def test_serial_listener_exception_is_not_contained(self) -> None:
        """A decision is not an observation: a broken decider must not be swallowed."""
        calls: list[str] = []
        self.bus.on(SERIAL, lambda payload: (_ for _ in ()).throw(Boom("x")), order=100)
        self.bus.on(SERIAL, lambda payload: calls.append("after") or "v", order=200)
        with self.assertRaises(Boom):
            self.bus.serial(SERIAL, "q")
        self.assertEqual(calls, [])
        self.assertEqual(self.reported, [])


# -- bail ----------------------------------------------------------------------


class BailTest(EventBusTestCase):
    def test_first_truthy_result_wins(self) -> None:
        calls: list[str] = []

        def decide(name: str, value: Any) -> Callable[[Any], Any]:
            def run(payload: str) -> Any:
                calls.append(name)
                return value

            return run

        self.bus.on(BAIL, decide("empty", ""), order=100)
        self.bus.on(BAIL, decide("none", None), order=200)
        self.bus.on(BAIL, decide("veto", "stop"), order=300)
        self.bus.on(BAIL, decide("never", "late"), order=400)

        self.assertEqual(self.bus.bail(BAIL, "q"), "stop")
        self.assertEqual(calls, ["empty", "none", "veto"])

    def test_falsy_return_is_an_abstention(self) -> None:
        calls: list[str] = []
        for name, value in (("empty", ""), ("none", None)):
            self.bus.on(
                BAIL,
                lambda payload, name=name, value=value: (
                    calls.append(name) or value
                ),
            )
        self.assertIsNone(self.bus.bail(BAIL, "q"))
        self.assertEqual(calls, ["empty", "none"])

    def test_bail_result_type_is_enforced_on_the_winner(self) -> None:
        self.bus.on(BAIL, lambda payload: 17, order=100)
        with self.assertRaises(EventContractError) as caught:
            self.bus.bail(BAIL, "q")
        self.assertIn("result must be str", str(caught.exception))


# -- waterfall -----------------------------------------------------------------


class WaterfallTest(EventBusTestCase):
    def test_chain_runs_outermost_first_and_terminal_last(self) -> None:
        trace: list[str] = []

        def wrap(name: str) -> Callable[[Any, Any], Any]:
            def listener(value: tuple, forward: Callable[..., Any]) -> tuple:
                trace.append(f"enter:{name}")
                inner = forward(value + (name,))
                trace.append(f"exit:{name}")
                return inner + (f"{name}-out",)

            return listener

        self.bus.on(WATERFALL, wrap("outer"), order=100)
        self.bus.on(WATERFALL, wrap("inner"), order=200)

        result = self.bus.waterfall(
            WATERFALL, ("seed",), lambda value: value + ("terminal",)
        )

        self.assertEqual(
            trace,
            ["enter:outer", "enter:inner", "exit:inner", "exit:outer"],
        )
        self.assertEqual(
            result,
            ("seed", "outer", "inner", "terminal", "inner-out", "outer-out"),
        )
        self.assertEqual(self.bus.short_circuits, ())

    def test_terminal_is_used_directly_when_no_listeners(self) -> None:
        result = self.bus.waterfall(
            WATERFALL, ("seed",), lambda value: value + ("terminal",)
        )
        self.assertEqual(result, ("seed", "terminal"))

    def test_next_without_argument_forwards_the_received_value(self) -> None:
        seen: list[tuple] = []
        self.bus.on(
            WATERFALL,
            lambda value, forward: forward(),
            order=100,
        )
        self.bus.on(
            WATERFALL,
            lambda value, forward: (seen.append(value) or forward(value)),
            order=200,
        )
        result = self.bus.waterfall(WATERFALL, ("seed",), lambda value: value)
        self.assertEqual(seen, [("seed",)])
        self.assertEqual(result, ("seed",))

    def test_next_with_argument_forwards_the_replacement(self) -> None:
        seen: list[tuple] = []
        terminal_saw: list[tuple] = []
        self.bus.on(
            WATERFALL,
            lambda value, forward: forward(("replaced",)),
            order=100,
        )
        self.bus.on(
            WATERFALL,
            lambda value, forward: (seen.append(value) or forward(value)),
            order=200,
        )

        def terminal(value: tuple) -> tuple:
            terminal_saw.append(value)
            return value

        result = self.bus.waterfall(WATERFALL, ("seed",), terminal)
        self.assertEqual(seen, [("replaced",)])
        self.assertEqual(terminal_saw, [("replaced",)])
        self.assertEqual(result, ("replaced",))

    def test_next_returns_the_inner_result_to_the_listener(self) -> None:
        captured: list[tuple] = []

        def listener(value: tuple, forward: Callable[..., Any]) -> tuple:
            inner = forward(value)
            captured.append(inner)
            return inner

        self.bus.on(WATERFALL, listener)
        result = self.bus.waterfall(
            WATERFALL, ("seed",), lambda value: value + ("terminal",)
        )
        self.assertEqual(captured, [("seed", "terminal")])
        self.assertEqual(result, ("seed", "terminal"))

    def test_short_circuit_skips_the_rest_and_is_recorded(self) -> None:
        calls: list[str] = []

        def stopper(value: tuple, forward: Callable[..., Any]) -> tuple:
            calls.append("stopper")
            return ("short",)

        self.bus.on(WATERFALL, stopper, order=100, owner="pkg", label="stopper")
        self.bus.on(
            WATERFALL,
            lambda value, forward: calls.append("downstream") or forward(value),
            order=200,
        )

        result = self.bus.waterfall(
            WATERFALL, ("seed",), lambda value: calls.append("terminal") or value
        )

        self.assertEqual(result, ("short",))
        self.assertEqual(calls, ["stopper"])
        self.assertEqual(self.bus.short_circuits, (("test/waterfall", "stopper"),))

    def test_short_circuit_of_an_inner_listener_still_runs_the_outer_wrapper(
        self,
    ) -> None:
        calls: list[str] = []

        def outer(value: tuple, forward: Callable[..., Any]) -> tuple:
            calls.append("outer")
            return forward(value) + ("outer-out",)

        self.bus.on(WATERFALL, outer, order=100, label="outer")
        self.bus.on(
            WATERFALL,
            lambda value, forward: calls.append("stopper") or ("short",),
            order=200,
            label="stopper",
        )

        result = self.bus.waterfall(
            WATERFALL, ("seed",), lambda value: calls.append("terminal") or value
        )

        self.assertEqual(calls, ["outer", "stopper"])
        self.assertEqual(result, ("short", "outer-out"))
        self.assertEqual(self.bus.short_circuits, (("test/waterfall", "stopper"),))

    def test_delegating_listener_is_not_recorded_as_a_short_circuit(self) -> None:
        self.bus.on(WATERFALL, lambda value, forward: forward(value), label="good")
        self.bus.waterfall(WATERFALL, ("seed",), lambda value: value)
        self.assertEqual(self.bus.short_circuits, ())

    def test_calling_next_twice_raises_waterfall_reentry(self) -> None:
        def twice(value: tuple, forward: Callable[..., Any]) -> tuple:
            forward(value)
            return forward(value)

        self.bus.on(WATERFALL, twice, order=100, owner="pkg", label="greedy")

        with self.assertRaises(WaterfallReentry) as caught:
            self.bus.waterfall(WATERFALL, ("seed",), lambda value: value)

        self.assertEqual(caught.exception.event, "test/waterfall")
        self.assertEqual(caught.exception.listener, "greedy")
        self.assertIn("more than once", str(caught.exception))
        self.assertIsInstance(caught.exception, EventContractError)

    def test_reentry_is_detected_per_listener_not_per_dispatch(self) -> None:
        """Two different listeners each calling next() once is legal."""
        self.bus.on(WATERFALL, lambda value, forward: forward(value), order=100)
        self.bus.on(WATERFALL, lambda value, forward: forward(value), order=200)
        result = self.bus.waterfall(
            WATERFALL, ("seed",), lambda value: value + ("terminal",)
        )
        self.assertEqual(result, ("seed", "terminal"))

    def test_the_same_listener_may_delegate_again_on_a_later_dispatch(self) -> None:
        self.bus.on(WATERFALL, lambda value, forward: forward(value), label="stable")
        for _ in range(3):
            self.assertEqual(
                self.bus.waterfall(WATERFALL, ("seed",), lambda value: value),
                ("seed",),
            )
        self.assertEqual(self.bus.short_circuits, ())

    def test_short_circuits_records_one_entry_per_dispatch(self) -> None:
        """The record is a log of occurrences, not a set of offending listeners."""
        self.bus.on(WATERFALL, lambda value, forward: ("short",), label="stopper")
        for _ in range(4):
            self.bus.waterfall(WATERFALL, ("seed",), lambda value: value)
        self.assertEqual(
            self.bus.short_circuits,
            (("test/waterfall", "stopper"),) * 4,
        )

    def test_a_captured_next_must_not_outlive_its_dispatch(self) -> None:
        """A listener that stashes ``next`` instead of calling it defers the chain.

        The dispatch itself behaves correctly -- the downstream listener never runs and
        the abstention is recorded -- but the delegate closure stays callable afterwards.
        """
        captured: list[Callable[..., Any]] = []
        downstream: list[str] = []

        def capturer(value: tuple, forward: Callable[..., Any]) -> tuple:
            captured.append(forward)
            return ("short",)

        def down(value: tuple, forward: Callable[..., Any]) -> tuple:
            downstream.append("ran")
            return forward(value)

        self.bus.on(WATERFALL, capturer, order=100, label="capturer")
        self.bus.on(WATERFALL, down, order=200, label="down")

        result = self.bus.waterfall(
            WATERFALL, ("seed",), lambda value: value + ("terminal",)
        )
        self.assertEqual(result, ("short",))
        self.assertEqual(downstream, [])
        self.assertEqual(self.bus.short_circuits, (("test/waterfall", "capturer"),))

        self.bus.close()
        self.assertEqual(self.bus.listeners(WATERFALL), ())

        # The chain is valid only for the duration of its dispatch. A stashed delegate must not
        # drive listeners whose plugins are already unwound, and deferring delegation must not
        # become a way to skip the checks that run after the listener returns.
        with self.assertRaises(WaterfallReentry) as caught:
            captured[0](("late",))
        self.assertIn("after its dispatch returned", str(caught.exception))
        self.assertEqual(downstream, [])


class WaterfallMonotonicTest(EventBusTestCase):
    def test_dropping_part_of_the_inner_result_raises_monotonic_violation(self) -> None:
        def dropper(value: tuple, forward: Callable[..., Any]) -> tuple:
            inner = forward(value)
            return tuple(item for item in inner if item != "required")

        self.bus.on(MONOTONIC, dropper, order=100, owner="pkg", label="dropper")

        with self.assertRaises(MonotonicViolation) as caught:
            self.bus.waterfall(
                MONOTONIC, ("seed",), lambda value: value + ("required",)
            )

        self.assertEqual(caught.exception.event, "test/monotonic")
        self.assertEqual(caught.exception.listener, "dropper")
        self.assertIn("dropped required", str(caught.exception))
        self.assertIsInstance(caught.exception, EventContractError)

    def test_growing_the_inner_result_is_allowed(self) -> None:
        self.bus.on(
            MONOTONIC,
            lambda value, forward: forward(value) + ("added",),
            label="grower",
        )
        result = self.bus.waterfall(
            MONOTONIC, ("seed",), lambda value: value + ("required",)
        )
        self.assertEqual(result, ("seed", "required", "added"))

    def test_returning_the_inner_result_unchanged_is_allowed(self) -> None:
        self.bus.on(MONOTONIC, lambda value, forward: forward(value), label="passthru")
        result = self.bus.waterfall(MONOTONIC, ("seed",), lambda value: value)
        self.assertEqual(result, ("seed",))

    def test_the_violating_listener_is_named_even_when_nested(self) -> None:
        self.bus.on(
            MONOTONIC,
            lambda value, forward: forward(value) + ("outer",),
            order=100,
            label="outer",
        )
        self.bus.on(
            MONOTONIC,
            lambda value, forward: tuple(
                item for item in forward(value) if item != "required"
            ),
            order=200,
            label="inner-dropper",
        )
        with self.assertRaises(MonotonicViolation) as caught:
            self.bus.waterfall(
                MONOTONIC, ("seed",), lambda value: value + ("required",)
            )
        self.assertEqual(caught.exception.listener, "inner-dropper")

    def test_short_circuiting_listener_is_recorded_not_flagged_as_shrinking(
        self,
    ) -> None:
        """No inner result exists, so the shrink check has nothing to compare against."""
        self.bus.on(
            MONOTONIC,
            lambda value, forward: ("replacement",),
            label="stopper",
        )
        result = self.bus.waterfall(
            MONOTONIC, ("seed",), lambda value: value + ("required",)
        )
        self.assertEqual(result, ("replacement",))
        self.assertEqual(self.bus.short_circuits, (("test/monotonic", "stopper"),))

    def test_shrink_check_is_not_consulted_for_a_plain_waterfall(self) -> None:
        self.bus.on(
            WATERFALL,
            lambda value, forward: tuple(
                item for item in forward(value) if item != "required"
            ),
            label="dropper",
        )
        result = self.bus.waterfall(
            WATERFALL, ("seed",), lambda value: value + ("required",)
        )
        self.assertEqual(result, ("seed",))


# -- payload and result type checks --------------------------------------------


class TypeContractTest(EventBusTestCase):
    def test_emit_rejects_a_mismatched_payload_before_any_listener_runs(self) -> None:
        calls: list[str] = []
        self.bus.on(EMIT, lambda payload: calls.append("ran"))
        with self.assertRaises(EventContractError) as caught:
            self.bus.emit(EMIT, 42)
        self.assertIn("payload must be str, got int", str(caught.exception))
        self.assertIn("test/emit", str(caught.exception))
        self.assertEqual(calls, [])
        self.assertEqual(self.reported, [])

    def test_parallel_serial_and_bail_reject_a_mismatched_payload(self) -> None:
        for mode, event, bad in (
            ("parallel", PARALLEL, 42),
            ("serial", SERIAL, 42),
            ("bail", BAIL, 42),
        ):
            with self.subTest(mode=mode):
                with self.assertRaises(EventContractError):
                    getattr(self.bus, mode)(event, bad)

    def test_waterfall_rejects_a_mismatched_payload(self) -> None:
        calls: list[str] = []
        with self.assertRaises(EventContractError) as caught:
            self.bus.waterfall(
                WATERFALL, "not a tuple", lambda value: calls.append("t") or value
            )
        self.assertIn("payload must be tuple, got str", str(caught.exception))
        self.assertEqual(calls, [])

    def test_serial_rejects_a_mismatched_result(self) -> None:
        self.bus.on(SERIAL, lambda payload: ["wrong"], order=100)
        with self.assertRaises(EventContractError) as caught:
            self.bus.serial(SERIAL, "q")
        self.assertIn("result must be str, got list", str(caught.exception))

    def test_waterfall_rejects_a_mismatched_listener_result(self) -> None:
        self.bus.on(WATERFALL, lambda value, forward: "not a tuple", label="bad")
        with self.assertRaises(EventContractError) as caught:
            self.bus.waterfall(WATERFALL, ("seed",), lambda value: value)
        self.assertIn("result must be tuple, got str", str(caught.exception))

    def test_waterfall_rejects_a_mismatched_terminal_result(self) -> None:
        with self.assertRaises(EventContractError):
            self.bus.waterfall(WATERFALL, ("seed",), lambda value: "not a tuple")

    def test_a_payload_subclass_is_accepted(self) -> None:
        class Special(str):
            pass

        seen: list[Any] = []
        self.bus.on(EMIT, lambda payload: seen.append(payload))
        self.bus.emit(EMIT, Special("sub"))
        self.assertEqual(seen, ["sub"])

    def test_declaring_a_mode_result_mismatch_fails_at_construction(self) -> None:
        with self.assertRaises(EventContractError):
            Event(name="test/no-result", mode="serial", payload=str)
        with self.assertRaises(EventContractError):
            Event(name="test/extra-result", mode="emit", payload=str, result=str)
        with self.assertRaises(EventContractError):
            Event(name="test/bad-monotonic", mode="emit", payload=str, monotonic=True)


# -- close ---------------------------------------------------------------------


class CloseTest(EventBusTestCase):
    def test_close_empties_the_listener_registry(self) -> None:
        self.bus.on(EMIT, lambda payload: None)
        self.bus.on(WATERFALL, lambda value, forward: forward(value))
        self.assertTrue(self.bus.has_listeners(EMIT))

        self.bus.close()

        self.assertTrue(self.bus.closed)
        self.assertEqual(self.bus.listeners(EMIT), ())
        self.assertEqual(self.bus.listeners(WATERFALL), ())
        self.assertFalse(self.bus.has_listeners(EMIT))
        self.assertFalse(self.bus.has_listeners(WATERFALL))

    def test_dispatch_after_close_never_reaches_a_late_listener(self) -> None:
        calls: list[str] = []
        self.bus.close()

        # A dying subprocess can still hand a plugin back a listener registration; the
        # closed guard, not an empty registry, is what makes the dispatch a no-op.
        self.bus.on(EMIT, lambda payload: calls.append("emit"))
        self.bus.on(PARALLEL, lambda payload: calls.append("parallel"))
        self.bus.on(SERIAL, lambda payload: calls.append("serial") or "v")
        self.bus.on(BAIL, lambda payload: calls.append("bail") or "v")

        self.assertIsNone(self.bus.emit(EMIT, "go"))
        self.assertIsNone(self.bus.parallel(PARALLEL, "go"))
        self.assertIsNone(self.bus.serial(SERIAL, "go"))
        self.assertIsNone(self.bus.bail(BAIL, "go"))

        self.assertEqual(calls, [])
        self.assertEqual(self.reported, [])

    def test_waterfall_after_close_returns_the_terminal_value(self) -> None:
        calls: list[str] = []
        self.bus.close()
        self.bus.on(WATERFALL, lambda value, forward: calls.append("listener") or value)

        result = self.bus.waterfall(
            WATERFALL, ("seed",), lambda value: value + ("terminal",)
        )

        self.assertEqual(result, ("seed", "terminal"))
        self.assertEqual(calls, [])
        self.assertEqual(self.bus.short_circuits, ())

    def test_close_is_idempotent(self) -> None:
        self.bus.on(EMIT, lambda payload: None)
        self.bus.close()
        self.bus.close()
        self.assertTrue(self.bus.closed)
        self.assertIsNone(self.bus.emit(EMIT, "go"))

    def test_close_after_parallel_shuts_the_pool_down(self) -> None:
        seen: list[str] = []
        self.bus.on(PARALLEL, lambda payload: seen.append("ran"))
        self.bus.parallel(PARALLEL, "go")
        self.assertEqual(seen, ["ran"])

        self.bus.close()

        self.bus.on(PARALLEL, lambda payload: seen.append("late"))
        self.bus.parallel(PARALLEL, "go")
        self.assertEqual(seen, ["ran"])

    def test_registration_on_a_closed_bus_leaves_the_registry_empty(self) -> None:
        self.bus.close()
        disposer = self.bus.on(EMIT, lambda payload: None)

        # The bus is closed during teardown, before owned work is killed, so a late registration
        # must not leave the registry holding a callback whose plugin is already unwound.
        self.assertFalse(self.bus.has_listeners(EMIT))
        self.assertEqual(self.bus.listeners(EMIT), ())

        self.assertIsNone(self.bus.emit(EMIT, "go"))
        # The returned disposer is still safe to call.
        self.assertIsNone(disposer())
        self.assertEqual(self.bus.listeners(EMIT), ())

    def test_closed_bus_still_validates_the_waterfall_payload(self) -> None:
        """The closed guard sits after the payload check for a waterfall only."""
        self.bus.close()
        with self.assertRaises(EventContractError):
            self.bus.waterfall(WATERFALL, "not a tuple", lambda value: value)
        self.assertIsNone(self.bus.emit(EMIT, 42))  # emit short-circuits first

    def test_disposer_survives_close(self) -> None:
        disposer = self.bus.on(EMIT, lambda payload: None)
        self.bus.close()
        disposer()  # must not raise even though the registry was cleared
        self.assertEqual(self.bus.listeners(EMIT), ())


if __name__ == "__main__":
    unittest.main()
