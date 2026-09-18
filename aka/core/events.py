"""The typed event bus.

Events are the extension points. A listener observes, wraps, or decides depending on the
event's declared dispatch mode, and the mode is enforced at both registration and dispatch
so a producer cannot quietly change an event's contract.

``BaseException`` is never caught here. ``EnvironmentUnavailable`` deliberately subclasses
it to escape ``except Exception``, and the bus must not undo that.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from .effects import Disposer
from .errors import EventContractError, MonotonicViolation, WaterfallReentry
from .keys import Event

_MISSING = object()

#: How many recent non-delegating waterfall listeners to remember.
SHORT_CIRCUIT_HISTORY = 256

ErrorSink = Callable[[str, BaseException], None]


@dataclass(frozen=True)
class Listener:
    event: str
    callback: Callable[..., Any]
    order: int
    owner: str
    sequence: int
    label: str = field(default="")

    def describe(self) -> str:
        return self.label or f"{self.owner or '<anonymous>'}#{self.sequence}"


class EventBus:
    """Listener registry plus the five dispatch modes."""

    def __init__(self, *, error_sink: ErrorSink | None = None, workers: int = 4):
        self._listeners: dict[str, list[Listener]] = {}
        self._sequence = 0
        self._closed = False
        self._workers = max(1, workers)
        self._pool: ThreadPoolExecutor | None = None
        self._error_sink = error_sink
        # Bounded: a campaign dispatches for hours, and this is a diagnostic tail, not a ledger.
        self._short_circuits: deque[tuple[str, str]] = deque(maxlen=SHORT_CIRCUIT_HISTORY)

    # -- registration ----------------------------------------------------

    def on(
        self,
        event: Event,
        callback: Callable[..., Any],
        *,
        order: int = 500,
        owner: str = "",
        label: str = "",
    ) -> Disposer:
        if not callable(callback):
            raise EventContractError(event.name, "listener is not callable")
        if self._closed:
            # The bus is closed during teardown, before owned work is killed. Registering here
            # would leave the registry holding a callback whose plugin is already unwound.
            return lambda: None
        self._sequence += 1
        listener = Listener(
            event=event.name,
            callback=callback,
            order=order,
            owner=owner,
            sequence=self._sequence,
            label=label,
        )
        self._listeners.setdefault(event.name, []).append(listener)

        def disposer() -> None:
            bucket = self._listeners.get(event.name)
            if bucket and listener in bucket:
                bucket.remove(listener)

        return disposer

    def listeners(self, event: Event) -> tuple[Listener, ...]:
        bucket = self._listeners.get(event.name, ())
        return tuple(sorted(bucket, key=lambda item: (item.order, item.sequence)))

    def has_listeners(self, event: Event) -> bool:
        return bool(self._listeners.get(event.name))

    def _is_live(self, listener: Listener) -> bool:
        """Is this listener still registered? Dispatch iterates a snapshot."""
        bucket = self._listeners.get(listener.event)
        return bool(bucket) and any(current is listener for current in bucket)

    @property
    def short_circuits(self) -> tuple[tuple[str, str], ...]:
        """``(event, listener)`` pairs where a waterfall listener did not delegate."""
        return tuple(self._short_circuits)

    # -- dispatch --------------------------------------------------------

    def emit(self, event: Event, payload: Any) -> None:
        self._require(event, "emit")
        if self._closed:
            return
        self._check_payload(event, payload)
        for listener in self.listeners(event):
            if not self._is_live(listener):
                # A listener disposed earlier in this same dispatch belongs to a plugin that is
                # already unwound; calling it would reach into torn-down state and the failure
                # would be swallowed into the error sink as if the observer were merely buggy.
                continue
            try:
                listener.callback(payload)
            except Exception as exc:  # noqa: BLE001 - one bad observer must not starve the rest
                self._report(event, listener, exc)

    def parallel(self, event: Event, payload: Any) -> None:
        self._require(event, "parallel")
        if self._closed:
            return
        self._check_payload(event, payload)
        listeners = self.listeners(event)
        if not listeners:
            return
        pool = self._ensure_pool()
        if pool is None:
            return
        try:
            futures = [pool.submit(listener.callback, payload) for listener in listeners]
        except RuntimeError:
            # close() shut the pool down between the guard and the submit: the tree is coming
            # apart, so an observer notification is no longer owed.
            return
        for listener, future in zip(listeners, futures):
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - collect every failure, run every listener
                self._report(event, listener, exc)

    def serial(self, event: Event, payload: Any) -> Any | None:
        self._require(event, "serial")
        return self._first(event, payload, lambda value: value is not None)

    def bail(self, event: Event, payload: Any) -> Any | None:
        self._require(event, "bail")
        return self._first(event, payload, bool)

    def waterfall(self, event: Event, payload: Any, terminal: Callable[[Any], Any]) -> Any:
        self._require(event, "waterfall")
        self._check_payload(event, payload)
        if self._closed:
            return terminal(payload)
        chain = self.listeners(event)

        def build(index: int) -> Callable[[Any], Any]:
            if index == len(chain):
                return terminal
            listener = chain[index]
            inner = build(index + 1)

            def step(value: Any) -> Any:
                if not self._is_live(listener):
                    # Disposed earlier in this dispatch: pass the value through untouched rather
                    # than calling into a plugin that is already unwound.
                    return inner(value)
                calls = 0
                captured: Any = _MISSING
                spent = False

                def delegate(forward: Any = _MISSING) -> Any:
                    nonlocal calls, captured
                    if spent:
                        raise WaterfallReentry(
                            event.name,
                            f"{listener.describe()} (called next() after its dispatch returned)",
                        )
                    calls += 1
                    if calls > 1:
                        raise WaterfallReentry(event.name, listener.describe())
                    forwarded = value if forward is _MISSING else forward
                    if event.monotonic and event.shrink_check is not None:
                        # A listener may add to what it forwards, never remove from it: shrinking
                        # on the way down would drop a decision the downstream listeners were
                        # entitled to see, which the return-path check alone cannot catch.
                        detail = event.shrink_check(forwarded, value)
                        if detail:
                            raise MonotonicViolation(
                                event.name,
                                f"{listener.describe()} (forwarded to next())",
                                detail,
                            )
                    captured = inner(forwarded)
                    return captured

                try:
                    result = listener.callback(value, delegate)
                finally:
                    # The chain is valid only for the duration of this dispatch. Letting a stashed
                    # delegate run later would drive listeners whose plugins are already unwound
                    # and skip every check below.
                    spent = True
                if calls == 0:
                    self._short_circuits.append((event.name, listener.describe()))
                elif captured is not _MISSING and event.monotonic and event.shrink_check:
                    detail = event.shrink_check(result, captured)
                    if detail:
                        raise MonotonicViolation(
                            event.name, listener.describe(), detail
                        )
                self._check_result(event, result, allow_none=False)
                return result

            return step

        result = build(0)(payload)
        self._check_result(event, result, allow_none=False)
        return result

    def _first(
        self, event: Event, payload: Any, accept: Callable[[Any], bool]
    ) -> Any | None:
        if self._closed:
            return None
        self._check_payload(event, payload)
        for listener in self.listeners(event):
            if not self._is_live(listener):
                continue
            result = listener.callback(payload)
            if accept(result):
                self._check_result(event, result)
                return result
        return None

    # -- lifecycle -------------------------------------------------------

    def close(self) -> None:
        """Stop accepting dispatches and drop every listener.

        Teardown closes the registry before killing owned work, so a late notification
        from a dying subprocess is a silent no-op rather than a call into a disposed
        plugin.
        """
        self._closed = True
        self._listeners.clear()
        self._short_circuits.clear()
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    @property
    def closed(self) -> bool:
        return self._closed

    # -- internals -------------------------------------------------------

    def _ensure_pool(self) -> ThreadPoolExecutor | None:
        if self._closed:
            # Never create a pool for a closed bus: it would outlive close() unshut.
            return None
        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self._workers, thread_name_prefix="aka-event"
            )
        return self._pool

    @staticmethod
    def _require(event: Event, mode: str) -> None:
        if event.mode != mode:
            raise EventContractError(
                event.name,
                f"declared {event.mode} dispatch cannot be dispatched as {mode}",
            )

    @staticmethod
    def _check_payload(event: Event, payload: Any) -> None:
        if not isinstance(payload, event.payload):
            raise EventContractError(
                event.name,
                f"payload must be {event.payload.__name__}, got "
                f"{type(payload).__name__}",
            )

    @staticmethod
    def _check_result(event: Event, result: Any, *, allow_none: bool = True) -> None:
        if event.result is None:
            return
        if result is None:
            if allow_none:
                # serial and bail treat a falsy return as an abstention.
                return
            raise EventContractError(
                event.name,
                f"a waterfall listener returned None; it must return {event.result.__name__}, "
                "either the value it got from next() or a replacement",
            )
        if not isinstance(result, event.result):
            raise EventContractError(
                event.name,
                f"result must be {event.result.__name__}, got {type(result).__name__}",
            )

    def _report(self, event: Event, listener: Listener, exc: BaseException) -> None:
        if self._error_sink is not None:
            self._error_sink(f"{event.name} listener {listener.describe()}", exc)


__all__ = ["ErrorSink", "EventBus", "Listener"]
