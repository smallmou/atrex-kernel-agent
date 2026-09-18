"""The event catalog and the prompt section order allocation.

Every event AKA dispatches is declared here or in :mod:`aka.core.internal`, exactly once, with
its dispatch mode. The mode is part of the event's public contract: the bus refuses to
dispatch a ``waterfall`` as an ``emit``, so a producer cannot quietly change it.

Prompt section order is a **centrally owned allocation**, not a number each contributor picks.
``MODE_POLICY`` sits first because today's renderer prepends the mode policy instead of
substituting it (``orchestrator/session_io.py``); giving it the lowest order reproduces that
placement structurally rather than as a special case in one function.
"""

from __future__ import annotations

from typing import Mapping

from aka.core.internal import (
    CORE_EVENTS,
    INTERNAL_CONFIG,
    INTERNAL_ERROR,
    INTERNAL_FIBER,
)
from aka.core.keys import Event, Mode, ShrinkCheck, TokenTable

EVENTS = TokenTable("event")

for _token in CORE_EVENTS.values():
    EVENTS.add(_token)


def event(
    name: str,
    mode: Mode,
    payload: type,
    *,
    result: type | None = None,
    monotonic: bool = False,
    shrink_check: ShrinkCheck | None = None,
    doc: str = "",
) -> Event:
    """Declare one event. Duplicate names fail at import."""
    return EVENTS.add(
        Event(
            name=name,
            mode=mode,
            payload=payload,
            result=result,
            monotonic=monotonic,
            shrink_check=shrink_check,
            doc=doc,
        )
    )


def keys() -> Mapping[str, Event]:
    return {name: EVENTS[name] for name in EVENTS.names()}


#: Prompt section placement. A contributor asks for its name, never for a number.
_SECTION_ORDER: Mapping[str, int] = {
    "MODE_POLICY": 100,
    "HARDWARE": 200,
    "EVALUATOR": 300,
    "SANDBOX": 400,
    "AGENT_RUNTIME": 500,
    "PLAN_GENERATOR": 600,
    "PLUGINS": 700,
    "KNOWLEDGE": 750,
    "NUMERICS": 800,
    "PROFILING": 850,
    "JOURNAL": 900,
}


def section_order(name: str) -> int:
    """The allocated order for one prompt section name."""
    try:
        return _SECTION_ORDER[name]
    except KeyError:
        raise KeyError(
            f'prompt section "{name}" has no allocated order; add it to '
            f"aka/events.py so placement stays reviewable in one place"
        ) from None


def section_names() -> tuple[str, ...]:
    return tuple(
        name for name, _ in sorted(_SECTION_ORDER.items(), key=lambda item: item[1])
    )


__all__ = [
    "EVENTS",
    "INTERNAL_CONFIG",
    "INTERNAL_ERROR",
    "INTERNAL_FIBER",
    "event",
    "keys",
    "section_names",
    "section_order",
]
