"""A consumer that waits for ``ctx.driver`` and records what it observed.

Exercises the consumer half of a seam: a required injection, a scoped contribution, an event
subscription, and an invariant companion -- all of which must unwind when the row unloads.
"""

from __future__ import annotations

from typing import Any, Mapping

from aka.core.context import Context
from aka.core.internal import INTERNAL_FIBER, FiberTransition
from aka.core.invariants import InvariantFailure

name = "driver-observer"
inject: tuple[str, ...] = ("driver",)

PACKAGE_NAME = "aka.testing.driver-observer"

#: Grows as rows load, so a test can assert what the observer saw across a reload.
OBSERVED: list[str] = []
TRANSITIONS: list[str] = []


def install(ctx: Context, fail: InvariantFailure) -> None:
    def check(transition: FiberTransition) -> None:
        if transition.current == "active" and transition.previous == "active":
            fail(f"row {transition.entry_id} re-entered active without unloading")

    ctx.on(INTERNAL_FIBER, check)


def apply(ctx: Context, config: Mapping[str, Any]) -> None:
    ctx.invariants.register(PACKAGE_NAME, install)
    # Capture rather than re-read: a disposer runs while the tree is coming apart, and the
    # provider it observed may already be gone.
    observed = ctx.driver.campaign_name
    OBSERVED.append(observed)
    ctx.contribute("observers", name, observed, order=100)
    ctx.on(INTERNAL_FIBER, lambda transition: TRANSITIONS.append(transition.entry_id))

    def forget() -> None:
        if observed in OBSERVED:
            OBSERVED.remove(observed)

    ctx.effect(forget, label="forget observation")


__all__ = ["OBSERVED", "PACKAGE_NAME", "TRANSITIONS", "apply", "inject", "install", "name"]
