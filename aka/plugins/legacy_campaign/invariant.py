"""Runtime invariant companion for the legacy campaign driver."""

from __future__ import annotations

from aka.core.context import Context
from aka.core.invariants import InvariantFailure

PACKAGE_NAME = "aka.plugins.legacy-campaign"


def install(ctx: Context, fail: InvariantFailure) -> None:
    # No runtime invariant: this row constructs the legacy campaign object and publishes it as
    # ctx.driver, and does nothing else. It produces no event stream, and the durable state a
    # campaign writes -- Git history, memory/vN.json, .atrex_long_horizon/ -- is owned by the
    # orchestrator modules still being migrated, so any assertion here would be checking
    # another package's contract. The rows that take that state over in stage three carry the
    # checks with them.
    return None


__all__ = ["PACKAGE_NAME", "install"]
