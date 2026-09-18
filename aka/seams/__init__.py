"""Capability seam definitions.

A **seam** is a swappable capability with three roles: a *Definition* declaring the interface,
a *Provider* implementing it, and a *Consumer* using it. A package may combine roles, but one
role alone is not a seam -- adding a capability means designing all three.

This package holds Definitions only: abstract base classes, the frozen request/response
dataclasses that cross the boundary, and the ``ServiceKey`` token. It imports no provider and
no orchestrator module, so a Definition can never drag an implementation in behind it.

``inject`` and ``provide`` in a plugin declaration are checked against :data:`SEAMS`, so a
misspelled service name is a load error rather than an ``AttributeError`` three hours into a
campaign.
"""

from __future__ import annotations

from typing import Mapping

from aka.core.keys import Cardinality, ServiceKey, TokenTable

SEAMS = TokenTable("seam")


def seam(
    name: str,
    definition: str,
    cardinality: Cardinality = "single",
    *,
    module: str = "",
    doc: str = "",
) -> ServiceKey:
    """Declare one seam. Duplicate names fail at import."""
    return SEAMS.add(
        ServiceKey(
            name=name,
            definition=definition,
            cardinality=cardinality,
            module=module,
            doc=doc,
        )
    )


def keys() -> Mapping[str, ServiceKey]:
    """The seam table, as the loader and ``declare()`` consume it."""
    return {name: SEAMS[name] for name in SEAMS.names()}


from . import driver as _driver  # noqa: E402,F401  (registers its seam on import)

DRIVER = _driver.DRIVER

__all__ = ["DRIVER", "SEAMS", "keys", "seam"]
