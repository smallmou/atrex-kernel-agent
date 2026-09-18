"""Atrex Kernel Agent: an everything-is-a-plugin harness for GPU kernel optimization.

``aka.core`` is a small container -- services, fibers, effects, typed events, JSON composition
-- with no domain knowledge. ``aka.seams`` holds capability Definitions. ``aka.plugins`` holds
the Providers and Consumers. A campaign is a plugin tree composed from ``aka/profiles``.

The entry points are ``aka.boot.compose`` and ``aka.boot.boot``; import them from
``aka.boot`` explicitly. This package deliberately re-exports nothing: ``boot`` names both
that submodule and the function inside it, so a package-level alias would resolve to one or
the other depending on import order.

Repository-scoped external plugins keep their own contract under ``plugins/``; see
``docs/plugins.md``.
"""

from __future__ import annotations
