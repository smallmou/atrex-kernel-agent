"""Two-level scoping for named contributions.

Tools, prompt sections, skills, session-environment variables, and gates all need the same
merge rule: a global layer plus a scope chain, where the nearest layer's entry wins a
duplicate name outright and an inherited set can be restricted without touching its owner.
That rule is implemented once here and reused, rather than growing five times.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .effects import Disposer
from .errors import DuplicateProvide


@dataclass(frozen=True)
class ScopedEntry:
    """One named contribution filed into a scope layer."""

    name: str
    value: Any
    owner: str = ""
    order: int = 500


@dataclass(frozen=True)
class _Filter:
    """One owner's restriction. ``allow=None`` means "this owner sets no allowlist".

    An *empty* allowlist is not the same as no allowlist: a caller that computed an empty
    allowed set means "admit nothing", so it must not silently widen to "admit everything".
    """

    allow: frozenset[str] | None
    deny: frozenset[str]

    def admits(self, name: str) -> bool:
        if self.allow is not None and name not in self.allow:
            return False
        return name not in self.deny


class Scope:
    """One layer of the contribution chain."""

    def __init__(self, name: str, parent: "Scope | None" = None):
        self.name = name
        self.parent = parent
        self._entries: dict[str, dict[str, ScopedEntry]] = {}
        # Several owners may restrict the same kind on the same scope; every restriction stays
        # in force until its own disposer runs, so they compose rather than overwrite.
        self._filters: dict[str, list[_Filter]] = {}

    def child(self, name: str) -> "Scope":
        return Scope(name, parent=self)

    def chain(self) -> tuple["Scope", ...]:
        scopes: list[Scope] = []
        cursor: Scope | None = self
        while cursor is not None:
            scopes.append(cursor)
            cursor = cursor.parent
        return tuple(reversed(scopes))

    # -- contributions ---------------------------------------------------

    def contribute(self, kind: str, entry: ScopedEntry) -> Disposer:
        layer = self._entries.setdefault(kind, {})
        existing = layer.get(entry.name)
        if existing is not None:
            raise DuplicateProvide(
                f"{kind}.{entry.name}", existing.owner or self.name, entry.owner
            )
        layer[entry.name] = entry

        def disposer() -> None:
            if layer.get(entry.name) is entry:
                del layer[entry.name]

        return disposer

    def restrict(
        self,
        kind: str,
        *,
        allow: Sequence[str] | None = None,
        deny: Sequence[str] = (),
    ) -> Disposer:
        """Filter entries inherited by this scope. Local contributions are unaffected.

        Restrictions compose: an entry must be admitted by every live restriction on this scope,
        so two owners narrowing the same kind intersect instead of cancelling. The returned
        disposer removes only its own restriction, by identity, so unwinding out of order cannot
        drop a live one or resurrect a dead one.

        ``allow=None`` adds no allowlist; ``allow=()`` admits nothing.
        """
        restriction = _Filter(
            allow=None if allow is None else frozenset(allow), deny=frozenset(deny)
        )
        live = self._filters.setdefault(kind, [])
        live.append(restriction)

        def disposer() -> None:
            for index, current in enumerate(live):
                if current is restriction:
                    del live[index]
                    return

        return disposer

    # -- reads -----------------------------------------------------------

    def merged(self, kind: str) -> Mapping[str, ScopedEntry]:
        accumulated: dict[str, ScopedEntry] = {}
        for scope in self.chain():
            restrictions = scope._filters.get(kind, ())
            if restrictions:
                accumulated = {
                    name: entry
                    for name, entry in accumulated.items()
                    if all(restriction.admits(name) for restriction in restrictions)
                }
            accumulated.update(scope._entries.get(kind, {}))
        return accumulated

    def resolve(self, kind: str) -> tuple[ScopedEntry, ...]:
        entries = self.merged(kind).values()
        return tuple(sorted(entries, key=lambda entry: (entry.order, entry.name)))

    def local(self, kind: str) -> tuple[ScopedEntry, ...]:
        return tuple(self._entries.get(kind, {}).values())

    def kinds(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))


__all__ = ["Scope", "ScopedEntry"]
