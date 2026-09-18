"""Turning a resolved composition into a running plugin tree.

Boot is best-effort with a required set. A required row that cannot become active disposes the
whole tree and fails loud; any other row that fails or stays waiting is a warning, and its
service simply reads as absent, which is exactly how a plugin that was never listed behaves.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .composition import ResolvedComposition, Row
from .declaration import declare
from .errors import BootFailure, CompositionError, CoreError
from .fiber import DEFAULT_SETTLE_ROUNDS, FiberState, Root
from .keys import ServiceKey


@dataclass(frozen=True)
class BootReport:
    """What the tree looks like after settling."""

    root: Root
    composition: ResolvedComposition
    active: tuple[str, ...] = ()
    failed: tuple[tuple[str, str], ...] = ()
    pending: tuple[tuple[str, tuple[str, ...]], ...] = ()
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def service(self, name: str) -> Any | None:
        """Read one service out of the booted tree.

        The host reads by name rather than through a plugin context: it is the caller of the
        tree, not a participant in it.
        """
        impl = self.root.realm.resolve(name)
        return None if impl is None else impl.value

    def dispose(self) -> None:
        self.root.dispose()


def import_plugin(module_path: str) -> Any:
    try:
        return importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - an unresolvable row is a reported failure
        raise CompositionError(module_path, f"cannot import: {exc}") from exc


def load(
    composition: ResolvedComposition,
    *,
    seams: Mapping[str, ServiceKey] | None = None,
    required: Sequence[str] | None = None,
    settle_rounds: int = DEFAULT_SETTLE_ROUNDS,
    workers: int = 4,
    stderr: Any = None,
) -> BootReport:
    root = Root(
        seams=seams,
        variables=composition.variables,
        settle_rounds=settle_rounds,
        workers=workers,
        stderr=stderr,
    )
    required_ids = (
        frozenset(required) if required is not None else composition.required
    )
    unresolved: list[tuple[str, str]] = []
    warnings: list[str] = []

    for row in composition.enabled:
        try:
            _mount(root, row, seams)
        except CoreError as exc:
            unresolved.append((row.id, f"{type(exc).__name__}: {exc}"))

    try:
        root.settle()
    except BaseException:
        root.dispose()
        raise

    failed = tuple(sorted(root.failed + tuple(unresolved)))
    pending = root.pending
    active = root.active

    # A required row must actually be ACTIVE. Intersecting only with failed and pending would
    # silently accept a required id that names a disabled row, a removed row, or a typo.
    reasons = dict(failed)
    reasons.update(
        {
            entry_id: f"waiting for {', '.join(missing) or 'an unavailable service'}"
            for entry_id, missing in pending
        }
    )
    blocking = tuple(
        (
            entry_id,
            reasons.get(
                entry_id,
                "no enabled row with this id is active; check the composition for a disabled, "
                "removed, or misspelled row",
            ),
        )
        for entry_id in sorted(required_ids)
        if entry_id not in active
    )
    if blocking:
        root.dispose()
        raise BootFailure(tuple(sorted(blocking)))

    for entry_id, reason in failed:
        warnings.append(f'optional row "{entry_id}" failed: {reason}')
    for entry_id, missing in pending:
        warnings.append(
            f'optional row "{entry_id}" is waiting for '
            f"{', '.join(missing) or 'an unavailable service'}"
        )
    for message in warnings:
        root.warn(message)

    return BootReport(
        root=root,
        composition=composition,
        active=active,
        failed=failed,
        pending=pending,
        warnings=tuple(warnings),
    )


def _mount(root: Root, row: Row, seams: Mapping[str, ServiceKey] | None) -> None:
    module = import_plugin(row.name)
    if seams is not None:
        # Row-level inject and isolate bypass declare()'s seam check, so validate them here:
        # otherwise a typo produces a row that waits forever, or a realm that isolates nothing
        # and leaks a "private" service into the shared one.
        unknown = sorted((set(row.inject) | set(row.isolate)) - set(seams))
        if unknown:
            raise CompositionError(
                f"entry:{row.id}",
                f"references unregistered seams: {', '.join(unknown)}",
            )
    declaration = declare(module, known_seams=seams).with_extra_inject(row.inject)
    root.mount(
        declaration,
        row.config,
        entry_id=row.id,
        required=row.required,
        isolate=row.isolate,
    )


def describe(report: BootReport) -> str:
    """A short human summary of a booted tree."""
    lines = [
        f"profile {report.composition.profile}: "
        f"{len(report.active)} active, {len(report.failed)} failed, "
        f"{len(report.pending)} waiting"
    ]
    for entry_id in report.active:
        lines.append(f"  active   {entry_id}")
    for entry_id, reason in report.failed:
        lines.append(f"  failed   {entry_id}: {reason}")
    for entry_id, missing in report.pending:
        lines.append(f"  waiting  {entry_id}: {', '.join(missing)}")
    return "\n".join(lines)


__all__ = ["BootReport", "FiberState", "describe", "import_plugin", "load"]
