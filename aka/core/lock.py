"""The composition lock for a campaign workspace.

External subprocess plugins already have ``.atrex_plugins/lock.json``, written by
``plugin_runtime.registry``. That file's shape is frozen: ``check_lock`` compares it
byte-for-byte, so adding a field to it would make every existing workspace unresumable.

The in-process plugin tree therefore records itself in a **sibling** file,
``.atrex_plugins/composition.json``. Its resume rule is deliberately not blanket equality:
flags like ``--numerical-gate`` and ``--verify-repeats`` are legally changed when resuming a
campaign today, and a strict lock would break the CLI contract. Only rows that decide what a
resumed campaign *is* fail closed; everything else reports a diff and continues.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from plugin_runtime.registry import STATE_DIR, tree_digest

from .composition import ResolvedComposition, Row
from .errors import CompositionError

LOCK_NAME = "composition.json"
LOCK_VERSION = 1

#: Rows whose change alters what a resumed campaign is, rather than how it is tuned.
RESUME_SENSITIVE: frozenset[str] = frozenset(
    (
        "workspace-git",
        "journal",
        "memory",
        "campaign-driver",
        "episode-engine",
        "framework-baseline",
        "numerics",
        "legacy-campaign",
    )
)


def package_dir(module_path: str) -> Path | None:
    """Locate a plugin module's package directory without executing it."""
    try:
        spec = importlib.util.find_spec(module_path)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    locations = list(spec.submodule_search_locations or ())
    if locations:
        return Path(locations[0])
    return Path(spec.origin).parent if spec.origin else None


def _row_fingerprint(row: Row) -> str:
    payload = json.dumps(
        {
            "id": row.id,
            "name": row.name,
            "config": row.config,
            "required": row.required,
            "isolate": list(row.isolate),
            "group": row.group,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_digests(rows: Iterable[Row]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for row in rows:
        directory = package_dir(row.name)
        if directory is None or not directory.is_dir():
            continue
        key = str(directory)
        if key not in digests:
            digests[key] = tree_digest(directory)
    return digests


def snapshot(composition: ResolvedComposition) -> dict[str, Any]:
    rows = composition.enabled
    return {
        "lock_version": LOCK_VERSION,
        "profile": composition.profile,
        "entries": [
            {"id": row.id, "name": row.name, "fingerprint": _row_fingerprint(row)}
            for row in rows
        ],
        "modules": _module_digests(rows),
    }


def lock_path(workspace: Path) -> Path:
    return workspace / STATE_DIR / LOCK_NAME


def read(workspace: Path) -> Mapping[str, Any] | None:
    path = lock_path(workspace)
    if not path.exists():
        return None
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CompositionError(str(path), f"unreadable composition lock: {exc}") from exc
    if not isinstance(stored, dict):
        raise CompositionError(str(path), "composition lock must be a JSON object")
    entries = stored.get("entries", [])
    if not isinstance(entries, list) or any(
        not isinstance(entry, dict) or not isinstance(entry.get("id"), str)
        for entry in entries
    ):
        raise CompositionError(
            str(path), "composition lock entries must each carry a string id"
        )
    if not isinstance(stored.get("modules", {}), dict):
        raise CompositionError(str(path), "composition lock modules must be an object")
    return stored


def write(workspace: Path, current: Mapping[str, Any]) -> Path:
    path = lock_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def differences(stored: Mapping[str, Any], current: Mapping[str, Any]) -> tuple[str, ...]:
    stored_rows = {
        entry["id"]: entry for entry in stored.get("entries", []) if isinstance(entry, dict)
    }
    current_rows = {entry["id"]: entry for entry in current["entries"]}
    diffs: list[str] = []
    for entry_id in sorted(set(stored_rows) | set(current_rows)):
        was, now = stored_rows.get(entry_id), current_rows.get(entry_id)
        if was is None:
            diffs.append(f"{entry_id}: added")
        elif now is None:
            diffs.append(f"{entry_id}: removed")
        elif was.get("name") != now.get("name"):
            diffs.append(f"{entry_id}: module {was.get('name')} -> {now.get('name')}")
        elif was.get("fingerprint") != now.get("fingerprint"):
            diffs.append(f"{entry_id}: configuration changed")
    # Compare the union, not the intersection: a plugin package that moved to a different install
    # path, or vanished, is exactly the case where a resumed campaign would silently run different
    # code.
    stored_modules = stored.get("modules", {})
    current_modules = current["modules"]
    for directory in sorted(set(stored_modules) | set(current_modules)):
        was, now = stored_modules.get(directory), current_modules.get(directory)
        if was == now:
            continue
        if was is None:
            diffs.append(f"{directory}: plugin code added")
        elif now is None:
            diffs.append(f"{directory}: plugin code no longer present")
        else:
            diffs.append(f"{directory}: plugin code changed")
    return tuple(diffs)


def reconcile(
    workspace: Path,
    composition: ResolvedComposition,
    *,
    resume_sensitive: frozenset[str] = RESUME_SENSITIVE,
) -> tuple[str, ...]:
    """Adopt, verify, or reject the workspace's in-process composition.

    Returns the diffs worth reporting. Raises when a resume-sensitive row changed.
    """
    current = snapshot(composition)
    stored = read(workspace)
    if stored is None:
        write(workspace, current)
        return ()
    diffs = differences(stored, current)
    if not diffs:
        return ()
    # A module-code diff is reported by directory, so map it back to the rows that load from it:
    # changed code under a resume-sensitive row is exactly as decisive as a changed config there.
    owners: dict[str, set[str]] = {}
    for entry in current["entries"]:
        directory = package_dir(entry["name"])
        if directory is not None:
            owners.setdefault(str(directory), set()).add(entry["id"])
    sensitive = tuple(
        diff
        for diff in diffs
        if (subject := diff.split(":", 1)[0]) in resume_sensitive
        or owners.get(subject, set()) & resume_sensitive
    )
    if sensitive:
        raise CompositionError(
            str(lock_path(workspace)),
            "this workspace was created by a different plugin composition: "
            + "; ".join(sensitive)
            + ". Restore the original composition or start a new workspace.",
        )
    return diffs


__all__ = [
    "LOCK_NAME",
    "LOCK_VERSION",
    "RESUME_SENSITIVE",
    "differences",
    "lock_path",
    "package_dir",
    "read",
    "reconcile",
    "snapshot",
    "write",
]
