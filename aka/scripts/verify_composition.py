#!/usr/bin/env python3
"""Verify every shipped composition resolves into a loadable tree.

Run from the repository root::

    python3 aka/scripts/verify_composition.py

For each profile under ``aka/profiles/`` this resolves the layered JSON, imports every row's
module, runs ``declare()``, and then checks the properties that only the whole composition can
answer:

* every row id is unique and every patch target resolved (``resolve()`` itself enforces this);
* every required injection in the tree is provided by some enabled row, so no row can wait
  forever for a service nobody publishes;
* the required set is exactly the rows marked ``required``;
* every declared ``provide`` names a registered seam, and no two enabled rows provide the same
  single-cardinality seam outside an isolation realm.

It deliberately does not *boot* a shipped profile: booting the default profile constructs a real
campaign, which needs an operator directory and a workspace. The real boot path is covered by
``aka/test_composition_smoke.py``, which boots a test profile with stub rows.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aka.boot import PROFILES_DIR, compose  # noqa: E402
from aka.core.declaration import declare  # noqa: E402
from aka.core.errors import CoreError  # noqa: E402
from aka.core.loader import import_plugin  # noqa: E402
from aka.seams import keys as seam_keys  # noqa: E402


def profiles(profiles_dir: Path = PROFILES_DIR) -> list[str]:
    return sorted(path.stem for path in profiles_dir.glob("*.json"))


def verify(profile: str, *, profiles_dir: Path = PROFILES_DIR) -> list[str]:
    seams = dict(seam_keys())
    problems: list[str] = []
    try:
        composition = compose(profile, profiles_dir=profiles_dir)
    except CoreError as exc:
        return [f"{profile}: {exc}"]

    provided: dict[str, list[str]] = defaultdict(list)
    required_injections: dict[str, list[str]] = defaultdict(list)
    for row in composition.enabled:
        try:
            module = import_plugin(row.name)
            declaration = declare(module, known_seams=seams).with_extra_inject(row.inject)
        except CoreError as exc:
            problems.append(f"{profile}/{row.id}: {exc}")
            continue
        for service in declaration.provide:
            provided[service].append(row.id)
        for service in declaration.inject:
            required_injections[service].append(row.id)

    for service, consumers in sorted(required_injections.items()):
        if service not in provided:
            problems.append(
                f"{profile}: no enabled row provides \"{service}\", required by "
                f"{', '.join(sorted(consumers))}"
            )

    for service, owners in sorted(provided.items()):
        if len(owners) > 1 and seams[service].cardinality == "single":
            isolated = {
                row.id: row.isolate for row in composition.enabled if row.id in owners
            }
            if not all(service in names for names in isolated.values()):
                problems.append(
                    f'{profile}: single-implementation seam "{service}" is provided by '
                    f"{', '.join(sorted(owners))} without an isolation realm"
                )

    declared_required = frozenset(row.id for row in composition.enabled if row.required)
    if composition.required != declared_required:
        problems.append(
            f"{profile}: required set {sorted(composition.required)} does not match the rows "
            f"marked required {sorted(declared_required)}"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    names = list(argv or []) or profiles()
    if not names:
        print("no profiles found under aka/profiles/", file=sys.stderr)
        return 1
    problems: list[str] = []
    for profile in names:
        problems.extend(verify(profile))
    for problem in problems:
        print(f"[verify-composition] {problem}", file=sys.stderr)
    print(
        f"[verify-composition] checked {len(names)} profile(s): {len(problems)} problem(s)"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
