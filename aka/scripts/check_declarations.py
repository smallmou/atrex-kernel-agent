#!/usr/bin/env python3
"""Check every in-process plugin's declaration without applying it.

Run from the repository root::

    python3 aka/scripts/check_declarations.py

This is the mechanical half of the plugin contract. It imports each plugin package, runs
``declare()`` -- which enforces the declaration rules -- and then checks the package-level
obligations that ``declare()`` cannot see:

* an ``invariant.py`` companion exists and exports ``PACKAGE_NAME`` and ``install``;
* an installer that never calls ``fail`` is explained: its first comment line must start
  ``No runtime invariant:`` and say, for that package specifically, why nothing is checkable.
  Publication is exhaustive; assertions are not. An unexplained empty installer is the failure
  mode this rule exists to prevent, because it looks like coverage and is not;
* at least one colocated ``test_*.py``.

Nothing is applied, so the check has no side effects and needs no GPU, workspace, or network.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aka.core.declaration import declare  # noqa: E402
from aka.core.errors import CoreError  # noqa: E402
from aka.seams import keys as seam_keys  # noqa: E402

PLUGIN_ROOT = REPO_ROOT / "aka" / "plugins"
NO_INVARIANT_PREFIX = "No runtime invariant:"


def plugin_packages(root: Path = PLUGIN_ROOT) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )


def _installer_explains_itself(source: Path) -> tuple[bool, str]:
    """Is this an installer that asserts something, or an explained empty one?"""
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, SyntaxError) as exc:
        return False, f"cannot parse {source.name}: {exc}"
    install = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "install"
        ),
        None,
    )
    if install is None:
        return False, f"{source.name} exports no install() function"
    uses_reporter = any(
        isinstance(node, ast.Name) and node.id == "fail"
        for node in ast.walk(install)
    )
    if uses_reporter:
        return True, ""
    text = source.read_text(encoding="utf-8").splitlines()
    # Comments are not AST nodes, so read the source region between the signature and the first
    # statement -- that is where an explained empty installer puts its reason.
    first_statement = install.body[0].lineno if install.body else install.lineno + 1
    comments = [
        line.strip().lstrip("#").strip()
        for line in text[install.lineno : first_statement]
        if line.strip().startswith("#")
    ]
    if comments and comments[0].startswith(NO_INVARIANT_PREFIX):
        if len(comments[0]) > len(NO_INVARIANT_PREFIX) + 20 or len(comments) > 1:
            return True, ""
        return False, (
            f"{source.name} declares no runtime invariant but does not explain why"
        )
    return False, (
        f"{source.name} has an install() that never calls fail(); either assert something this "
        f'package owns, or start its first comment line with "{NO_INVARIANT_PREFIX}" and explain '
        "what is not checkable here and why"
    )


def check(package: Path, seams: dict) -> list[str]:
    problems: list[str] = []
    module_path = f"aka.plugins.{package.name}"
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # noqa: BLE001 - an unimportable plugin is a reported problem
        return [f"{module_path}: cannot import: {type(exc).__name__}: {exc}"]

    try:
        declaration = declare(module, known_seams=seams)
    except CoreError as exc:
        problems.append(f"{module_path}: {exc}")
    else:
        if declaration.name != package.name.replace("_", "-"):
            problems.append(
                f'{module_path}: name "{declaration.name}" does not match its directory'
            )

    invariant = package / "invariant.py"
    if not invariant.is_file():
        problems.append(
            f"{module_path}: missing invariant.py; every plugin publishes one, even when it only "
            f'explains why nothing is checkable ("{NO_INVARIANT_PREFIX} ...")'
        )
    else:
        try:
            companion = importlib.import_module(f"{module_path}.invariant")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{module_path}.invariant: cannot import: {exc}")
        else:
            if not isinstance(getattr(companion, "PACKAGE_NAME", None), str):
                problems.append(f"{module_path}.invariant: exports no PACKAGE_NAME")
            if not callable(getattr(companion, "install", None)):
                problems.append(f"{module_path}.invariant: exports no install()")
        ok, reason = _installer_explains_itself(invariant)
        if not ok:
            problems.append(f"{module_path}: {reason}")

    if not sorted(package.glob("test_*.py")):
        problems.append(
            f"{module_path}: has no colocated test_*.py (tests/ is gitignored, so tests live "
            "next to the code)"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    seams = dict(seam_keys())
    packages = plugin_packages()
    if not packages:
        print("no in-process plugin packages found under aka/plugins/", file=sys.stderr)
        return 1
    problems: list[str] = []
    for package in packages:
        problems.extend(check(package, seams))
    for problem in problems:
        print(f"[check-declarations] {problem}", file=sys.stderr)
    print(
        f"[check-declarations] checked {len(packages)} plugin package(s): "
        f"{len(problems)} problem(s)"
    )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
