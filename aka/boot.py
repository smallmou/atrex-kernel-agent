"""Booting AKA from a composition.

Two calls are the whole entry point: :func:`compose` resolves the layered JSON composition,
:func:`boot` mounts it and settles the tree. Everything else is a plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from aka.core.composition import ResolvedComposition, resolve
from aka.core.loader import BootReport, load
from aka.seams import keys as seam_keys

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
DEFAULT_PROFILE = "default"


def compose(
    profile: str = DEFAULT_PROFILE,
    *,
    profiles_dir: Path | None = None,
    patch_files: Sequence[Path] = (),
    patches: Sequence[Mapping[str, Any]] = (),
    variables: Mapping[str, str] | None = None,
) -> ResolvedComposition:
    """Resolve one profile through its bundles, its patches, and the caller's overlays."""
    merged = {"repo_root": str(REPO_ROOT), **dict(variables or {})}
    return resolve(
        profiles_dir or PROFILES_DIR,
        profile,
        patch_files=patch_files,
        patches=patches,
        variables=merged,
    )


def boot(
    composition: ResolvedComposition,
    *,
    required: Sequence[str] | None = None,
    workers: int = 4,
    stderr: Any = None,
) -> BootReport:
    """Mount a resolved composition and settle it.

    A required row that cannot become active disposes the tree and raises ``BootFailure``.
    Any other row that fails or waits is a warning; its service simply reads as absent.
    """
    return load(
        composition,
        seams=seam_keys(),
        required=required,
        workers=workers,
        stderr=stderr,
    )


__all__ = ["DEFAULT_PROFILE", "PROFILES_DIR", "REPO_ROOT", "boot", "compose"]
