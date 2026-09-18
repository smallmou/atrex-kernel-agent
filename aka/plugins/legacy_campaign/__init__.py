"""The legacy campaign driver plugin.

The package module is the plugin: a composition row names ``aka.plugins.legacy_campaign`` and
the loader reads the declaration from here.
"""

from __future__ import annotations

from .plugin import (
    Config,
    Defaults,
    apply,
    inject,
    interpolate,
    name,
    optional_inject,
    provide,
)

__all__ = [
    "Config",
    "Defaults",
    "apply",
    "inject",
    "interpolate",
    "name",
    "optional_inject",
    "provide",
]
