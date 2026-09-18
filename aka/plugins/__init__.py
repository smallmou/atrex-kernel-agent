"""In-process plugin packages.

Each subdirectory is one plugin: the package module itself carries the declaration
(``name``, ``apply``, and optionally ``Config``/``Defaults``/``inject``/``provide``), so a
composition row names ``aka.plugins.<package>`` and nothing has to be discovered by globbing.

This root is deliberately separate from the repository-level ``plugins/`` directory. That one
holds external subprocess plugins whose ``.atrex_plugins/lock.json`` snapshot is compared
byte-for-byte on resume; adding a directory there would make every existing campaign workspace
unresumable.
"""

from __future__ import annotations
