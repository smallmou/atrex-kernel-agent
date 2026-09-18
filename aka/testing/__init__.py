"""Test-only plugins and compositions.

These exist so the real composition path can be exercised without a GPU, an operator directory,
or a coding-agent CLI. A hand-built context is not a substitute: the loader, the declaration
checks, the settle loop, and the JSON layering are exactly the parts most likely to break, and
only a real profile boot covers them.
"""

from __future__ import annotations
