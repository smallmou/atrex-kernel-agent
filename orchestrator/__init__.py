"""AKA orchestration package.

Kept an explicit package rather than an implicit namespace package so colocated
``test_*.py`` modules are reachable from the repository's discovery command,
``python3 -m unittest discover -s . -p 'test_*.py' -t .``. ``orchestrator/optimize.py``
still binds its live module into this package when run as a script, so direct execution
and package import continue to share one module identity.
"""
