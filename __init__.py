"""perception/lenses -- base lens executor and derived lenses.

`base` holds LensExecutor/PolarityLens; `derived` holds the higher
lenses. Import the submodules directly:
  from perception.lenses.base import LensExecutor
  The package __init__ is intentionally EMPTY: eagerly importing both
submodules re-creates a cycle (derived -> binder -> lenses shim).
"""
from __future__ import annotations

