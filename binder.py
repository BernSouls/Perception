"""Compatibility wrapper for the legacy non-.py module `binder`.

This repository currently contains a module named `binder` without a `.py`
extension. This file makes that module importable as a real Python module
while preserving the original file as-is.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_legacy_path = Path(__file__).with_name("binder")

if not _legacy_path.exists():
    raise FileNotFoundError(f"Legacy binder module not found at: {_legacy_path}")

_spec = importlib.util.spec_from_file_location("legacy_binder", _legacy_path)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Unable to load legacy binder module from: {_legacy_path}")

_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

for _name in dir(_module):
    if _name.startswith("__") and _name not in {"__all__", "__doc__", "__file__", "__name__"}:
        continue
    globals()[_name] = getattr(_module, _name)

__all__ = getattr(_module, "__all__", [])
