"""Python entry point for the legacy extensionless `binder` source.

The repository's original binder implementation is stored in a file named
`binder` without a Python extension. Execute that source in this module's
namespace so normal imports such as `import binder` and `from binder import
Binder` work without changing the existing implementation or its import
behavior.
"""

from pathlib import Path

_legacy_path = Path(__file__).with_name("binder")
if not _legacy_path.is_file():
    raise FileNotFoundError(f"Legacy binder source not found: {_legacy_path}")

_source = _legacy_path.read_text(encoding="utf-8")
exec(compile(_source, str(_legacy_path), "exec"), globals())
