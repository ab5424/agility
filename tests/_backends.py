"""Backend availability probes shared by the test suite.

Importing some optional backends (notably ``ovito``) can hard-crash the
interpreter on machines without a usable graphics stack (e.g. headless
Windows CI runners raise a native access violation inside OVITO's Qt/Direct3D
initialisation).  A native crash cannot be caught in-process, so availability
is checked by attempting the import in a *subprocess* instead.
"""

from __future__ import annotations

import subprocess
import sys
from functools import cache
from importlib.util import find_spec


@cache
def _backend_importable(name: str) -> bool:
    """Return ``True`` if ``import name`` succeeds in a fresh interpreter."""
    if find_spec(name) is None:
        return False
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", f"import {name}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def ovito_available() -> bool:
    """Return ``True`` if the ovito package can be imported without crashing."""
    return _backend_importable("ovito")


def ase_available() -> bool:
    """Return ``True`` if the ase package can be imported."""
    return _backend_importable("ase")


def pymatgen_available() -> bool:
    """Return ``True`` if the pymatgen package can be imported."""
    return _backend_importable("pymatgen")
