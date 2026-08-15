"""The triangle mesh for this generator — the shared implementation.

There were three near-identical `Mesh` classes in this repository. They agreed
on the representation (vertices N×3, triangles M×3) and disagreed about which
checks existed, so a mesh from one generator could not be validated with
another's tools. `geokit.mesh` is the union of them, and every generator now
produces it.

Re-exported here so `from sdfkit.mesh import Mesh` keeps working.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_REPO_ROOT = _Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

from geokit.mesh import Mesh  # noqa: E402,F401

__all__ = ["Mesh"]
