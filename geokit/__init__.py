"""geokit — what seven geometry generators in this repository share.

    from geokit.mesh import Mesh
    from geokit.blender import find_blender, run_script

One triangle mesh with the manifold checks, and one path into Blender. Both
existed in three to seven copies before. Stdlib plus numpy, so a generator
importing them standalone gains nothing it did not already need.
"""

from .mesh import Mesh

__version__ = "1.0.0"
__all__ = ["Mesh"]
