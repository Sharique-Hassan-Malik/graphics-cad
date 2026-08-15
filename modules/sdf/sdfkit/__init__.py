"""sdfkit — model with signed distance fields, mesh with marching tetrahedra."""

from .marching import triangulate
from .mesh import Mesh
from .sdf import (
    SDF, Box, Cylinder, Difference, Intersection, Round, Shell, SmoothUnion,
    Sphere, Torus, Translate, Union,
)

__all__ = [
    "SDF", "Sphere", "Box", "Torus", "Cylinder",
    "Union", "Intersection", "Difference", "SmoothUnion", "Shell", "Round", "Translate",
    "triangulate", "Mesh",
]
