"""Turn a heightmap into a watertight solid terrain slab, and write it to STL.

The top surface is the heightmap; a flat underside and four side walls close it
into a solid a slicer or physics engine will accept. "Watertight" is checked the
honest way — every edge used by exactly two triangles — because a terrain that is
only a surface (an open sheet) has boundary edges all along its rim and is not a
printable object.
"""

from __future__ import annotations

import struct

import numpy as np


class Mesh:
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    @property
    def vertex_count(self):
        return len(self.vertices)

    @property
    def face_count(self):
        return len(self.faces)

    def _edge_counts(self):
        counts: dict[tuple[int, int], int] = {}
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def is_watertight(self) -> bool:
        return all(n == 2 for n in self._edge_counts().values())

    def euler_characteristic(self) -> int:
        return self.vertex_count - len(self._edge_counts()) + self.face_count

    def face_normals(self):
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        n = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, lengths, out=np.zeros_like(n), where=lengths > 0)

    def to_stl_binary(self) -> bytes:
        normals = self.face_normals()
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        out = bytearray()
        out += b"procedural-terrain".ljust(80, b"\0")
        out += struct.pack("<I", self.face_count)
        for i in range(self.face_count):
            out += struct.pack("<3f", *normals[i])
            out += struct.pack("<3f", *v0[i])
            out += struct.pack("<3f", *v1[i])
            out += struct.pack("<3f", *v2[i])
            out += struct.pack("<H", 0)
        return bytes(out)

    def save_stl(self, path: str) -> str:
        with open(path, "wb") as handle:
            handle.write(self.to_stl_binary())
        return path


def terrain_mesh(heightmap: np.ndarray, width: float = 10.0, height_scale: float = 3.0,
                 base: float = 0.0) -> Mesh:
    """A watertight terrain slab from a heightmap. `width` is the world size of
    the longest side; `height_scale` maps the [0,1] heightmap to world height."""
    rows, cols = heightmap.shape
    cell = width / max(rows, cols)
    z = base + heightmap * height_scale
    floor = base - 0.5 * height_scale

    verts = []
    top = np.zeros((rows, cols), int)
    bot = np.zeros((rows, cols), int)
    for r in range(rows):
        for c in range(cols):
            top[r, c] = len(verts)
            verts.append((c * cell, (rows - 1 - r) * cell, float(z[r, c])))
    for r in range(rows):
        for c in range(cols):
            bot[r, c] = len(verts)
            verts.append((c * cell, (rows - 1 - r) * cell, floor))

    faces = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            a, b, d, e = top[r, c], top[r, c + 1], top[r + 1, c + 1], top[r + 1, c]
            faces += [(a, b, d), (a, d, e)]
            a2, b2, d2, e2 = bot[r, c], bot[r, c + 1], bot[r + 1, c + 1], bot[r + 1, c]
            faces += [(a2, d2, b2), (a2, e2, d2)]

    def wall(t0, t1, b0, b1):
        faces.append((t0, t1, b1))
        faces.append((t0, b1, b0))

    for c in range(cols - 1):
        wall(top[0, c], top[0, c + 1], bot[0, c], bot[0, c + 1])
        wall(top[rows - 1, c + 1], top[rows - 1, c], bot[rows - 1, c + 1], bot[rows - 1, c])
    for r in range(rows - 1):
        wall(top[r + 1, 0], top[r, 0], bot[r + 1, 0], bot[r, 0])
        wall(top[r, cols - 1], top[r + 1, cols - 1], bot[r, cols - 1], bot[r + 1, cols - 1])

    return Mesh(verts, faces)
