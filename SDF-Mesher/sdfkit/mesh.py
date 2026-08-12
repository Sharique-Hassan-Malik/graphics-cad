"""A triangle mesh and the topological checks that decide whether an extracted
isosurface is a real solid rather than a soup of triangles.

Marching tetrahedra emits one or two triangles per boundary-crossing tetrahedron,
independently — a "triangle soup" whose shared edges hold *coincident but
distinct* vertices. `welded()` fuses those, after which the surface is a closed
2-manifold and the counting checks below become meaningful: every edge used by
exactly two faces (watertight), a signed volume that matches the analytic answer,
and an Euler characteristic that fingerprints the topology (a ball is 2, a torus
is 0).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass
class Mesh:
    vertices: np.ndarray   # N x 3
    faces: np.ndarray      # M x 3 int

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def bounds(self):
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def size(self):
        lo, hi = self.bounds()
        return hi - lo

    # -- topology ------------------------------------------------------------

    def _edge_counts(self) -> dict[tuple[int, int], int]:
        counts: dict[tuple[int, int], int] = {}
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def is_watertight(self) -> bool:
        return all(n == 2 for n in self._edge_counts().values())

    def is_edge_manifold(self) -> bool:
        return all(n <= 2 for n in self._edge_counts().values())

    def boundary_edges(self):
        return [e for e, n in self._edge_counts().items() if n == 1]

    def euler_characteristic(self) -> int:
        return self.vertex_count - len(self._edge_counts()) + self.face_count

    def genus(self):
        """For a closed orientable surface, χ = 2 − 2g. Returns g if χ is even."""
        chi = self.euler_characteristic()
        return (2 - chi) // 2 if chi % 2 == 0 else None

    def is_consistently_oriented(self) -> bool:
        seen: set[tuple[int, int]] = set()
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                if (u, v) in seen:
                    return False
                seen.add((u, v))
        return True

    # -- geometry ------------------------------------------------------------

    def volume(self) -> float:
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        return float(np.einsum("ij,ij->", np.cross(v1 - v0, v2 - v0), v0) / 6.0)

    def area(self) -> float:
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

    def face_normals(self) -> np.ndarray:
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        n = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, lengths, out=np.zeros_like(n), where=lengths > 0)

    def face_centroids(self) -> np.ndarray:
        return self.vertices[self.faces].mean(axis=1)

    # -- construction --------------------------------------------------------

    def welded(self, tol: float = 1e-6) -> "Mesh":
        """Fuse vertices closer than `tol` into one, turning triangle soup into a
        connected surface. Degenerate triangles (a collapsed edge) are dropped."""
        if self.vertex_count == 0:
            return Mesh(self.vertices, self.faces)
        quantised = np.round(self.vertices / tol).astype(np.int64)
        _, first, inverse = np.unique(quantised, axis=0, return_index=True, return_inverse=True)
        new_vertices = self.vertices[first]
        new_faces = inverse.reshape(-1)[self.faces]
        keep = (new_faces[:, 0] != new_faces[:, 1]) & \
               (new_faces[:, 1] != new_faces[:, 2]) & \
               (new_faces[:, 0] != new_faces[:, 2])
        return Mesh(new_vertices, new_faces[keep])

    def oriented(self, outward=None) -> "Mesh":
        """Make every face wind consistently, then face them the right way.

        Propagate a seed face's winding across the (welded, manifold) surface by
        flood fill: two faces sharing an edge are consistent exactly when that
        edge runs opposite ways through them, so any neighbour that agrees is
        flipped. This is purely topological — it needs no surface normal — which
        makes it robust across the sharp creases of CSG results, where a
        finite-difference gradient is unreliable.

        The remaining choice is each connected component's *global* sign. If an
        `outward` direction field is given (an SDF gradient, which points out of
        the material everywhere), each component is flipped to agree with it by a
        robust majority vote over its faces — correct even for an internal cavity,
        whose outward-facing normals point *into* the void and so must contribute
        negative volume. Without such a field, each component is simply flipped to
        positive volume, which is right for any shape that has no cavities.
        """
        faces = self.faces.copy()
        n = len(faces)
        if n == 0:
            return Mesh(self.vertices, faces)

        edge_faces: dict[tuple[int, int], list[int]] = {}
        for fi in range(n):
            a, b, c = faces[fi]
            for u, v in ((a, b), (b, c), (c, a)):
                edge_faces.setdefault((u, v) if u < v else (v, u), []).append(fi)

        visited = np.zeros(n, dtype=bool)
        component = np.full(n, -1, dtype=np.int64)
        label = 0
        for seed in range(n):
            if visited[seed]:
                continue
            visited[seed] = True
            component[seed] = label
            stack = [seed]
            while stack:
                f = stack.pop()
                a, b, c = faces[f]
                for u, v in ((a, b), (b, c), (c, a)):
                    key = (u, v) if u < v else (v, u)
                    for g in edge_faces.get(key, ()):
                        if g == f or visited[g]:
                            continue
                        ga, gb, gc = faces[g]
                        # If g runs the shared edge the *same* way as f, flip g.
                        if (u, v) in ((ga, gb), (gb, gc), (gc, ga)):
                            faces[g] = faces[g][::-1]
                        visited[g] = True
                        component[g] = label
                        stack.append(g)
            label += 1

        # Per-component global sign.
        current = Mesh(self.vertices, faces)
        if outward is not None:
            normals = current.face_normals()
            centroids = current.face_centroids()
            g = outward(centroids)
            agree = np.einsum("ij,ij->i", normals, g)   # >0 where normal points outward
            for l in range(label):
                idx = np.where(component == l)[0]
                if agree[idx].sum() < 0:                 # majority disagrees → flip
                    faces[idx] = faces[idx][:, ::-1]
        else:
            v0, v1, v2 = (self.vertices[faces[:, i]] for i in range(3))
            face_vol = np.einsum("ij,ij->i", np.cross(v1 - v0, v2 - v0), v0) / 6.0
            for l in range(label):
                idx = np.where(component == l)[0]
                if face_vol[idx].sum() < 0:
                    faces[idx] = faces[idx][:, ::-1]
        return Mesh(self.vertices, faces)

    # -- export --------------------------------------------------------------

    def to_stl_binary(self) -> bytes:
        normals = self.face_normals()
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        out = bytearray()
        out += b"sdf-mesher".ljust(80, b"\0")
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

    def to_obj(self) -> str:
        lines = [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in self.vertices]
        lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in self.faces]
        return "\n".join(lines) + "\n"
