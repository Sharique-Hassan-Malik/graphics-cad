"""A triangle mesh, and the checks that separate a printable part from a picture.

The difference between "looks like a gear" and "is a gear you can 3D print" is
almost entirely topology, and none of it is visible in a render. A mesh is
manufacturable only if it is a closed, orientable 2-manifold:

  * **watertight** — every edge is shared by exactly two triangles. An edge used
    once is a hole; an edge used three times is a non-manifold seam. Either makes
    the "inside" undefined, and a slicer either refuses the file or fills it with
    guesses.
  * **consistently oriented** — every triangle winds the same way (outward), so
    the surface has a well-defined inside. The signed volume is positive exactly
    when this holds, which is why `volume()` doubles as an orientation check.

This module is deliberately free of any modelling library. A mesh is two arrays —
vertices and triangle indices — and every property below is computed from them
directly, so "is this watertight?" is answered by counting, not by trusting a
kernel.

Every generator in this repository produces one of these. There were three
near-identical `Mesh` classes before — the terrain one, the SDF one and this —
agreeing on the representation and disagreeing about which checks existed, so a
mesh from one generator could not be validated with another's tools. This is the
one with the manifold checks, and now everything uses it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    """Vertices (N x 3 float) and triangles (M x 3 int indices into vertices)."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self):
        self.vertices = np.asarray(self.vertices, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if self.faces.size and self.faces.max() >= len(self.vertices):
            raise ValueError("a face references a vertex index past the end of the vertex array")

    # -- basic stats ---------------------------------------------------------

    @property
    def vertex_count(self) -> int:
        return len(self.vertices)

    @property
    def face_count(self) -> int:
        return len(self.faces)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    def size(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    # -- topology ------------------------------------------------------------

    def _edge_use_counts(self) -> dict[tuple[int, int], int]:
        """How many faces use each undirected edge. The heart of every check below."""
        counts: dict[tuple[int, int], int] = {}
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                key = (u, v) if u < v else (v, u)
                counts[key] = counts.get(key, 0) + 1
        return counts

    def is_watertight(self) -> bool:
        """True iff every edge is used by exactly two faces."""
        return all(n == 2 for n in self._edge_use_counts().values())

    def is_edge_manifold(self) -> bool:
        """No edge shared by more than two faces (a hole is allowed, a seam is not)."""
        return all(n <= 2 for n in self._edge_use_counts().values())

    def boundary_edges(self) -> list[tuple[int, int]]:
        """Edges used by only one face — the holes. Empty iff watertight."""
        return [e for e, n in self._edge_use_counts().items() if n == 1]

    def euler_characteristic(self) -> int:
        """V - E + F. A closed genus-0 surface (a solid with no holes) has χ = 2.

        A solid with a through-hole (a torus topology, like a bored gear) has
        χ = 0. So χ is a fingerprint of the shape's topology, and a value that is
        neither 2 nor 0 for a part that should be one of those means the mesh is
        broken — a fast check that needs no geometry.
        """
        v = self.vertex_count
        e = len(self._edge_use_counts())
        f = self.face_count
        return v - e + f

    def is_consistently_oriented(self) -> bool:
        """Every shared edge must appear once in each direction across its two faces.

        If two adjacent triangles wind the same way, they share a *directed* edge
        in the same direction — which means one of them faces inward. Checking
        directed-edge uniqueness catches a flipped triangle that watertightness
        alone would miss.
        """
        seen: set[tuple[int, int]] = set()
        for a, b, c in self.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                if (u, v) in seen:
                    return False  # this directed edge used twice → same winding
                seen.add((u, v))
        return True

    # -- geometry ------------------------------------------------------------

    def volume(self) -> float:
        """Signed volume via the divergence theorem (sum of tetrahedra to origin).

        Positive when the surface is closed and outward-facing, so it is both the
        part's volume and a witness that the mesh is a valid solid. A near-zero or
        negative result on a shape that should enclose space is a topology bug.
        """
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(np.einsum("ij,ij->", np.cross(v1 - v0, v2 - v0), v0) / 6.0)

    def area(self) -> float:
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        return float(0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum())

    def centroid(self) -> np.ndarray:
        return self.vertices[self.faces].mean(axis=(0, 1))

    # -- construction --------------------------------------------------------

    def transformed(self, matrix: np.ndarray) -> "Mesh":
        """Apply a 4x4 homogeneous transform."""
        homo = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
        moved = (homo @ np.asarray(matrix, dtype=np.float64).T)[:, :3]
        return Mesh(moved, self.faces.copy())

    def translated(self, dx=0.0, dy=0.0, dz=0.0) -> "Mesh":
        return Mesh(self.vertices + np.array([dx, dy, dz]), self.faces.copy())

    @staticmethod
    def concatenate(meshes: list["Mesh"]) -> "Mesh":
        """Combine several meshes, offsetting face indices. No welding of coincident
        vertices — the caller is responsible for shared boundaries being identical."""
        verts, faces, offset = [], [], 0
        for m in meshes:
            verts.append(m.vertices)
            faces.append(m.faces + offset)
            offset += len(m.vertices)
        return Mesh(np.vstack(verts), np.vstack(faces))

    # -- brought across from the SDF mesher's copy, so nothing was lost in
    # -- the merge: genus and per-face vectors were only in that one.

    def face_normals(self) -> np.ndarray:
        v0, v1, v2 = (self.vertices[self.faces[:, i]] for i in range(3))
        n = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return np.divide(n, lengths, out=np.zeros_like(n), where=lengths > 0)

    def face_centroids(self) -> np.ndarray:
        return self.vertices[self.faces].mean(axis=1)

    # -- construction --------------------------------------------------------

    def genus(self):
        """For a closed orientable surface, χ = 2 − 2g. Returns g if χ is even."""
        chi = self.euler_characteristic()
        return (2 - chi) // 2 if chi % 2 == 0 else None

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

    def welded(self, tol: float = 1e-7) -> "Mesh":
        """Merge vertices closer than `tol`, so shared boundaries become shared
        topology. Watertightness checks need this: two triangles that meet at a
        seam are only 'adjacent' if they reference the *same* vertex index."""
        quantised = np.round(self.vertices / tol).astype(np.int64)
        _, first_index, inverse = np.unique(
            quantised, axis=0, return_index=True, return_inverse=True)
        new_vertices = self.vertices[first_index]
        new_faces = inverse.reshape(-1)[self.faces]
        # Drop degenerate triangles created by welding (two corners merged).
        keep = (new_faces[:, 0] != new_faces[:, 1]) & \
               (new_faces[:, 1] != new_faces[:, 2]) & \
               (new_faces[:, 0] != new_faces[:, 2])
        return Mesh(new_vertices, new_faces[keep])

    # -- export --------------------------------------------------------------

    def to_stl_binary(self) -> bytes:
        """Binary STL — the lingua franca of 3D printing. Written by hand: it is an
        80-byte header, a face count, then 50 bytes per triangle (a normal and
        three vertices as float32, plus two padding bytes)."""
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)

        out = bytearray()
        out += b"parametric-parts".ljust(80, b"\0")
        out += struct.pack("<I", len(self.faces))
        for i in range(len(self.faces)):
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

    @staticmethod
    def from_stl_binary(data: bytes) -> "Mesh":
        """Parse binary STL back, for round-trip verification."""
        (count,) = struct.unpack("<I", data[80:84])
        verts, faces = [], []
        pos = 84
        for i in range(count):
            # Each 50-byte record is normal(3f) + v0(3f) + v1(3f) + v2(3f) + attr(2).
            # Skip the 12-byte normal, read the three vertices (9 floats).
            tri = struct.unpack("<9f", data[pos + 12: pos + 48])
            base = len(verts)
            verts.extend([tri[0:3], tri[3:6], tri[6:9]])
            faces.append([base, base + 1, base + 2])
            pos += 50
        return Mesh(np.array(verts), np.array(faces))
