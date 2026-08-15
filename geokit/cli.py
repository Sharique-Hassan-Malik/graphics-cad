"""`geo` — one command over seven generators.

    geo modules                     what is here, and how to run each alone
    geo make parts --part gear      run a generator (its own flags, unchanged)
    geo check model.stl             is this mesh actually manufacturable?
    geo render model.stl            put any mesh in front of Blender

`check` and `render` work on the output of *any* generator, which is the point
of there being one `Mesh`: before, each one could only validate its own.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from . import registry
from .blender import cycles_setup, find_blender, finish, frame_camera, mesh_to_pydata, run_script
from .mesh import Mesh


def _wrap(text: str, width: int) -> list[str]:
    words, lines, line = text.split(), [], []
    for word in words:
        if sum(len(w) + 1 for w in line) + len(word) > width and line:
            lines.append(" ".join(line))
            line = []
        line.append(word)
    if line:
        lines.append(" ".join(line))
    return lines


def _cmd_modules(args: argparse.Namespace) -> int:
    print()
    for gen in registry.generators():
        why = registry.unavailable(gen)
        print(f"  {gen.name:9} {gen.produces:10} {'ready' if not why else why}")
        print(f"  {'':9} {gen.title}")
        for line in _wrap(gen.summary, 70):
            print(f"  {'':9} {line}")
        print(f"  {'':9} cd modules/{gen.name} && {gen.standalone}")
        print()
    blender = find_blender()
    print(f"  Blender: {blender or 'not found — generators still work, renders are emitted as scripts'}")
    print()
    return 0


def _cmd_make(args: argparse.Namespace) -> int:
    """Delegate to the generator's own CLI, arguments untouched."""
    registry.add_to_path(args.generator)
    module = importlib.import_module(f"{registry.generator(args.generator).package}.cli")
    return int(module.main(args.args) or 0)


def _load_mesh(path: str, *, weld: bool = True, tol: float = 1e-6) -> tuple[Mesh, Mesh]:
    """Load an STL, and weld it.

    Binary STL stores three vertices per triangle with no sharing, so a
    round-trip through it turns every mesh into loose triangles: no edge is
    used twice and every topology check says "not watertight" — for a mesh that
    was closed when it was written. Welding restores the shared vertices, so
    the checks are about the geometry rather than about the file format.

    Both meshes come back, because the vertex counts before and after are worth
    seeing: they are how much the format inflated it.
    """
    raw = Mesh.from_stl_binary(Path(path).read_bytes())
    return (raw.welded(tol=tol) if weld else raw), raw


def _cmd_check(args: argparse.Namespace) -> int:
    """Report the topology that decides whether a mesh is printable.

    None of this is visible in a render, which is exactly why it is worth a
    command: a mesh can look like a gear and be unusable.
    """
    mesh, raw = _load_mesh(args.mesh, weld=not args.no_weld, tol=args.tolerance)
    watertight = mesh.is_watertight()
    manifold = mesh.is_edge_manifold()
    oriented = mesh.is_consistently_oriented()

    facts = {
        "vertices": mesh.vertex_count,
        "vertices_in_file": raw.vertex_count,
        "faces": mesh.face_count,
        "watertight": watertight,
        "edge_manifold": manifold,
        "consistently_oriented": oriented,
        "euler_characteristic": mesh.euler_characteristic(),
        "genus": mesh.genus() if watertight else None,
        "volume": round(mesh.volume(), 6),
        "area": round(mesh.area(), 6),
        "boundary_edges": len(mesh.boundary_edges()),
    }

    if args.json:
        print(json.dumps(facts, indent=2))
    else:
        print()
        for key, value in facts.items():
            mark = ""
            if key in ("watertight", "edge_manifold", "consistently_oriented"):
                mark = "  ✓" if value else "  ✖"
            print(f"  {key:24} {value}{mark}")
        print()
        if watertight and manifold and oriented:
            print("  Closed, orientable and manifold — a slicer will accept this.\n")
        else:
            print("  Not manufacturable: the inside is undefined, so a slicer either")
            print("  refuses the file or fills it with guesses.\n")

    return 0 if (watertight and manifold and oriented) else 1


def _cmd_render(args: argparse.Namespace) -> int:
    """Render any mesh through the shared Blender path."""
    mesh, _ = _load_mesh(args.mesh)
    name = Path(args.mesh).stem
    out_dir = Path(args.out)
    blend = str(out_dir / f"{name}.blend")
    png = str(out_dir / f"{name}.png")

    centre = mesh.centroid()
    radius = float((mesh.size() ** 2).sum() ** 0.5) / 2 or 1.0

    script = (
        "import bpy, mathutils, math\n"
        "bpy.ops.wm.read_factory_settings(use_empty=True)\n"
        + mesh_to_pydata(mesh, name)
        + "light_data = bpy.data.lights.new('key', type='SUN')\n"
        "light_data.energy = 4.0\n"
        "light = bpy.data.objects.new('key', light_data)\n"
        "light.rotation_euler = (0.9, 0.2, 0.8)\n"
        "bpy.context.collection.objects.link(light)\n"
        + frame_camera(centre, radius)
        + cycles_setup(samples=args.samples)
        + finish(blend, png)
    )

    result = run_script(script, out_dir, name=name, blend_path=blend, png_path=png)
    print()
    for key, value in result.to_dict().items():
        print(f"  {key:16} {value}")
    if not result.ran and result.note:
        print(f"\n  {result.note}")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geo",
        description="Seven geometry generators, one mesh type, one way into Blender.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("modules", help="the generators and their own CLIs")
    sub.add_parser("make", help="run a generator (its own flags)", add_help=False)

    check = sub.add_parser("check", help="report a mesh's topology")
    check.add_argument("mesh", help="a binary STL")
    check.add_argument("--json", action="store_true")
    check.add_argument("--no-weld", action="store_true",
                       help="check the file as stored — every STL will look open")
    check.add_argument("--tolerance", type=float, default=1e-6,
                       help="vertex welding tolerance (default: 1e-6)")

    render = sub.add_parser("render", help="render a mesh with Blender")
    render.add_argument("mesh", help="a binary STL")
    render.add_argument("--out", default="./render")
    render.add_argument("--samples", type=int, default=24)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # `make` hands everything after the generator name to that generator's own
    # parser, so `geo make parts --help` prints partkit's help, not this one's.
    if argv and argv[0] == "make":
        if len(argv) < 2:
            print("geo: make needs a generator name; try `geo modules`", file=sys.stderr)
            return 2
        namespace = argparse.Namespace(generator=argv[1], args=argv[2:])
        try:
            return _cmd_make(namespace)
        except KeyError as exc:
            print(f"geo: {exc}", file=sys.stderr)
            return 2

    args = build_parser().parse_args(argv)
    if args.command == "modules":
        return _cmd_modules(args)
    if args.command == "check":
        return _cmd_check(args)
    return _cmd_render(args)


if __name__ == "__main__":
    raise SystemExit(main())
