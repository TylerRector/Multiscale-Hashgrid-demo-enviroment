"""level/exporter.py — deterministic World -> Mitsuba scene. One source of truth (§principle 1).

The SAME World drives this exporter, the samplers, and the game. Geometry here must match
the game's DDA world exactly: axis-aligned unit tiles, floor at y=0, ceiling at y=wall_height.
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from common import World


# Material classification from a hit (position, normal). Robust for an axis-aligned tile world:
# used by both the exporter's cross-check and the tracer's per-vertex albedo lookup.
def classify_material(p: np.ndarray, n: np.ndarray, wall_height: float) -> str:
    if n[1] > 0.5 and p[1] < 0.5 * wall_height:
        return "floor"
    if n[1] < -0.5:
        return "ceil"
    return "wall"


def _conductor_bsdf(f0, roughness):
    """spec(70).md §3: conductor mapped to (rough)conductor, f0_rgb as the shared tint (R3)."""
    val = {"type": "rgb", "value": [float(x) for x in f0]}
    if roughness and roughness > 1e-4:
        return {"type": "roughconductor", "distribution": "ggx", "alpha": float(roughness),
                "specular_reflectance": val}
    return {"type": "conductor", "specular_reflectance": val}


def build_scene_dict(world: World, spp: int = 8, res: int = 128,
                     include_objects: bool = False, roughness_override: float = -1.0):
    import mitsuba as mi
    T = mi.ScalarTransform4f
    cols, rows = world.cols, world.rows
    cell = world.cell
    wh = world.wall_height

    def diffuse(mat):
        rgb = [float(x) for x in world.albedo[mat]]
        return {"type": "diffuse", "reflectance": {"type": "rgb", "value": rgb}}

    scene: dict = {
        "type": "scene",
        "integrator": {"type": "path", "max_depth": 6},
        # floor: normal +y at y=0, covering [0,cols]x[0,rows]
        "floor": {
            "type": "rectangle",
            "to_world": T().translate([cols * cell / 2, 0.0, rows * cell / 2])
                            .rotate([1, 0, 0], -90).scale([cols * cell / 2, rows * cell / 2, 1]),
            "bsdf": diffuse("floor"),
        },
        # ceiling: normal -y at y=wall_height
        "ceil": {
            "type": "rectangle",
            "to_world": T().translate([cols * cell / 2, wh, rows * cell / 2])
                            .rotate([1, 0, 0], 90).scale([cols * cell / 2, rows * cell / 2, 1]),
            "bsdf": diffuse("ceil"),
        },
    }
    # one cube per wall tile
    for r in range(rows):
        for c in range(cols):
            if world.grid[r][c] == "#":
                cx, cz = (c + 0.5) * cell, (r + 0.5) * cell
                scene[f"wall_{r}_{c}"] = {
                    "type": "cube",
                    "to_world": T().translate([cx, wh / 2, cz]).scale([cell / 2, wh / 2, cell / 2]),
                    "bsdf": diffuse("wall"),
                }
    if include_objects:
        for ob in (world.objects or []):
            rough = ob["roughness"] if roughness_override < 0 else roughness_override
            bsdf = _conductor_bsdf(ob["f0_rgb"], rough)
            if ob["shape"] == "sphere":
                cen = ob["center"]; rad = float(ob["radius"])
                scene[ob["id"]] = {"type": "sphere", "center": [float(x) for x in cen],
                                   "radius": rad, "bsdf": bsdf}
            elif ob["shape"] == "box":
                bmin = np.array(ob["min"], np.float32); bmax = np.array(ob["max"], np.float32)
                c = (bmin + bmax) / 2; h = (bmax - bmin) / 2
                scene[ob["id"]] = {"type": "cube",
                                   "to_world": T().translate([float(c[0]), float(c[1]), float(c[2])])
                                                   .scale([float(h[0]), float(h[1]), float(h[2])]),
                                   "bsdf": bsdf}
    return scene


def load_scene(world: World, include_objects: bool = False, roughness_override: float = -1.0):
    import mitsuba as mi
    return mi.load_dict(build_scene_dict(world, include_objects=include_objects,
                                         roughness_override=roughness_override))


def add_light_and_sensor(world: World, light_pos, light_intensity, cam_origin, cam_target,
                         res: int = 128, spp: int = 32, include_objects: bool = False,
                         roughness_override: float = -1.0):
    """Build a scene dict WITH a point light + a pinhole sensor — for the NEE acceptance render."""
    import mitsuba as mi
    T = mi.ScalarTransform4f
    d = build_scene_dict(world, include_objects=include_objects, roughness_override=roughness_override)
    d["integrator"] = {"type": "path", "max_depth": 6}
    d["torch"] = {
        "type": "point",
        "position": [float(x) for x in light_pos],
        "intensity": {"type": "rgb", "value": [float(x) for x in light_intensity]},
    }
    d["sensor"] = {
        "type": "perspective",
        "fov": 75,
        "to_world": T().look_at(origin=[float(x) for x in cam_origin],
                                target=[float(x) for x in cam_target],
                                up=[0, 1, 0]),
        "film": {"type": "hdrfilm", "width": res, "height": res, "pixel_format": "rgb"},
        "sampler": {"type": "independent", "sample_count": spp},
    }
    return mi.load_dict(d)


if __name__ == "__main__":
    import mitsuba as mi
    mi.set_variant("llvm_ad_rgb")
    from common import load_world
    w = load_world()
    sc = load_scene(w)
    print("exported scene OK:", sc)
