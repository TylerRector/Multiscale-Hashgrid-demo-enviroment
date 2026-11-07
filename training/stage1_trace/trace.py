"""stage1_trace/trace.py — multi-camera light-agnostic path dump (plan Stage 1).

BSDF sampling only (cosine-weighted diffuse), NO next-event estimation, Russian roulette.
Mitsuba is the deterministic intersection engine (scene from level.toml). We own the SoA dump.

Output: runs/<run>/trace/  (schema 'trace'). CLI-runnable pure function: level+config -> blob.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import mitsuba as mi

import schemas
from common import World, load_config, load_world, run_dir, Ledger, now, git_rev
from level.exporter import load_scene, classify_material


def _np3(x):
    a = np.array(x)
    return a.T if a.shape[0] == 3 else a


def blue_noise_poses(world: World, n: int, rng: np.random.Generator, eye: float):
    """Best-candidate ('blue noise') poses over walkable cell centers, random yaw."""
    cells = world.walkable_cells()
    centers = np.array([world.cell_center_world(r, c) for (r, c) in cells])  # (M,2) x,z
    chosen = []
    idx0 = rng.integers(len(centers))
    chosen.append(idx0)
    for _ in range(n - 1):
        best, bestd = -1, -1.0
        cand = rng.integers(0, len(centers), size=min(16, len(centers)))
        for ci in cand:
            d = min(np.sum((centers[ci] - centers[cj]) ** 2) for cj in chosen)
            if d > bestd:
                bestd, best = d, ci
        chosen.append(int(best))
    poses = []
    for ci in chosen[:n]:
        x, z = centers[ci]
        # jitter within cell, keep clear of walls
        jx = (rng.random() - 0.5) * 0.4 * world.cell
        jz = (rng.random() - 0.5) * 0.4 * world.cell
        yaw = rng.random() * 2 * np.pi
        origin = np.array([x + jx, eye, z + jz], np.float32)
        target = origin + np.array([np.cos(yaw), 0.0, np.sin(yaw)], np.float32)
        poses.append((origin, target))
    return poses


def primary_dirs(origin, target, res, fov_deg=75.0):
    fwd = target - origin
    fwd = fwd / np.linalg.norm(fwd)
    up = np.array([0, 1, 0], np.float32)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    tup = np.cross(right, fwd)
    tan = np.tan(np.radians(fov_deg) / 2)
    # pixel centers in [-1,1]
    js = (np.arange(res) + 0.5) / res * 2 - 1
    sx, sy = np.meshgrid(js, -js)  # sy flipped so +y is up
    sx = sx.reshape(-1); sy = sy.reshape(-1)
    d = (fwd[None, :] + (sx * tan)[:, None] * right[None, :] + (sy * tan)[:, None] * tup[None, :])
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return d.astype(np.float32)


def metal_classifier(world):
    """spec(70).md §3/§4: geometric conductor test (independent of Mitsuba internals). Returns a
    vectorized fn P(N,3) -> (is_metal(N) bool, f0(N,3)). A hit is metal if it lies on an object surface."""
    sph, box = [], []
    for o in (world.objects or []):
        f0 = np.array(o["f0_rgb"], np.float32)
        if o["shape"] == "sphere":
            sph.append((np.array(o["center"], np.float32), float(o["radius"]), f0))
        elif o["shape"] == "box":
            box.append((np.array(o["min"], np.float32), np.array(o["max"], np.float32), f0))

    def cls(P, eps=2e-2):
        N = P.shape[0]; ism = np.zeros(N, bool); f0 = np.zeros((N, 3), np.float32)
        for c, r, f in sph:
            hit = np.abs(np.linalg.norm(P - c[None, :], axis=1) - r) < eps
            ism |= hit; f0[hit] = f
        for mn, mx, f in box:
            inside = np.all((P >= mn[None, :] - eps) & (P <= mx[None, :] + eps), axis=1)
            onface = inside & (np.min(np.minimum(np.abs(P - mn[None, :]), np.abs(P - mx[None, :])), axis=1) < eps)
            ism |= onface; f0[onface] = f
        return ism, f0
    return cls


def orbit_poses(world, per_object, rng, eye):
    """spec(70).md §4: `per_object` blue-noise cameras ringed around each object, aimed at it."""
    poses = []
    for o in (world.objects or []):
        if o["shape"] == "sphere":
            c = np.array(o["center"], np.float32); rad = float(o["radius"]) + 1.4
        else:
            c = (np.array(o["min"], np.float32) + np.array(o["max"], np.float32)) / 2; rad = 1.6
        base = rng.random() * 2 * np.pi
        for k in range(per_object):
            ang = base + 2 * np.pi * k / per_object + (rng.random() - 0.5) * 0.2
            x = c[0] + rad * np.cos(ang); z = c[2] + rad * np.sin(ang)
            cc = int(x // world.cell); rr = int(z // world.cell)
            if world.is_wall(rr, cc):
                continue  # keep the ring camera in the walkable interior
            origin = np.array([x, eye, z], np.float32)
            target = np.array([c[0], eye, c[2]], np.float32)
            poses.append((origin, target))
    return poses


def cosine_sample(normals, rng):
    """Cosine-weighted hemisphere directions around each normal. normals (K,3)."""
    K = normals.shape[0]
    u1 = rng.random(K); u2 = rng.random(K)
    r = np.sqrt(u1); theta = 2 * np.pi * u2
    lx = r * np.cos(theta); ly = r * np.sin(theta); lz = np.sqrt(np.maximum(0, 1 - u1))
    # build tangent frame per normal
    n = normals
    helper = np.where(np.abs(n[:, 0:1]) > 0.9, np.array([0, 1, 0], np.float32), np.array([1, 0, 0], np.float32))
    t = np.cross(helper, n); t /= (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
    b = np.cross(n, t)
    d = lx[:, None] * t + ly[:, None] * b + lz[:, None] * n
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    return d.astype(np.float32)


def trace(profile: str, run: str, config_path="config/config.toml"):
    mi.set_variant("llvm_ad_rgb")
    cfg = load_config(profile, config_path)
    seed = cfg["run"]["seed"]
    s1 = cfg["stage1"]
    spec = cfg.get("spec", {})
    world = load_world(cfg["run"]["level"])
    # spec(70).md M3: include conductors so BSDF sampling bounces off metal (metal→scene transport)
    metal_on = bool(spec.get("reattribute_metal", False))
    scene = load_scene(world, include_objects=metal_on)
    metal_cls = metal_classifier(world) if metal_on else None
    rng = np.random.default_rng(seed)

    n_cam = s1["n_cameras"]; res = s1["res"]; spp = s1["spp"]
    depth = s1["depth"]; rr_start = s1.get("rr_start_depth", 2)
    eye = world.eye_height

    poses = blue_noise_poses(world, n_cam, rng, eye)
    n_orbit = int(spec.get("orbit_cams_per_obj", 0))
    if metal_on and n_orbit > 0:
        poses = poses + orbit_poses(world, n_orbit, rng, eye)
        print(f"[stage1] spec: +{len(poses)-n_cam} orbit cams around {len(world.objects)} objects "
              f"-> {len(poses)} total (conductors in scene)")

    # accumulators (SoA)
    A_pos, A_n, A_rho, A_valid, A_cam, A_spec = [], [], [], [], [], []
    # per-vertex records tagged with a global pixel id -> CSR later
    V_pix, V_pos, V_n, V_thr = [], [], [], []

    def wide_intersect(o, d):
        o = np.ascontiguousarray(o, np.float32); d = np.ascontiguousarray(d, np.float32)
        ray = mi.Ray3f(mi.Point3f(o[:, 0], o[:, 1], o[:, 2]),
                       mi.Vector3f(d[:, 0], d[:, 1], d[:, 2]))
        si = scene.ray_intersect(ray)
        return (_np3(si.p), _np3(si.sh_frame.n), np.array(si.is_valid()), np.array(si.t))

    t0 = now()
    global_pix = 0
    for ci, (origin, target) in enumerate(poses):
        dirs = primary_dirs(origin, target, res)
        # average over spp by jittering primary dirs slightly (box filter) — cheap AA of geometry
        # (light-agnostic dump: spp mainly matters for the indirect throughput MC, handled by bounces)
        npix = dirs.shape[0]
        o = np.repeat(origin[None, :], npix, axis=0)
        p, n, valid, t = wide_intersect(o, dirs)
        rho = np.zeros((npix, 3), np.float32)
        for k in range(npix):
            if valid[k]:
                rho[k] = world.albedo[classify_material(p[k], n[k], world.wall_height)]
        # spec M3: re-attribution — a metal first hit is view-dependent, so drop it as a diffuse
        # tuple (aov_valid=0, aov_spec=1). Diffuse coverage of that wall comes from other cams; the
        # metal's light on nearby walls is captured by diffuse paths that bounce onto it (below).
        spec_hit = np.zeros(npix, bool)
        if metal_cls is not None:
            ism, _f0 = metal_cls(p)
            spec_hit = ism & valid
            valid = valid & ~ism
        A_pos.append(p.astype(np.float16)); A_n.append(n.astype(np.float16))
        A_rho.append(rho.astype(np.float16))
        A_valid.append(valid.astype(np.uint8)); A_cam.append(np.full(npix, ci, np.int32))
        A_spec.append(spec_hit.astype(np.uint8))

        # ---- light-agnostic sub-path per pixel: gatherable vertices with throughput ----
        pix_ids = global_pix + np.arange(npix)
        alive = valid.copy()
        pos_k = p.copy(); n_k = n.copy(); rho_k = rho.copy()
        upstream = np.ones((npix, 3), np.float32)   # prod_{j<k} albedo_j
        for dk in range(depth):
            a = np.where(alive)[0]
            if a.size == 0:
                break
            # record vertex k (gatherable): diffuse -> upstream*albedo/pi ; conductor (spec M3) ->
            # upstream*f0 (Fresnel tint, no 1/pi) so metal-mediated light lands tinted on walls.
            if metal_cls is not None:
                ism_a, f0_a = metal_cls(pos_k[a])
                seg_thr = np.where(ism_a[:, None], upstream[a] * f0_a, upstream[a] * (rho_k[a] / np.pi))
                refl_a = np.where(ism_a[:, None], f0_a, rho_k[a])
            else:
                seg_thr = upstream[a] * (rho_k[a] / np.pi)
                refl_a = rho_k[a]
            V_pix.append(pix_ids[a]); V_pos.append(pos_k[a].astype(np.float16))
            V_n.append(n_k[a].astype(np.float16)); V_thr.append(seg_thr.astype(np.float32))
            # extend path by cosine-sampled bounce
            newdir = cosine_sample(n_k[a], rng)
            offo = pos_k[a] + n_k[a] * 1e-3
            p2, n2, v2, t2 = wide_intersect(offo, newdir)
            upstream[a] = upstream[a] * refl_a       # diffuse: albedo; conductor: f0 tint
            # Russian roulette after rr_start
            surv = np.ones(a.size, bool)
            if dk + 1 >= rr_start:
                q = np.clip(upstream[a].max(axis=1), 0.05, 0.95)
                u = rng.random(a.size)
                surv = u < q
                upstream[a] = np.where(surv[:, None], upstream[a] / q[:, None], upstream[a])
            cont = v2 & surv
            # write back
            full_alive = np.zeros(npix, bool)
            idx = a[cont]
            full_alive[idx] = True
            pos_k[a[cont]] = p2[cont]; n_k[a[cont]] = n2[cont]
            # albedo at continued vertices
            for j in idx:
                rho_k[j] = world.albedo[classify_material(pos_k[j], n_k[j], world.wall_height)]
            alive = full_alive
        global_pix += npix

    # -------- assemble SoA + CSR offsets (per pixel) --------
    aov_pos = np.concatenate(A_pos); aov_n = np.concatenate(A_n); aov_rho = np.concatenate(A_rho)
    aov_valid = np.concatenate(A_valid); aov_cam = np.concatenate(A_cam)
    aov_spec = np.concatenate(A_spec)
    total_pix = aov_pos.shape[0]

    if V_pix:
        vp = np.concatenate(V_pix); vpos = np.concatenate(V_pos)
        vn = np.concatenate(V_n); vthr = np.concatenate(V_thr)
        order = np.argsort(vp, kind="stable")
        vp = vp[order]; vpos = vpos[order]; vn = vn[order]; vthr = vthr[order]
        counts = np.bincount(vp, minlength=total_pix)
    else:
        vpos = np.zeros((0, 3), np.float16); vn = np.zeros((0, 3), np.float16)
        vthr = np.zeros((0, 3), np.float32); counts = np.zeros(total_pix, np.int64)
    seg_offsets = np.zeros(total_pix + 1, np.int64)
    seg_offsets[1:] = np.cumsum(counts)

    out = os.path.join(run_dir(run), "trace")
    schemas.write_blob(
        out, "trace", seed, schemas.TRACE_FIELDS,
        {
            "aov_pos": aov_pos, "aov_n": aov_n, "aov_rho": aov_rho,
            "aov_valid": aov_valid, "aov_cam": aov_cam, "aov_spec": aov_spec,
            "seg_pos": vpos, "seg_n": vn, "seg_thr": vthr,
            "seg_offsets": seg_offsets,
        },
        provenance={
            "stage": "stage1_trace", "profile": profile, "git": git_rev(),
            "n_cameras": len(poses), "res": res, "spp": spp, "depth": depth,
            "metal": metal_on, "orbit_cams_per_obj": n_orbit,
            "cameras": [{"origin": o.tolist(), "target": t.tolist()} for (o, t) in poses],
        },
    )
    dt = now() - t0
    Ledger(run).stage("stage1_trace", t_wall=dt, pixels=int(total_pix),
                      valid=int(aov_valid.sum()), segments=int(vpos.shape[0]),
                      cameras=n_cam, res=res)
    print(f"[stage1] {n_cam} cams x {res}^2 -> {total_pix} px, "
          f"{aov_valid.sum()} valid, {vpos.shape[0]} segments in {dt:.1f}s -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="micro")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    trace(a.profile, a.run)
