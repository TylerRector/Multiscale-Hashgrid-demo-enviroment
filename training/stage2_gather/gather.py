"""stage2_gather/gather.py — GATHERLIGHT + denoise -> pre-gathered training table (plan Stage 2).

Light-agnostic gather: for a sampled light position v, evaluate the contribution at every
supervised surface point by summing the stored path segments and testing visibility with
shadow rays. Works for ANY v without re-tracing (the paper's key decoupling, plan §1).

Light sampler = walkable volume at torch height  U  static-light ranges (plan Stage 2;
dodges the Fig.-14 cavity failure). CI asserts sampler ⊆ walkable mask.

cpu profile: emits the flat pre-gathered table (schema 'table'). Chunked shadow rays keep VRAM/RAM bounded.
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
from level.exporter import load_scene


def _np3(x):
    a = np.array(x)
    return a.T if a.shape[0] == 3 else a


class LightSampler:
    """Uniform over walkable volume at torch height, unioned with static-light positions."""

    def __init__(self, world: World, rng: np.random.Generator, wall_margin: float = 0.0):
        self.world = world
        self.rng = rng
        self.wall_margin = float(wall_margin)   # fixplan P4b: min torch-wall clearance
        self.cells = world.walkable_cells()
        self.centers = np.array([world.cell_center_world(r, c) for (r, c) in self.cells])
        self.z0 = world.light_domain.get("z_min", world.eye_height - 0.05)
        self.z1 = world.light_domain.get("z_max", world.eye_height + 0.05)
        self.static = [np.array(s["pos"], np.float32) for s in world.static_lights]
        self.static_I = [np.array(s["intensity"], np.float32) for s in world.static_lights]
        self.torch_I = np.array(world.torch["intensity"], np.float32)

    def sample(self, near=None, radius=2.0):
        """Return (pos(3), intensity(3), is_torch).

        ~25% authored static light (global). Otherwise the TORCH: bound to the player, so it is
        sampled in the walkable volume NEAR `near` (the camera origin being gathered) within
        `radius` cells — matching gameplay and keeping the training distribution in the
        torch-close-to-player regime the game actually renders. Falls back to global if no `near`.
        """
        if self.static and self.rng.random() < 0.25:
            k = self.rng.integers(len(self.static))
            return self.static[k].astype(np.float32), self.static_I[k], False
        y = self.z0 + self.rng.random() * (self.z1 - self.z0)
        if near is not None:
            for _ in range(16):  # rejection-sample a walkable torch pose near the player
                ang = self.rng.random() * 2 * np.pi
                rad = radius * self.world.cell * np.sqrt(self.rng.random())
                x = near[0] + rad * np.cos(ang); z = near[2] + rad * np.sin(ang)
                cand = np.array([x, y, z], np.float32)
                if self.in_walkable(cand):
                    return cand, self.torch_I, True
        # fallback: uniform over walkable centers
        ci = self.rng.integers(len(self.centers))
        x, z = self.centers[ci]
        x += (self.rng.random() - 0.5) * 0.6 * self.world.cell
        z += (self.rng.random() - 0.5) * 0.6 * self.world.cell
        return np.array([x, y, z], np.float32), self.torch_I, True

    def in_walkable(self, pos) -> bool:
        c = int(pos[0] // self.world.cell); r = int(pos[2] // self.world.cell)
        if self.world.is_wall(r, c):
            return False
        m = self.wall_margin
        if m <= 0.0:
            return True
        # P4b: require `m` clearance from any wall — the eroded walkable mask. Test the
        # margin box corners (an axis-aligned tile world: corners bound the box).
        for dx in (-m, m):
            for dz in (-m, m):
                cc = int((pos[0] + dx) // self.world.cell)
                rr = int((pos[2] + dz) // self.world.cell)
                if self.world.is_wall(rr, cc):
                    return False
        return True


def bilateral_denoise(img, albedo, normal, sigma_s=1.5, sigma_r=0.15):
    """Joint-bilateral fallback for OIDN (plan risk: OIDN unavailable). img,albedo,normal: (H,W,3)."""
    H, W, _ = img.shape
    out = np.zeros_like(img)
    wsum = np.zeros((H, W, 1), np.float32)
    rad = 2
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            sy0, sy1 = max(0, dy), H + min(0, dy)
            dy0, dy1 = max(0, -dy), H + min(0, -dy)
            sx0, sx1 = max(0, dx), W + min(0, dx)
            dx0, dx1 = max(0, -dx), W + min(0, -dx)
            ga = albedo[dy0:dy1, dx0:dx1]; gn = normal[dy0:dy1, dx0:dx1]
            ca = albedo[sy0:sy1, sx0:sx1]; cn = normal[sy0:sy1, sx0:sx1]
            wr = np.exp(-(np.sum((ga - ca) ** 2, axis=2, keepdims=True)) / (2 * sigma_r ** 2))
            wn = np.clip(np.sum(gn * cn, axis=2, keepdims=True), 0, 1)
            ws = np.exp(-(dx * dx + dy * dy) / (2 * sigma_s ** 2))
            w = wr * wn * ws
            out[sy0:sy1, sx0:sx1] += w * img[dy0:dy1, dx0:dx1]
            wsum[sy0:sy1, sx0:sx1] += w
    return out / np.maximum(wsum, 1e-6)


def atrous_denoise(img, albedo, normal, pos, valid, iters=5,
                   sig_n=0.5, sig_a=0.08, sig_p=0.5):
    """Edge-avoiding a-trous wavelet denoise (SVGF-style), guided by albedo+normal+position.

    fixplan P1: the 1-path-per-pixel targets carry heavy MC noise that the weak 5x5 bilateral
    leaves in; the hashgrid then memorizes it (the world-locked mottling). This multi-scale
    edge-stopping filter genuinely removes it while preserving geometry/albedo edges. Background
    (invalid) pixels are masked out so black does not bleed onto lit surfaces.
    img/albedo/normal/pos: (H,W,3); valid: (H,W) bool.
    """
    H, W, _ = img.shape
    m = valid.astype(np.float32)[:, :, None]
    out = img * m
    kern = np.array([1.0, 4.0, 6.0, 4.0, 1.0], np.float32)  # B3-spline a-trous kernel
    for it in range(iters):
        step = 1 << it
        acc = np.zeros_like(out)
        wsum = np.zeros((H, W, 1), np.float32)
        for ky in range(-2, 3):
            for kx in range(-2, 3):
                dy, dx = ky * step, kx * step
                sy0, sy1 = max(0, dy), H + min(0, dy)
                dy0, dy1 = max(0, -dy), H + min(0, -dy)
                sx0, sx1 = max(0, dx), W + min(0, dx)
                dx0, dx1 = max(0, -dx), W + min(0, -dx)
                if sy1 <= sy0 or sx1 <= sx0:
                    continue
                gc = out[dy0:dy1, dx0:dx1]; ga = albedo[dy0:dy1, dx0:dx1]
                gn = normal[dy0:dy1, dx0:dx1]; gp = pos[dy0:dy1, dx0:dx1]
                cc = out[sy0:sy1, sx0:sx1]; ca = albedo[sy0:sy1, sx0:sx1]
                cn = normal[sy0:sy1, sx0:sx1]; cp = pos[sy0:sy1, sx0:sx1]
                mv = m[dy0:dy1, dx0:dx1]
                wa = np.exp(-np.sum((ga - ca) ** 2, axis=2, keepdims=True) / (2 * sig_a ** 2))
                wn = np.clip(np.sum(gn * cn, axis=2, keepdims=True), 0, 1) ** (1.0 / max(sig_n, 1e-3))
                wp = np.exp(-np.sum((gp - cp) ** 2, axis=2, keepdims=True) / (2 * sig_p ** 2))
                h = kern[ky + 2] * kern[kx + 2]
                w = h * wa * wn * wp * mv
                acc[sy0:sy1, sx0:sx1] += w * gc
                wsum[sy0:sy1, sx0:sx1] += w
        out = np.where(wsum > 1e-8, acc / np.maximum(wsum, 1e-8), out)
    return out * m


def gather_pixels_for_light(scene, seg_pos, seg_n, seg_thr, pix_of_seg, npix, v, intensity,
                            chunk):
    """Vectorized light-agnostic gather with chunked shadow rays. Returns per-pixel RGB (npix,3)."""
    acc = np.zeros((npix, 3), np.float32)
    S = seg_pos.shape[0]
    for c0 in range(0, S, chunk):
        c1 = min(S, c0 + chunk)
        sp = seg_pos[c0:c1].astype(np.float32)
        sn = seg_n[c0:c1].astype(np.float32)
        thr = seg_thr[c0:c1]
        to = v[None, :] - sp
        d2 = np.sum(to * to, axis=1) + 1e-6
        dist = np.sqrt(d2)
        wdir = to / dist[:, None]
        cosL = np.clip(np.sum(sn * wdir, axis=1), 0, None)
        # shadow ray from surface point to light
        offo = sp + sn * 1e-3
        ray = mi.Ray3f(mi.Point3f(offo[:, 0], offo[:, 1], offo[:, 2]),
                       mi.Vector3f(wdir[:, 0], wdir[:, 1], wdir[:, 2]))
        ray.maxt = dist - 2e-3
        occluded = np.array(scene.ray_test(ray))
        vis = (~occluded).astype(np.float32)
        contrib = thr * (cosL * vis / d2)[:, None] * intensity[None, :]
        np.add.at(acc, pix_of_seg[c0:c1], contrib)
    return acc


def gather(profile: str, run: str, config_path="config/config.toml"):
    mi.set_variant("llvm_ad_rgb")
    cfg = load_config(profile, config_path)
    seed = cfg["run"]["seed"]
    s2 = cfg["stage2"]
    world = load_world(cfg["run"]["level"])
    scene = load_scene(world)
    rng = np.random.default_rng(seed + 7)

    trace_dir = os.path.join(run_dir(run), "trace")
    tr, tmeta = schemas.read_blob(trace_dir, "trace")
    aov_pos = np.asarray(tr["aov_pos"]); aov_n = np.asarray(tr["aov_n"])
    aov_rho = np.asarray(tr["aov_rho"]); aov_valid = np.asarray(tr["aov_valid"])
    aov_cam = np.asarray(tr["aov_cam"])
    seg_pos = np.asarray(tr["seg_pos"]); seg_n = np.asarray(tr["seg_n"])
    seg_thr = np.asarray(tr["seg_thr"]); seg_off = np.asarray(tr["seg_offsets"])
    npix = aov_pos.shape[0]
    res = tmeta.provenance["res"]

    # segment -> pixel map from CSR offsets
    counts = np.diff(seg_off)
    pix_of_seg = np.repeat(np.arange(npix), counts)

    fix = cfg.get("fix", {})
    # P4b: torch-wall clearance (config override wins over the level default)
    wall_margin = float(fix.get("light_wall_margin",
                                world.light_domain.get("light_wall_margin", 0.0)))
    sampler = LightSampler(world, rng, wall_margin=wall_margin)
    n_pairs = s2["light_samples"]
    chunk = s2.get("gather_chunk", 262144)
    denoise = s2.get("denoise", "bilateral")
    if denoise == "oidn":
        try:
            import oidn  # noqa
        except Exception:
            print("[stage2] OIDN unavailable -> torch/numpy joint-bilateral fallback")
            denoise = "bilateral"
    # P1: strong edge-avoiding a-trous denoise of the noisy 1-spp targets.
    clean = bool(fix.get("clean_supervision", False))
    lineage = "atrous" if clean else ("bilateral" if denoise == "bilateral" else denoise)
    if wall_margin > 0:
        print(f"[stage2] P4b light-wall margin = {wall_margin:.3f} (eroded walkable sampler)")
    if clean:
        print("[stage2] P1 clean supervision: a-trous edge-avoiding denoise -> lineage 'atrous'")

    cam_ids = np.unique(aov_cam)
    cam_origins = {i: np.array(c["origin"], np.float32) for i, c in enumerate(tmeta.provenance["cameras"])}
    # coverage accumulator over level footprint
    cols, rows = world.cols, world.rows
    cov = np.zeros((rows * 4, cols * 4), np.float32)

    rows_x, rows_n, rows_rho, rows_v, rows_t = [], [], [], [], []
    t0 = now()
    for pair in range(n_pairs):
        ci = int(cam_ids[rng.integers(len(cam_ids))])
        v, intensity, is_torch = sampler.sample(near=cam_origins.get(ci))
        if is_torch:  # CI invariant: walkable-volume sampler ⊆ walkable mask (plan Stage 2)
            assert sampler.in_walkable(v), f"torch sample {v} not in walkable mask"
        mask = (aov_cam == ci) & (aov_valid > 0)
        pidx = np.where(mask)[0]
        if pidx.size == 0:
            continue
        # gather over this camera's segments only
        seg_sel = np.isin(pix_of_seg, pidx)
        sidx = np.where(seg_sel)[0]
        local_pix = np.searchsorted(pidx, pix_of_seg[sidx])
        acc = gather_pixels_for_light(
            scene, seg_pos[sidx], seg_n[sidx], seg_thr[sidx], local_pix, pidx.size, v, intensity, chunk)
        # denoise as a res x res image (guided by albedo + normal AOVs).
        # A camera's pixels are one contiguous res*res block in trace order.
        cam_pix = np.where(aov_cam == ci)[0]
        full = np.zeros((npix, 3), np.float32)
        full[pidx] = acc
        img = full[cam_pix].reshape(res, res, 3).copy()
        alb = aov_rho[cam_pix].astype(np.float32).reshape(res, res, 3)
        nrm = aov_n[cam_pix].astype(np.float32).reshape(res, res, 3)
        if clean:
            xpos = aov_pos[cam_pix].astype(np.float32).reshape(res, res, 3)
            vmask = (aov_valid[cam_pix] > 0).reshape(res, res)
            img = atrous_denoise(img, alb, nrm, xpos, vmask)
        elif denoise == "bilateral":
            img = bilateral_denoise(img, alb, nrm)
        # keep only valid pixels of this camera as training rows
        take = np.where(aov_valid[cam_pix] > 0)[0]
        rows_x.append(aov_pos[cam_pix][take]); rows_n.append(aov_n[cam_pix][take])
        rows_rho.append(aov_rho[cam_pix][take])
        rows_v.append(np.repeat(v[None, :], take.size, axis=0).astype(np.float16))
        rows_t.append(img.reshape(-1, 3)[take].astype(np.float32))
        # coverage
        xs = aov_pos[cam_pix][take][:, 0].astype(np.float32)
        zs = aov_pos[cam_pix][take][:, 2].astype(np.float32)
        gx = np.clip((xs / (cols * world.cell) * cov.shape[1]).astype(int), 0, cov.shape[1] - 1)
        gz = np.clip((zs / (rows * world.cell) * cov.shape[0]).astype(int), 0, cov.shape[0] - 1)
        np.add.at(cov, (gz, gx), 1.0)

    X = np.concatenate(rows_x); Nn = np.concatenate(rows_n); Rho = np.concatenate(rows_rho)
    V = np.concatenate(rows_v); Tg = np.concatenate(rows_t)
    # shuffle on disk order (plan §2.2: shuffled, memory-mapped at train time)
    perm = rng.permutation(X.shape[0])
    X, Nn, Rho, V, Tg = X[perm], Nn[perm], Rho[perm], V[perm], Tg[perm]

    out = os.path.join(run_dir(run), "table")
    schemas.write_blob(out, "table", seed, schemas.TABLE_FIELDS,
                       {"x": X, "n": Nn, "rho": Rho, "v": V, "target": Tg},
                       provenance={"stage": "stage2_gather", "profile": profile, "git": git_rev(),
                                   "pairs": n_pairs, "denoise": denoise, "rows": int(X.shape[0]),
                                   "target_lineage": lineage, "wall_margin": wall_margin})
    # save coverage heatmap
    np.save(os.path.join(run_dir(run), "coverage.npy"), cov)
    dt = now() - t0
    frac_cov = float((cov > 0).mean())
    Ledger(run).stage("stage2_gather", t_wall=dt, rows=int(X.shape[0]),
                      pairs=n_pairs, coverage_frac=frac_cov, denoise=denoise)
    print(f"[stage2] {n_pairs} (cam,light) pairs -> {X.shape[0]} table rows, "
          f"coverage {frac_cov*100:.0f}% in {dt:.1f}s -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="micro")
    ap.add_argument("--run", required=True)
    a = ap.parse_args()
    gather(a.profile, a.run)
