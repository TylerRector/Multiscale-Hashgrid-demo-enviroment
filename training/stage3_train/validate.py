"""stage3_train/validate.py — held-out (pose, light) validation vs Mitsuba GT (plan M2/M3).

The critical axis is HELD-OUT camera poses (view generalization — the thing the paper never
had to test). We sample NEW poses (unseen in Stage 1), render Mitsuba NEE ground truth, and
compare against the NRP prediction evaluated at each pixel's first-hit surface point.
Reports mean PSNR; writes side-by-side PNGs.
"""
from __future__ import annotations

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import mitsuba as mi

from common import load_config, load_world, run_dir, Ledger
from level.exporter import load_scene, add_light_and_sensor, classify_material
from stage1_trace.trace import blue_noise_poses, primary_dirs, _np3
from stage3_train.model import NRP


def load_model(run, device="cpu"):
    mdir = os.path.join(run_dir(run), "model")
    with open(os.path.join(mdir, "model.json")) as f:
        arch = json.load(f)
    m = NRP(net=arch["net"], hg_levels=arch["hashgrid"]["levels"],
            hg_log2=arch["hashgrid"]["log2_entries"], hg_feat=arch["hashgrid"]["features"],
            aabb_min=tuple(arch["aabb_min"]), aabb_max=tuple(arch["aabb_max"]),
            hg_max_res=arch["hashgrid"].get("max_res", 128))
    sd = np.load(os.path.join(mdir, "model.npz"))
    m.load_state_dict({k: torch.from_numpy(sd[k]) for k in sd.files})
    m.eval().to(device)
    return m, arch


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-8 else 10 * np.log10((max(a.max(), b.max(), 1.0) ** 2) / mse)


def tonemap(x):
    return np.clip(x / (1 + x), 0, 1) ** (1 / 2.2)


def validate(profile, run, config_path="config/config.toml", n_views=None, save_png=True):
    mi.set_variant("llvm_ad_rgb")
    cfg = load_config(profile, config_path)
    world = load_world(cfg["run"]["level"])
    scene = load_scene(world)
    res = cfg["stage1"]["res"]
    n_pairs = n_views if n_views is not None else cfg["stage3"]["validation"]["held_out_pairs"]
    target_db = cfg["stage3"]["validation"]["psnr_target_db"]

    model, arch = load_model(run)
    # HELD-OUT poses: different seed so they are not the Stage 1 training cameras
    rng = np.random.default_rng(cfg["run"]["seed"] + 50051)
    poses = blue_noise_poses(world, max(4, n_pairs // 8), rng, world.eye_height)
    torch_I = np.array(world.torch["intensity"], np.float32)

    psnrs = []
    os.makedirs(os.path.join(run_dir(run), "val"), exist_ok=True)
    done = 0
    for pi, (origin, target) in enumerate(poses):
        # torch carried ahead of the player at torch height (matches gameplay + the training
        # light domain: walkable volume, not a degenerate headlamp exactly at the eye).
        fwd = target - origin; fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
        v = (origin + 0.5 * fwd).astype(np.float32)
        v[1] = world.light_domain.get("z_min", world.eye_height)
        # first-hit AOVs for this held-out view
        dirs = primary_dirs(origin, target, res)
        o = np.repeat(origin[None, :], dirs.shape[0], axis=0).astype(np.float32)
        ray = mi.Ray3f(mi.Point3f(o[:, 0], o[:, 1], o[:, 2]),
                       mi.Vector3f(dirs[:, 0], dirs[:, 1], dirs[:, 2]))
        si = scene.ray_intersect(ray)
        p = _np3(si.p); nrm = _np3(si.sh_frame.n); valid = np.array(si.is_valid())
        rho = np.zeros_like(p)
        for k in range(p.shape[0]):
            if valid[k]:
                rho[k] = world.albedo[classify_material(p[k], nrm[k], world.wall_height)]
        # NRP prediction
        with torch.no_grad():
            pred = model(torch.from_numpy(p.astype(np.float32)),
                         torch.from_numpy(nrm.astype(np.float32)),
                         torch.from_numpy(rho.astype(np.float32)),
                         torch.from_numpy(np.repeat(v[None, :], p.shape[0], 0).astype(np.float32))).numpy()
        pred[~valid] = 0.0
        pred_img = pred.reshape(res, res, 3)
        # Mitsuba GT
        ref = np.array(mi.render(add_light_and_sensor(world, v, torch_I, origin, target, res=res, spp=64),
                                 spp=64)).reshape(res, res, 3)
        ref[~valid.reshape(res, res)] = 0.0
        m = valid.reshape(res, res)
        psnrs.append(psnr(tonemap(pred_img), tonemap(ref)))
        if pi < 2:
            print(f"  view {pi}: pred mean {pred_img[m].mean():.3f}  gt mean {ref[m].mean():.3f}  "
                  f"psnr {psnrs[-1]:.1f} dB")
        if save_png and pi < 4:
            try:
                from PIL import Image
                strip = np.concatenate([tonemap(pred_img), tonemap(ref)], axis=1)
                Image.fromarray((strip * 255).astype(np.uint8)).save(
                    os.path.join(run_dir(run), "val", f"heldout_{pi:02d}.png"))
            except Exception:
                pass
        done += 1
        if done >= n_pairs:
            break

    mean_psnr = float(np.mean(psnrs))
    Ledger(run).stage("stage3_validate", t_wall=None, held_out_views=len(psnrs),
                       mean_psnr_db=mean_psnr, target_db=target_db,
                       pass_=bool(mean_psnr >= target_db))
    verdict = "PASS" if mean_psnr >= target_db else "BELOW TARGET"
    print(f"[validate] held-out poses: mean PSNR {mean_psnr:.1f} dB over {len(psnrs)} views "
          f"(target {target_db} dB) -> {verdict}")
    return mean_psnr


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="micro"); ap.add_argument("--run", required=True)
    ap.add_argument("--views", type=int, default=None)
    a = ap.parse_args()
    validate(a.profile, a.run, n_views=a.views)
