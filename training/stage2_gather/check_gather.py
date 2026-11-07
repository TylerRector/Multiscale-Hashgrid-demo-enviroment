"""Acceptance: our light-agnostic gather ~= Mitsuba render of the same (pose, light) (plan M1).

Compares direct+1-bounce gather from stored segments against a Mitsuba `path` render with an
actual point light at the same position. Reports relative L1 over lit pixels.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mitsuba as mi

import schemas
from common import load_world, load_config, run_dir
from level.exporter import load_scene, add_light_and_sensor
from stage2_gather.gather import gather_pixels_for_light


def main(profile, run):
    mi.set_variant("llvm_ad_rgb")
    cfg = load_config(profile)
    world = load_world(cfg["run"]["level"])
    scene = load_scene(world)
    tr, tmeta = schemas.read_blob(os.path.join(run_dir(run), "trace"), "trace")
    res = tmeta.provenance["res"]
    cams = tmeta.provenance["cameras"]
    aov_cam = np.asarray(tr["aov_cam"]); aov_valid = np.asarray(tr["aov_valid"])
    seg_pos = np.asarray(tr["seg_pos"]); seg_n = np.asarray(tr["seg_n"])
    seg_thr = np.asarray(tr["seg_thr"]); seg_off = np.asarray(tr["seg_offsets"])
    npix = aov_cam.shape[0]
    counts = np.diff(seg_off); pix_of_seg = np.repeat(np.arange(npix), counts)

    ci = 0
    origin = np.array(cams[ci]["origin"], np.float32)
    target = np.array(cams[ci]["target"], np.float32)
    # place a torch light at the camera (headlamp) — guaranteed visible, strong signal
    v = origin.copy()
    intensity = np.array(world.torch["intensity"], np.float32)

    cam_pix = np.where(aov_cam == ci)[0]
    seg_sel = np.isin(pix_of_seg, cam_pix)
    sidx = np.where(seg_sel)[0]
    local = np.searchsorted(cam_pix, pix_of_seg[sidx])
    acc = gather_pixels_for_light(scene, seg_pos[sidx], seg_n[sidx], seg_thr[sidx],
                                  local, cam_pix.size, v, intensity, 262144)
    ours = acc.reshape(res, res, 3)

    ref_scene = add_light_and_sensor(world, v, intensity, origin, target, res=res, spp=64)
    img = np.array(mi.render(ref_scene, spp=64)).reshape(res, res, 3)

    lit = (img.mean(axis=2) > 1e-3) & (aov_valid[cam_pix].reshape(res, res) > 0)
    rel = np.abs(ours[lit] - img[lit]).sum() / (np.abs(img[lit]).sum() + 1e-6)
    print(f"[check] pose {ci} headlamp gather vs Mitsuba: rel-L1 = {rel:.3f} "
          f"(ours mean {ours[lit].mean():.3f}, ref mean {img[lit].mean():.3f}) over {lit.sum()} lit px")
    print("[check] PASS (< 0.5, MC + missing-multibounce tolerance)" if rel < 0.5 else "[check] investigate")
    return rel


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--profile", default="micro"); ap.add_argument("--run", required=True)
    a = ap.parse_args(); main(a.profile, a.run)
