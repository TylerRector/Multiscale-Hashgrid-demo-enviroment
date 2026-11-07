"""stage3_train/train.py — the overnight core (plan Stage 3).

Trains N(x,n,rho,v) on the pre-gathered table (cpu) or, on gpu profiles, would consume a live
gathered pool (same schema; not needed for the cpu deliverable). rel-MSE loss with stop-gradient
(paper Eq. 4), cosine LR, fp16 AMP on GPU. Checkpoints (model+opt+rng+iter) every ckpt_every_sec
and on SIGTERM; resumable. Smoke: extrapolates iters/sec after 500 and refuses runs past --max-hours.
"""
from __future__ import annotations

import argparse, json, os, signal, sys, math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import schemas
from common import load_config, load_world, run_dir, Ledger, now, git_rev
from stage3_train.model import NRP

_STOP = {"flag": False}


def _sigterm(*_):
    _STOP["flag"] = True


def rel_mse(pred, target, eps):
    # paper Eq. 4: relative MSE with stop-gradient on the denominator
    denom = pred.detach() ** 2 + eps
    return ((pred - target) ** 2 / denom).mean()


def rel_mse_tonemap(pred, target, eps):
    # fixplan P4a: rel-MSE in the Reinhard-tonemapped domain T(I)=I/(1+I). Keeps Eq.-4 structure
    # (stop-grad denominator) but in the display space validation/bench use, so the network cannot
    # win by inflating brightness to shrink its own denominator (the HDR near-field blowout, C3).
    tp = pred / (1.0 + pred)
    tt = target / (1.0 + target)
    denom = tp.detach() ** 2 + eps
    return ((tp - tt) ** 2 / denom).mean()


def rightsize_grid(tab, N, aabb_min, aabb_max, base_levels, base_log2, min_samples, cap_res):
    """fixplan P2: derive hashgrid capacity from supervision density.

    Bin table samples into the finest spatial grid level and pick the finest resolution at which
    >=90% of occupied cells carry >= min_samples. Also shrink log2_entries toward occupancy and
    drop levels when the finest cell is coarse. Returns the derived (levels, log2, max_res) + stats.
    """
    import math
    M = int(min(N, 400_000))
    ridx = np.random.default_rng(0).integers(0, N, size=M); ridx.sort()
    x = np.asarray(tab["x"][ridx]).astype(np.float32)
    amin = np.array(aabb_min, np.float32); amax = np.array(aabb_max, np.float32)
    x01 = np.clip((x - amin) / np.maximum(amax - amin, 1e-6), 0.0, 1.0)
    chosen = None
    stats = {}
    cand = sorted({cap_res, cap_res // 2, cap_res // 4, cap_res // 8, cap_res // 16, 8, 4},
                  reverse=True)
    for R in cand:
        if R < 4:
            continue
        cell = np.minimum((x01 * R).astype(np.int64), R - 1)
        key = (cell[:, 0] * R + cell[:, 1]) * R + cell[:, 2]
        _, cnt = np.unique(key, return_counts=True)
        frac_ok = float(np.mean(cnt >= min_samples)); n_occ = int(cnt.size)
        stats = {"max_res": R, "occ": n_occ, "frac_ok": frac_ok, "median": float(np.median(cnt))}
        if frac_ok >= 0.90:
            chosen = stats
            break
    if chosen is None:
        chosen = stats  # coarsest tried; report honestly even if < 0.90
    R, n_occ = chosen["max_res"], chosen["occ"]
    log2 = min(base_log2, max(11, int(math.ceil(math.log2(max(2, n_occ * 2))))))
    if R <= 16:
        levels = min(base_levels, 4)
    elif R <= 32:
        levels = min(base_levels, 5)
    elif R <= 64:
        levels = min(base_levels, 6)
    else:
        levels = base_levels
    chosen.update({"levels": levels, "log2": log2, "min_samples": int(min_samples)})
    return chosen


def load_table(run):
    tab, meta = schemas.read_blob(os.path.join(run_dir(run), "table"), "table", mmap=True)
    return tab, meta


def train(profile: str, run: str, config_path="config/config.toml", max_hours=None, snapshot_every=0):
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)
    cfg = load_config(profile, config_path)
    s3 = cfg["stage3"]
    world = load_world(cfg["run"]["level"])
    device = cfg["_profile"].get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[stage3] cuda requested but unavailable -> cpu")
        device = "cpu"
    amp = bool(s3.get("amp", False)) and device == "cuda"
    max_hours = max_hours if max_hours is not None else s3.get("max_hours", 8.0)

    fix = cfg.get("fix", {})
    tab, tmeta = load_table(run)
    N = tab["x"].shape[0]

    # P1: data-enforced coupling — refuse to train the hashgrid on non-denoised targets.
    lineage = tmeta.provenance.get("target_lineage", "raw")
    if bool(fix.get("require_clean_targets", False)) and lineage not in ("atrous", "avg8", "oidn"):
        print(f"[stage3] REFUSING: target_lineage='{lineage}' is not genuinely denoised "
              f"(require_clean_targets=true). Re-gather with [fix].clean_supervision.")
        sys.exit(4)

    # AABB from level footprint (+ torch height band) for input normalization
    cols, rows = world.cols, world.rows
    aabb_min = (0.0, 0.0, 0.0)
    aabb_max = (cols * world.cell, world.wall_height, rows * world.cell)

    # grid geometry defaults (fixplan P2/P3 can derive/override these)
    hg_levels = s3["hashgrid_levels"]; hg_log2 = s3["hashgrid_log2_entries"]
    hg_feat = s3["hashgrid_features"]; hg_max_res = 128
    hg_feat_init = float(fix.get("grid_feature_init", 1e-4))
    grid_wd = float(fix.get("grid_weight_decay", 0.0))
    min_samp = int(fix.get("min_samples_per_cell", 16))
    grid_stats = None
    if bool(fix.get("grid_rightsize", False)):
        grid_stats = rightsize_grid(tab, N, aabb_min, aabb_max, hg_levels, hg_log2,
                                    min_samp, cap_res=hg_max_res)
        hg_levels = grid_stats["levels"]; hg_log2 = grid_stats["log2"]; hg_max_res = grid_stats["max_res"]
        print(f"[stage3] P2 rightsize: finest_res {hg_max_res}, levels {hg_levels}, "
              f"log2_entries {hg_log2} | occupied {grid_stats['occ']} cells, "
              f"{grid_stats['frac_ok']*100:.0f}% have >= {min_samp} samples (median {grid_stats['median']:.0f})")
        assert grid_stats["frac_ok"] >= 0.90, (
            f"P2 invariant: only {grid_stats['frac_ok']*100:.0f}% of active finest cells reach "
            f"{min_samp} samples (< 90%)")

    torch.manual_seed(cfg["run"]["seed"])
    model = NRP(net=s3["net"], hg_levels=hg_levels, hg_log2=hg_log2, hg_feat=hg_feat,
                aabb_min=aabb_min, aabb_max=aabb_max, hg_max_res=hg_max_res,
                hg_feat_init=hg_feat_init).to(device)
    if grid_wd > 0.0:
        # P3: weight decay on the hashgrid feature table ONLY (MLP excluded).
        grid_p = [model.hash.tables]
        other_p = [p for n, p in model.named_parameters() if not n.startswith("hash.")]
        opt = torch.optim.Adam([{"params": other_p, "weight_decay": 0.0},
                                {"params": grid_p, "weight_decay": grid_wd}], lr=s3["lr"])
        print(f"[stage3] P3 grid weight_decay={grid_wd:g} (hashgrid table only), feat_init={hg_feat_init:g}")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=s3["lr"])
    loss_fn = rel_mse_tonemap if bool(fix.get("tonemap_loss", False)) else rel_mse
    if loss_fn is rel_mse_tonemap:
        print("[stage3] P4a loss domain: rel-MSE on Reinhard-tonemapped pred/target")
    iters = s3["iters"]; batch = s3["batch"]; eps = s3["relmse_eps"]
    lr0 = s3["lr"]; lrf = s3["lr"] * s3["lr_final_frac"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    ckpt_path = os.path.join(run_dir(run), "ckpt.pt")
    start = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start = ck["iter"]; torch.set_rng_state(ck["rng"])
        print(f"[stage3] resumed from iter {start}")

    rng = np.random.default_rng(cfg["run"]["seed"] + 99)

    def get_batch():
        idx = rng.integers(0, N, size=batch)
        idx.sort()
        def g(name):
            return torch.from_numpy(np.asarray(tab[name][idx]).astype(np.float32)).to(device)
        return g("x"), g("n"), g("rho"), g("v"), g("target")

    def cos_lr(it):
        t = min(1.0, it / max(1, iters))
        return lrf + 0.5 * (lr0 - lrf) * (1 + math.cos(math.pi * t))

    def save_progress(step):
        if snapshot_every <= 0:
            return
        from pathlib import Path
        dest = Path(run_dir(run)) / "progress" / f"step_{step:06d}" / "model"
        dest.mkdir(parents=True, exist_ok=True)
        np.savez(dest / "model.npz", **{
            k: val.detach().cpu().numpy() for k, val in model.state_dict().items()})
        arch = {"net": s3["net"], "hashgrid": {
            "levels": hg_levels, "log2_entries": hg_log2, "features": hg_feat,
            "res": model.hash.res, "n_entries": model.hash.n_entries, "max_res": hg_max_res},
            "aabb_min": aabb_min, "aabb_max": aabb_max,
            "in_dim": model.mlp[0].in_features, "iters": step,
            "atlas_fp16": bool(fix.get("atlas_fp16", False))}
        (dest / "model.json").write_text(json.dumps(arch, indent=2))

    if snapshot_every > 0:
        save_progress(start)
    progress_log = []

    t0 = now(); last_ckpt = t0; probe_t = None; loss_ema = None
    for it in range(start, iters):
        for pg in opt.param_groups:
            pg["lr"] = cos_lr(it)
        x, n, rho, v, tgt = get_batch()
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", enabled=amp):
            pred = model(x, n, rho, v)
            loss = loss_fn(pred, tgt, eps)
        if amp:
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        else:
            loss.backward(); opt.step()
        if snapshot_every > 0 and ((it + 1) in (1, 10, 50, 100) or
                                   (it + 1) % snapshot_every == 0 or it + 1 == iters):
            save_progress(it + 1)
            progress_log.append({"step": it + 1, "batch_loss": float(loss.item())})
            with open(os.path.join(run_dir(run), "progress", "loss.json"), "w") as f:
                json.dump(progress_log, f, indent=2)
        loss_ema = loss.item() if loss_ema is None else 0.99 * loss_ema + 0.01 * loss.item()

        if it == start + 500:
            probe_t = now()
            ips = 500 / (probe_t - t0)
            proj_h = (iters - start) / ips / 3600
            print(f"[stage3] iters/sec={ips:.1f}, projected {proj_h:.2f}h for {iters} iters")
            if proj_h > max_hours:
                print(f"[stage3] REFUSING: projected {proj_h:.2f}h > --max-hours {max_hours}. "
                      f"Shrink net/iters or use gpu profile.")
                sys.exit(3)
        if it % max(1, iters // 10) == 0:
            print(f"[stage3] it {it}/{iters} loss_ema {loss_ema:.4f} lr {cos_lr(it):.2e}")
        if now() - last_ckpt > s3.get("ckpt_every_sec", 300) or _STOP["flag"]:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "iter": it + 1, "rng": torch.get_rng_state()}, ckpt_path)
            last_ckpt = now()
            if _STOP["flag"]:
                print(f"[stage3] SIGTERM -> checkpointed at iter {it+1}, exiting resumably"); sys.exit(0)

    dt = now() - t0
    # export weights bundle (Stage 4 input)
    out = os.path.join(run_dir(run), "model")
    os.makedirs(out, exist_ok=True)
    sd = {k: val.detach().cpu().numpy() for k, val in model.state_dict().items()}
    np.savez(os.path.join(out, "model.npz"), **sd)
    arch = {"net": s3["net"], "hashgrid": {"levels": hg_levels,
            "log2_entries": hg_log2, "features": hg_feat,
            "res": model.hash.res, "n_entries": model.hash.n_entries, "max_res": hg_max_res},
            "aabb_min": aabb_min, "aabb_max": aabb_max, "in_dim": model.mlp[0].in_features,
            "profile": profile, "iters": iters, "git": git_rev(),
            "target_lineage": lineage, "grid_weight_decay": grid_wd,
            "grid_rightsize": grid_stats, "tonemap_loss": bool(fix.get("tonemap_loss", False)),
            "atlas_fp16": bool(fix.get("atlas_fp16", False))}
    with open(os.path.join(out, "model.json"), "w") as f:
        json.dump(arch, f, indent=2)
    n_params = sum(p.numel() for p in model.parameters())
    Ledger(run).stage("stage3_train", t_wall=dt, iters=iters, final_loss=loss_ema,
                       params=int(n_params), device=device, net=s3["net"],
                       lineage=lineage, hg_levels=hg_levels, hg_log2=hg_log2, hg_max_res=hg_max_res,
                       grid_weight_decay=grid_wd, tonemap_loss=bool(fix.get("tonemap_loss", False)),
                       min_samples_frac=(grid_stats["frac_ok"] if grid_stats else None))
    print(f"[stage3] done {iters} iters in {dt/60:.1f} min, {n_params} params, "
          f"final loss_ema {loss_ema:.4f} -> {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="micro"); ap.add_argument("--run", required=True)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--snapshot-every", type=int, default=0)
    a = ap.parse_args()
    train(a.profile, a.run, max_hours=a.max_hours, snapshot_every=a.snapshot_every)
