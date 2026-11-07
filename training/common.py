"""common.py — shared helpers: config x profile merge, level parsing, world geometry, ledger.

No stage branches on hardware; stages branch only on config values loaded here (plan §2.1).
"""
from __future__ import annotations

import copy
import json
import os
import time
import tomllib
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- config

def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(profile: str, config_path: str = "config/config.toml") -> dict:
    """Effective config = base config.toml overlaid by profiles/<profile>.toml."""
    with open(os.path.join(ROOT, config_path), "rb") as f:
        base = tomllib.load(f)
    ppath = os.path.join(ROOT, "config", "profiles", f"{profile}.toml")
    with open(ppath, "rb") as f:
        prof = tomllib.load(f)
    cfg = _deep_merge(base, prof)
    cfg["_profile"] = prof.get("profile", {"name": profile})
    return cfg


# --------------------------------------------------------------------------- level / world

@dataclass
class World:
    name: str
    grid: list[str]        # rows of '#'/'.'
    rows: int
    cols: int
    cell: float
    eye_height: float
    wall_height: float
    albedo: dict            # 'wall'/'floor'/'ceil' -> rgb
    torch: dict
    static_lights: list[dict]
    light_domain: dict
    objects: list[dict] = None   # spec(70).md §3: metallic reflective objects (sphere|box)

    def is_wall(self, r: int, c: int) -> bool:
        if r < 0 or c < 0 or r >= self.rows or c >= self.cols:
            return True
        return self.grid[r][c] == "#"

    def walkable_cells(self) -> list[tuple[int, int]]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.grid[r][c] == "."]

    def cell_center_world(self, r: int, c: int) -> tuple[float, float]:
        # world X = col+0.5, world Z = row+0.5 (times cell size)
        return ((c + 0.5) * self.cell, (r + 0.5) * self.cell)

    def bounds_xz(self) -> tuple[float, float, float, float]:
        return (0.0, 0.0, self.cols * self.cell, self.rows * self.cell)


def load_world(level_path: str = "level/level.toml") -> World:
    with open(os.path.join(ROOT, level_path), "rb") as f:
        t = tomllib.load(f)
    m = t["meta"]
    grid = m["map"]
    w = len(grid[0])
    assert all(len(row) == w for row in grid), "level map rows must be equal length"
    return World(
        name=m["name"],
        grid=grid,
        rows=len(grid),
        cols=w,
        cell=float(m["cell_size"]),
        eye_height=float(m["eye_height"]),
        wall_height=float(m["wall_height"]),
        albedo={k: np.array(v, dtype=np.float32) for k, v in t["albedo"].items()},
        torch=t["torch"],
        static_lights=t.get("static_lights", []),
        light_domain=t.get("light_domain", {}),
        objects=t.get("objects", []),
    )


# --------------------------------------------------------------------------- ledger / runs

def run_dir(run_id: str) -> str:
    d = os.path.join(ROOT, "runs", run_id)
    os.makedirs(d, exist_ok=True)
    return d


class Ledger:
    """Per-run ledger: records per-stage wall time + progress units. Plan §2.4."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.path = os.path.join(run_dir(run_id), "ledger.json")
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.data = json.load(f)
        else:
            self.data = {"run_id": run_id, "stages": {}}

    def stage(self, name: str, **metrics) -> None:
        self.data["stages"][name] = {"t_wall": metrics.pop("t_wall", None), **metrics}
        self._flush()

    def _flush(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)


def now() -> float:
    return time.time()


def git_rev() -> str:
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "nogit"
