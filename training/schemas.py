"""schemas.py — versioned SoA blob definitions. THE interface between stages.

Data-driven principle #2/#4: stages communicate ONLY through these memory-mapped
Structure-of-Arrays blobs. No per-path Python objects cross a stage boundary.

A "blob" here = a directory containing:
  - meta.json         : {schema, version, seed, counts, dtype map, provenance}
  - <field>.npy       : one memory-mappable array per SoA field

Every writer stamps meta.json; every reader asserts (schema, version) before use.
Bump SCHEMA_VERSION and document in SCHEMA.md on any field change.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

SCHEMA_VERSION = 3  # v2: table `target_lineage`; v3: trace `aov_spec` (metal first-hit flag, spec M3)

# ----------------------------------------------------------------------------
# Field dtype contracts (SoA). fp16 for geometry, fp32 for radiometry accum.
# ----------------------------------------------------------------------------

# Stage 1 output: per-camera light-agnostic path dump.
# Paths are stored as flat segment vertices with CSR-style offsets per pixel.
TRACE_FIELDS = {
    # first-hit AOVs, one row per traced pixel (the surface point we will supervise)
    "aov_pos":   ("f2", 3),   # x: world-space first-hit position   (fp16)
    "aov_n":     ("f2", 3),   # n: shading normal at first hit       (fp16)
    "aov_rho":   ("f2", 3),   # rho: diffuse albedo at first hit     (fp16)
    "aov_valid": ("u1", 1),   # 1 if pixel hit geometry AND that hit is diffuse (spec M3: metal→0)
    "aov_cam":   ("i4", 1),   # camera index that produced this pixel
    "aov_spec":  ("u1", 1),   # v3: 1 if the first hit was a conductor (metal) surface, else 0
    # path segments (flattened across all pixels), CSR-indexed by seg_offsets
    "seg_pos":   ("f2", 3),   # vertex position along the light-transport path
    "seg_n":     ("f2", 3),   # vertex shading normal (for cos-to-light + shadow offset)
    "seg_thr":   ("f4", 3),   # throughput to this vertex INCLUDING local diffuse BRDF
                              #   = (prod_{j<k} albedo_j) * (albedo_k / pi)
    # CSR offsets: pixel p owns segments [seg_offsets[p], seg_offsets[p+1])
    "seg_offsets": ("i8", 1),
}

# Stage 2 output (cpu profile): pre-gathered flat training table.
# One row = one supervised (surface point, light config) tuple. Plan §2.2.
TABLE_FIELDS = {
    "x":      ("f2", 3),   # surface position
    "n":      ("f2", 3),   # normal
    "rho":    ("f2", 3),   # albedo
    "v":      ("f2", 3),   # light (torch) position — the 3D light param
    "target": ("f4", 3),   # denoised RGB contribution (ground-truth-ish label)
}
# v2: the table's meta.json provenance additionally carries
#   "target_lineage": one of {"raw", "bilateral", "atrous", "avg8", "oidn"}
# recording HOW the target labels were denoised. Stage 3 refuses to train the
# hashgrid when the lineage is not genuinely denoised and require_clean_targets
# is set (fixplan P1 — the data-enforced coupling rule).

# Stage 3 output: trained model bundle (weights .npz + meta). Not SoA; documented for completeness.
# Stage 4 output: GLSL + atlas .png + atlas meta. Documented in SCHEMA.md.


def emitter_radiance(light: dict) -> np.ndarray:
    """corrections C1 (D1) — the ONE definition of a light's emitter radiance (linear RGB).

    Root cause of D1 (orb whiteout): the analytic light-hit branch was scaling by a
    network-contribution term, not the emitter's radiance. This is the single source consumed by all
    three: (1) Stage-2 gather (target intensity), (2) network output scaling, (3) the shader / runtime
    light-hit path, which composites `L += tint * F(cosθ) * emitter_radiance(light)` — Fresnel-tinted
    so the torch seen at a grazing copper reflection reads copper, not white.

    For a Mitsuba point emitter, radiant `intensity` IS the value used here; `radiance_rgb` in
    level.toml pins it explicitly. Falls back to `intensity` if `radiance_rgb` is absent.
    """
    v = light.get("radiance_rgb", light.get("intensity"))
    return np.asarray(v, dtype=np.float32)


@dataclass
class BlobMeta:
    schema: str
    version: int
    seed: int
    counts: dict[str, int] = field(default_factory=dict)
    dtypes: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {
            "schema": self.schema,
            "version": self.version,
            "seed": self.seed,
            "counts": self.counts,
            "dtypes": self.dtypes,
            "provenance": self.provenance,
        }


def _np_dtype(code: str) -> np.dtype:
    return np.dtype(code)


def write_blob(path: str, schema: str, seed: int, fields: dict, arrays: dict[str, np.ndarray],
               provenance: dict | None = None) -> None:
    """Write a SoA blob directory. `fields` is one of the *_FIELDS contracts."""
    os.makedirs(path, exist_ok=True)
    counts: dict[str, int] = {}
    dtypes: dict[str, Any] = {}
    for name, (code, ncomp) in fields.items():
        if name not in arrays:
            raise KeyError(f"blob {schema}: missing field '{name}'")
        arr = np.ascontiguousarray(arrays[name], dtype=_np_dtype(code))
        want_shape = (arr.shape[0],) if ncomp == 1 else (arr.shape[0], ncomp)
        if ncomp == 1 and arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if ncomp != 1 and (arr.ndim != 2 or arr.shape[1] != ncomp):
            raise ValueError(f"blob {schema}: field '{name}' expected {ncomp} comps, got {arr.shape}")
        np.save(os.path.join(path, f"{name}.npy"), arr)
        counts[name] = int(arr.shape[0])
        dtypes[name] = [code, ncomp]
    meta = BlobMeta(schema=schema, version=SCHEMA_VERSION, seed=seed,
                    counts=counts, dtypes=dtypes, provenance=provenance or {})
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta.to_json(), f, indent=2)


def read_blob(path: str, schema: str, mmap: bool = True) -> tuple[dict[str, np.ndarray], BlobMeta]:
    """Read a SoA blob directory back. Asserts schema + version match."""
    with open(os.path.join(path, "meta.json")) as f:
        m = json.load(f)
    if m["schema"] != schema:
        raise ValueError(f"blob at {path}: schema '{m['schema']}' != expected '{schema}'")
    if m["version"] != SCHEMA_VERSION:
        raise ValueError(f"blob at {path}: version {m['version']} != {SCHEMA_VERSION}")
    meta = BlobMeta(schema=m["schema"], version=m["version"], seed=m["seed"],
                    counts=m["counts"], dtypes=m["dtypes"], provenance=m.get("provenance", {}))
    arrays: dict[str, np.ndarray] = {}
    for name in m["counts"]:
        fp = os.path.join(path, f"{name}.npy")
        arrays[name] = np.load(fp, mmap_mode="r" if mmap else None)
    return arrays, meta
