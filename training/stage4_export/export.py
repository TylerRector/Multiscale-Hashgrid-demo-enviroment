"""stage4_export/export.py — trained NRP -> fused GLSL 330 fragment shader + hashgrid/weight atlas.

Produces (schema 'export', see SCHEMA.md):
  export/atlas.npy   : float32 texture data (hashgrid tables followed by MLP weights/biases)
  export/export.json : atlas layout, arch, AABB, light cap
  shaders/nrp.frag   : GLSL 330 fragment shader (in-shader hash + trilinear + unrolled MLP)

A numpy reference forward (`nrp_forward_np`) reads the atlas EXACTLY as the shader does; the
parity check (check_parity.py) asserts it matches the torch model, and — if moderngl is present —
that the compiled GLSL matches the numpy forward on a real G-buffer.
"""
from __future__ import annotations

import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from common import run_dir, Ledger, git_rev

# fixed hash primes — MUST match model.HashGrid3D and the GLSL
PRIMES = (1, 2654435761, 805459861)


def _layer_dims(state):
    """Recover MLP layer (in,out) shapes from state_dict weight keys, in order."""
    ws = sorted([k for k in state.files if k.startswith("mlp.") and k.endswith(".weight")],
                key=lambda s: int(s.split(".")[1]))
    dims = []
    for k in ws:
        W = state[k]
        dims.append((W.shape[1], W.shape[0], k))
    return dims


def export(run, config_path="config/config.toml"):
    mdir = os.path.join(run_dir(run), "model")
    with open(os.path.join(mdir, "model.json")) as f:
        arch = json.load(f)
    state = np.load(os.path.join(mdir, "model.npz"))

    levels = arch["hashgrid"]["levels"]; feat = arch["hashgrid"]["features"]
    n_entries = arch["hashgrid"]["n_entries"]; hres = arch["hashgrid"]["res"]
    tables = state["hash.tables"].astype(np.float32)  # (levels, n_entries, feat)

    dims = _layer_dims(state)  # list of (in,out,key)
    # ---- pack a flat float32 atlas: [hashgrid tables] ++ [per-layer W (row-major), b] ----
    parts = [tables.reshape(-1)]
    layer_meta = []
    off = tables.size
    for (din, dout, wkey) in dims:
        lid = wkey.split(".")[1]
        W = state[f"mlp.{lid}.weight"].astype(np.float32)  # (dout, din)
        b = state[f"mlp.{lid}.bias"].astype(np.float32)    # (dout,)
        layer_meta.append({"in": int(din), "out": int(dout),
                           "w_off": int(off), "b_off": int(off + W.size)})
        parts.append(W.reshape(-1)); parts.append(b.reshape(-1))
        off += W.size + b.size
    atlas = np.concatenate(parts).astype(np.float32)

    # fixplan P2: store the atlas features as fp16 (halves the asset; GL loaders upcast on upload).
    fp16 = bool(arch.get("atlas_fp16", False))
    if fp16:
        atlas = atlas.astype(np.float16)
    atlas_dtype = "float16" if fp16 else "float32"
    parity_tol = 5e-2 if fp16 else 1e-3

    exp = {
        "arch": arch, "hashgrid": {"levels": levels, "features": feat, "n_entries": n_entries,
                                    "res": hres, "table_off": 0, "table_len": int(tables.size)},
        "layers": layer_meta, "primes": list(PRIMES),
        "aabb_min": arch["aabb_min"], "aabb_max": arch["aabb_max"],
        "in_dim": arch["in_dim"], "atlas_len": int(atlas.size),
        "atlas_dtype": atlas_dtype, "parity_tol": parity_tol,
        "active_light_cap": 4, "git": git_rev(),
    }
    outdir = os.path.join(run_dir(run), "export"); os.makedirs(outdir, exist_ok=True)
    np.save(os.path.join(outdir, "atlas.npy"), atlas)
    with open(os.path.join(outdir, "export.json"), "w") as f:
        json.dump(exp, f, indent=2)

    frag = codegen_glsl(exp)
    shdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "game", "shaders")
    os.makedirs(shdir, exist_ok=True)
    with open(os.path.join(shdir, "nrp.frag"), "w") as f:
        f.write(frag)
    # also copy shader next to the export for provenance
    with open(os.path.join(outdir, "nrp.frag"), "w") as f:
        f.write(frag)

    size_kb = atlas.nbytes / 1024
    Ledger(run).stage("stage4_export", t_wall=None, atlas_floats=int(atlas.size),
                       atlas_kb=round(size_kb, 1), layers=len(layer_meta))
    print(f"[stage4] atlas {atlas.size} floats ({size_kb:.0f} KB), {len(layer_meta)} MLP layers "
          f"-> {outdir} + game/shaders/nrp.frag")
    return outdir


# --------------------------------------------------------------------------- numpy reference forward

def _hash_np(ix, iy, iz, n_entries):
    h = (ix.astype(np.uint32) * np.uint32(PRIMES[0])) ^ (iy.astype(np.uint32) * np.uint32(PRIMES[1])) \
        ^ (iz.astype(np.uint32) * np.uint32(PRIMES[2]))
    return (h % np.uint32(n_entries)).astype(np.int64)


def nrp_forward_np(exp, atlas, x, n, rho, v):
    """Reference forward reading the packed atlas EXACTLY as the GLSL shader does. Batched (B,3)."""
    aabb_min = np.array(exp["aabb_min"], np.float32); aabb_max = np.array(exp["aabb_max"], np.float32)
    def norm01(p):
        return np.clip((p - aabb_min) / np.maximum(aabb_max - aabb_min, 1e-6), 0, 1)
    hg = exp["hashgrid"]; levels = hg["levels"]; feat = hg["features"]
    n_entries = hg["n_entries"]; res = hg["res"]
    tables = atlas[hg["table_off"]:hg["table_off"] + hg["table_len"]].reshape(levels, n_entries, feat)
    x01 = norm01(x); B = x.shape[0]
    feats = np.zeros((B, levels * feat), np.float32)
    for l in range(levels):
        r = res[l]; p = x01 * (r - 1); p0 = np.floor(p).astype(np.int64); f = p - p0
        acc = np.zeros((B, feat), np.float32)
        for cx in (0, 1):
            for cy in (0, 1):
                for cz in (0, 1):
                    ix = np.clip(p0[:, 0] + cx, 0, r - 1); iy = np.clip(p0[:, 1] + cy, 0, r - 1)
                    iz = np.clip(p0[:, 2] + cz, 0, r - 1)
                    wx = f[:, 0] if cx else 1 - f[:, 0]
                    wy = f[:, 1] if cy else 1 - f[:, 1]
                    wz = f[:, 2] if cz else 1 - f[:, 2]
                    w = (wx * wy * wz)[:, None]
                    idx = _hash_np(ix, iy, iz, n_entries)
                    acc += w * tables[l][idx]
        feats[:, l * feat:(l + 1) * feat] = acc
    vin = norm01(v)
    h = np.concatenate([feats, n.astype(np.float32), rho.astype(np.float32), vin], axis=1)
    for i, lm in enumerate(exp["layers"]):
        W = atlas[lm["w_off"]:lm["w_off"] + lm["in"] * lm["out"]].reshape(lm["out"], lm["in"])
        b = atlas[lm["b_off"]:lm["b_off"] + lm["out"]]
        h = h @ W.T + b
        if i < len(exp["layers"]) - 1:
            h = np.maximum(h, 0.0)
    return np.log1p(np.exp(-np.abs(h))) + np.maximum(h, 0.0)  # numerically-stable softplus


# --------------------------------------------------------------------------- GLSL codegen

def codegen_glsl(exp) -> str:
    hg = exp["hashgrid"]; levels = hg["levels"]; feat = hg["features"]
    n_entries = hg["n_entries"]; res = hg["res"]; in_dim = exp["in_dim"]
    layers = exp["layers"]; cap = exp["active_light_cap"]
    max_w = max(l["out"] for l in layers)
    res_str = ", ".join(str(r) for r in res)
    nL = len(layers)
    woff_str = ", ".join(str(l["w_off"]) for l in layers)
    boff_str = ", ".join(str(l["b_off"]) for l in layers)
    lin_str = ", ".join(str(l["in"]) for l in layers)
    lout_str = ", ".join(str(l["out"]) for l in layers)
    return f"""#version 330 core
// AUTO-GENERATED by stage4_export/export.py — fused world-space NRP shading.
// N(x, n, rho, v) -> RGB, summed over active torch/static lights (linear per light).
// Atlas layout (float32, packed): [hashgrid tables] ++ per-layer [W row-major][b].
in vec3 g_pos;   // world-space first-hit position (DDA G-buffer)
in vec3 g_n;     // shading normal
in vec3 g_rho;   // albedo
out vec4 frag;

uniform sampler2D uAtlas;   // packed weights+tables as an Nx1-ish float texture
uniform int   uAtlasW;      // atlas texture width
uniform vec3  uAABBmin;
uniform vec3  uAABBmax;
uniform int   uNLights;
uniform vec3  uLightPos[{cap}];
uniform vec3  uLightInt[{cap}];

const int LEVELS = {levels};
const int FEAT   = {feat};
const int NENT   = {n_entries};
const int INDIM  = {in_dim};
const int RES[{levels}]   = int[{levels}]({res_str});
const uint P0 = 1u, P1 = 2654435761u, P2 = 805459861u;
const int NLAYERS = {nL};
const int LWOFF[{nL}] = int[{nL}]({woff_str});   // weight offset per layer
const int LBOFF[{nL}] = int[{nL}]({boff_str});   // bias offset per layer
const int LIN[{nL}]   = int[{nL}]({lin_str});    // input dim per layer
const int LOUT[{nL}]  = int[{nL}]({lout_str});   // output dim per layer

float atlas(int i) {{
    return texelFetch(uAtlas, ivec2(i % uAtlasW, i / uAtlasW), 0).r;
}}
uint hash(uint x, uint y, uint z) {{ return ((x*P0) ^ (y*P1) ^ (z*P2)) % uint(NENT); }}

void encode(vec3 x01, out float feat_out[INDIM]) {{
    for (int l = 0; l < LEVELS; ++l) {{
        int r = RES[l];
        vec3 p = x01 * float(r - 1);
        ivec3 p0 = ivec3(floor(p));
        vec3 f = p - vec3(p0);
        float a0 = 0.0, a1 = 0.0;   // FEAT==2 assumed for the atlas RG packing
        for (int c = 0; c < 8; ++c) {{
            ivec3 o = ivec3(c & 1, (c >> 1) & 1, (c >> 2) & 1);
            ivec3 q = clamp(p0 + o, ivec3(0), ivec3(r - 1));
            float w = (o.x==1 ? f.x : 1.0-f.x) * (o.y==1 ? f.y : 1.0-f.y) * (o.z==1 ? f.z : 1.0-f.z);
            uint idx = hash(uint(q.x), uint(q.y), uint(q.z));
            int base = (l * NENT + int(idx)) * FEAT;
            a0 += w * atlas(base);
            a1 += w * atlas(base + 1);
        }}
        feat_out[l*FEAT + 0] = a0;
        feat_out[l*FEAT + 1] = a1;
    }}
}}

vec3 nrp(vec3 x, vec3 nrm, vec3 rho, vec3 v) {{
    vec3 x01 = clamp((x - uAABBmin) / max(uAABBmax - uAABBmin, vec3(1e-6)), 0.0, 1.0);
    vec3 v01 = clamp((v - uAABBmin) / max(uAABBmax - uAABBmin, vec3(1e-6)), 0.0, 1.0);
    float h[INDIM];
    encode(x01, h);
    int k = LEVELS*FEAT;
    h[k+0]=nrm.x; h[k+1]=nrm.y; h[k+2]=nrm.z;
    h[k+3]=rho.x; h[k+4]=rho.y; h[k+5]=rho.z;
    h[k+6]=v01.x; h[k+7]=v01.y; h[k+8]=v01.z;
    float cur[{max_w}]; float nxt[{max_w}];
    for (int i=0;i<INDIM;++i) cur[i]=h[i];
    for (int L=0; L<NLAYERS; ++L) {{
        int din=LIN[L], dout=LOUT[L], woff=LWOFF[L], boff=LBOFF[L];
        for (int o=0;o<dout;++o) {{
            float s = atlas(boff+o);
            for (int i=0;i<din;++i) s += atlas(woff + o*din + i) * cur[i];
            nxt[o] = (L < NLAYERS-1) ? max(s,0.0) : s;
        }}
        for (int o=0;o<dout;++o) cur[o]=nxt[o];
    }}
    // softplus output (non-negative radiance)
    return vec3(log(1.0+exp(-abs(cur[0])))+max(cur[0],0.0),
                log(1.0+exp(-abs(cur[1])))+max(cur[1],0.0),
                log(1.0+exp(-abs(cur[2])))+max(cur[2],0.0));
}}

void main() {{
    vec3 c = vec3(0.0);
    // spec(70).md §2 (M0): N() already carries the per-light intensity E (baked at gather, Eq. 3
    // applies E exactly once). Sum the NRP contributions directly — do NOT multiply by uLightInt
    // again (uLightInt is used only for analytic light hits in the specular chase).
    for (int i=0;i<uNLights && i<{cap}; ++i)
        c += nrp(g_pos, normalize(g_n), g_rho, uLightPos[i]);
    // simple Reinhard tonemap + gamma for display
    c = pow(c/(1.0+c), vec3(1.0/2.2));
    frag = vec4(c, 1.0);
}}
"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True)
    a = ap.parse_args(); export(a.run)
