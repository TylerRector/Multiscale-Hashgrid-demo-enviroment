"""stage3_train/model.py — world-space NRP N(x, n, rho, v) -> RGB contribution.

x -> multiresolution 3D hash encoding (Instant-NGP style, plain torch).
[hash(x), n, rho, v] -> small MLP -> RGB (>=0). Plan §3 / Eq. 3.

Kept deliberately tiny (3x32 floor / 4x64 primary) so the whole model (with hashgrid) is ~0.5 MB
and exports to an unrolled GLSL fragment shader (Stage 4).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# fixed small primes for the spatial hash (Instant-NGP)
_PRIMES = torch.tensor([1, 2654435761, 805459861], dtype=torch.int64)


def net_shape(name: str) -> tuple[int, int]:
    return {"3x32": (3, 32), "4x64": (4, 64), "6x128": (6, 128)}[name]


class HashGrid3D(nn.Module):
    def __init__(self, levels=8, log2_entries=15, features=2, base_res=4, max_res=128,
                 feat_init=1e-4):
        super().__init__()
        self.levels = levels
        self.features = features
        self.n_entries = 1 << log2_entries
        self.max_res = max_res
        b = np.exp((np.log(max_res) - np.log(base_res)) / max(1, levels - 1))
        self.res = [int(round(base_res * (b ** l))) for l in range(levels)]
        # one table per level. feat_init is the U(-a, a) amplitude (fixplan P3 / Instant-NGP hygiene).
        self.tables = nn.Parameter(
            torch.zeros(levels, self.n_entries, features).uniform_(-feat_init, feat_init))

    def _hash(self, ix, iy, iz):
        h = (ix * _PRIMES[0].item()) ^ (iy * _PRIMES[1].item()) ^ (iz * _PRIMES[2].item())
        return (h % self.n_entries).long()

    def forward(self, x01):
        """x01: (B,3) in [0,1]. Returns (B, levels*features) trilinearly-interpolated features."""
        B = x01.shape[0]
        feats = []
        for l in range(self.levels):
            r = self.res[l]
            p = x01 * (r - 1)
            p0 = torch.floor(p).long()
            f = p - p0.float()
            acc = torch.zeros(B, self.features, device=x01.device, dtype=self.tables.dtype)
            for cx in (0, 1):
                for cy in (0, 1):
                    for cz in (0, 1):
                        ix = (p0[:, 0] + cx).clamp(0, r - 1)
                        iy = (p0[:, 1] + cy).clamp(0, r - 1)
                        iz = (p0[:, 2] + cz).clamp(0, r - 1)
                        wx = f[:, 0] if cx else (1 - f[:, 0])
                        wy = f[:, 1] if cy else (1 - f[:, 1])
                        wz = f[:, 2] if cz else (1 - f[:, 2])
                        w = wx * wy * wz
                        idx = self._hash(ix, iy, iz)
                        acc = acc + w.unsqueeze(1) * self.tables[l][idx]
            feats.append(acc)
        return torch.cat(feats, dim=1)


class NRP(nn.Module):
    def __init__(self, net="4x64", hg_levels=8, hg_log2=15, hg_feat=2,
                 aabb_min=(0, 0, 0), aabb_max=(1, 1, 1), hg_max_res=128, hg_feat_init=1e-4):
        super().__init__()
        depth, width = net_shape(net)
        self.hash = HashGrid3D(hg_levels, hg_log2, hg_feat, max_res=hg_max_res, feat_init=hg_feat_init)
        in_dim = hg_levels * hg_feat + 3 + 3 + 3  # hash(x) + n + rho + v
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, 3)]
        self.mlp = nn.Sequential(*layers)
        self.register_buffer("aabb_min", torch.tensor(aabb_min, dtype=torch.float32))
        self.register_buffer("aabb_max", torch.tensor(aabb_max, dtype=torch.float32))
        self.net_name = net

    def norm01(self, p):
        return (p - self.aabb_min) / (self.aabb_max - self.aabb_min).clamp(min=1e-6)

    def forward(self, x, n, rho, v):
        h = self.hash(self.norm01(x).clamp(0, 1))
        vin = self.norm01(v).clamp(0, 1)
        feat = torch.cat([h, n, rho, vin], dim=1)
        out = self.mlp(feat)
        return torch.nn.functional.softplus(out)  # non-negative radiance
