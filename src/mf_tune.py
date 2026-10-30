"""Tuning / reconstruction with the matrix-free operator on already-formed (Poisson or Gaussian) data y."""
import numpy as np
import torch
from .mf_solvers import MFTikhonov, MFTVADMM
from .pnp import pnp_pgd
from .metrics import per_image_metrics

LAM_TIK = np.logspace(-5, 0, 21)
LAM_TV = np.logspace(-3.5, -1, 11)        # Stage-2 dose sweep selected 1.8e-3 .. 3.2e-2 (coarser than v1 to save matrix-free time)
SD = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
CK = (0, 1, 2, 3, 5, 10, 25, 50, 100, 200)


class MFSolvers:
    def __init__(self, op):
        self.op, self.tik, self.tv = op, MFTikhonov(op), MFTVADMM(op)
        self.eta = 1.0 / op.lipschitz()

    def tune(self, den, val, yv, sd_grid=SD, ck=CK):
        psnr = lambda x: per_image_metrics(x.clamp(0, 1), val)[0].mean()
        p = {"Tikhonov": float(max(LAM_TIK, key=lambda l: psnr(self.tik(yv, l)))),
             "TV": float(max(LAM_TV, key=lambda l: psnr(self.tv(yv, l))))}
        best = -1.0
        for sd in sd_grid:
            for k, x in pnp_pgd(self.op, yv, den, sd, self.eta, ck)[0].items():
                s = float(psnr(x))
                if s > best:
                    best, p["PnP"] = s, (sd, k)
        return p

    def reconstruct(self, den, y, p):
        sd, k = p["PnP"]
        return {"FBP": self.op.fbp(y).clamp(0, 1), "Tikhonov": self.tik(y, p["Tikhonov"]), "TV": self.tv(y, p["TV"]),
                "PnP-DnCNN": pnp_pgd(self.op, y, den, sd, self.eta, (k,))[0][k]}
