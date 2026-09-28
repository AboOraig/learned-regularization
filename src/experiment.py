"""Shared helpers: tune all methods on validation data, then reconstruct with FROZEN params."""
import numpy as np
from .baselines import Tikhonov, TVADMM
from .pnp import pnp_pgd
from .metrics import per_image_metrics

LAM_TIK = np.logspace(-5, 0, 21)
LAM_TV = np.logspace(-5, -1, 17)
SD = [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
CK = (0, 1, 2, 3, 5, 10, 25, 50, 100, 200, 400)


def tune_all(op, den, val, yv):
    """Pick hyperparameters maximizing mean validation PSNR. Returns dict of params."""
    eta = 1.0 / op.lipschitz()
    tik, tv = Tikhonov(op), TVADMM(op)
    psnr = lambda x: per_image_metrics(x, val)[0].mean()
    p = {"Tikhonov": max(LAM_TIK, key=lambda l: psnr(tik(yv, l))),
         "TV": max(LAM_TV, key=lambda l: psnr(tv(yv, l)))}
    best = -1.0
    for sd in SD:
        outs, _ = pnp_pgd(op, yv, den, sd, eta, CK)
        for k, x in outs.items():
            s = psnr(x)
            if s > best:
                best, p["PnP"] = s, (sd, k)
    return p


def reconstruct_all(model_op, den, y, p):
    """Reconstruct with `model_op` (possibly mismatched) using frozen params p."""
    sd, k = p["PnP"]
    outs, _ = pnp_pgd(model_op, y, den, sd, 1.0 / model_op.lipschitz(), (k,))
    return {"FBP": model_op.fbp(y).clamp(0, 1),
            "Tikhonov": Tikhonov(model_op)(y, p["Tikhonov"]),
            "TV": TVADMM(model_op)(y, p["TV"]),
            "PnP-DnCNN": outs[k]}
