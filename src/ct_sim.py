"""Crime-free CT data simulation for LoDoPaB-like geometry: the 362x362 image is up-sampled x2 (bicubic), projected with the
parallel-beam operator on the 724x724 grid (detector 2*n_det, pixel size / 2), the fine detector bins are averaged in pairs
(= one coarse bin), and Poisson photon noise is added exactly as in LoDoPaB (N ~ Poisson(f N0 exp(-mu y)), zero counts -> 0.1,
y = -ln(N/(f N0))/mu)."""
import numpy as np
import torch
import torch.nn.functional as F
from .radon_pb import ParallelBeamOperator
from .lodopab_noise import N0, MU_MAX, obs_from_counts


class CTSimulator:
    def __init__(self, op, device=None, chunk_batch=2):
        """op: the coarse ParallelBeamOperator whose geometry is simulated."""
        self.op, self.device = op, device or op.device
        self.fine = ParallelBeamOperator(2 * op.img_size, 2 * op.n_det, angles_deg=op.angles, pixel_size=op.pixel_size / 2,
                                         device=self.device, sign=op.sign, flip_det=op.flip_det, chunk_angles=16, chunk_batch=chunk_batch)

    def clean(self, x):
        """x: (B, n) in [0,1] -> clean sinogram (B, K*n_det) in the coarse operator's units."""
        H = self.op.img_size
        up = F.interpolate(x.reshape(-1, 1, H, H), scale_factor=2, mode="bicubic", align_corners=False).clamp(0, 1)
        y = self.fine.forward(up.reshape(len(x), -1))
        y = y.reshape(len(x), self.op.K, self.op.n_det, 2).mean(-1)
        return y.reshape(len(x), -1)


def poisson_obs(y_clean, dose, seed, n0=N0, mu=MU_MAX):
    """LoDoPaB-style noisy observation at dose f = `dose` (fraction of 4096 photons per ray)."""
    g = torch.Generator().manual_seed(int(seed))
    lam = (dose * n0 * torch.exp(-mu * y_clean.double())).cpu()
    N = torch.poisson(lam, generator=g)
    return obs_from_counts(N, dose * n0, mu).to(y_clean.device)
