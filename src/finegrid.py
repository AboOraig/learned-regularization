"""Crime-free data simulation: generate measurements on a FINER grid, reconstruct on the 28x28 operator.

The v1 experiments generate y = A x + n with the SAME dense matrix A that the solvers then invert (an "inverse
crime"). Here the clean data come from an up-sampled version of the image projected with the same Radon routine on
an f-times finer grid, then binned onto the coarse detector. The reconstruction operator A is unchanged, so the
difference y_fine - A x is a genuine (discretisation + interpolation) model error.

Geometry / conventions (verified by tests/test_finegrid.py):
- f must be ODD, so each coarse pixel centre coincides with a fine pixel centre and the rotation centres agree.
- coarse detector bin j (offset u = j - n_det//2 coarse pixels) is the average of the f fine detector bins centred
  on f*u (a box of one coarse pixel, as in the coarse operator), and the pixel-sum -> pixel-area conversion gives
      y_coarse_equivalent = (1/scale) * (1/f^2) * sum_t sino_fine[centre_f + f*u + t],   t = -(f-1)/2 .. (f-1)/2
  where scale = op.scale (= img_size) is the normalisation already used by RadonOperator.
- up_mode: 'nearest' (pixel-constant object: isolates the projector discretisation), 'bilinear' (default) or
  'bicubic' (smoother 'continuous-like' object; clamped to [0, 1]).
The ground truth used for evaluation stays the original 28x28 image.
"""
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import radon


class FineGridSimulator:
    def __init__(self, op, factor=5, up_mode="bilinear"):
        if factor % 2 != 1 or factor < 1:
            raise ValueError("factor must be a positive odd integer (3, 5, 7, ...)")
        if up_mode not in ("nearest", "bilinear", "bicubic"):
            raise ValueError("up_mode must be nearest | bilinear | bicubic")
        self.op, self.f, self.mode = op, factor, up_mode
        n, f = op.img_size, factor
        self.n_f = n * f
        c_f = (n // 2) * f + (f - 1) // 2                      # coarse centre pixel, in fine pixel coordinates
        m = max(c_f, self.n_f - c_f)
        self.H = 2 * m                                          # array whose index H//2 is the coarse centre pixel
        self.top = self.H // 2 - c_f
        assert 0 <= self.top and self.H - self.n_f - self.top >= 0
        diag = np.sqrt(2) * self.H
        self.D_f = self.H + int(np.ceil(diag - self.H))         # fine detector length (skimage, circle=False)
        self.centre_f = self.D_f // 2
        n_det = op.sino_shape[0]
        self.c_det = n_det // 2
        # index of the f fine bins that make up each coarse bin
        u = np.arange(n_det) - self.c_det
        t = np.arange(f) - (f - 1) // 2
        idx = self.centre_f + f * u[:, None] + t[None, :]        # (n_det, f)
        self.valid = (idx >= 0) & (idx < self.D_f)
        self.idx = np.clip(idx, 0, self.D_f - 1)

    def upsample(self, x):
        """x: (B, n*n) in [0,1] -> (B, n_f, n_f) numpy float64."""
        n, f = self.op.img_size, self.f
        xt = x.detach().cpu().float().reshape(-1, 1, n, n)
        if self.mode == "nearest":
            up = F.interpolate(xt, scale_factor=f, mode="nearest")
        else:
            up = F.interpolate(xt, scale_factor=f, mode=self.mode, align_corners=False).clamp(0, 1)
        return up[:, 0].double().numpy()

    def forward(self, x):
        """Clean fine-grid measurements in the SAME layout/units as op.forward(x): (B, m) torch tensor."""
        up = self.upsample(x)
        out = []
        for img in up:
            arr = np.zeros((self.H, self.H))
            arr[self.top:self.top + self.n_f, self.top:self.top + self.n_f] = img
            sino = radon(arr, theta=self.op.angles, circle=False)          # (D_f, n_angles)
            binned = (sino[self.idx] * self.valid[..., None]).sum(1)       # (n_det, n_angles): sum over the f fine bins
            out.append((binned / (self.f ** 2) / self.op.scale).ravel())
        return torch.tensor(np.stack(out), dtype=torch.float32, device=self.op.device)


def make_simulator(op, fine_factor=0, up_mode="bilinear"):
    """Returns a function x -> clean measurements. fine_factor=0 reproduces the v1 behaviour (inverse crime)."""
    if not fine_factor:
        return op.forward
    return FineGridSimulator(op, fine_factor, up_mode).forward
