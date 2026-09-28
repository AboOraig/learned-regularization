"""Sparse-view Radon operator stored as an explicit dense matrix A (m x n).
Pros: exact adjoint (A.T), differentiable, easy to swap for a mismatched operator.
Images are flattened to length n = H*W. Sinograms to length m = n_det * n_angles.
A is scaled by 1/img_size so sinogram values are O(1); noise sigma is defined
on this normalized sinogram.
"""
import numpy as np
import torch
from skimage.transform import radon, iradon


class RadonOperator:
    def __init__(self, img_size=28, n_angles=20, angle_offset=0.0, angles=None, device="cpu"):
        self.img_size = img_size
        self.n = img_size * img_size
        if angles is None:
            angles = np.linspace(0.0, 180.0, n_angles, endpoint=False) + angle_offset
        self.angles = np.asarray(angles, dtype=float)
        self.scale = float(img_size)
        cols = []
        for i in range(self.n):
            e = np.zeros(self.n)
            e[i] = 1.0
            s = radon(e.reshape(img_size, img_size), theta=self.angles, circle=False)
            cols.append(s.ravel())
        self.sino_shape = s.shape  # (n_det, n_angles)
        A = np.stack(cols, axis=1) / self.scale
        self.A = torch.tensor(A, dtype=torch.float32, device=device)
        self.m = self.A.shape[0]
        self.device = device

    def forward(self, x):   # x: (B, n) -> (B, m)
        return x @ self.A.T

    def adjoint(self, y):   # y: (B, m) -> (B, n)
        return y @ self.A

    def lipschitz(self, iters=100):
        """Largest eigenvalue of A^T A (power iteration) -> step size eta < 2/L."""
        v = torch.randn(1, self.n, device=self.device)
        for _ in range(iters):
            v = self.adjoint(self.forward(v))
            v = v / v.norm()
        return float((self.adjoint(self.forward(v)) * v).sum())

    def fbp(self, y):
        """Filtered back-projection (ramp filter) with this operator's angles."""
        out = []
        for yi in y.cpu().numpy():
            sino = yi.reshape(self.sino_shape) * self.scale
            r = iradon(sino, theta=self.angles, circle=False, filter_name="ramp",
                       output_size=self.img_size)
            out.append(r.ravel())
        return torch.tensor(np.stack(out), dtype=torch.float32, device=self.device)


def add_noise(y, sigma, seed):
    g = torch.Generator().manual_seed(seed)
    return y + sigma * torch.randn(y.shape, generator=g).to(y.device)
