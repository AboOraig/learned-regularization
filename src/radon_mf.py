"""Matrix-free sparse-view Radon operator (no dense A is ever formed).

Forward: embed the HxW image in a D x D canvas (D = ceil(sqrt2*H), as skimage radon with
circle=False), rotate by each angle with bilinear grid_sample, sum along detector rows.
Adjoint: exact transpose of that discretisation, obtained by autograd (vjp) -- so
<Ax,y> = <x,A^T y> holds to float precision by construction.
Drop-in for RadonOperator: forward, adjoint, lipschitz, fbp, n, m, angles, scale, sino_shape.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F
from skimage.transform import iradon


class MatrixFreeRadon:
    def __init__(self, img_size=28, n_angles=20, angle_offset=0.0, angles=None,
                 device="cpu", sign=-1.0, chunk=256):
        self.img_size = img_size
        self.n = img_size * img_size
        if angles is None:
            angles = np.linspace(0.0, 180.0, n_angles, endpoint=False) + angle_offset
        self.angles = np.asarray(angles, dtype=float)
        self.K = len(self.angles)
        self.D = int(math.ceil(math.sqrt(2) * img_size))
        self.scale = float(img_size)
        self.sino_shape = (self.D, self.K)
        self.m = self.D * self.K
        self.device = device
        self.chunk = chunk
        th = torch.tensor(np.deg2rad(self.angles) * sign, dtype=torch.float32, device=device)
        c, s = torch.cos(th), torch.sin(th)
        R = torch.zeros(self.K, 2, 2, device=device)
        R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1] = c, -s, s, c
        # skimage rotates about pixel D//2 (half a pixel off the canvas centre for even D):
        # rotation about the centre plus the translation (I - R) delta.  Verified to match
        # RadonOperator to ~3e-7 relative error (tests/test_matrix_free.py).
        delta = torch.full((2,), 0.5, device=device) * 2.0 / self.D
        theta = torch.zeros(self.K, 2, 3, device=device)
        theta[:, :, :2] = R
        theta[:, :, 2] = delta - R @ delta
        self.grid = F.affine_grid(theta, (self.K, 1, self.D, self.D), align_corners=False)
        self.pad = (self.D - img_size) // 2
        self.pad2 = self.D - img_size - self.pad

    def _fwd(self, x):  # x: (B,n) -> (B,m)
        B = x.shape[0]
        img = x.reshape(B, 1, self.img_size, self.img_size)
        img = F.pad(img, (self.pad, self.pad2, self.pad, self.pad2))
        img = img.unsqueeze(1).expand(B, self.K, 1, self.D, self.D).reshape(B * self.K, 1, self.D, self.D)
        grid = self.grid.unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * self.K, self.D, self.D, 2)
        rot = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sino = rot.sum(dim=2).reshape(B, self.K, self.D)       # sum over rows -> (B,K,det)
        return sino.permute(0, 2, 1).reshape(B, self.m) / self.scale  # (det,angle) raveled

    def forward(self, x):
        with torch.no_grad():
            return torch.cat([self._fwd(x[i:i + self.chunk]) for i in range(0, x.shape[0], self.chunk)], 0)

    def adjoint(self, y):
        outs = []
        for i in range(0, y.shape[0], self.chunk):
            yi = y[i:i + self.chunk]
            with torch.enable_grad():
                x0 = torch.zeros(yi.shape[0], self.n, device=self.device, requires_grad=True)
                (g,) = torch.autograd.grad(self._fwd(x0), x0, grad_outputs=yi)
            outs.append(g.detach())
        return torch.cat(outs, 0)

    def normal(self, x):
        return self.adjoint(self.forward(x))

    def lipschitz(self, iters=100, seed=0):
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(1, self.n, generator=g).to(self.device)
        for _ in range(iters):
            v = self.normal(v)
            v = v / v.norm()
        return float((self.normal(v) * v).sum())

    def fbp(self, y):
        out = []
        for yi in y.cpu().numpy():
            sino = yi.reshape(self.sino_shape) * self.scale
            r = iradon(sino, theta=self.angles, circle=False, filter_name="ramp", output_size=self.img_size)
            out.append(r.ravel())
        return torch.tensor(np.stack(out), dtype=torch.float32, device=self.device)
