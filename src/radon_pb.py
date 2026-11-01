"""General matrix-free parallel-beam operator for square images (used for LoDoPaB-CT, 362x362, 513 detector bins).

Geometry (pixel units, origin = centre of the image): the image is resampled on a n_det x n_det canvas whose pixel
CENTRES lie at integer offsets (k - (n_det-1)/2) from the image centre; the canvas is rotated by each angle with bilinear
interpolation and summed along its rows.  Detector bin t therefore sits at lateral position (t - (n_det-1)/2) pixels.
With an odd n_det this agrees with skimage.transform.radon(circle=False) for odd image sizes (tests/test_radon_pb.py).
Line integrals are multiplied by `pixel_size` (physical length of one pixel), so  A x  has the units of  integral of x ds.
Sinogram layout: (n_angles, n_det) raveled, as in the LoDoPaB observation files.
Adjoint = exact transpose of the discretisation (autograd), chunked over angles to bound memory.
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


class ParallelBeamOperator:
    def __init__(self, img_size=362, n_det=513, angles_deg=None, n_angles=1000, angle_offset_deg=0.0,
                 pixel_size=1.0, device="cpu", sign=-1.0, flip_det=False, chunk_angles=16, chunk_batch=8):
        self.img_size, self.n = img_size, img_size * img_size
        self.n_det, self.device = n_det, device
        if angles_deg is None:
            angles_deg = np.linspace(0.0, 180.0, n_angles, endpoint=False) + angle_offset_deg
        self.angles = np.asarray(angles_deg, dtype=float)
        self.K = len(self.angles)
        self.m = self.K * n_det
        self.sino_shape = (self.K, n_det)
        self.pixel_size = float(pixel_size)
        self.sign, self.flip_det = float(sign), bool(flip_det)
        self.chunk_angles, self.chunk_batch = chunk_angles, chunk_batch

    # ---- one angle chunk, differentiable --------------------------------------------------------------
    def _grid(self, a0, a1):
        th = torch.tensor(np.deg2rad(self.angles[a0:a1]) * self.sign, dtype=torch.float32, device=self.device)
        c, s = torch.cos(th), torch.sin(th)
        k = self.n_det / self.img_size
        theta = torch.zeros(len(th), 2, 3, device=self.device)
        theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = c * k, -s * k, s * k, c * k
        return F.affine_grid(theta, (len(th), 1, self.n_det, self.n_det), align_corners=False)

    def _fwd_chunk(self, x, a0, a1):          # x: (B, n) -> (B, a1-a0, n_det)
        B, Kc, H, D = x.shape[0], a1 - a0, self.img_size, self.n_det
        img = x.reshape(B, 1, 1, H, H).expand(B, Kc, 1, H, H).reshape(B * Kc, 1, H, H)
        grid = self._grid(a0, a1).unsqueeze(0).expand(B, -1, -1, -1, -1).reshape(B * Kc, D, D, 2)
        rot = F.grid_sample(img, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        s = rot.sum(dim=2).reshape(B, Kc, D) * self.pixel_size
        return s.flip(-1) if self.flip_det else s

    def forward(self, x):
        outs = []
        with torch.no_grad():
            for b in range(0, x.shape[0], self.chunk_batch):
                xb = x[b:b + self.chunk_batch]
                outs.append(torch.cat([self._fwd_chunk(xb, a, min(a + self.chunk_angles, self.K))
                                       for a in range(0, self.K, self.chunk_angles)], 1).reshape(xb.shape[0], -1))
        return torch.cat(outs, 0)

    def adjoint(self, y):
        outs = []
        for b in range(0, y.shape[0], self.chunk_batch):
            yb = y[b:b + self.chunk_batch].reshape(-1, self.K, self.n_det)
            acc = torch.zeros(yb.shape[0], self.n, device=self.device)
            for a in range(0, self.K, self.chunk_angles):
                a1 = min(a + self.chunk_angles, self.K)
                with torch.enable_grad():
                    x0 = torch.zeros(yb.shape[0], self.n, device=self.device, requires_grad=True)
                    (g,) = torch.autograd.grad(self._fwd_chunk(x0, a, a1), x0, grad_outputs=yb[:, a:a1])
                acc += g.detach()
            outs.append(acc)
        return torch.cat(outs, 0)

    def normal(self, x):
        return self.adjoint(self.forward(x))

    def lipschitz(self, iters=30, seed=0):
        g = torch.Generator().manual_seed(seed)
        v = torch.randn(1, self.n, generator=g).to(self.device)
        for _ in range(iters):
            v = self.normal(v); v = v / v.norm()
        return float((self.normal(v) * v).sum())

    def fbp(self, y, filter_name="ramp", freq_scale=1.0):
        """Filtered back-projection for angles spread uniformly over [0, 180): filter in cycles/length, then
        x = (pi/K)(1/pixel_size) A^T(filtered).  filter_name: 'ramp' or 'hann' (ramp x Hann window, cut-off
        freq_scale x Nyquist).  Returns (B, n)."""
        yy = y.reshape(-1, self.K, self.n_det)
        npad = 1 << int(math.ceil(math.log2(2 * self.n_det)))
        f = torch.fft.fftfreq(npad, d=self.pixel_size).to(y.device)
        w = f.abs()
        if filter_name == "hann":
            fc = freq_scale / (2.0 * self.pixel_size)
            w = w * torch.where(f.abs() < fc, 0.5 * (1 + torch.cos(math.pi * f / fc)), torch.zeros_like(f))
        elif filter_name != "ramp":
            raise ValueError("filter_name must be 'ramp' or 'hann'")
        filt = torch.fft.ifft(torch.fft.fft(yy, n=npad, dim=-1) * w, dim=-1).real[..., :self.n_det]
        return self.adjoint(filt.reshape(len(yy), -1)) * (math.pi / (self.K * self.pixel_size))
