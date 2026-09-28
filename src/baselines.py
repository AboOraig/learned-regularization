"""Classical reconstruction baselines: Tikhonov (L2) and isotropic TV (ADMM).
Both use precomputed dense inverses (n = 784 is small), so batches are fast."""
import torch


class Tikhonov:
    """argmin 0.5||Ax-y||^2 + 0.5*lam||x||^2  (closed form)."""
    def __init__(self, op):
        self.op = op
        self.AtA = op.A.T @ op.A
        self.I = torch.eye(op.n, device=op.device)
        self._cache = {}

    def __call__(self, y, lam):
        if lam not in self._cache:
            self._cache[lam] = torch.linalg.inv(self.AtA + lam * self.I)
        return (self.op.adjoint(y)) @ self._cache[lam].T


def _grad_matrix(h, w, device):
    """Dense forward-difference operator D (2n x n), Neumann boundary."""
    n = h * w
    Dx = torch.zeros(n, n); Dy = torch.zeros(n, n)
    for i in range(h):
        for j in range(w):
            k = i * w + j
            if j < w - 1:
                Dx[k, k] = -1; Dx[k, k + 1] = 1
            if i < h - 1:
                Dy[k, k] = -1; Dy[k, k + w] = 1
    return torch.cat([Dx, Dy], 0).to(device)


class TVADMM:
    """argmin 0.5||Ax-y||^2 + lam*TV(x), isotropic TV, ADMM with splitting z = Dx."""
    def __init__(self, op, rho=0.05, iters=100):
        self.op, self.rho, self.iters = op, rho, iters
        self.D = _grad_matrix(op.img_size, op.img_size, op.device)
        M = op.A.T @ op.A + rho * self.D.T @ self.D
        self.Minv = torch.linalg.inv(M)

    def __call__(self, y, lam):
        op, D, rho, n = self.op, self.D, self.rho, self.op.n
        Aty = op.adjoint(y)
        x = Aty.clone()
        z = x @ D.T
        u = torch.zeros_like(z)
        for _ in range(self.iters):
            x = (Aty + rho * (z - u) @ D) @ self.Minv.T
            Dx = x @ D.T
            v = Dx + u
            gx, gy = v[:, :n], v[:, n:]
            mag = torch.sqrt(gx ** 2 + gy ** 2).clamp_min(1e-12)
            f = torch.clamp(1 - (lam / rho) / mag, min=0)
            z = torch.cat([gx * f, gy * f], 1)
            u = u + Dx - z
        return x
