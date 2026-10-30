"""Matrix-free classical baselines (only forward/adjoint are used) + Poisson noise + randomized SVD."""
import math
import torch


def cg(apply, b, iters=200, tol=1e-6):
    """Batched conjugate gradients for SPD `apply`; x0 = 0. b: (B,n)."""
    x = torch.zeros_like(b); r = b.clone(); p = r.clone()
    rs = (r * r).sum(1, keepdim=True); b2 = (b * b).sum(1, keepdim=True).clamp_min(1e-30)
    for _ in range(iters):
        Ap = apply(p)
        a = rs / (p * Ap).sum(1, keepdim=True).clamp_min(1e-30)
        x = x + a * p; r = r - a * Ap
        rn = (r * r).sum(1, keepdim=True)
        if bool((rn / b2).max() < tol ** 2):
            break
        p = r + (rn / rs.clamp_min(1e-30)) * p; rs = rn
    return x


class MFTikhonov:
    """argmin 0.5||Ax-y||^2 + 0.5 lam ||x||^2 via CG (no dense inverse)."""
    def __init__(self, op, iters=200, tol=1e-6):
        self.op, self.iters, self.tol = op, iters, tol

    def __call__(self, y, lam):
        return cg(lambda v: self.op.normal(v) + lam * v, self.op.adjoint(y), self.iters, self.tol)


def _grad(x, h, w):
    X = x.reshape(-1, h, w)
    gx = torch.zeros_like(X); gy = torch.zeros_like(X)
    gx[:, :, :-1] = X[:, :, 1:] - X[:, :, :-1]
    gy[:, :-1, :] = X[:, 1:, :] - X[:, :-1, :]
    return gx.reshape(x.shape[0], -1), gy.reshape(x.shape[0], -1)


def _grad_T(gx, gy, h, w):
    Gx = gx.reshape(-1, h, w); Gy = gy.reshape(-1, h, w)
    out = torch.zeros_like(Gx)
    out[:, :, 1:] += Gx[:, :, :-1]; out[:, :, :-1] -= Gx[:, :, :-1]
    out[:, 1:, :] += Gy[:, :-1, :]; out[:, :-1, :] -= Gy[:, :-1, :]
    return out.reshape(gx.shape[0], -1)


class MFTVADMM:
    """Isotropic-TV ADMM; x-update solved by CG on (A^T A + rho D^T D). Same model as TVADMM."""
    def __init__(self, op, rho=0.05, iters=100, cg_iters=30):
        self.op, self.rho, self.iters, self.cg_iters = op, rho, iters, cg_iters
        self.h = self.w = op.img_size

    def __call__(self, y, lam):
        op, rho, h, w = self.op, self.rho, self.h, self.w
        def M(v):
            gx, gy = _grad(v, h, w)
            return op.normal(v) + rho * _grad_T(gx, gy, h, w)
        Aty = op.adjoint(y); x = Aty.clone()
        zx, zy = _grad(x, h, w); ux = torch.zeros_like(zx); uy = torch.zeros_like(zy)
        for _ in range(self.iters):
            rhs = Aty + rho * _grad_T(zx - ux, zy - uy, h, w)
            x = cg(M, rhs, self.cg_iters, 1e-5)
            dx, dy = _grad(x, h, w)
            vx, vy = dx + ux, dy + uy
            mag = torch.sqrt(vx ** 2 + vy ** 2).clamp_min(1e-12)
            f = torch.clamp(1 - (lam / rho) / mag, min=0)
            zx, zy = vx * f, vy * f
            ux, uy = ux + dx - zx, uy + dy - zy
        return x


def poisson_from_clean(p, I0, seed, mu=1.0):
    """Beer-Lambert low-dose model applied to a clean (normalised) sinogram p.
    I = I0 exp(-mu p); N ~ Poisson(I); p_hat = -log(max(N,1)/I0)/mu.
    Var(p_hat) ~ exp(mu p)/(I0 mu^2): signal dependent, grows with the line integral. Returns (p_hat, N)."""
    g = torch.Generator().manual_seed(seed)
    lam = (I0 * torch.exp(-mu * p)).cpu()
    N = torch.poisson(lam, generator=g).to(p.device)
    return -torch.log(N.clamp_min(1.0) / I0) / mu, N


def poisson_sinogram(op, x, I0, seed, mu_scale=1.0):
    return poisson_from_clean(op.forward(x), I0, seed, mu_scale)


def pnp_pgd_weighted(op, y, N, I0, denoiser, sigma_d, eta_scale, checkpoints, ):
    """PnP-PGD with a Poisson-weighted data term 0.5 sum w_i (Ax-y)_i^2, w = N/I0 (~ inverse variance of
    p_hat up to a constant), clamped to [1/I0, 1]. Step 1/(L max w). Returns {k: x}."""
    from .denoiser import denoise as denoise_batch
    w = (N / I0).clamp(1.0 / I0, 1.0)
    eta = eta_scale / (op.lipschitz() * float(w.max()))
    x = op.fbp(y).clamp(0, 1); out = {}
    if 0 in checkpoints:
        out[0] = denoise_batch(denoiser, x, sigma_d)
    for k in range(1, max(checkpoints) + 1):
        x = denoise_batch(denoiser, x - eta * op.adjoint(w * (op.forward(x) - y)), sigma_d)
        if k in checkpoints:
            out[k] = x.clone()
    return out


def strong_basis(op, tau=0.05, k=600, seed=0):
    """Orthonormal basis of the well-determined subspace {s/s_max >= tau} from forward/adjoint only."""
    s, V = randomized_svd(op, k=k, seed=seed)
    keep = s / s[0] >= tau
    assert not bool(keep[-1]), "increase k: top-k must reach below tau*s_max"
    return V[:, keep]


def error_split(Vs, xr, xt):
    """||e||, error in the strong (data-determined) subspace, error in its complement (weak + null)."""
    e = xr - xt
    st = (e @ Vs).norm(dim=1); tot = e.norm(dim=1)
    return tot, st, (tot ** 2 - st ** 2).clamp_min(0).sqrt()


def randomized_svd(op, k=400, oversample=40, power=4, seed=0):
    """Top-k singular triplets of A from forward/adjoint only (Halko). Returns s, V (n x k)."""
    g = torch.Generator().manual_seed(seed)
    Om = torch.randn(op.n, k + oversample, generator=g).to(op.device)
    Y = op.forward(Om.T).T                       # (m, l)
    Q, _ = torch.linalg.qr(Y)
    for _ in range(power):
        Z, _ = torch.linalg.qr(op.adjoint(Q.T).T)
        Q, _ = torch.linalg.qr(op.forward(Z.T).T)
    B = op.adjoint(Q.T)                          # (l, n) = (Q^T A)
    _, s, Vh = torch.linalg.svd(B, full_matrices=False)
    return s[:k], Vh[:k].T
