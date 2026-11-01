"""Primal-dual (Chambolle-Pock) isotropic-TV reconstruction: one forward and one adjoint per iteration.
   min_{0<=x<=1}  0.5 ||A x - y||^2 + lam * TV(x),   TV(x) = sum_pixels |grad x| (forward differences, Neumann).
The problem is normalised by a^2 = ||A||^2 (so lam is in normalised units: 0.5||A x - y||^2/a^2 + lam TV): the operator norm
of the stacked operator [A/a; grad] is then <= 3."""
import torch


def _grad(x, h, w):
    X = x.reshape(-1, h, w)
    gx = torch.zeros_like(X); gy = torch.zeros_like(X)
    gx[:, :, :-1] = X[:, :, 1:] - X[:, :, :-1]
    gy[:, :-1, :] = X[:, 1:, :] - X[:, :-1, :]
    return gx, gy


def _div_T(gx, gy):                       # adjoint of _grad -> (B, h, w)
    out = torch.zeros_like(gx)
    out[:, :, 1:] += gx[:, :, :-1]; out[:, :, :-1] -= gx[:, :, :-1]
    out[:, 1:, :] += gy[:, :-1, :]; out[:, :-1, :] -= gy[:, :-1, :]
    return out


def tv_objective(op, x, y, lam, a2):
    gx, gy = _grad(x, op.img_size, op.img_size)
    tv = torch.sqrt(gx ** 2 + gy ** 2).sum((1, 2))
    return 0.5 * ((op.forward(x) - y) ** 2).sum(1) / a2 + lam * tv


def tv_pdhg(op, y, lam, iters=150, x0=None, L=None, box=(0.0, 1.0)):
    h = w = op.img_size
    a2 = L if L is not None else op.lipschitz()
    a = a2 ** 0.5
    B = y.shape[0]
    x = (x0 if x0 is not None else torch.zeros(B, op.n, device=y.device)).clone().clamp(*box)
    xbar = x.clone()
    p = torch.zeros_like(y)
    qx = torch.zeros(B, h, w, device=y.device); qy = torch.zeros_like(qx)
    tau = sigma = 0.99 / 3.0
    yn = y / a
    for _ in range(iters):
        p = (p + sigma * (op.forward(xbar) / a - yn)) / (1 + sigma)
        gx, gy = _grad(xbar, h, w)
        qx, qy = qx + sigma * gx, qy + sigma * gy
        nrm = torch.sqrt(qx ** 2 + qy ** 2).clamp_min(1e-12)
        s = torch.clamp(lam / nrm, max=1.0)
        qx, qy = qx * s, qy * s
        xold = x
        x = (x - tau * (op.adjoint(p) / a + _div_T(qx, qy).reshape(B, -1))).clamp(*box)
        xbar = 2 * x - xold
    return x
