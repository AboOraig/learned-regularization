"""Plug-and-Play proximal gradient: x_{k+1} = D_sigma( x_k - eta * A^T (A x_k - y) )."""
import torch
from .denoiser import denoise


def _psnr(x, t):
    mse = ((x.clamp(0, 1) - t) ** 2).mean(1).clamp_min(1e-12)
    return (10 * torch.log10(1.0 / mse)).mean().item()


def pnp_pgd(model_op, y, denoiser, sigma_d, eta, checkpoints=(100,), x_true=None):
    """model_op: operator used INSIDE the reconstruction (may differ from the one that made y).
    Returns ({iter: x}, history) ; history has data residual (and PSNR if x_true given) per iteration."""
    x = model_op.fbp(y).clamp(0, 1)
    out, hist = {}, {"residual": [], "psnr": []}
    if 0 in checkpoints:  # control: no data-consistency iterations at all
        out[0] = denoise(denoiser, x, sigma_d)
    for k in range(1, max(checkpoints) + 1):
        grad = model_op.adjoint(model_op.forward(x) - y)
        x = denoise(denoiser, x - eta * grad, sigma_d)
        hist["residual"].append((model_op.forward(x) - y).norm(dim=1).mean().item())
        if x_true is not None:
            hist["psnr"].append(_psnr(x, x_true))
        if k in checkpoints:
            out[k] = x.clone()
    return out, hist
