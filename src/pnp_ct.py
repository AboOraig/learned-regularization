"""PnP-PGD for the matrix-free CT operator and full-size images:  x <- D_sigma( x - eta A^T (A x - y) ),  eta = 1/L."""
import torch


@torch.no_grad()
def denoise_img(model, x_flat, sigma, size, chunk=8):
    out = []
    for b in range(0, x_flat.shape[0], chunk):
        x = x_flat[b:b + chunk].reshape(-1, 1, size, size)
        s = torch.full((len(x),), float(sigma), device=x.device)
        out.append(model(x, s).clamp(0, 1).reshape(len(x), -1))
    return torch.cat(out)


def pnp_ct(op, y, den, sigma_d, eta, checkpoints, x0):
    """Returns {k: x_k} for k in checkpoints (k >= 1), starting from x0."""
    x = x0.clone().clamp(0, 1)
    out = {}
    for k in range(1, max(checkpoints) + 1):
        x = denoise_img(den, x - eta * op.adjoint(op.forward(x) - y), sigma_d, op.img_size)
        if k in checkpoints:
            out[k] = x.clone()
    return out


def anneal_schedule(s0, s1, T):
    """Geometric decay of the denoiser level from s0 (iteration 1) to s1 (iteration T)."""
    if T == 1:
        return [float(s1)]
    return [float(s0 * (s1 / s0) ** (i / (T - 1))) for i in range(T)]


def pnp_ct_sched(op, y, den, sigmas, eta, checkpoints, x0):
    """PnP-PGD with a per-iteration denoiser level sigmas[k-1] (len(sigmas) >= max(checkpoints))."""
    x = x0.clone().clamp(0, 1)
    out = {}
    for k in range(1, max(checkpoints) + 1):
        x = denoise_img(den, x - eta * op.adjoint(op.forward(x) - y), sigmas[k - 1], op.img_size)
        if k in checkpoints:
            out[k] = x.clone()
    return out
