"""Noise model of the LoDoPaB observations: y = -ln(N / N0) / mu_max, N ~ Poisson(N0 exp(-mu_max A x)), N0 = 4096 photons,
mu_max = 81.35858 / m, with zero counts replaced by 0.1 before the logarithm (these constants are from the LoDoPaB
paper as I remember them; run_stage3_step2.py --exp noise tests them on the data).
Lower-dose data from the real observations: recover the integer counts, thin them binomially (a Poisson variable thinned
with probability f is Poisson with f times the mean), and transform back with N0 -> f N0."""
import numpy as np
import torch

N0, MU_MAX, ZERO_REPL = 4096.0, 81.35858, 0.1


def counts_from_obs(y, N0=N0, mu=MU_MAX):
    raw = N0 * torch.exp(-mu * y.double())
    return torch.where(raw < 0.5, torch.zeros_like(raw), torch.round(raw))


def obs_from_counts(N, N0=N0, mu=MU_MAX):
    N = torch.where(N == 0, torch.full_like(N, ZERO_REPL), N)
    return (-torch.log(N / N0) / mu).float()


def integrality(y, N0=N0, mu=MU_MAX, tol=0.05):
    """Fraction of bins whose recovered count is within tol of an integer, among bins with raw count >= 3
    (a random real number is that close with probability 2*tol = 0.10)."""
    raw = N0 * torch.exp(-mu * y.double())
    m = raw >= 3
    d = (raw[m] - torch.round(raw[m])).abs()
    return float((d < tol).double().mean()), float(d.mean()), int(m.sum())


def thin_obs(y, f, seed=0, N0=N0, mu=MU_MAX):
    """Observation at dose f*N0 photons built from the native one (f in (0, 1])."""
    if f >= 1.0:
        return y
    N = counts_from_obs(y, N0, mu).cpu()
    g = torch.Generator().manual_seed(seed)
    Nt = torch.binomial(N, torch.full_like(N, float(f)), generator=g)
    return obs_from_counts(Nt, f * N0, mu).to(y.device)
