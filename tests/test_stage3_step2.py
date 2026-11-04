import numpy as np, torch
from src.radon_pb import ParallelBeamOperator
from src.tv_pdhg import tv_pdhg, tv_objective, _grad, _div_T
from src.lodopab_noise import counts_from_obs, obs_from_counts, thin_obs, integrality, N0, MU_MAX


def test_grad_adjoint():
    x, qx, qy = torch.randn(2, 24, 24), torch.randn(2, 24, 24), torch.randn(2, 24, 24)
    gx, gy = _grad(x.reshape(2, -1), 24, 24)
    assert abs(float((gx * qx + gy * qy).sum() - (x * _div_T(qx, qy)).sum())) < 1e-3


def _phantom(H):
    yy, xx = np.mgrid[:H, :H]
    return torch.tensor(((np.hypot(yy - H / 2, xx - H / 2) < H * .35) * 0.6 + (np.hypot(yy - H * .4, xx - H * .55) < H * .12) * 0.3).astype(np.float32)).reshape(1, -1)


def test_pdhg_converges_and_beats_fbp():
    H = 32
    op = ParallelBeamOperator(H, 47, n_angles=24, pixel_size=1.0)
    x = _phantom(H); y = op.forward(x)
    g = torch.Generator().manual_seed(0)
    yn = y + 0.02 * y.abs().max() * torch.randn(y.shape, generator=g)
    a2 = op.lipschitz()
    lam = 2e-3
    xs = tv_pdhg(op, yn, lam, iters=100, L=a2)
    xl = tv_pdhg(op, yn, lam, iters=1500, L=a2)
    fs, fl = tv_objective(op, xs, yn, lam, a2).item(), tv_objective(op, xl, yn, lam, a2).item()
    assert fs <= fl * 1.15 and fl <= tv_objective(op, torch.zeros_like(x), yn, lam, a2).item()
    err = lambda r: float((r.clamp(0, 1) - x).norm() / x.norm())
    assert err(xl) < err(op.fbp(yn))


def test_hann_fbp_reduces_noise():
    H = 32
    op = ParallelBeamOperator(H, 47, n_angles=90, pixel_size=1.0)
    x = _phantom(H); y = op.forward(x)
    g = torch.Generator().manual_seed(1)
    yn = y + 0.15 * y.abs().max() * torch.randn(y.shape, generator=g)      # heavy noise: the low-pass filter must win
    e = lambda r: float((r - x).norm() / x.norm())
    assert e(op.fbp(yn, "hann", 0.7)) < e(op.fbp(yn, "ramp"))
    assert torch.allclose(op.fbp(yn), op.fbp(yn, "ramp"))


def test_counts_roundtrip_and_zero_replacement():
    g = torch.Generator().manual_seed(0)
    lam = torch.rand(20000, generator=g) * 4000 + 0.05
    N = torch.poisson(lam, generator=g).double()
    y = obs_from_counts(N)
    assert torch.equal(counts_from_obs(y), N)
    assert abs(float(y.max()) - np.log(N0 / 0.1) / MU_MAX) < 1e-6 or (N == 0).sum() == 0
    frac, _, n = integrality(y)
    assert frac > 0.99 and n > 1000
    assert integrality(y + 0.0007 * torch.rand(y.shape, generator=g))[0] < 0.5


def test_thinning_is_poisson_with_scaled_mean():
    g = torch.Generator().manual_seed(2)
    lam = torch.full((200000,), 400.0)
    N = torch.poisson(lam, generator=g).double()
    y = obs_from_counts(N)
    f = 0.25
    Nt = counts_from_obs(thin_obs(y, f, seed=3), N0=f * N0)
    assert abs(Nt.mean().item() - f * 400) < 1.0
    assert abs(Nt.var().item() / Nt.mean().item() - 1) < 0.03
    assert torch.equal(thin_obs(y, 1.0), y)
