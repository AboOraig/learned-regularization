import torch
from src.operators import RadonOperator
from src.radon_mf import MatrixFreeRadon
from src.mf_solvers import MFTikhonov, MFTVADMM, poisson_sinogram, randomized_svd, cg
from src.baselines import Tikhonov, TVADMM

torch.manual_seed(0)
D = RadonOperator(28, 20); M = MatrixFreeRadon(28, 20)


def test_matches_dense():
    x = torch.rand(16, 784)
    r = (M.forward(x) - D.forward(x)).norm(dim=1) / D.forward(x).norm(dim=1)
    assert r.max() < 1e-5


def test_adjoint_dot():
    u, v = torch.rand(4, 784), torch.rand(4, M.m)
    a, b = (M.forward(u) * v).sum(), (u * M.adjoint(v)).sum()
    assert abs(a - b) / abs(a) < 1e-5
    assert (M.adjoint(v) - D.adjoint(v)).abs().max() < 1e-3


def test_lipschitz():
    assert abs(M.lipschitz() - D.lipschitz()) / D.lipschitz() < 1e-3


def test_other_angles():
    m2, d2 = MatrixFreeRadon(28, 7), RadonOperator(28, 7)
    x = torch.rand(4, 784)
    assert ((m2.forward(x) - d2.forward(x)).norm() / d2.forward(x).norm()) < 1e-5


def test_cg_tikhonov_matches_closed_form():
    y = D.forward(torch.rand(3, 784))
    a = Tikhonov(D)(y, 1e-2); b = MFTikhonov(M, iters=400, tol=1e-7)(y, 1e-2)
    assert (a - b).norm() / a.norm() < 1e-3


def test_tv_matches_dense():
    y = D.forward(torch.rand(2, 784))
    a = TVADMM(D, iters=30)(y, 1e-2); b = MFTVADMM(M, iters=30, cg_iters=60)(y, 1e-2)
    assert (a - b).norm() / a.norm() < 1e-2


def test_poisson_statistics():
    x = torch.rand(200, 784) * 0.3
    p = M.forward(x)
    ph, N = poisson_sinogram(M, x, I0=1e4, seed=1)
    assert abs((ph - p).mean()) < 5e-3
    ph2, _ = poisson_sinogram(M, x, I0=1e2, seed=1)
    assert (ph2 - p).std() > 3 * (ph - p).std()
    a, _ = poisson_sinogram(M, x, 1e3, seed=5); b, _ = poisson_sinogram(M, x, 1e3, seed=5)
    assert torch.equal(a, b)


def test_randomized_svd():
    s, V = randomized_svd(M, k=40, seed=0)
    s_ref = torch.linalg.svdvals(D.A)[:40]
    assert (s - s_ref).abs().max() / s_ref[0] < 1e-2


def test_weighted_pnp_identity_denoiser_reduces_residual():
    class Id(torch.nn.Module):
        def forward(self, x, s): return x
    from src.mf_solvers import pnp_pgd_weighted
    x = torch.rand(2, 784) * 0.5
    ph, N = poisson_sinogram(M, x, 1e3, seed=2)
    out = pnp_pgd_weighted(M, ph, N, 1e3, Id(), 0.05, 1.0, (1, 20))
    r = lambda z: (M.forward(z) - ph).norm()
    assert r(out[20]) < r(out[1])


def test_error_split_pythagoras():
    from src.mf_solvers import strong_basis, error_split
    Vs = strong_basis(M, 0.05)
    xt = torch.rand(5, 784); xr = xt + 0.1 * torch.randn(5, 784)
    tot, st, out = error_split(Vs, xr, xt)
    assert torch.allclose(st ** 2 + out ** 2, tot ** 2, atol=1e-4)
    from src.nullspace import OperatorSubspaces
    dense = OperatorSubspaces(D.A)
    assert Vs.shape[1] == 784 - dense.dim(0.05)
    # same subspace: projections agree
    Pd = xr - xt - dense.project(xr - xt, 0.05)       # strong part = e - weak part
    assert ((Pd.norm(dim=1) - st).abs() / tot).max() < 1e-2
