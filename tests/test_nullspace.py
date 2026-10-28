"""Run from the repo root:  python -m pytest tests -q
(needs `pip install pytest`; no dataset download required)."""
import os, sys
import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.operators import RadonOperator
from src.nullspace import OperatorSubspaces, SUBSPACES, subspace_metrics


@pytest.fixture(scope="module")
def op():
    return RadonOperator(28, 20)          # the nominal operator used in all experiments


@pytest.fixture(scope="module")
def subs(op):
    return OperatorSubspaces(op.A)


def test_exact_null_space_is_really_null(op, subs):
    V = torch.tensor(subs.basis(SUBSPACES["exact_null"]), dtype=torch.float32)
    assert V.shape[1] > 0, "expected a non-trivial null space for 20 angles"
    resid = (op.A @ V).abs().max().item()
    assert resid < 1e-5 * subs.smax, resid


def test_null_cut_sits_in_a_clear_gap(subs):
    g = subs.gap_report()
    # the smallest 'kept' singular value must be well above the largest 'null' one (or null is empty)
    assert g["smallest_above"] > 100 * max(g["largest_below"], 1e-14), g
    assert g["n_null"] > 0


def test_projector_is_orthogonal_projection(subs):
    for tau in SUBSPACES.values():
        V = subs.basis(tau)
        P = V @ V.T
        assert np.allclose(P, P.T, atol=1e-10)
        assert np.allclose(P @ P, P, atol=1e-8)
        assert np.isclose(np.trace(P), V.shape[1], atol=1e-8)


def test_gradient_step_never_changes_null_component(op, subs):
    """The PnP data step x - eta*A^T(Ax - y) leaves the exact-null component untouched."""
    g = torch.Generator().manual_seed(0)
    x = torch.rand(8, op.n, generator=g)
    y = torch.rand(8, op.m, generator=g)
    eta = 1.0 / op.lipschitz()
    x_new = x - eta * op.adjoint(op.forward(x) - y)
    d = subs.project((x_new - x), SUBSPACES["exact_null"])
    assert np.abs(d).max() < 1e-5, np.abs(d).max()


def test_metrics_zero_for_perfect_reconstruction(subs):
    x = torch.rand(4, 784)
    m = subspace_metrics(subs, x, x, SUBSPACES["tau0.05"])
    assert np.allclose(m["null_err_energy"], 0) and np.allclose(m["total_err_energy"], 0)


def test_metrics_recover_planted_null_error(subs):
    tau = SUBSPACES["tau0.05"]
    V = subs.basis(tau)
    rng = np.random.RandomState(0)
    v = V @ rng.randn(V.shape[1], 3) * 0.1              # (n, 3) vectors inside the subspace
    x = torch.rand(3, 784).double()
    x_hat = x + torch.tensor(v.T)
    m = subspace_metrics(subs, x_hat, x, tau)
    assert np.allclose(m["null_err_energy"], (v ** 2).sum(0), rtol=1e-8)
    assert np.allclose(m["null_err_frac"], 1.0, atol=1e-8)   # the whole error lives in the subspace


def test_error_decomposes_exactly(subs):
    tau = SUBSPACES["tau0.05"]
    x, xh = torch.rand(6, 784).double(), torch.rand(6, 784).double()
    m = subspace_metrics(subs, xh, x, tau)
    assert np.allclose(m["null_err_energy"] + m["range_err_energy"], m["total_err_energy"], rtol=1e-9)
    # err_in_subspace = recon + true - 2*inner, exactly (this is what the pooled cosine in the summary relies on)
    assert np.allclose(m["null_err_energy"], m["recon_energy"] + m["true_energy"] - 2 * m["inner"], rtol=1e-8, atol=1e-10)


def test_cosine_and_injection_on_constructed_cases(subs):
    tau = SUBSPACES["tau0.05"]
    x = torch.rand(5, 784).double()
    Px = torch.tensor(subs.project(x, tau))
    m = subspace_metrics(subs, 3 * Px, x, tau)                 # same direction, 3x amplitude
    assert np.allclose(m["cos_sub"], 1.0, atol=1e-9) and np.allclose(m["inject_ratio"], 9.0, rtol=1e-9)
    m = subspace_metrics(subs, -Px, x, tau)                    # opposite direction, same energy
    assert np.allclose(m["cos_sub"], -1.0, atol=1e-9) and np.allclose(m["inject_ratio"], 1.0, rtol=1e-9)
    m = subspace_metrics(subs, x - Px, x, tau)                 # nothing at all inside the subspace
    assert np.allclose(m["recon_energy"], 0, atol=1e-18) and np.isnan(m["cos_sub"]).all() and np.allclose(m["inject_ratio"], 0, atol=1e-12)


def test_subspaces_are_nested(subs):
    dims = [subs.dim(t) for t in sorted(SUBSPACES.values())]
    assert dims == sorted(dims)