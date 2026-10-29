"""Run from the repo root:  python -m pytest tests -q   (no dataset download needed)."""
import os, sys
import numpy as np
import pytest
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.operators import RadonOperator
from src.finegrid import FineGridSimulator, make_simulator


def smooth_images(n=4, seed=0):
    """Blob-like test images in [0,1] (no dataset needed)."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n):
        im = gaussian_filter(rng.rand(28, 28), 2.0)
        im = (im - im.min()) / (im.max() - im.min())
        yy, xx = np.mgrid[:28, :28]
        im = im * (((yy - 14) ** 2 + (xx - 14) ** 2) < 11 ** 2)        # keep it inside the field of view
        out.append(im.ravel())
    return torch.tensor(np.stack(out), dtype=torch.float32)


def rel_err(y, yc):
    return ((y - yc).norm(dim=1) / yc.norm(dim=1)).mean().item()


@pytest.fixture(scope="module")
def op():
    return RadonOperator(28, 20)


def test_even_factor_and_bad_mode_rejected(op):
    with pytest.raises(ValueError):
        FineGridSimulator(op, 4)
    with pytest.raises(ValueError):
        FineGridSimulator(op, 3, "nope")


def test_layout_and_linearity(op):
    sim = FineGridSimulator(op, 3, "nearest")
    x = smooth_images(3)
    y = sim.forward(x)
    assert y.shape == op.forward(x).shape
    assert torch.allclose(sim.forward(0 * x), torch.zeros_like(y), atol=1e-7)
    assert torch.allclose(sim.forward(2 * x.clamp(0, 0.5)), 2 * sim.forward(x.clamp(0, 0.5)), atol=1e-5)


@pytest.mark.parametrize("f", [3, 5])
def test_pixel_constant_object_matches_coarse_operator(op, f):
    """With a pixel-constant up-sampled object the fine-grid data must agree with A x up to the (small) projector
    discretisation difference. A wrong centring / scale / bin mapping would give a much larger error."""
    x = smooth_images(4)
    assert rel_err(FineGridSimulator(op, f, "nearest").forward(x), op.forward(x)) < 0.03


def test_test_has_power_against_misalignment(op):
    """Shifting the detector-bin mapping by ONE fine bin must make the agreement clearly worse."""
    x = smooth_images(4)
    sim = FineGridSimulator(op, 5, "nearest")
    good = rel_err(sim.forward(x), op.forward(x))
    sim.idx = np.clip(sim.idx + 1, 0, sim.D_f - 1)
    bad = rel_err(sim.forward(x), op.forward(x))
    assert bad > 3 * good, (good, bad)


@pytest.mark.parametrize("n_angles", [5, 90])
def test_other_angle_counts(n_angles):
    op = RadonOperator(28, n_angles)
    x = smooth_images(2)
    assert rel_err(FineGridSimulator(op, 3, "nearest").forward(x), op.forward(x)) < 0.03


def test_factor_consistency(op):
    x = smooth_images(3)
    e3 = rel_err(FineGridSimulator(op, 3, "nearest").forward(x), op.forward(x))
    e5 = rel_err(FineGridSimulator(op, 5, "nearest").forward(x), op.forward(x))
    assert abs(e3 - e5) < 0.5 * max(e3, e5) + 1e-3


def test_smooth_upsampling_adds_model_error_but_stays_small(op):
    x = smooth_images(3)
    en = rel_err(FineGridSimulator(op, 5, "nearest").forward(x), op.forward(x))
    eb = rel_err(FineGridSimulator(op, 5, "bilinear").forward(x), op.forward(x))
    assert eb < 0.10 and eb > 0.0 and en < 0.03


def test_make_simulator_off_is_identity(op):
    x = smooth_images(2)
    assert torch.equal(make_simulator(op, 0)(x), op.forward(x))
