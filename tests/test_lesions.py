"""Step 5: lesion insertion, contrast kernel, detection AUC, false-lesion counting, observability."""
import numpy as np, pytest, torch
from src import lesions as L
from src.nullspace import OperatorSubspaces
from src.operators import RadonOperator


def test_gaussian_lesion_peak_and_clipping():
    c = torch.tensor([[10, 12], [6, 21]])
    g = L.gaussian_lesion(c, 1.5).reshape(2, 28, 28)
    assert g[0, 10, 12] == pytest.approx(1.0) and g[1, 6, 21] == pytest.approx(1.0)
    assert g[0, 10, 13] < 1.0
    x = torch.full((2, 784), 0.9)
    assert L.insert_lesion(x, c, 1.5, 0.5).max() <= 1.0 and L.insert_lesion(x, c, 1.5, 0.5).min() >= 0.0


def test_contrast_kernel_is_zero_sum_and_constant_image_gives_zero():
    for rho in (2, 3):
        assert L.contrast_kernel(rho).sum().abs() < 1e-6
    x = torch.full((3, 784), 0.7)
    assert L.contrast_at(x, 2, torch.tensor([[8, 8], [14, 14], [20, 6]])).abs().max() < 1e-6


def test_contrast_response_peaks_at_lesion_centre_and_scales_with_contrast():
    c = torch.tensor([[14, 14]])
    for sl in (1.0, 1.75):
        rho = L.lesion_rho(sl)
        r1 = L.contrast_at(L.gaussian_lesion(c, sl) * 0.25, rho, c).item()
        r2 = L.contrast_at(L.gaussian_lesion(c, sl) * 0.5, rho, c).item()
        assert r1 > 0 and r2 == pytest.approx(2 * r1, rel=1e-5)
        off = L.contrast_at(L.gaussian_lesion(c, sl) * 0.5, rho, torch.tensor([[14, 19]])).item()
        assert off < 0.4 * r2                      # a centre 5 px away sees far less of the lesion
        assert L.nominal_response(sl, 0.5) == pytest.approx(r2, rel=1e-5)


def test_centre_range_fits_every_kernel():
    lo, hi = L.centre_range((1.0, 1.75))
    assert (lo, hi) == (6, 21)
    x = torch.rand(2, 784)
    L.contrast_at(x, 3, torch.tensor([[lo, lo], [hi, hi]]))        # must not index out of range


def test_auc_basics():
    assert L.auc([2, 3, 4], [0, 1, 1.5]) == 1.0
    assert L.auc([0, 1], [2, 3]) == 0.0
    assert L.auc([1, 1], [1, 1]) == pytest.approx(0.5)
    a = np.random.RandomState(0).randn(50); b = np.random.RandomState(1).randn(50) + 0.5
    assert L.auc(a, b) == pytest.approx(1 - L.auc(b, a))
    assert L.paired_auc([2, 0], [1, 1]) == 0.5
    lo, hi = L.bootstrap_auc_ci(b, a, n_boot=100)
    assert lo <= L.auc(b, a) <= hi


def test_false_lesion_counting():
    truth = torch.zeros(2, 784)
    c = torch.tensor([[14, 14], [10, 18]])
    blob = L.gaussian_lesion(c, 1.0) * 0.5
    theta = 0.5 * L.nominal_response(1.0, 0.5)
    f, n = L.count_false_lesions(blob, truth, 2, theta)
    assert f.tolist() == [1, 1] and n.tolist() == [1, 1]            # a blob that is not in the truth is a false lesion
    f2, n2 = L.count_false_lesions(truth + blob, truth + blob, 2, theta)
    assert f2.tolist() == [0, 0] and n2.tolist() == [1, 1]          # the same blob present in the truth is supported
    f3, _ = L.count_false_lesions(truth, truth, 2, theta)
    assert f3.tolist() == [0, 0]


def test_observability_extremes():
    op = RadonOperator(28, 20, device="cpu"); subs = OperatorSubspaces(op.A)
    weak = torch.tensor(subs.basis(0.05)[:, :3].T)                  # directions the data barely see
    strong = torch.tensor(subs.V[:, :3].T)                          # best-measured directions
    assert L.observability(subs, weak, 0.05).max() < 1e-6
    assert L.observability(subs, strong, 0.05).min() > 1 - 1e-6
    assert np.isnan(L.observability(subs, torch.zeros(1, 784), 0.05)).all()
