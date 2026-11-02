import numpy as np, torch
from src.radon_pb import ParallelBeamOperator
from src.ct_sim import CTSimulator, poisson_obs
from src import lesions as L
from src import lesions_ct as C


def _smooth(H):
    yy, xx = np.mgrid[:H, :H]
    return torch.tensor((0.5 * np.exp(-((yy - H * .45) ** 2 + (xx - H * .55) ** 2) / (2 * (H / 6) ** 2))
                         + 0.3 * np.exp(-((yy - H * .6) ** 2 + (xx - H * .35) ** 2) / (2 * (H / 9) ** 2))).astype(np.float32)).reshape(1, -1)


def test_simulator_matches_coarse_operator_on_smooth_image():
    H = 32
    op = ParallelBeamOperator(H, 47, n_angles=12, pixel_size=0.5)
    x = _smooth(H)
    y0, y1 = op.forward(x), CTSimulator(op).clean(x)
    assert float((y1 - y0).norm() / y0.norm()) < 0.03


def test_poisson_obs_statistics_and_zero_replacement():
    y = torch.full((1, 200000), 0.05)
    o = poisson_obs(y, 1.0, 0)
    lam = 4096 * np.exp(-81.35858 * 0.05)
    N = 4096 * torch.exp(-81.35858 * o.double())
    assert abs(N.mean().item() - lam) < 0.5 and abs(N.var().item() / lam - 1) < 0.05
    z = poisson_obs(torch.full((1, 1000), 0.2), 1.0, 1)             # expected counts ~ 0.0003: almost all zero -> replaced by 0.1
    assert abs(float(z.max()) - np.log(4096 / 0.1) / 81.35858) < 1e-4
    assert torch.equal(poisson_obs(y, 0.25, 3), poisson_obs(y, 0.25, 3))


def test_multi_lesion_insertion_and_responses():
    S = 64
    x = torch.zeros(2, S * S) + 0.2
    centers = torch.tensor([[[20, 20], [44, 44]], [[32, 32], [0, 0]]]); valid = torch.tensor([[True, True], [True, False]])
    xl = C.insert_multi(x, centers, valid, 2.0, 0.1, size=S)
    rho = L.lesion_rho(2.0)
    t = C.responses(xl, rho, centers, S)
    nom = L.nominal_response(2.0, 0.1, S)
    assert abs(float(t[0, 0]) - nom) < 0.002 and abs(float(t[0, 1]) - nom) < 0.002 and abs(float(t[1, 0]) - nom) < 0.002
    assert float((xl[1] - x[1]).abs().sum()) > 0 and abs(float(C.responses(x, rho, centers, S).abs().max())) < 1e-6


def test_false_lesions_zero_for_truth_and_positive_for_invented_blob():
    S = 64
    x = torch.zeros(1, S * S) + 0.2
    rho, th = L.lesion_rho(2.0), 0.5 * L.nominal_response(2.0, 0.1, S)
    f0, _ = L.count_false_lesions(x, x, rho, th, size=S)
    inv = C.insert_multi(x, torch.tensor([[[30, 30]]]), torch.tensor([[True]]), 2.0, 0.1, size=S)
    f1, _ = L.count_false_lesions(inv, x, rho, th, size=S)
    assert int(f0) == 0 and int(f1) == 1


def test_sampler_separation_and_mask():
    m = C.square_mask(120, 20, 99)
    cen, val = C.sample_centers([m, m], 4, 30, seed=1)
    assert val.all()
    for b in range(2):
        pts = cen[b].numpy()
        assert all(m[p[0], p[1]] for p in pts)
        assert min(np.hypot(*(pts[i] - pts[j])) for i in range(4) for j in range(i)) >= 30
    img = np.zeros((120, 120)); img[10:110, 10:110] = 0.3
    assert C.body_mask(img, 5)[60, 60] and not C.body_mask(img, 5)[12, 12]


def test_cluster_auc():
    t1 = np.array([[2., 2.], [1., 1.]]); t0 = np.array([[1., 1.], [1., 1.]])
    assert C.cluster_paired_auc(t1, t0, np.ones((2, 2))) == (1 + 1 + .5 + .5) / 4
    assert C.cluster_paired_auc(t1, t0, np.ones((2, 2)), idx=np.array([0, 0])) == 1.0
    lo, hi = C.cluster_boot(lambda i: C.cluster_paired_auc(t1, t0, np.ones((2, 2)), i), 2, 200)
    assert lo <= 0.75 <= hi


def test_shapes_phantom():
    from src.phantoms import shapes_phantom
    a, b = shapes_phantom(0), shapes_phantom(1)
    assert a.shape == (362 * 362,) and 0 <= a.min() and a.max() <= 1 and not np.array_equal(a, b)
    assert np.array_equal(a, shapes_phantom(0)) and a.std() > 0.1


def test_paired_diff_and_unsupported_map():
    rng = np.random.default_rng(0); S, M = 30, 4
    R = {"valid|2|0.06": np.ones((S, M), bool), "psnr|A": np.zeros(S), "psnr|B": np.zeros(S)}
    R["t0|2|0.06|A"] = rng.normal(size=(S, M)); R["t0|2|0.06|B"] = R["t0|2|0.06|A"].copy()
    R["t1|2|0.06|A"] = R["t0|2|0.06|A"] + 3.0; R["t1|2|0.06|B"] = R["t0|2|0.06|B"] + 0.0
    R["fl|2|0.06|A"] = np.full(S, 1.0); R["fl|2|0.06|B"] = np.full(S, 3.0)
    d, lo, hi = C.paired_diff(R, [(2.0, 0.06)], "A", "B", "auc", 200)
    assert d > 0.4 and lo > 0.3
    d, lo, hi = C.paired_diff(R, [(2.0, 0.06)], "A", "B", "fl", 200)
    assert d == -2.0 and lo == hi == -2.0
    S_, rho = 64, L.lesion_rho(2.0)
    x = torch.zeros(1, S_ * S_) + 0.2
    inv = C.insert_multi(x, torch.tensor([[[30, 30]]]), torch.tensor([[True]]), 2.0, 0.1, size=S_)
    nom = L.nominal_response(2.0, 0.1, S_)
    Rm, un = C.unsupported_map(inv, x, rho, nom, size=S_)
    assert int(((Rm >= 0.5 * nom) & un).sum()) == 1 and int(((Rm >= 1.5 * nom) & un).sum()) == 0
    Rm, un = C.unsupported_map(inv, inv, rho, nom, size=S_)
    assert int(((Rm >= 0.5 * nom) & un).sum()) == 0


def test_tune_climb_extends_and_stops():
    from src.tune_ct import climb
    f = lambda v: -(v - 5.0) ** 2
    b, sc = climb([1.0, 2.0], f, lambda b: b - 1 if b > 0 else None, lambda b: b + 1, 5, "t", log=lambda s: None)
    assert b == 5.0 and 6.0 in sc
    b, sc = climb([1.0, 2.0], f, lambda b: b - 1, lambda b: b + 1, 0, "t", log=lambda s: None)
    assert b == 2.0 and len(sc) == 2                           # no extension when rounds = 0


def test_paired_diff_threshold_kinds():
    rng = np.random.default_rng(1); S = 20
    R = {"psnr|A": np.zeros(S), "psnr|B": np.zeros(S), "fl1|2|0.06|A": np.full(S, 2.0), "fl1|2|0.06|B": np.full(S, 5.0)}
    d, lo, hi = C.paired_diff(R, [(2.0, 0.06)], "A", "B", "fl1", 100)
    assert d == -3.0 and lo == hi == -3.0
