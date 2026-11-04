import os, tempfile
import numpy as np, torch
from skimage.transform import radon
from src.radon_pb import ParallelBeamOperator


def test_matches_skimage_odd_geometry():
    H = 27; D = int(np.ceil(np.sqrt(2) * H)); D += (D + 1) % 2 if D % 2 == 0 else 0     # odd canvas
    ang = np.linspace(0, 180, 9, endpoint=False)
    x = torch.rand(2, H * H)
    op = ParallelBeamOperator(H, D, angles_deg=ang)
    ours = op.forward(x).reshape(2, len(ang), D)
    for b in range(2):
        ref = radon(x[b].reshape(H, H).numpy().astype(float), theta=ang, circle=False)     # (det_ref, angles)
        d0 = (ref.shape[0] - D) // 2
        ref = ref[d0:d0 + D] if ref.shape[0] >= D else np.pad(ref, ((-d0, -d0), (0, 0)))
        assert np.abs(ours[b].T.numpy() - ref).max() / np.abs(ref).max() < 2e-3


def test_adjoint_dot_and_chunking():
    ang = np.linspace(0, 180, 20, endpoint=False)
    a = ParallelBeamOperator(31, 45, angles_deg=ang, chunk_angles=4, chunk_batch=2, pixel_size=0.3)
    b = ParallelBeamOperator(31, 45, angles_deg=ang, chunk_angles=20, chunk_batch=8, pixel_size=0.3)
    u, v = torch.rand(3, 31 * 31), torch.rand(3, a.m)
    assert torch.allclose(a.forward(u), b.forward(u), atol=1e-5)
    lhs, rhs = (a.forward(u) * v).sum(), (u * a.adjoint(v)).sum()
    assert abs(lhs - rhs) / abs(lhs) < 1e-5


def test_flip_det_reverses():
    ang = np.linspace(0, 180, 6, endpoint=False); x = torch.rand(1, 29 * 29)
    p = ParallelBeamOperator(29, 41, angles_deg=ang).forward(x).reshape(6, 41)
    q = ParallelBeamOperator(29, 41, angles_deg=ang, flip_det=True).forward(x).reshape(6, 41)
    assert torch.allclose(p.flip(-1), q, atol=1e-6)


def test_fbp_reconstructs_smooth_phantom():
    H = 64; D = 91; K = 180
    yy, xx = np.mgrid[:H, :H]; r = np.hypot(yy - 31.5, xx - 31.5)
    img = (np.exp(-((yy - 28) ** 2 + (xx - 36) ** 2) / 60) + 0.5 * (r < 20)).astype(np.float32)
    op = ParallelBeamOperator(H, D, n_angles=K, pixel_size=1.0)
    x = torch.tensor(img).reshape(1, -1)
    rec = op.fbp(op.forward(x))
    assert (rec - x).norm() / x.norm() < 0.12


def test_loader_on_synthetic_files():
    import h5py
    from src.lodopab import LoDoPaB, PER_FILE
    with tempfile.TemporaryDirectory() as d:
        n = PER_FILE + 5                       # 2 files, second with 5 valid slices
        for k in range(2):
            for kind, shape in (("ground_truth", (PER_FILE, 8, 8)), ("observation", (PER_FILE, 3, 4))):
                os.makedirs(f"{d}/{kind}_test", exist_ok=True)
                with h5py.File(f"{d}/{kind}_test/{kind}_test_{k:03d}.hdf5", "w") as h:
                    h.create_dataset("data", data=np.full(shape, 10 * k, np.float32) + np.arange(PER_FILE)[:, None, None])
        open(f"{d}/patient_ids_rand_test.csv", "w").write("\n".join(str(i // 3) for i in range(n)) + "\n")
        ds = LoDoPaB(d, "test")
        assert len(ds) == n
        g = ds.gt([0, 127, 128, 132]); assert g.shape == (4, 8, 8)
        assert [float(v[0, 0]) for v in g] == [0, 127, 10, 14]
        assert ds.obs([130]).shape == (1, 3, 4)
        assert len(ds.patient_ids) == n
        ds.close()
