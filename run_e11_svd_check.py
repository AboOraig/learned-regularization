"""E11 control: randomized SVD (matrix-free) versus the dense SVD, and the resulting strong/weak error split.

Reports (i) the largest singular-value error of the top-k randomized SVD relative to s_1 (k = 400 and the k = 600 used by strong_basis),
(ii) the dimension of the well-determined subspace {s_i/s_1 >= 0.05} from both, (iii) the largest principal angle between the two subspaces,
(iv) the difference between the dense and matrix-free strong-subspace error of real reconstructions (Tikhonov, TV-ADMM, PnP k=25 on the same
200 noisy test images as run_stage2.py --exp equiv), as a fraction of ||e||.
Usage (repository root):  python run_e11_svd_check.py        Output: results/control/e11_svd_check.csv
"""
import os, numpy as np, pandas as pd, torch
from src.operators import RadonOperator, add_noise
from src.radon_mf import MatrixFreeRadon
from src.finegrid import make_simulator
from src.mf_solvers import MFTikhonov, MFTVADMM, randomized_svd, strong_basis, error_split
from src.baselines import Tikhonov, TVADMM
from src.nullspace import OperatorSubspaces
from src.pnp import pnp_pgd
from src.denoiser import load_denoiser
from src.data import load_mnist

dev = "cuda" if torch.cuda.is_available() else "cpu"
D = RadonOperator(28, 20, device=dev); M = MatrixFreeRadon(28, 20, device=dev)
sub = OperatorSubspaces(D.A)
s_ref = torch.linalg.svdvals(D.A)
rows = []
for k in (400, 600):
    s, _ = randomized_svd(M, k=k, seed=0)
    err = ((s - s_ref[:k]).abs().max() / s_ref[0]).item()
    rows.append(dict(item=f"max |s_i - s_i(dense)| / s_1, first {k}", value=err))
Vs = strong_basis(M, 0.05)
rows.append(dict(item="dim strong subspace, matrix-free", value=Vs.shape[1]))
rows.append(dict(item="dim strong subspace, dense", value=784 - sub.dim(0.05)))
# principal angles between the strong subspaces: strong basis of the dense SVD
U, S, Vh = torch.linalg.svd(D.A, full_matrices=True)
Vd = Vh.T[:, : int((S / S[0] >= 0.05).sum())]
cosines = torch.linalg.svdvals(Vd.T @ Vs)
rows.append(dict(item="largest principal angle between strong subspaces (sin)", value=float(torch.sqrt((1 - cosines.min() ** 2).clamp_min(0)))))
# error split of real reconstructions
_, _, test = load_mnist(); x = test[:200].to(dev)
sim = make_simulator(M, 5, "bicubic"); y = add_noise(sim(x), 0.05, seed=1)
den = load_denoiser("results/dncnn.pt", dev)
recs = {"Tikhonov": MFTikhonov(M, 400, 1e-7)(y, 1e-2), "TV-ADMM": MFTVADMM(M)(y, 1e-2),
        "PnP(k=25)": pnp_pgd(M, y, den, 0.1, 1 / M.lipschitz(), (25,))[0][25]}
for name, xr in recs.items():
    tot, st, _ = error_split(Vs, xr, x)
    e = xr - x
    st_dense = (e @ Vd).norm(dim=1)
    d = ((st - st_dense).abs() / tot)
    rows.append(dict(item=f"strong-error difference / ||e||, {name}: max over images", value=d.max().item()))
    rows.append(dict(item=f"strong-error difference / ||e||, {name}: mean", value=d.mean().item()))
    rows.append(dict(item=f"strong share of error energy, {name} (dense)", value=((st_dense**2).sum() / (tot**2).sum()).item()))
df = pd.DataFrame(rows); os.makedirs("results/control", exist_ok=True)
df.to_csv("results/control/e11_svd_check.csv", index=False); print(df.to_string(index=False))