"""Stage 2: matrix-free operator + Poisson (low-dose) noise.

--exp equiv : dense vs matrix-free pipelines on the same Gaussian-noise data (should agree), plus timings.
--exp dose  : Poisson dose sweep (I0 photons/ray), methods FBP/Tikhonov/TV/PnP/PnP-weighted, several image sets.
All reconstruction uses ONLY forward/adjoint of MatrixFreeRadon (no dense A, no dense SVD).
Data are simulated crime-free on a finer grid (--fine_factor, default 5, bicubic) then Poisson-corrupted.
Hyperparameters are tuned on a held-out MNIST validation set at each dose and FROZEN for every test set.
"""
import argparse, os, time, platform
import numpy as np, pandas as pd, torch
from src.radon_mf import MatrixFreeRadon
from src.operators import RadonOperator, add_noise
from src.finegrid import make_simulator
from src.mf_solvers import (MFTikhonov, MFTVADMM, poisson_from_clean, pnp_pgd_weighted,
                            strong_basis, error_split)
from src.pnp import pnp_pgd
from src.denoiser import load_denoiser
from src.data import load_mnist, load_fashion, load_emnist_letters, shepp_logan
from src.metrics import per_image_metrics
from src.experiment import LAM_TIK, LAM_TV

p = argparse.ArgumentParser()
p.add_argument("--exp", default="dose", choices=["equiv", "dose"])
p.add_argument("--model", default="results/dncnn.pt")
p.add_argument("--n_angles", type=int, default=20)
p.add_argument("--doses", default="1000,300,100,30,10")
p.add_argument("--mu", type=float, default=2.0, help="attenuation scale: I = I0 exp(-mu * A x)")
p.add_argument("--sets", default="mnist,fashion,emnist,shepp")
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--fine_factor", type=int, default=5)
p.add_argument("--up_mode", default="bicubic")
p.add_argument("--sd_grid", default="0.01,0.02,0.05,0.1,0.15,0.2,0.3")
p.add_argument("--ck", default="0,1,2,3,5,10,25,50,100,200")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results/stage2")
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | python {platform.python_version()} | device {dev}")
os.makedirs(a.out, exist_ok=True)
SD = [float(v) for v in a.sd_grid.split(",")]; CK = tuple(int(v) for v in a.ck.split(","))
op = MatrixFreeRadon(28, a.n_angles, device=dev)
sim = make_simulator(op, a.fine_factor, a.up_mode)
den = load_denoiser(a.model, dev)
_, val, test = load_mnist()
val, test = val[:a.n_val].to(dev), test[:a.n_test].to(dev)
psnr = lambda x, t: per_image_metrics(x.clamp(0, 1), t)[0].mean()
sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)


def timed(f):
    sync(); t = time.time(); r = f(); sync(); return r, time.time() - t


if a.exp == "equiv":
    dop = RadonOperator(28, a.n_angles, device=dev)
    from src.baselines import Tikhonov, TVADMM
    x = test
    y = add_noise(sim(x), 0.05, seed=1)
    print(f"forward rel diff dense vs MF: {((op.forward(x)-dop.forward(x)).norm()/dop.forward(x).norm()).item():.2e}")
    for name, fa, fb in [("Tikhonov", lambda: Tikhonov(dop)(y, 1e-2), lambda: MFTikhonov(op, 400, 1e-7)(y, 1e-2)),
                         ("TV-ADMM", lambda: TVADMM(dop)(y, 1e-2), lambda: MFTVADMM(op)(y, 1e-2)),
                         ("PnP(k=25)", lambda: pnp_pgd(dop, y, den, 0.1, 1/dop.lipschitz(), (25,))[0][25],
                                       lambda: pnp_pgd(op, y, den, 0.1, 1/op.lipschitz(), (25,))[0][25])]:
        (xa, ta), (xb, tb) = timed(fa), timed(fb)
        print(f"{name:10s} dense {ta:6.2f}s  MF {tb:6.2f}s  PSNR dense {psnr(xa,x):.3f} MF {psnr(xb,x):.3f}  "
              f"rel diff {((xa-xb).norm()/xa.norm()).item():.2e}")
    print("peak GPU mem (MB):", torch.cuda.max_memory_allocated() / 1e6 if dev == "cuda" else "n/a")
    raise SystemExit

sets = {}
for s in a.sets.split(","):
    sets[s] = (test if s == "mnist" else load_fashion() if s == "fashion" else
               load_emnist_letters() if s == "emnist" else shepp_logan())
sets = {k: (v[:a.n_test] if torch.is_tensor(v) else torch.as_tensor(v)[:a.n_test]).float().reshape(-1, 784).to(dev)
        for k, v in sets.items()}
Vs = strong_basis(op, 0.05)
print(f"strong subspace dim (s/s_max>=0.05): {Vs.shape[1]} of 784 (matrix-free randomized SVD)")
tik, tv = MFTikhonov(op), MFTVADMM(op)


def reconstruct(y, N, I0, par):
    sd, k = par["PnP"]; sdw, kw = par["PnPw"]
    return {"FBP": op.fbp(y).clamp(0, 1), "Tikhonov": tik(y, par["Tikhonov"]), "TV": tv(y, par["TV"]),
            "PnP": pnp_pgd(op, y, den, sd, 1 / op.lipschitz(), (k,))[0][k],
            "PnP-weighted": pnp_pgd_weighted(op, y, N, I0, den, sdw, 1.0, (kw,))[kw]}


rows = []
for I0 in [float(v) for v in a.doses.split(",")]:
    t0 = time.time()
    yv, Nv = poisson_from_clean(sim(val), I0, a.seed, a.mu)
    par = {"Tikhonov": max(LAM_TIK, key=lambda l: psnr(tik(yv, l), val)),
           "TV": max(LAM_TV, key=lambda l: psnr(tv(yv, l), val))}
    for tag, runner in (("PnP", lambda sd: {k: x for k, x in pnp_pgd(op, yv, den, sd, 1 / op.lipschitz(), CK)[0].items()}),
                        ("PnPw", lambda sd: pnp_pgd_weighted(op, yv, Nv, I0, den, sd, 1.0, CK))):
        best = -1
        for sd in SD:
            for k, x in runner(sd).items():
                s = float(psnr(x, val))
                if s > best: best, par[tag] = s, (sd, k)
    clean_val = sim(val)
    print(f"I0={I0:g}: noise std (val) {float((yv - clean_val).std()):.4f} | params {par} | tuned in {time.time()-t0:.0f}s", flush=True)
    for name, xt in sets.items():
        ct = sim(xt); y, N = poisson_from_clean(ct, I0, a.seed + 1, a.mu)
        model_err = (ct - op.forward(xt)).norm(dim=1).mean().item()
        for meth, xr in reconstruct(y, N, I0, par).items():
            xr = xr.clamp(0, 1); ps, ss, _ = per_image_metrics(xr, xt)
            tot, st, out = error_split(Vs, xr, xt)
            for i in range(len(ps)):
                rows.append(dict(I0=I0, noise_std=float((y - ct).std()), dataset=name, method=meth, img=i, psnr=ps[i],
                                 ssim=ss[i], err=float(tot[i]), err_strong=float(st[i]), err_weak=float(out[i]),
                                 model_err=model_err, **{f"p_{k}": str(v) for k, v in par.items()}))
df = pd.DataFrame(rows); df.to_csv(f"{a.out}/dose_sweep.csv", index=False)
g = df.groupby(["dataset", "I0", "method"])[["psnr", "err_strong", "err_weak"]].mean().round(3)
pv = g["psnr"].unstack("method")[["FBP", "Tikhonov", "TV", "PnP", "PnP-weighted"]]
pv["PnP-TV"] = pv["PnP"] - pv["TV"]; pv["PnPw-TV"] = pv["PnP-weighted"] - pv["TV"]
print(pv.round(2).to_string())
ratio = (g["err_weak"].unstack("method")["PnP"] / g["err_weak"].unstack("method")["TV"]).round(2).rename("weak-err PnP/TV")
print(ratio.to_string())
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, len(sets), figsize=(4 * len(sets), 3.2), sharey=False)
    ax = np.atleast_1d(ax)
    for axi, ds in zip(ax, sets):
        for m in ["FBP", "Tikhonov", "TV", "PnP", "PnP-weighted"]:
            q = g.loc[ds, "psnr"].unstack("method")[m].sort_index(); axi.plot(q.index, q.values, "o-", label=m)
        axi.set_xscale("log"); axi.set_title(ds); axi.set_xlabel("I0 (photons/ray)"); axi.set_ylabel("PSNR (dB)")
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{a.out}/dose_sweep.png", dpi=130)
except Exception as e:
    print("plot skipped:", e)
