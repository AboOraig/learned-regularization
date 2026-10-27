"""Stage 1 / step 2: null-space ("hallucination") metrics for FBP / Tikhonov / TV / PnP-DnCNN.

Re-runs the v1 protocol and adds, per image, how much of the reconstruction ERROR lies in the part of
image space that the measurements do not determine (src/nullspace.py).

  --exp noise   E1 conditions (20 angles, sigma sweep),  hyperparameters tuned per condition on MNIST val
  --exp angles  E2 conditions (sigma=0.05, angle sweep), hyperparameters tuned per condition on MNIST val
  --exp shift   E4 conditions: params tuned ONCE on MNIST val at (sigma, 20 angles) and FROZEN (as run_robustness.py)
  --exp all     all three (default)

Usage:  python run_nullspace.py                      # everything
        python run_nullspace.py --exp shift          # only distribution shift
        python run_nullspace.py --exp shift --sets mnist,shepp   # subset of test sets (no downloads needed beyond MNIST)
Outputs (results/): nullspace_metrics.csv (per image), nullspace_pnp_vs_tv.csv (summary), nullspace_*.png
"""
import argparse, os, platform
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from src.operators import RadonOperator, add_noise
from src.denoiser import load_denoiser
from src.experiment import tune_all, reconstruct_all
from src.metrics import per_image_metrics
from src.nullspace import OperatorSubspaces, SUBSPACES, subspace_metrics
from src.data import load_mnist

p = argparse.ArgumentParser()
p.add_argument("--exp", default="all", choices=["noise", "angles", "shift", "all"])
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sigma", type=float, default=0.05, help="nominal noise for angles/shift experiments")
p.add_argument("--sets", default="mnist,fashion,emnist,shepp", help="comma list for --exp shift")
p.add_argument("--model", default="results/dncnn.pt")
p.add_argument("--out", default="results")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | numpy {np.__version__} | python {platform.python_version()} | device {dev}")
os.makedirs(a.out, exist_ok=True)
_, val, test = load_mnist(); val, test = val[:a.n_val].to(dev), test[:a.n_test].to(dev)
den = load_denoiser(a.model, dev)
HEADLINE = "tau0.05"     # subspace shown in the printed table and figures; the others are sensitivity checks
rows = []


def score(recon, x_true, subs, meta):
    """Append per-image PSNR/SSIM + subspace metrics (3 subspaces) for every method."""
    for name, xr in recon.items():
        xr = xr.clamp(0, 1)
        ps, ss, _ = per_image_metrics(xr, x_true)
        for sname, tau in SUBSPACES.items():
            m = subspace_metrics(subs, xr, x_true, tau)
            for i in range(len(ps)):
                rows.append(dict(**meta, method=name, subspace=sname, dim=subs.dim(tau), img=i,
                                 psnr=ps[i], ssim=ss[i], **{k: v[i] for k, v in m.items()}))


def run_sweep(exp):
    conds = ([dict(n_angles=20, sigma=s) for s in [0, 0.01, 0.05, 0.1, 0.2]] if exp == "noise"
             else [dict(n_angles=n, sigma=a.sigma) for n in [5, 10, 20, 40, 90]])
    for c in conds:
        op = RadonOperator(28, c["n_angles"], device=dev)
        params = tune_all(op, den, val, add_noise(op.forward(val), c["sigma"], seed=a.seed))
        yt = add_noise(op.forward(test), c["sigma"], seed=a.seed + 1)
        subs = OperatorSubspaces(op.A)
        print(f"[{exp}] {c} params={params} subspace dims={ {k: subs.dim(t) for k, t in SUBSPACES.items()} }", flush=True)
        score(reconstruct_all(op, den, yt, params), test, subs, dict(experiment=exp, dataset="MNIST", **c))


def run_shift():
    op = RadonOperator(28, 20, device=dev)
    params = tune_all(op, den, val, add_noise(op.forward(val), a.sigma, seed=a.seed))   # tuned once, then frozen
    subs = OperatorSubspaces(op.A)
    print(f"[shift] frozen params={params} subspace dims={ {k: subs.dim(t) for k, t in SUBSPACES.items()} }", flush=True)
    from src.data import load_fashion, load_emnist_letters, shepp_logan
    avail = {"mnist": ("MNIST (in-dist)", lambda: test),
             "fashion": ("Fashion-MNIST", lambda: load_fashion()[:a.n_test].to(dev)),
             "emnist": ("EMNIST-letters", lambda: load_emnist_letters()[:a.n_test].to(dev)),
             "shepp": ("Shepp-Logan (x20 noise)", lambda: shepp_logan().repeat(20, 1).to(dev))}
    for i, key in enumerate(avail):               # i fixed by position => same noise seeds as run_robustness.py
        if key not in a.sets.split(","):
            continue
        name, getx = avail[key]
        x = getx()
        y = add_noise(op.forward(x), a.sigma, seed=a.seed + 10 + i)
        score(reconstruct_all(op, den, y, params), x, subs, dict(experiment="shift", dataset=name, n_angles=20, sigma=a.sigma))


for e in (["noise", "angles", "shift"] if a.exp == "all" else [a.exp]):
    run_sweep(e) if e in ("noise", "angles") else run_shift()

df = pd.DataFrame(rows)
df.to_csv(f"{a.out}/nullspace_metrics.csv", index=False)
KEYS = ["experiment", "dataset", "sigma", "n_angles", "subspace", "dim"]

# ---- summary: PnP vs TV inside each subspace --------------------------------------------------------------
out = []
for k, g in df.groupby(KEYS):
    piv = g.pivot(index="img", columns="method", values="null_err_energy")
    if not {"TV", "PnP-DnCNN"} <= set(piv.columns):
        continue
    d = piv["PnP-DnCNN"] - piv["TV"]
    try:
        pv = wilcoxon(piv["PnP-DnCNN"], piv["TV"]).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        pv = np.nan
    gm = g.groupby("method")[["psnr", "total_err_energy", "recon_energy", "true_energy"]].mean()
    out.append(dict(zip(KEYS, k),
                    n=len(piv), null_err_TV=piv["TV"].mean(), null_err_PnP=piv["PnP-DnCNN"].mean(),
                    null_err_ratio_PnP_over_TV=piv["PnP-DnCNN"].mean() / piv["TV"].mean(),
                    p_wilcoxon=pv, true_energy_in_subspace=gm.loc["TV", "true_energy"],
                    recon_energy_TV=gm.loc["TV", "recon_energy"], recon_energy_PnP=gm.loc["PnP-DnCNN", "recon_energy"],
                    psnr_TV=gm.loc["TV", "psnr"], psnr_PnP=gm.loc["PnP-DnCNN", "psnr"]))
summ = pd.DataFrame(out)
summ.to_csv(f"{a.out}/nullspace_pnp_vs_tv.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
show = ["experiment", "dataset", "sigma", "n_angles", "n", "null_err_TV", "null_err_PnP", "null_err_ratio_PnP_over_TV",
        "p_wilcoxon", "true_energy_in_subspace", "psnr_TV", "psnr_PnP"]
for sname in SUBSPACES:
    s = summ[summ.subspace == sname]
    print(f"\n=== subspace '{sname}' (dim {s['dim'].iloc[0] if s['dim'].nunique() == 1 else 'varies with operator'}) :"
          " error energy inside the subspace, ratio<1 means PnP has LESS prior-driven error than TV ===")
    print(s[show].round(4).to_string(index=False))
print("\nCaveat: Shepp-Logan rows are 20 noise draws of ONE image -> p-values there are pseudo-replicated.")

# ---- figures -----------------------------------------------------------------------------------------------
H = df[df.subspace == HEADLINE]
COL = {"FBP": "tab:gray", "Tikhonov": "tab:green", "TV": "tab:orange", "PnP-DnCNN": "tab:blue"}
sh = H[H.experiment == "shift"]
if len(sh):
    ds = list(sh.dataset.unique()); methods = list(COL)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4)); w = 0.8 / len(methods)
    for j, (col, ttl) in enumerate([("null_err_energy", f"error energy inside subspace '{HEADLINE}'"),
                                    ("null_err_frac", "share of total error that lies inside the subspace")]):
        g = sh.groupby(["dataset", "method"])[col].agg(["mean", "sem"]).reset_index()
        for m_i, m in enumerate(methods):
            d = g[g.method == m].set_index("dataset").reindex(ds)
            ax[j].bar(np.arange(len(ds)) + m_i * w, d["mean"], w, yerr=1.96 * d["sem"], capsize=2, color=COL[m], label=m)
        ax[j].set_xticks(np.arange(len(ds)) + 0.4 - w / 2); ax[j].set_xticklabels(ds, rotation=12, fontsize=8)
        ax[j].set_title(ttl, fontsize=10); ax[j].grid(alpha=.3, axis="y")
    ax[0].legend(); plt.tight_layout(); plt.savefig(f"{a.out}/nullspace_shift.png", dpi=150); plt.close()

sw = [(e, "sigma", "measurement noise sigma") if e == "noise" else (e, "n_angles", "number of angles")
      for e in ("noise", "angles") if (H.experiment == e).any()]
if sw:
    fig, ax = plt.subplots(1, len(sw), figsize=(6 * len(sw), 4), squeeze=False)
    for j, (e, xc, xl) in enumerate(sw):
        for m in COL:
            g = H[(H.experiment == e) & (H.method == m)].groupby(xc).null_err_energy.agg(["mean", "sem"])
            ax[0, j].errorbar(g.index, g["mean"], yerr=1.96 * g["sem"], marker="o", capsize=3, color=COL[m], label=m)
        ax[0, j].set_xlabel(xl); ax[0, j].set_ylabel(f"error energy in '{HEADLINE}'"); ax[0, j].grid(alpha=.3)
    ax[0, 0].legend(); plt.tight_layout(); plt.savefig(f"{a.out}/nullspace_sweeps.png", dpi=150); plt.close()
print(f"\nSaved: {a.out}/nullspace_metrics.csv, nullspace_pnp_vs_tv.csv, nullspace_shift.png, nullspace_sweeps.png")
