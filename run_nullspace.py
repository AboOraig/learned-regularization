"""Stage 1 / step 2(b): null-space ("hallucination") metrics for FBP / Tikhonov / TV / PnP-DnCNN.

Re-runs the v1 protocol and adds, per image, how the reconstruction ERROR splits between the part of image
space that the measurements do not determine (the "subspace", src/nullspace.py) and the part they do, plus how
much energy each method puts into the subspace and how well it is aligned with the truth.

  --exp noise   E1 conditions (20 angles, sigma sweep),  hyperparameters tuned per condition on MNIST val
  --exp angles  E2 conditions (sigma=0.05, angle sweep), hyperparameters tuned per condition on MNIST val
  --exp shift   E4 conditions: params tuned ONCE on MNIST val at (sigma, 20 angles) and FROZEN (as run_robustness.py)
  --exp all     all three (default)

Usage:  python run_nullspace.py                      # everything
        python run_nullspace.py --exp shift          # only distribution shift
        python run_nullspace.py --exp shift --sets mnist,shepp   # subset of test sets
Outputs (results/):
  nullspace_metrics.csv       per image x method x subspace (includes the tuned hyperparameters)
  nullspace_pnp_vs_tv.csv     PnP vs TV summary (error split, injection, alignment, tuned params, p-values)
  nullspace_method_means.csv  pooled means for ALL four methods
  nullspace_shift.png / nullspace_sweeps.png / nullspace_signature.png
Empty subspaces (e.g. the exact null space at >= 40 angles) are skipped, not reported as NaN rows.
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


def score(recon, x_true, subs, meta, params):
    """Append per-image PSNR/SSIM + subspace metrics for every method; tuned hyperparameters stored on every row."""
    ptag = dict(tik_lam=float(params["Tikhonov"]), tv_lam=float(params["TV"]),
                pnp_sd=float(params["PnP"][0]), pnp_k=int(params["PnP"][1]))
    for name, xr in recon.items():
        xr = xr.clamp(0, 1)
        ps, ss, _ = per_image_metrics(xr, x_true)
        for sname, tau in SUBSPACES.items():
            d = subs.dim(tau)
            if d == 0:
                continue                      # empty subspace: nothing to measure
            m = subspace_metrics(subs, xr, x_true, tau)
            for i in range(len(ps)):
                rows.append(dict(**meta, **ptag, method=name, subspace=sname, dim=d, img=i,
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
        score(reconstruct_all(op, den, yt, params), test, subs, dict(experiment=exp, dataset="MNIST", **c), params)


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
        score(reconstruct_all(op, den, y, params), x, subs,
              dict(experiment="shift", dataset=name, n_angles=20, sigma=a.sigma), params)


for e in (["noise", "angles", "shift"] if a.exp == "all" else [a.exp]):
    run_sweep(e) if e in ("noise", "angles") else run_shift()

df = pd.DataFrame(rows)
df.to_csv(f"{a.out}/nullspace_metrics.csv", index=False)
KEYS = ["experiment", "dataset", "sigma", "n_angles", "subspace", "dim"]
COLS = ["null_err_energy", "range_err_energy", "total_err_energy", "recon_energy", "true_energy", "inner"]


def wil(x, y):
    if not np.any(np.asarray(x) != np.asarray(y)):
        return 1.0
    try:
        return wilcoxon(x, y).pvalue
    except ValueError:
        return np.nan


# ---- pooled means for all methods -----------------------------------------------------------------------------
mm = df.groupby(KEYS + ["method"])[COLS + ["psnr", "ssim"]].mean().reset_index()
mm["null_share"] = mm.null_err_energy / mm.total_err_energy
mm["inject"] = mm.recon_energy / mm.true_energy
mm["cos_pooled"] = mm.inner / np.sqrt(mm.true_energy * mm.recon_energy)
mm.to_csv(f"{a.out}/nullspace_method_means.csv", index=False)

# ---- summary: PnP vs TV inside each subspace ------------------------------------------------------------------
out = []
for k, g in df.groupby(KEYS):
    P = {c: g.pivot(index="img", columns="method", values=c) for c in COLS + ["cos_sub"]}
    if not {"TV", "PnP-DnCNN"} <= set(P["null_err_energy"].columns):
        continue
    row = dict(zip(KEYS, k), n=len(P["null_err_energy"]))
    for m, tag in (("TV", "TV"), ("PnP-DnCNN", "PnP")):
        mean = {c: P[c][m].mean() for c in COLS}
        row.update({f"null_err_{tag}": mean["null_err_energy"], f"range_err_{tag}": mean["range_err_energy"],
                    f"total_err_{tag}": mean["total_err_energy"],
                    f"null_share_{tag}": mean["null_err_energy"] / mean["total_err_energy"],
                    f"recon_energy_{tag}": mean["recon_energy"],
                    f"inject_{tag}": mean["recon_energy"] / mean["true_energy"],
                    f"cos_pooled_{tag}": mean["inner"] / np.sqrt(mean["true_energy"] * mean["recon_energy"]),
                    f"cos_median_{tag}": float(np.nanmedian(P["cos_sub"][m])),
                    f"psnr_{tag}": g[g.method == m].psnr.mean()})
    row["true_energy_in_subspace"] = P["true_energy"]["TV"].mean()
    for c, nm in (("null", "null_err_energy"), ("range", "range_err_energy"), ("total", "total_err_energy")):
        row[f"{c}_err_ratio_PnP_over_TV"] = row[f"{c}_err_PnP"] / row[f"{c}_err_TV"]
        row["p_wilcoxon" if c == "null" else f"p_wilcoxon_{c}"] = wil(P[nm]["PnP-DnCNN"], P[nm]["TV"])
    pp = g.iloc[0]
    row.update(tik_lam=pp.tik_lam, tv_lam=pp.tv_lam, pnp_sd=pp.pnp_sd, pnp_k=pp.pnp_k)
    out.append(row)
summ = pd.DataFrame(out)
summ.to_csv(f"{a.out}/nullspace_pnp_vs_tv.csv", index=False)

pd.set_option("display.width", 260, "display.max_columns", 40)
S = summ[summ.subspace == HEADLINE]
print(f"\n=== headline subspace '{HEADLINE}' (ratios are PnP/TV; <1 = PnP has less error) ===")
print(S[["experiment", "dataset", "sigma", "n_angles", "dim", "n", "null_err_ratio_PnP_over_TV", "range_err_ratio_PnP_over_TV",
         "total_err_ratio_PnP_over_TV", "inject_TV", "inject_PnP", "cos_pooled_TV", "cos_pooled_PnP",
         "pnp_sd", "pnp_k"]].round(3).to_string(index=False))
print("\n=== sensitivity: null-subspace error ratio PnP/TV by subspace choice ===")
print(summ.pivot_table(index=["experiment", "dataset", "sigma", "n_angles"], columns="subspace",
                       values="null_err_ratio_PnP_over_TV").round(3).to_string())
print("\nCaveats: Shepp-Logan rows are 20 noise draws of ONE image (p-values pseudo-replicated; the smallest possible"
      "\nWilcoxon p for n=20 is 1.9e-6). Cosines/injection are POOLED (ratio of means). Subspace dims differ per operator.")

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

    # hallucination signature: energy injected into the subspace (relative to truth) vs alignment with the truth
    M = mm[(mm.subspace == HEADLINE) & (mm.experiment == "shift")]
    fig, ax = plt.subplots(figsize=(7, 5)); mk = dict(zip(ds, ["o", "s", "^", "D", "v", "P"]))
    for _, r in M.iterrows():
        ax.scatter(r.inject, r.cos_pooled, color=COL[r.method], marker=mk[r.dataset], s=70, edgecolor="k", linewidth=.5)
    ax.axvline(1, color="k", ls=":", lw=1); ax.set_xlabel("energy put in subspace / true energy there   (>1: louder than the truth)")
    ax.set_ylabel("alignment with the truth (pooled cosine)"); ax.set_ylim(-0.05, 1.0); ax.grid(alpha=.3)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=c, label=m) for m, c in COL.items()] +
                      [Line2D([], [], marker=mk[d], ls="", color="gray", label=d) for d in ds], fontsize=7, loc="upper right")
    ax.set_title("Hallucination signature: loud AND misaligned = bottom right", fontsize=10)
    plt.tight_layout(); plt.savefig(f"{a.out}/nullspace_signature.png", dpi=150); plt.close()

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
print(f"\nSaved to {a.out}/: nullspace_metrics.csv, nullspace_pnp_vs_tv.csv, nullspace_method_means.csv, "
      "nullspace_shift.png, nullspace_sweeps.png, nullspace_signature.png")