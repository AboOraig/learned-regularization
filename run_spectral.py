"""Stage 1 / step 2c: (1) spectral error profile, (2) iteration-budget control for PnP under distribution shift.

Question: under shift (Fashion-MNIST, Shepp-Logan) most of PnP's extra error is NOT in the 'tau0.05' subspace.
  (1) WHERE along the singular-value spectrum (bands of s_i/s_max) does each method's error live?
  (2) Is the failure an artefact of the early stopping tuned on MNIST (k=5: the data step barely touches directions
      below ~0.3*s_max), or does the denoiser's own fixed point produce it?  -> run PnP with k = 1 ... 400 at the
      SAME sigma_d and watch error, alignment and data residual as the data step engages.

Setup = the E4 shift setup of run_nullspace.py: nominal operator (20 angles), noise sigma, hyperparameters tuned
ONCE on MNIST val and frozen; same noise seeds. By default the frozen hyperparameters are READ from
results/nullspace_pnp_vs_tv.csv (written by run_nullspace.py) so both analyses use identical settings; use --retune
to tune again from scratch (slow).

Usage:  python run_spectral.py
        python run_spectral.py --sets mnist,fashion          # subset
        python run_spectral.py --sds 0.05,0.2                # also run other sigma_d values (extra curves)
Outputs (results/): spectral_bands.csv, iteration_control.csv, spectral_profile.png, iteration_control.png
How to read it: if PnP's error on Fashion/Shepp-Logan FALLS toward TV's as k grows (and the data residual falls toward
the noise level), early stopping tuned in-distribution is the culprit. If error stays high / alignment stays low even
when the data residual is small, the denoiser's fixed point itself hallucinates.
"""
import argparse, os, platform
import numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.operators import RadonOperator, add_noise
from src.denoiser import load_denoiser
from src.baselines import Tikhonov, TVADMM
from src.experiment import tune_all
from src.pnp import pnp_pgd
from src.metrics import per_image_metrics
from src.nullspace import OperatorSubspaces, SUBSPACES, subspace_metrics, band_metrics, band_labels, BAND_EDGES
from src.data import load_mnist

p = argparse.ArgumentParser()
p.add_argument("--n_val", type=int, default=100)
p.add_argument("--n_test", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sigma", type=float, default=0.05)
p.add_argument("--sets", default="mnist,fashion,emnist,shepp")
p.add_argument("--ks", default="1,2,5,10,25,50,100,200,400", help="PnP iteration counts to record")
p.add_argument("--sds", default="", help="extra sigma_d values besides the tuned one, comma list")
p.add_argument("--model", default="results/dncnn.pt")
p.add_argument("--params_csv", default="results/nullspace_pnp_vs_tv.csv")
p.add_argument("--retune", action="store_true")
p.add_argument("--out", default="results")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"torch {torch.__version__} | numpy {np.__version__} | python {platform.python_version()} | device {dev}")
os.makedirs(a.out, exist_ok=True)
_, val, test = load_mnist(); val, test = val[:a.n_val].to(dev), test[:a.n_test].to(dev)
den = load_denoiser(a.model, dev)
op = RadonOperator(28, 20, device=dev)
subs = OperatorSubspaces(op.A)
eta = 1.0 / op.lipschitz()
TAU = SUBSPACES["tau0.05"]
labels = band_labels()


def frozen_params():
    if not a.retune and os.path.exists(a.params_csv):
        t = pd.read_csv(a.params_csv)
        r = t[(t.experiment == "shift") & (t.dataset == "MNIST (in-dist)") & (t.subspace == "tau0.05") & (np.abs(t.sigma - a.sigma) < 1e-9)]
        if len(r):
            r = r.iloc[0]
            print(f"frozen params read from {a.params_csv}")
            return {"Tikhonov": float(r.tik_lam), "TV": float(r.tv_lam), "PnP": (float(r.pnp_sd), int(r.pnp_k))}
        print(f"no matching shift row in {a.params_csv} -> tuning")
    return tune_all(op, den, val, add_noise(op.forward(val), a.sigma, seed=a.seed))


params = frozen_params()
sd0, k0 = params["PnP"]
sds = sorted({sd0, *[float(s) for s in a.sds.split(",") if s]})
ks = sorted({int(k) for k in a.ks.split(",")} | {k0})
print(f"frozen params: Tikhonov lam={params['Tikhonov']:.4g}, TV lam={params['TV']:.4g}, PnP sigma_d={sd0}, k={k0}; "
      f"recording PnP at k={ks}, sigma_d={sds}")

from src.data import load_fashion, load_emnist_letters, shepp_logan
avail = {"mnist": ("MNIST (in-dist)", lambda: test),
         "fashion": ("Fashion-MNIST", lambda: load_fashion()[:a.n_test].to(dev)),
         "emnist": ("EMNIST-letters", lambda: load_emnist_letters()[:a.n_test].to(dev)),
         "shepp": ("Shepp-Logan (x20 noise)", lambda: shepp_logan().repeat(20, 1).to(dev))}

sum_rows, band_rows = [], []
noise_norm = a.sigma * np.sqrt(op.m) if a.sigma > 0 else np.nan     # expected ||n|| of the measurement noise


def record(ds, method, k, sd, xr, x_true, y):
    xr = xr.clamp(0, 1)
    ps, ss, _ = per_image_metrics(xr, x_true)
    m = subspace_metrics(subs, xr, x_true, TAU)
    b = band_metrics(subs, xr, x_true)
    resid = (op.forward(xr) - y).norm(dim=1).cpu().numpy().mean()
    rec, tru, inn = m["recon_energy"].mean(), m["true_energy"].mean(), m["inner"].mean()
    sum_rows.append(dict(dataset=ds, method=method, k=k, sigma_d=sd, is_tuned=bool(method == "PnP-DnCNN" and sd == sd0 and k == k0),
                         n=len(ps), psnr=ps.mean(), ssim=ss.mean(), total_err=m["total_err_energy"].mean(),
                         null_err=m["null_err_energy"].mean(), range_err=m["range_err_energy"].mean(),
                         recon_energy=rec, true_energy=tru, inject=rec / tru, cos_pooled=inn / np.sqrt(tru * rec),
                         resid=resid, resid_over_noise=resid / noise_norm))
    for j, lab in enumerate(labels):
        if b["dims"][j] == 0:
            continue
        r_, t_, i_ = b["recon"][:, j].mean(), b["true"][:, j].mean(), b["inner"][:, j].mean()
        band_rows.append(dict(dataset=ds, method=method, k=k, sigma_d=sd, band=lab, lo=BAND_EDGES[j], hi=BAND_EDGES[j + 1],
                              dims=int(b["dims"][j]), err=b["err"][:, j].mean(), true_energy=t_, recon_energy=r_,
                              cos_pooled=i_ / np.sqrt(t_ * r_) if t_ * r_ > 1e-18 else np.nan))


for i, key in enumerate(avail):                  # i fixed by position => same noise seeds as run_nullspace.py / run_robustness.py
    if key not in a.sets.split(","):
        continue
    ds, getx = avail[key]
    x = getx()
    y = add_noise(op.forward(x), a.sigma, seed=a.seed + 10 + i)
    record(ds, "FBP", np.nan, np.nan, op.fbp(y), x, y)
    record(ds, "Tikhonov", np.nan, np.nan, Tikhonov(op)(y, params["Tikhonov"]), x, y)
    record(ds, "TV", np.nan, np.nan, TVADMM(op)(y, params["TV"]), x, y)
    for sd in sds:
        outs, _ = pnp_pgd(op, y, den, sd, eta, tuple(ks))
        for k in ks:
            record(ds, "PnP-DnCNN", k, sd, outs[k], x, y)
    print(f"done {ds}", flush=True)

S, B = pd.DataFrame(sum_rows), pd.DataFrame(band_rows)
S.to_csv(f"{a.out}/iteration_control.csv", index=False)
B.to_csv(f"{a.out}/spectral_bands.csv", index=False)
dsets = list(S.dataset.unique())

# ---- printed summary ---------------------------------------------------------------------------------------------
pd.set_option("display.width", 220, "display.max_columns", 30)
print(f"\nExpected measurement-noise norm ||n|| ~ {noise_norm:.2f}. 'sub' = subspace tau0.05 (dim {subs.dim(TAU)}); "
      "ratios are PnP/TV; '*' marks the MNIST-tuned k.")
for ds in dsets:
    t = S[S.dataset == ds]
    tv = t[t.method == "TV"].iloc[0]
    print(f"\n--- {ds} ---   TV: psnr {tv.psnr:.2f} | total err {tv.total_err:.2f} | sub err {tv.null_err:.2f} | "
          f"inject {tv.inject:.2f} | cos {tv.cos_pooled:.2f} | resid {tv.resid:.2f}")
    for sd in sds:
        q = t[(t.method == "PnP-DnCNN") & (t.sigma_d == sd)].sort_values("k")
        tab = pd.DataFrame(dict(k=[f"{int(k)}{'*' if (sd == sd0 and k == k0) else ''}" for k in q.k], psnr=q.psnr.values,
                                total_over_TV=(q.total_err / tv.total_err).values, sub_over_TV=(q.null_err / tv.null_err).values,
                                inject=q.inject.values, cos=q.cos_pooled.values, resid=q.resid.values))
        print(f"PnP sigma_d={sd}:"); print(tab.round(3).to_string(index=False))
        best = q.loc[q.total_err.idxmin()]
        print(f"  -> lowest total error at k={int(best.k)} (ratio to TV {best.total_err / tv.total_err:.2f}); "
              f"at k={int(q.k.max())}: ratio {q.iloc[-1].total_err / tv.total_err:.2f}, cos {q.iloc[-1].cos_pooled:.2f}, inject {q.iloc[-1].inject:.2f}")
print("\nCaveat: Shepp-Logan = 20 noise draws of ONE image; read it as an illustration. Pooled (ratio-of-means) quantities.")

# ---- figure 1: spectral error profile ----------------------------------------------------------------------------
kmax = max(ks)
fig, axs = plt.subplots(1, len(dsets), figsize=(4.4 * len(dsets), 4.2), sharey=True, squeeze=False)
for ax, ds in zip(axs[0], dsets):
    sel = B[B.dataset == ds]
    labs = [l for l in labels if l in set(sel.band)]
    dims = sel[sel.method == "TV"].set_index("band").reindex(labs).dims.values
    series = [("TV", sel[sel.method == "TV"], "tab:orange"),
              (f"PnP k={k0} (tuned)", sel[(sel.method == "PnP-DnCNN") & (sel.sigma_d == sd0) & (sel.k == k0)], "tab:blue")]
    if kmax != k0:
        series.append((f"PnP k={kmax}", sel[(sel.method == "PnP-DnCNN") & (sel.sigma_d == sd0) & (sel.k == kmax)], "tab:purple"))
    w = 0.8 / len(series); xs = np.arange(len(labs))
    for j, (nm, d_, c) in enumerate(series):
        ax.bar(xs + (j - (len(series) - 1) / 2) * w, d_.set_index("band").reindex(labs).err.values, w, color=c, label=nm)
    ax.plot(xs, sel[sel.method == "TV"].set_index("band").reindex(labs).true_energy.values, "k_", ms=16, mew=2, ls="", label="true energy in band")
    ax.set_yscale("log"); ax.set_title(ds, fontsize=10); ax.grid(alpha=.3, axis="y")
    ax.set_xticks(xs); ax.set_xticklabels([f"{l}\n({d})" for l, d in zip(labs, dims)], fontsize=7)
    ax.set_xlabel("band of s/s_max  (#directions)", fontsize=8)
axs[0, 0].set_ylabel("error energy in band (log)"); axs[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{a.out}/spectral_profile.png", dpi=150); plt.close()

# ---- figure 2: iteration-budget control --------------------------------------------------------------------------
cols = plt.cm.tab10(np.arange(len(dsets)))
panels = [("total_err", "total error energy", True), ("null_err", f"error inside subspace tau0.05", True),
          ("cos_pooled", "alignment with truth in subspace (cosine)", False), ("resid", "data residual ||A x - y||", False)]
fig, axs = plt.subplots(1, 4, figsize=(18, 4)); sdp = sd0
for ax, (col, ttl, logy) in zip(axs, panels):
    for ds, c in zip(dsets, cols):
        t = S[S.dataset == ds]
        q = t[(t.method == "PnP-DnCNN") & (t.sigma_d == sdp)].sort_values("k")
        ax.plot(q.k, q[col], "-o", color=c, ms=4, label=ds)
        ax.axhline(t[t.method == "TV"].iloc[0][col], color=c, ls="--", lw=1)
        ax.plot([k0], q[q.k == k0][col], "o", mfc="none", mec="k", ms=11)
    if col == "resid" and a.sigma > 0:
        ax.axhline(noise_norm, color="gray", ls=":", lw=1.5); ax.text(ks[0], noise_norm, " expected noise norm", fontsize=7, va="bottom")
    ax.set_xscale("log"); ax.set_xlabel(f"PnP iterations k  (sigma_d={sdp})"); ax.set_title(ttl, fontsize=10); ax.grid(alpha=.3)
    if logy: ax.set_yscale("log")
axs[0].legend(fontsize=7)
fig.text(0.5, -0.02, "dashed = TV (same colour = same dataset); ring = MNIST-tuned k", ha="center", fontsize=8)
plt.tight_layout(); plt.savefig(f"{a.out}/iteration_control.png", dpi=150, bbox_inches="tight"); plt.close()
print(f"\nSaved to {a.out}/: iteration_control.csv, spectral_bands.csv, spectral_profile.png, iteration_control.png")
