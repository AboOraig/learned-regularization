"""Stage 3 / step 5 statistics: paired differences between methods in every (K, dose) condition, bootstrap over patients (slices are one per patient),
grid tables and a figure.  Usage: python run_stage3_step5_summary.py [--dir results/stage3/s5]
Outputs: step5_means.csv, step5_paired.csv, step5_grid.png"""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src import lesions_ct as C

p = argparse.ArgumentParser(); p.add_argument("--dir", default="results/stage3/s5"); p.add_argument("--n_boot", type=int, default=5000)
a = p.parse_args()
METHODS = ["FBP-Hann", "TV", "TV-long", "PnP-const", "PnP-TVinit"]
PAIRS = [("PnP-const", "TV-long"), ("PnP-TVinit", "TV-long"), ("PnP-const", "FBP-Hann"), ("TV-long", "FBP-Hann"), ("PnP-const", "TV")]
conds = {}
for fn in glob.glob(f"{a.dir}/step5_K*_dose*.npz"):
    m = re.search(r"K(\d+)_dose([\d.]+)\.npz", fn)
    conds[(int(m.group(1)), float(m.group(2)))] = dict(np.load(fn, allow_pickle=True))
if not conds:
    raise SystemExit(f"no step5_K*_dose*.npz in {a.dir}")
means, rows = [], []
for (K, f), R in sorted(conds.items()):
    S = len(R["psnr|TV"])
    for met in ("psnr", "ssim"):
        for m in METHODS:
            means.append(dict(K=K, dose=f, metric=met, method=m, mean=float(R[f"{met}|{m}"].mean()), n=S))
        for A, B in PAIRS:
            d = R[f"{met}|{A}"] - R[f"{met}|{B}"]
            lo, hi = C.cluster_boot(lambda i: float(d[i].mean()), S, a.n_boot, 1)
            rows.append(dict(K=K, dose=f, metric=met.upper(), A=A, B=B, diff=float(d.mean()), lo=lo, hi=hi))
    types = sorted({(float(k.split("|")[1]), float(k.split("|")[2])) for k in R if k.startswith("valid|")})
    for kind, name in (("auc", "AUC"), ("fl", "false@0.5x"), ("fl1", "false@1x"), ("fl1.5", "false@1.5x")):
        if not types:
            break
        for A, B in PAIRS:
            if kind != "auc" and "FBP-Hann" in (A, B):
                continue
            d, lo, hi = C.paired_diff(R, types, A, B, kind, a.n_boot, 1)
            rows.append(dict(K=K, dose=f, metric=name, A=A, B=B, diff=d, lo=lo, hi=hi))
M, P = pd.DataFrame(means), pd.DataFrame(rows)
P["excludes_0"] = (P.lo > 0) | (P.hi < 0)
M.to_csv(f"{a.dir}/step5_means.csv", index=False); P.to_csv(f"{a.dir}/step5_paired.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("=== mean PSNR (dB) / SSIM per condition ===")
print(M.pivot_table(index=["metric", "method"], columns=["K", "dose"], values="mean").round(3).to_string())
for (A, B) in PAIRS[:2]:
    for met in ("PSNR", "SSIM", "AUC", "false@1x"):
        q = P[(P.A == A) & (P.B == B) & (P.metric == met)]
        if q.empty:
            continue
        q = q.assign(cell=lambda d: d["diff"].map("{:+.3f}".format) + np.where(d.excludes_0, "*", " ") + d.apply(lambda r: f" [{r.lo:+.3f},{r.hi:+.3f}]", axis=1))
        print(f"\n--- {met}: {A} minus {B}   (* = 95% bootstrap interval over patients excludes 0) ---")
        print(q.pivot(index="dose", columns="K", values="cell").sort_index(ascending=False).to_string())
print("\n=== sign summary (PnP-const minus TV-long) ===")
for met in ("PSNR", "SSIM", "AUC", "false@1x"):
    q = P[(P.A == "PnP-const") & (P.B == "TV-long") & (P.metric == met)]
    if len(q):
        print(f"{met:9s}: positive in {int((q['diff'] > 0).sum())}/{len(q)} conditions, interval excludes 0 in {int(q.excludes_0.sum())} (positive: {int(((q['diff'] > 0) & q.excludes_0).sum())})")
# --- figure
panels = [("PSNR", "PnP-const", "TV-long", "dB"), ("PSNR", "PnP-TVinit", "TV-long", "dB"), ("SSIM", "PnP-const", "TV-long", ""),
          ("AUC", "PnP-const", "TV-long", ""), ("AUC", "PnP-const", "FBP-Hann", ""), ("false@1x", "PnP-const", "TV-long", "blobs/img")]
panels = [t for t in panels if len(P[(P.metric == t[0]) & (P.A == t[1]) & (P.B == t[2])])]
fig, axs = plt.subplots(2, 3, figsize=(13, 7)); axs = axs.ravel()
Ks, Fs = sorted({k for k, _ in conds}), sorted({f for _, f in conds}, reverse=True)
for ax, (met, A, B, unit) in zip(axs, panels):
    q = P[(P.metric == met) & (P.A == A) & (P.B == B)]
    Z = np.full((len(Fs), len(Ks)), np.nan); T = [[""] * len(Ks) for _ in Fs]
    for _, r in q.iterrows():
        i, j = Fs.index(r.dose), Ks.index(r.K); Z[i, j] = r["diff"]
        T[i][j] = f"{r['diff']:+.3f}{'*' if r.excludes_0 else ''}\n[{r.lo:+.3f},{r.hi:+.3f}]"
    v = np.nanmax(np.abs(Z)) if np.isfinite(Z).any() else 1
    ax.imshow(Z, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    for i in range(len(Fs)):
        for j in range(len(Ks)):
            ax.text(j, i, T[i][j], ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(Ks)), [f"K={k}" for k in Ks]); ax.set_yticks(range(len(Fs)), [f"dose {f:g}" for f in Fs])
    ax.set_title(f"{met}: {A} - {B} {unit}", fontsize=9)
for ax in axs[len(panels):]:
    ax.axis("off")
fig.suptitle("Paired differences by condition (* = 95% interval over patients excludes 0; blue = first method higher)", fontsize=10)
fig.tight_layout(); fig.savefig(f"{a.dir}/step5_grid.png", dpi=120)
print(f"\nwrote {a.dir}/step5_means.csv, step5_paired.csv, step5_grid.png")
