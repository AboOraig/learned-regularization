"""Stage 1 / step 4: aggregate the per-model outputs of run_step4.py.
Prints (1) seed variability of PnP-TV on every test set, (2) the prior x test-set matrix (matched-prior control),
(3) the sigma_d grid-edge check for the noise sweep; writes step4_summary.csv, step4_noise.csv, step4_summary.png.
Usage: python summarize_step4.py [--dir results/step4]
"""
import argparse, glob, os, re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = argparse.ArgumentParser(); p.add_argument("--dir", default="results/step4"); a = p.parse_args()
H = "tau0.05"; pd.set_option("display.width", 250, "display.max_columns", 40)
FAMS = {"m03": "MNIST-trained, sigma_max 0.3", "m05": "MNIST-trained, sigma_max 0.5", "f03": "Fashion-trained, sigma_max 0.3"}
INDIST = {"m03": "MNIST (in-dist)", "m05": "MNIST (in-dist)", "f03": "Fashion-MNIST"}
COL = {"m03": "tab:blue", "m05": "tab:cyan", "f03": "tab:red"}


def load(kind):
    rows = []
    for f in sorted(glob.glob(f"{a.dir}/nullspace_pnp_vs_tv_{kind}_*.csv")):
        name = re.search(rf"{kind}_(\w+)\.csv", f).group(1); fam, seed = name.rsplit("_s", 1)
        sm = pd.read_csv(f); sm = sm[sm.subspace == H]
        mm = pd.read_csv(f.replace("pnp_vs_tv", "method_means")); mm = mm[mm.subspace == H]
        ps = mm.pivot_table(index=["dataset", "sigma", "n_angles"], columns="method", values="psnr")
        sm = sm.set_index(["dataset", "sigma", "n_angles"])
        d = pd.DataFrame({"psnr_TV": ps["TV"], "psnr_PnP": ps["PnP-DnCNN"], "total_ratio": sm["total_err_ratio_PnP_over_TV"],
                          "null_ratio": sm["null_err_ratio_PnP_over_TV"], "inject_PnP": sm["inject_PnP"], "cos_PnP": sm["cos_pooled_PnP"],
                          "sd": sm["pnp_sd"], "k": sm["pnp_k"]}).reset_index()
        d["gap"] = d.psnr_PnP - d.psnr_TV; d["family"] = fam; d["seed"] = int(seed); rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def ms(x, f="{:+.2f}"):
    x = np.asarray(x, float); return (f + " ± " + f.replace("+", "")).format(x.mean(), x.std(ddof=1) if len(x) > 1 else np.nan) + f"  [{f.format(x.min())}, {f.format(x.max())}] n={len(x)}"


sh = load("shift"); nz = load("noise")
if sh.empty:
    raise SystemExit(f"no shift outputs found in {a.dir}; run run_step4.py first")
sh["role"] = np.where(sh.apply(lambda r: INDIST[r.family] == r.dataset, axis=1), "in-dist", "shifted")
sh.to_csv(f"{a.dir}/step4_summary.csv", index=False)

print("=== (1) PnP - TV PSNR gap (dB) at sigma=0.05, 20 angles; mean ± sd over seeds [min, max] ===")
for fam in FAMS:
    d = sh[sh.family == fam]
    if d.empty: continue
    print(f"\n-- {fam}: {FAMS[fam]} (every method tuned on this family's own validation set)")
    for ds, g in d.groupby("dataset", sort=False):
        print(f"   {ds:26s} [{g.role.iloc[0]:8s}] gap {ms(g.gap)} | total err ratio PnP/TV {np.mean(g.total_ratio):.2f} "
              f"| tuned (sigma_d,k): {sorted(set(zip(g.sd, g.k)))}")

print("\n=== (2) prior x test set: mean PnP-TV gap (dB) over seeds; <0 means TV is better ===")
print(sh.pivot_table(index="family", columns="dataset", values="gap", aggfunc="mean").round(2).to_string())
print("\n    total-error ratio PnP/TV (<1 means PnP has less error):")
print(sh.pivot_table(index="family", columns="dataset", values="total_ratio", aggfunc="mean").round(2).to_string())

if not nz.empty:
    nz.to_csv(f"{a.dir}/step4_noise.csv", index=False)
    print("\n=== (3) noise sweep with the extended sigma_d grid (grid max 0.5): tuned sigma_d and PnP-TV gap ===")
    for fam in ("m03", "m05"):
        d = nz[nz.family == fam]
        if d.empty: continue
        for s, g in d.groupby("sigma"):
            print(f"   {fam} sigma={s:<5} gap {ms(g.gap)} | tuned sigma_d per seed {list(g.sd)} k {list(g.k)}")
    print("   (edge check: a tuned sigma_d equal to the largest value the model was trained for (0.3 / 0.5) still means 'at the edge')")

fig, axs = plt.subplots(1, 2 if not nz.empty else 1, figsize=(12 if not nz.empty else 6.5, 4.2), squeeze=False)
ax = axs[0, 0]; order = list(dict.fromkeys(sh.dataset)); off = {"m03": -0.2, "m05": 0.0, "f03": 0.2}
for fam in FAMS:
    d = sh[sh.family == fam]
    for i, ds in enumerate(order):
        g = d[d.dataset == ds]
        if len(g): ax.scatter(np.full(len(g), i) + off[fam], g.gap, color=COL[fam], s=26, label=FAMS[fam] if i == 0 else None)
ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(len(order))); ax.set_xticklabels([o.replace(" (x20 noise)", "").replace(" (in-dist)", "") for o in order], rotation=15, fontsize=8)
ax.set_ylabel("PnP − TV, PSNR (dB)"); ax.set_title("Shift, sigma=0.05 (dot = one denoiser seed)", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=7)
if not nz.empty:
    ax = axs[0, 1]
    for fam in ("m03", "m05"):
        d = nz[nz.family == fam]
        if len(d): ax.scatter(d.sigma + (-0.003 if fam == "m03" else 0.003), d.gap, color=COL[fam], s=26, label=FAMS[fam])
    ax.axhline(0, color="k", lw=.8); ax.set_xlabel("noise sigma"); ax.set_ylabel("PnP − TV, PSNR (dB)"); ax.set_title("In-distribution noise sweep", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{a.dir}/step4_summary.png", dpi=150); plt.close()
print(f"\nSaved: {a.dir}/step4_summary.csv, step4_noise.csv, step4_summary.png")
