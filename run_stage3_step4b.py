"""Stage 3 / step 4b: sharper statistics and a look at the false structures behind step 4.
Part A (always, seconds, needs only results/stage3/step4_<set>_dose<f>.npz): PAIRED differences between methods (same slices, same bootstrap
   resamples over slices) of location-known AUC and false blobs per image, per lesion type and pooled over the four types
   (equal weight). Primary comparisons are the pooled rows; per-type rows are exploratory (many comparisons, expect a few chance 'significant').
Part B (--gallery 1, needs GPU, ~10 min/dose): re-creates the lesion-FREE reconstructions of step 4 (same seeds), then
   (i) sweeps the false-blob threshold theta = m x (nominal lesion response) for m = 0.5, 0.75, 1, 1.5: a blob at m >= 1 is as strong as
       the lesion it imitates, so the sweep separates residual noise (dies out at small m) from lesion-sized invented structure (survives at m >= 1);
   (ii) saves galleries of the strongest unsupported blobs of TV and PnP-const on the phantoms and on the CT slices (truth, FBP, TV, PnP-const, PnP-TVinit crops).
Usage: python run_stage3_step4b.py                       (part A)
       python run_stage3_step4b.py --gallery 1           (A + B, both doses)
Outputs (results/stage3): step4b_paired.csv, step4b_theta_sweep.csv, step4b_gallery_<set>_dose<f>.png"""
import argparse, os, glob, json
import numpy as np, pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--out", default="results/stage3"); p.add_argument("--root", default="data")
p.add_argument("--model", default="results/stage3/dncnn_ct.pt")
p.add_argument("--doses", default="1,0.25"); p.add_argument("--angles", type=int, default=60)
p.add_argument("--gallery", type=int, default=0); p.add_argument("--n_ct", type=int, default=40); p.add_argument("--n_shapes", type=int, default=20)
p.add_argument("--tv_iters", type=int, default=150); p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_boot", type=int, default=2000); p.add_argument("--n_gal", type=int, default=4)
p.add_argument("--override", default="", help="hann=..,lam=..,const=sd/k,tvinit=sd/k  (all doses; for smoke tests only)")
a = p.parse_args()
fl = lambda s: [float(v) for v in s.split(",")]
from src import lesions_ct as C

METHODS = ["FBP-Hann", "TV", "TV-long", "PnP-const", "PnP-TVinit"]
PAIRS = [("PnP-const", "TV"), ("PnP-const", "TV-long"), ("PnP-TVinit", "TV-long"), ("PnP-TVinit", "TV"), ("PnP-const", "FBP-Hann"), ("TV", "FBP-Hann")]

# ------------------------------------------------------------------ part A
rows = []
for f in fl(a.doses):
    for name in ("ct", "shapes"):
        fn = f"{a.out}/step4_{name}_dose{f:g}.npz"
        if not os.path.exists(fn):
            print(f"missing {fn}"); continue
        R = dict(np.load(fn, allow_pickle=True))
        types = sorted({(float(k.split("|")[1]), float(k.split("|")[2])) for k in R if k.startswith("valid|")})
        for kind in ("auc", "fl"):
            for mA, mB in PAIRS:
                if kind == "fl" and "FBP-Hann" in (mA, mB):
                    continue                                   # FBP false blobs are residual noise, not comparable
                for tag, tl in [("pooled", types)] + [(f"s{s:g}_c{c:g}", [(s, c)]) for s, c in types]:
                    d, lo, hi = C.paired_diff(R, tl, mA, mB, kind, a.n_boot, 1)
                    rows.append(dict(set=name, dose=f, types=tag, metric="AUC" if kind == "auc" else "false_per_img", A=mA, B=mB, diff=d, lo=lo, hi=hi,
                                     excludes_0=bool(lo > 0 or hi < 0), n_slices=len(R["psnr|TV"])))
P = pd.DataFrame(rows); P.to_csv(f"{a.out}/step4b_paired.csv", index=False)
pd.set_option("display.width", 250)
print("\n=== PAIRED differences A - B, pooled over lesion types (95% bootstrap over slices; same resamples for A and B) ===")
print(P.query("types == 'pooled'").round(3).drop(columns=["types"]).to_string(index=False))
PT = P[P["types"] != "pooled"]; ex = PT[PT["excludes_0"]]
print(f"\n=== per-type rows whose interval excludes 0: {len(ex)} of {len(PT)} (about {0.05 * len(PT):.0f} expected by chance) ===")
print(ex.round(3).to_string(index=False))

# ------------------------------------------------------------------ part B
if not a.gallery:
    raise SystemExit
import torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.lodopab import LoDoPaB, make_operator, gt_oriented, one_per_patient, IMG
from src.tv_pdhg import tv_pdhg
from src.denoiser import load_denoiser
from src.pnp_ct import pnp_ct
from src.ct_sim import CTSimulator, poisson_obs
from src.phantoms import shapes_phantom
from src import lesions as L

dev = "cuda" if torch.cuda.is_available() else "cpu"
PARAMS = {1.0: dict(hann=0.3, lam=0.003, const=(0.02, 8), tvinit=(0.02, 4)),           # tuned in step 4 (printed in your console log)
          0.25: dict(hann=0.16, lam=0.01, const=(0.05, 8), tvinit=(0.02, 8))}
for f_ in list(PARAMS):                                                 # tuned parameters saved by run_stage3_step4.py take precedence
    pj = f"{a.out}/step4_params_dose{f_:g}.json"
    if os.path.exists(pj):
        d = json.load(open(pj)); PARAMS[f_] = dict(hann=d["hann"], lam=d["lam"], lam_long=d.get("lam_long", d["lam"]), const=tuple(d["const"]), tvinit=tuple(d["tvinit"]))
        print(f"using tuned parameters from {pj}: {PARAMS[f_]}")
if a.override:
    ov = dict(kv.split("=") for kv in a.override.split(","))
    sk = lambda v: tuple(float(t) if i == 0 else int(t) for i, t in enumerate(v.split("/")))
    for f in PARAMS:
        PARAMS[f] = dict(hann=float(ov["hann"]), lam=float(ov["lam"]), const=sk(ov["const"]), tvinit=sk(ov["tvinit"]))
den = load_denoiser(a.model, dev)
op, ai = make_operator(a.angles, dev); L_ = op.lipschitz(); eta = 1.0 / L_
sim = CTSimulator(op, dev)
dt = LoDoPaB(a.root, "test"); it = one_per_patient(dt, a.n_ct, a.seed + 1)
SETS = {"ct": torch.tensor(gt_oriented(dt, it)),
        "shapes": torch.tensor(np.stack([shapes_phantom(1000 + i) for i in range(a.n_shapes)]))}
MASS = [(2.0, 0.06), (4.0, 0.06)]; MULT = [0.5, 0.75, 1.0, 1.5]
NOM = {t: L.nominal_response(t[0], t[1], IMG) for t in MASS}
rng = np.random.default_rng(0)


def recon_all(y, prm):
    x0 = op.fbp(y, "hann", prm["hann"]).clamp(0, 1)
    tv = tv_pdhg(op, y, prm["lam"], a.tv_iters, x0, L_)
    sd, k = prm["const"]; pc = pnp_ct(op, y, den, sd, eta, (k,), x0)[k]
    sd, k = prm["tvinit"]; pt = pnp_ct(op, y, den, sd, eta, (k,), tv)[k]
    return {"FBP-Hann": x0, "TV": tv, "TV-long": tv_pdhg(op, y, prm.get("lam_long", prm["lam"]), 2 * a.tv_iters, x0, L_), "PnP-const": pc, "PnP-TVinit": pt}


sweep = []
for f in fl(a.doses):
    prm = PARAMS[f]
    for name, X in SETS.items():
        REC = {m: [] for m in METHODS}
        for b in range(0, len(X), 5):                                         # same chunking and seeds as step 4 -> same lesion-free noise
            x = X[b:b + 5].to(dev)
            r = recon_all(poisson_obs(sim.clean(x), f, seed=1000 * b + 1), prm)
            for m in METHODS:
                REC[m].append(r[m].half().cpu())
            print(f"  [{name}, dose {f:g}] {min(b + 5, len(X))}/{len(X)}", flush=True)
        REC = {m: torch.cat(v) for m, v in REC.items()}
        S = len(X)
        gal = {}
        for (s, c) in MASS:
            rho, nom = L.lesion_rho(s), NOM[(s, c)]
            cnt = {m: [] for m in METHODS}
            for m in METHODS:
                per = {mu: [] for mu in MULT}
                for b in range(0, S, 5):
                    Rm, un = C.unsupported_map(REC[m][b:b + 5].float().to(dev), X[b:b + 5].to(dev), rho, nom, size=IMG)
                    for mu in MULT:
                        per[mu].append(((Rm >= mu * nom) & un).sum((1, 2)).cpu().numpy())
                    if s == 4.0 and m in ("TV", "PnP-const"):
                        Rn = torch.where(un, Rm, torch.full_like(Rm, -1e9)).reshape(len(Rm), -1)
                        v, j = Rn.max(1)
                        for i in range(len(Rm)):
                            W = Rm.shape[2]; gal.setdefault(m, []).append((float(v[i]) / nom, b + i, int(j[i]) // W + L.margin(rho), int(j[i]) % W + L.margin(rho)))
                cnt[m] = {mu: np.concatenate(v) for mu, v in per.items()}
            for mu in MULT:
                for m in METHODS:
                    v = cnt[m][mu]
                    lo, hi = C.cluster_boot(lambda i: float(v[i].mean()), S, a.n_boot, 2)
                    sweep.append(dict(set=name, dose=f, sigma_l=s, contrast=c, theta_mult=mu, method=m, false_per_img=v.mean(), lo=lo, hi=hi, n=S, kind="absolute"))
                for mA, mB in [("PnP-const", "TV"), ("PnP-TVinit", "TV-long")]:
                    d = cnt[mA][mu] - cnt[mB][mu]
                    lo, hi = C.cluster_boot(lambda i: float(d[i].mean()), S, a.n_boot, 3)
                    sweep.append(dict(set=name, dose=f, sigma_l=s, contrast=c, theta_mult=mu, method=f"{mA} - {mB}", false_per_img=d.mean(), lo=lo, hi=hi, n=S, kind="paired_diff"))
        # ---- gallery: strongest unsupported blob per image, best n_gal images, for TV and PnP-const (sigma_l = 4 detector)
        fig, axs = plt.subplots(2 * a.n_gal, 5, figsize=(11, 2.3 * 2 * a.n_gal))
        for r_, (who, lst) in enumerate([(w, sorted(gal[w], reverse=True)[:a.n_gal]) for w in ("PnP-const", "TV")]):
            for q, (val, img, cy, cx) in enumerate(lst):
                row = r_ * a.n_gal + q
                y0, x0 = int(np.clip(cy - 32, 0, IMG - 64)), int(np.clip(cx - 32, 0, IMG - 64))
                crop = lambda im: im.reshape(IMG, IMG)[y0:y0 + 64, x0:x0 + 64].float().numpy()
                tr = crop(X[img])
                allv = np.concatenate([tr.ravel()] + [crop(REC[m][img]).ravel() for m in ("FBP-Hann", "TV", "PnP-const", "PnP-TVinit")])
                lo_, hi_ = max(0, np.percentile(allv, 0.5) - 0.02), min(1, np.percentile(allv, 99.5) + 0.02)   # window from truth AND reconstructions
                for col, (lab, im) in enumerate([("truth", X[img])] + [(m, REC[m][img]) for m in ("FBP-Hann", "TV", "PnP-const", "PnP-TVinit")]):
                    ax = axs[row, col]; ax.imshow(crop(im), cmap="gray", vmin=lo_, vmax=hi_); ax.set_xticks([]); ax.set_yticks([])
                    ax.add_patch(plt.Circle((cx - x0, cy - y0), 8, fill=False, color="r", lw=0.8))
                    ax.set_title(lab if row % a.n_gal == 0 and False else (f"{lab}" + (f"  [{who}: {val:.2f}x nominal]" if lab == who else "")), fontsize=7)
                axs[row, 0].set_ylabel(f"worst {who}\nimg {img}", fontsize=7)
        fig.suptitle(f"{name}, dose {f:g}: strongest UNSUPPORTED blob (sigma_l=4 detector, nominal = contrast 0.06 lesion); window = 0.5-99.5% of truth+recon crops", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{a.out}/step4b_gallery_{name}_dose{f:g}.png", dpi=110); plt.close(fig)
        # consistency check against step 4 (m = 0.5 must reproduce its false-lesion counts)
        fn = f"{a.out}/step4_{name}_dose{f:g}.npz"
        if os.path.exists(fn):
            R4 = np.load(fn, allow_pickle=True)
            for (s, c) in MASS:
                k = f"fl|{s:g}|{c:g}|PnP-const"
                if k in R4:
                    mine = [r for r in sweep if r["set"] == name and r["dose"] == f and r["sigma_l"] == s and r["theta_mult"] == 0.5 and r["method"] == "PnP-const"][-1]["false_per_img"]
                    print(f"  consistency [{name}, dose {f:g}, sigma_l {s:g}]: step 4 PnP-const false/img {R4[k].mean():.3f} vs rerun {mine:.3f}")
W = pd.DataFrame(sweep); W.to_csv(f"{a.out}/step4b_theta_sweep.csv", index=False)
print("\n=== false blobs per lesion-free image vs threshold (theta = mult x nominal response of a contrast-0.06 lesion; 95% CI over images) ===")
print(W.query("kind == 'absolute' and method != 'FBP-Hann'").round(3).drop(columns=["kind", "n"]).to_string(index=False))
print("\n=== paired difference (first minus second), same images ===")
print(W.query("kind == 'paired_diff'").round(3).drop(columns=["kind", "n"]).to_string(index=False))
