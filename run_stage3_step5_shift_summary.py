"""Stage 3 / step 5b statistics: method differences on the shape phantoms (shift) next to the same differences on the real CT slices of step 5
(same K, dose, parameters, lesion type), paired within a set (bootstrap over slices / phantoms), and the shift effect = (phantom diff) - (CT diff)
with independent bootstraps of the two sets (different images, so the sets are independent).
Usage: python run_stage3_step5_shift_summary.py [--s5 results/stage3/s5] [--shift results/stage3/s5/shift]
Outputs: <shift>/step5shift_paired.csv (long table), prints compact tables."""
import argparse, glob, re, os
import numpy as np, pandas as pd
from src import lesions_ct as C

p = argparse.ArgumentParser(); p.add_argument("--s5", default="results/stage3/s5"); p.add_argument("--shift", default="results/stage3/s5/shift")
p.add_argument("--n_boot", type=int, default=5000); a = p.parse_args()
PAIRS = [("PnP-const", "TV-long"), ("PnP-const", "TV"), ("PnP-const", "FBP-Hann"), ("PnP-TVinit", "TV-long"), ("TV-long", "FBP-Hann")]
KINDS = [("psnr", "PSNR"), ("ssim", "SSIM"), ("auc", "AUC"), ("fl", "false@0.5x"), ("fl1", "false@1x"), ("fl1.5", "false@1.5x")]
SUBS = {"all": None, "sigma2": 2.0, "sigma4": 4.0}


def stat(R, kind, m, sig, idx):
    if kind in ("psnr", "ssim"):
        return float(R[f"{kind}|{m}"][idx].mean())
    c = sorted({float(k.split("|")[2]) for k in R if k.startswith("valid|")})[0]
    sl = [s for s in sorted({float(k.split("|")[1]) for k in R if k.startswith("valid|")}) if sig is None or s == sig]
    return float(np.nanmean([C._stat(R, s, c, m, kind, idx) for s in sl]))


def boot(R, kind, A, B, sig, n_boot, seed):
    S = len(R["psnr|TV"]); rng = np.random.default_rng(seed)
    d = lambda i: stat(R, kind, A, sig, i) - stat(R, kind, B, sig, i)
    return d(np.arange(S)), np.array([d(rng.integers(0, S, S)) for _ in range(n_boot)])


rows = []
for fn in sorted(glob.glob(f"{a.shift}/step5shift_K*_dose*.npz")):
    K, f = int(re.search(r"K(\d+)", fn).group(1)), float(re.search(r"dose([\d.]+)\.npz", fn).group(1))
    Rs = dict(np.load(fn, allow_pickle=True)); Rc = dict(np.load(f"{a.s5}/step5_K{K}_dose{f:g}.npz", allow_pickle=True))
    for kind, name in KINDS:
        for sub, sig in SUBS.items():
            if sig is not None and kind in ("psnr", "ssim"):
                continue
            for A, B in PAIRS:
                dc, bc = boot(Rc, kind, A, B, sig, a.n_boot, 1); ds, bs = boot(Rs, kind, A, B, sig, a.n_boot, 2)
                q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
                (cl, ch), (sl_, sh), (xl, xh) = q(bc), q(bs), q(bs - bc)
                rows.append(dict(K=K, dose=f, metric=name, subset=sub, A=A, B=B, ct=dc, ct_lo=cl, ct_hi=ch, shape=ds, shape_lo=sl_, shape_hi=sh,
                                 shift_effect=ds - dc, eff_lo=xl, eff_hi=xh))
P = pd.DataFrame(rows)
if P.empty:
    raise SystemExit(f"no step5shift_*.npz in {a.shift}")
star = lambda lo, hi: "*" if (lo > 0 or hi < 0) else " "
P.to_csv(f"{a.shift}/step5shift_paired.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
print("=== phantom (shift) mean PSNR / SSIM ===")
mm = []
for fn in sorted(glob.glob(f"{a.shift}/step5shift_K*_dose*.npz")):
    K, f = int(re.search(r"K(\d+)", fn).group(1)), float(re.search(r"dose([\d.]+)\.npz", fn).group(1)); R = np.load(fn)
    for met in ("psnr", "ssim"):
        for m in ["FBP-Hann", "TV", "TV-long", "PnP-const", "PnP-TVinit"]:
            mm.append(dict(K=K, dose=f, metric=met, method=m, mean=float(R[f"{met}|{m}"].mean())))
print(pd.DataFrame(mm).pivot_table(index=["metric", "method"], columns=["K", "dose"], values="mean").round(3).to_string())
print("\n=== ABSOLUTE false blobs per lesion-free image at 1x nominal lesion response (mean over sigma_l = 2, 4), CT | shapes ===")
ab = []
for fn in sorted(glob.glob(f"{a.shift}/step5shift_K*_dose*.npz")):
    K, f = int(re.search(r"K(\d+)", fn).group(1)), float(re.search(r"dose([\d.]+)\.npz", fn).group(1))
    for setname, R in (("CT", dict(np.load(f"{a.s5}/step5_K{K}_dose{f:g}.npz", allow_pickle=True))), ("shapes", dict(np.load(fn, allow_pickle=True)))):
        for kind in ("fl1", "fl1.5"):
            for m in ["FBP-Hann", "TV", "TV-long", "PnP-const", "PnP-TVinit"]:
                ab.append(dict(K=K, dose=f, set=setname, kind=kind, method=m, mean=stat(R, kind, m, None, np.arange(len(R["psnr|TV"])))))
AB = pd.DataFrame(ab); AB.to_csv(f"{a.shift}/step5shift_false_absolute.csv", index=False)
for kind in ("fl1", "fl1.5"):
    print(f"--- threshold {kind[2:]}x ---")
    print(AB[AB.kind == kind].pivot_table(index=["set", "method"], columns=["K", "dose"], values="mean").round(2).to_string())
for A, B in PAIRS[:3]:
    for name in ("PSNR", "SSIM", "AUC", "false@1x", "false@1.5x"):
        if B == "FBP-Hann" and name in ("PSNR", "SSIM", "AUC"):
            continue
        for sub in ("all", "sigma2", "sigma4"):
            q = P[(P.A == A) & (P.B == B) & (P.metric == name) & (P.subset == sub)]
            if q.empty:
                continue
            print(f"\n--- {name} [{sub}] {A} minus {B}:  CT | shapes | shift effect (shapes - CT);  * = 95% interval excludes 0 ---")
            for _, r in q.sort_values(["K", "dose"], ascending=[True, False]).iterrows():
                print(f"K={int(r.K):3d} dose={r.dose:<4g}  CT {r.ct:+.3f}{star(r.ct_lo, r.ct_hi)} [{r.ct_lo:+.3f},{r.ct_hi:+.3f}]   "
                      f"shapes {r['shape']:+.3f}{star(r.shape_lo, r.shape_hi)} [{r.shape_lo:+.3f},{r.shape_hi:+.3f}]   "
                      f"effect {r.shift_effect:+.3f}{star(r.eff_lo, r.eff_hi)} [{r.eff_lo:+.3f},{r.eff_hi:+.3f}]")
print("\nSaved", f"{a.shift}/step5shift_paired.csv")
