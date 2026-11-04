"""Stage 3 / step 3b: learned prior (PnP-DnCNN) versus tuned classical baselines on LoDoPaB, same slices and data as step 2.
Re-tunes FBP-Hann and TV on WIDER grids (step 2 hit the grid edge for the Hann cut-off and, at quarter dose, for TV), tunes PnP
(sigma_d, iteration count) on the same validation slices, evaluates on the same test slices (one slice per patient) and reports
paired differences with bootstrap CIs over patients.
Usage: python run_stage3_step3.py --model results/stage3/dncnn_ct.pt [--angles 60] [--doses 1,0.25] [--n_val 6] [--n_test 40]
Outputs (results/stage3): step3_slices.csv, step3_summary.csv, step3_paired.csv, step3_examples.png"""
import argparse, os, time, platform
import numpy as np, pandas as pd, torch
from skimage.metrics import structural_similarity
from src.lodopab import LoDoPaB, make_operator, gt_oriented, one_per_patient, IMG
from src.lodopab_noise import thin_obs
from src.tv_pdhg import tv_pdhg
from src.denoiser import load_denoiser
from src.pnp_ct import pnp_ct

p = argparse.ArgumentParser()
p.add_argument("--root", default="data")
p.add_argument("--out", default="results/stage3")
p.add_argument("--model", default="results/stage3/dncnn_ct.pt")
p.add_argument("--val_split", default="validation")
p.add_argument("--test_split", default="test")
p.add_argument("--angles", default="60")
p.add_argument("--doses", default="1,0.25")
p.add_argument("--n_val", type=int, default=6)
p.add_argument("--n_test", type=int, default=40)
p.add_argument("--tv_iters", type=int, default=150)
p.add_argument("--tv_lams", default="3e-4,1e-3,3e-3,1e-2,3e-2,1e-1")
p.add_argument("--hann", default="0.08,0.12,0.16,0.2,0.3,0.4")
p.add_argument("--sd_grid", default="0.005,0.01,0.02,0.03,0.05,0.08")
p.add_argument("--ck", default="1,2,3,5,8,12,20,30,50")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_boot", type=int, default=5000)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
print(f"torch {torch.__version__} | python {platform.python_version()} | device {dev}", flush=True)
den = load_denoiser(a.model, dev)
ck = torch.load(a.model, map_location="cpu", weights_only=True)
print(f"denoiser: {ck.get('n_train_slices', '?')} training slices, {ck.get('n_train_patients', '?')} patients, sigma_max {ck.get('sigma_max', '?')}, steps {ck.get('steps', '?')}")
dv, dt = LoDoPaB(a.root, a.val_split), LoDoPaB(a.root, a.test_split)
iv, it = one_per_patient(dv, a.n_val, a.seed), one_per_patient(dt, a.n_test, a.seed + 1)        # same slices as step 2
GT = {"val": torch.tensor(gt_oriented(dv, iv)).to(dev), "test": torch.tensor(gt_oriented(dt, it)).to(dev)}
OBS = {"val": torch.tensor(dv.obs(iv)), "test": torch.tensor(dt.obs(it))}
psnr = lambda r, t: (-10 * torch.log10(((r.clamp(0, 1) - t) ** 2).mean(1).clamp_min(1e-12))).cpu().numpy()
ssim = lambda r, t: np.array([structural_similarity(r_.clamp(0, 1).reshape(IMG, IMG).cpu().numpy(), t_.reshape(IMG, IMG).cpu().numpy(), data_range=1.0) for r_, t_ in zip(r, t)])
lams = [float(v) for v in a.tv_lams.split(",")]; hanns = [float(v) for v in a.hann.split(",")]
SD = [float(v) for v in a.sd_grid.split(",")]; CK = tuple(int(v) for v in a.ck.split(","))
edge = lambda b, g: "  [WARNING: best value at the edge of the grid]" if b in (g[0], g[-1]) else ""
rows, examples = [], {}
for f in [float(v) for v in a.doses.split(",")]:
    Y = {k: thin_obs(OBS[k], f, seed=a.seed + 11 + (k == "test")) for k in OBS}
    for K in [int(v) for v in a.angles.split(",")]:
        t0 = time.time()
        op, ai = make_operator(K, dev)
        y = {k: Y[k][:, ai, :].reshape(len(Y[k]), -1).to(dev) for k in Y}
        tag = f"K={K}, dose={f:g} x 4096"; print(f"\n--- condition {tag} ---", flush=True)
        L = op.lipschitz(); eta = 1.0 / L
        sc = {h: psnr(op.fbp(y["val"], "hann", h), GT["val"]).mean() for h in hanns}; hb = max(sc, key=sc.get)
        print("Hann cut-off -> val PSNR: " + ", ".join(f"{h:g}: {v:.2f}" for h, v in sc.items()) + f"  => {hb:g}" + edge(hb, hanns), flush=True)
        x0v = op.fbp(y["val"], "hann", hb).clamp(0, 1); x0t = op.fbp(y["test"], "hann", hb).clamp(0, 1)
        ts = {l: psnr(tv_pdhg(op, y["val"], l, a.tv_iters, x0v, L), GT["val"]).mean() for l in lams}; lb = max(ts, key=ts.get)
        print("TV lambda -> val PSNR: " + ", ".join(f"{l:g}: {v:.2f}" for l, v in ts.items()) + f"  => {lb:g}" + edge(lb, lams), flush=True)
        best = (-1.0, None); tab = {}
        for sd in SD:
            outs = pnp_ct(op, y["val"], den, sd, eta, CK, x0v)
            for k, x in outs.items():
                s = psnr(x, GT["val"]).mean(); tab[(sd, k)] = s
                if s > best[0]:
                    best = (s, (sd, k))
        sdb, kb = best[1]
        print("PnP val PSNR by sigma_d (best over iterations): " + ", ".join(f"{sd:g}: {max(v for (s_, k_), v in tab.items() if s_ == sd):.2f}" for sd in SD)
              + f"  => sigma_d={sdb:g}, k={kb}" + edge(sdb, SD) + edge(kb, list(CK)), flush=True)
        rec = {"FBP-Hann": x0t, "TV": torch.cat([tv_pdhg(op, y["test"][b:b + 8], lb, a.tv_iters, x0t[b:b + 8], L) for b in range(0, len(it), 8)])}
        rec["PnP-DnCNN"] = torch.cat([pnp_ct(op, y["test"][b:b + 8], den, sdb, eta, (kb,), x0t[b:b + 8])[kb] for b in range(0, len(it), 8)])
        par = {"FBP-Hann": hb, "TV": lb, "PnP-DnCNN": f"{sdb:g}/{kb}"}
        for m, r in rec.items():
            P, S = psnr(r, GT["test"]), ssim(r, GT["test"])
            for i in range(len(it)):
                rows.append(dict(angles=K, dose=f, method=m, param=par[m], slice=int(it[i]), patient=int(dt.patient_ids[it[i]]), psnr=P[i], ssim=S[i]))
            print(f"  {m:10s}: PSNR {P.mean():.2f}  SSIM {S.mean():.3f}", flush=True)
        examples[tag] = {m: r[0].clamp(0, 1).reshape(IMG, IMG).cpu().numpy() for m, r in rec.items()}
        print(f"  (condition took {(time.time() - t0) / 60:.1f} min)", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{a.out}/step3_slices.csv", index=False)
rng = np.random.default_rng(0)
boot = lambda v: np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(a.n_boot)], [2.5, 97.5])
S, Pd = [], []
for (K, f), g in D.groupby(["angles", "dose"], sort=False):
    for m, gm in g.groupby("method", sort=False):
        lo, hi = boot(gm.psnr.values); slo, shi = boot(gm.ssim.values)
        S.append(dict(angles=K, dose=f, method=m, n=len(gm), psnr=gm.psnr.mean(), psnr_lo=lo, psnr_hi=hi, ssim=gm.ssim.mean(), ssim_lo=slo, ssim_hi=shi))
    w = g.pivot(index="patient", columns="method", values=["psnr", "ssim"])
    for a_, b_ in (("PnP-DnCNN", "TV"), ("PnP-DnCNN", "FBP-Hann"), ("TV", "FBP-Hann")):
        for met in ("psnr", "ssim"):
            d = (w[(met, a_)] - w[(met, b_)]).values; lo, hi = boot(d)
            Pd.append(dict(angles=K, dose=f, comparison=f"{a_} - {b_}", metric=met, mean=d.mean(), lo=lo, hi=hi, frac_better=float((d > 0).mean()), n=len(d)))
S, Pd = pd.DataFrame(S), pd.DataFrame(Pd)
S.to_csv(f"{a.out}/step3_summary.csv", index=False); Pd.to_csv(f"{a.out}/step3_paired.csv", index=False)
print("\n=== summary (mean over test slices, one per patient; 95% bootstrap CI) ===\n" + S.round(3).to_string(index=False))
print("\n=== paired differences (bootstrap over patients) ===\n" + Pd[Pd.metric == "psnr"].round(3).to_string(index=False))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
g0 = GT["test"][0].reshape(IMG, IMG).cpu().numpy()
fig, ax = plt.subplots(len(examples), 4, figsize=(12, 3.1 * len(examples)), squeeze=False)
for r, (tag, d) in enumerate(examples.items()):
    ims = [("ground truth", g0)] + list(d.items())
    for c in range(4):
        ax[r, c].imshow(ims[c][1], cmap="gray", vmin=0, vmax=1); ax[r, c].set_title(ims[c][0] if c == 0 else f"{ims[c][0]} ({tag})", fontsize=8); ax[r, c].axis("off")
plt.tight_layout(); plt.savefig(f"{a.out}/step3_examples.png", dpi=110); plt.close()
print(f"\nwrote {a.out}/step3_slices.csv, step3_summary.csv, step3_paired.csv, step3_examples.png")
