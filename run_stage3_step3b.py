"""Stage 3 / step 3b: is the in-distribution PnP deficit versus TV real, or a weakness of the PnP set-up?
For each denoiser (--models, comma list; e.g. the depth-8 and a deeper one) and each condition, with EVERYTHING tuned on the
same --n_val validation slices (one per patient) and evaluated on the same test slices as steps 2-3:
   FBP-Hann, TV                              (baselines; Hann cut-off and TV weight tuned)
   PnP-const     constant sigma_d, FBP-Hann initialisation, iteration count tuned          (= step 3)
   PnP-TVinit    constant sigma_d, TV-solution initialisation
   PnP-anneal    sigma_d decreasing geometrically from s0 to s1 over T iterations, FBP-Hann initialisation
   TV+denoise    one denoiser pass on the TV solution (sigma tuned)
Reports PSNR/SSIM per variant and paired differences to TV with bootstrap CIs over patients.
Usage: python run_stage3_step3b.py --models results/stage3/dncnn_ct.pt --doses 1
       python run_stage3_step3b.py --models results/stage3/dncnn_ct.pt,results/stage3/dncnn_ct_d17.pt --doses 1,0.25
Outputs (results/stage3): step3b_slices.csv, step3b_summary.csv, step3b_paired.csv"""
import argparse, itertools, os, time, platform
import numpy as np, pandas as pd, torch
from skimage.metrics import structural_similarity
from src.lodopab import LoDoPaB, make_operator, gt_oriented, one_per_patient, IMG
from src.lodopab_noise import thin_obs
from src.tv_pdhg import tv_pdhg
from src.denoiser import load_denoiser
from src.pnp_ct import pnp_ct, pnp_ct_sched, anneal_schedule, denoise_img

p = argparse.ArgumentParser()
p.add_argument("--root", default="data")
p.add_argument("--out", default="results/stage3")
p.add_argument("--models", default="results/stage3/dncnn_ct.pt")
p.add_argument("--val_split", default="validation")
p.add_argument("--test_split", default="test")
p.add_argument("--angles", default="60")
p.add_argument("--doses", default="1,0.25")
p.add_argument("--n_val", type=int, default=20)
p.add_argument("--n_test", type=int, default=40)
p.add_argument("--tv_iters", type=int, default=150)
p.add_argument("--tv_lams", default="1e-3,3e-3,1e-2,3e-2")
p.add_argument("--hann", default="0.12,0.16,0.2,0.3")
p.add_argument("--sd_grid", default="0.02,0.03,0.05,0.08")
p.add_argument("--ck", default="2,4,8,12,20,30,45,60")
p.add_argument("--anneal_s0", default="0.05,0.1")
p.add_argument("--anneal_s1", default="0.01,0.03")
p.add_argument("--anneal_T", default="20,40")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_boot", type=int, default=5000)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
print(f"torch {torch.__version__} | python {platform.python_version()} | device {dev}", flush=True)
fl = lambda s: [float(v) for v in s.split(",")]
lams, hanns, SD, CK = fl(a.tv_lams), fl(a.hann), fl(a.sd_grid), tuple(int(v) for v in a.ck.split(","))
S0, S1, TT = fl(a.anneal_s0), fl(a.anneal_s1), [int(v) for v in a.anneal_T.split(",")]
dens = {}
for path in a.models.split(","):
    cfg = torch.load(path, map_location="cpu", weights_only=True)
    tag = f"d{cfg['config']['depth']}"
    while tag in dens:
        tag += "b"
    dens[tag] = load_denoiser(path, dev)
    print(f"denoiser {tag}: {path} ({cfg.get('steps', '?')} steps, {cfg.get('n_train_slices', '?')} slices, sigma_max {cfg.get('sigma_max', '?')})")
dv, dt = LoDoPaB(a.root, a.val_split), LoDoPaB(a.root, a.test_split)
iv, it = one_per_patient(dv, a.n_val, a.seed), one_per_patient(dt, a.n_test, a.seed + 1)
GT = {"val": torch.tensor(gt_oriented(dv, iv)).to(dev), "test": torch.tensor(gt_oriented(dt, it)).to(dev)}
OBS = {"val": torch.tensor(dv.obs(iv)), "test": torch.tensor(dt.obs(it))}
psnr = lambda r, t: (-10 * torch.log10(((r.clamp(0, 1) - t) ** 2).mean(1).clamp_min(1e-12))).cpu().numpy()
ssim = lambda r, t: np.array([structural_similarity(r_.clamp(0, 1).reshape(IMG, IMG).cpu().numpy(), t_.reshape(IMG, IMG).cpu().numpy(), data_range=1.0) for r_, t_ in zip(r, t)])
edge = lambda b, g: "  [edge of grid]" if b in (g[0], g[-1]) else ""
cat8 = lambda fn, y, x0: torch.cat([fn(y[b:b + 10], x0[b:b + 10]) for b in range(0, len(y), 10)])
rows = []
for f in fl(a.doses):
    Y = {k: thin_obs(OBS[k], f, seed=a.seed + 11 + (k == "test")) for k in OBS}
    for K in [int(v) for v in a.angles.split(",")]:
        t0 = time.time()
        op, ai = make_operator(K, dev)
        y = {k: Y[k][:, ai, :].reshape(len(Y[k]), -1).to(dev) for k in Y}
        tag = f"K={K}, dose={f:g}"; print(f"\n=== condition {tag} ({a.n_val} validation, {a.n_test} test slices) ===", flush=True)
        L = op.lipschitz(); eta = 1.0 / L
        sc = {h: psnr(op.fbp(y["val"], "hann", h), GT["val"]).mean() for h in hanns}; hb = max(sc, key=sc.get)
        print("Hann cut-off -> val PSNR: " + ", ".join(f"{h:g}: {v:.2f}" for h, v in sc.items()) + f" => {hb:g}" + edge(hb, hanns), flush=True)
        x0v, x0t = op.fbp(y["val"], "hann", hb).clamp(0, 1), op.fbp(y["test"], "hann", hb).clamp(0, 1)
        tvv = {l: cat8(lambda yy, xx: tv_pdhg(op, yy, l, a.tv_iters, xx, L), y["val"], x0v) for l in lams}
        ts = {l: psnr(v, GT["val"]).mean() for l, v in tvv.items()}; lb = max(ts, key=ts.get)
        print("TV lambda -> val PSNR: " + ", ".join(f"{l:g}: {v:.2f}" for l, v in ts.items()) + f" => {lb:g}" + edge(lb, lams), flush=True)
        tv_val = tvv[lb]; tv_test = cat8(lambda yy, xx: tv_pdhg(op, yy, lb, a.tv_iters, xx, L), y["test"], x0t)
        rec, par = {"FBP-Hann": x0t, "TV": tv_test}, {"FBP-Hann": hb, "TV": lb}
        for mt, den in dens.items():
            for name, x0_val, x0_test in (("PnP-const", x0v, x0t), ("PnP-TVinit", tv_val, tv_test)):
                best = (-1.0, None)
                for sd_ in SD:
                    chunks = [pnp_ct(op, y["val"][b:b + 10], den, sd_, eta, CK, x0_val[b:b + 10]) for b in range(0, len(iv), 10)]
                    for kk in CK:
                        s_ = psnr(torch.cat([c[kk] for c in chunks]), GT["val"]).mean()
                        if s_ > best[0]:
                            best = (s_, (sd_, kk))
                sd, k = best[1]
                print(f"[{mt}] {name}: val PSNR {best[0]:.2f} at sigma_d={sd:g}, k={k}" + edge(sd, SD) + edge(k, list(CK)), flush=True)
                rec[f"{name}[{mt}]"] = torch.cat([pnp_ct(op, y["test"][b:b + 10], den, sd, eta, (k,), x0_test[b:b + 10])[k] for b in range(0, len(it), 10)])
                par[f"{name}[{mt}]"] = f"{sd:g}/{k}"
            # --- PnP-anneal
            best = (-1.0, None)
            for s0, s1, T in itertools.product(S0, S1, TT):
                if s1 >= s0:
                    continue
                sg = anneal_schedule(s0, s1, T)
                xv = torch.cat([pnp_ct_sched(op, y["val"][b:b + 10], den, sg, eta, (T,), x0v[b:b + 10])[T] for b in range(0, len(iv), 10)])
                s_ = psnr(xv, GT["val"]).mean()
                if s_ > best[0]:
                    best = (s_, (s0, s1, T))
            s0, s1, T = best[1]
            print(f"[{mt}] PnP-anneal: val PSNR {best[0]:.2f} at s0={s0:g}, s1={s1:g}, T={T}", flush=True)
            sg = anneal_schedule(s0, s1, T)
            rec[f"PnP-anneal[{mt}]"] = torch.cat([pnp_ct_sched(op, y["test"][b:b + 10], den, sg, eta, (T,), x0t[b:b + 10])[T] for b in range(0, len(it), 10)])
            par[f"PnP-anneal[{mt}]"] = f"{s0:g}>{s1:g}/{T}"
            # --- TV + one denoiser pass
            ds = {sd_: psnr(denoise_img(den, tv_val, sd_, IMG), GT["val"]).mean() for sd_ in SD + [0.01]}; db = max(ds, key=ds.get)
            print(f"[{mt}] TV+denoise: val PSNR by sigma: " + ", ".join(f"{k_:g}: {v:.2f}" for k_, v in sorted(ds.items())) + f" => {db:g}", flush=True)
            rec[f"TV+denoise[{mt}]"] = denoise_img(den, tv_test, db, IMG); par[f"TV+denoise[{mt}]"] = db
        for m, r in rec.items():
            P, S_ = psnr(r, GT["test"]), ssim(r, GT["test"])
            for i in range(len(it)):
                rows.append(dict(angles=K, dose=f, method=m, param=par[m], slice=int(it[i]), patient=int(dt.patient_ids[it[i]]), psnr=P[i], ssim=S_[i]))
            print(f"  {m:22s}: PSNR {P.mean():.2f}  SSIM {S_.mean():.3f}", flush=True)
        print(f"  (condition took {(time.time() - t0) / 60:.1f} min)", flush=True)
D = pd.DataFrame(rows); D.to_csv(f"{a.out}/step3b_slices.csv", index=False)
rng = np.random.default_rng(0)
boot = lambda v: np.percentile([v[rng.integers(0, len(v), len(v))].mean() for _ in range(a.n_boot)], [2.5, 97.5])
S, Pd = [], []
for (K, f), g in D.groupby(["angles", "dose"], sort=False):
    w = g.pivot(index="patient", columns="method", values=["psnr", "ssim"])
    for m in g.method.unique():
        lo, hi = boot(w[("psnr", m)].values)
        S.append(dict(angles=K, dose=f, method=m, n=len(w), psnr=w[("psnr", m)].mean(), psnr_lo=lo, psnr_hi=hi, ssim=w[("ssim", m)].mean()))
        if m != "TV":
            d = (w[("psnr", m)] - w[("psnr", "TV")]).values; lo, hi = boot(d)
            Pd.append(dict(angles=K, dose=f, method=m, mean_psnr_minus_TV=d.mean(), lo=lo, hi=hi, frac_better_than_TV=float((d > 0).mean()), n=len(d)))
S, Pd = pd.DataFrame(S), pd.DataFrame(Pd)
S.to_csv(f"{a.out}/step3b_summary.csv", index=False); Pd.to_csv(f"{a.out}/step3b_paired.csv", index=False)
print("\n=== summary ===\n" + S.round(3).to_string(index=False))
print("\n=== paired difference to TV (PSNR, dB; bootstrap over patients) ===\n" + Pd.round(3).to_string(index=False))
