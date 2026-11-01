"""Stage 3 / step 2: noise model of the real observations, and classical baselines on LoDoPaB (patient-level evaluation).

  --extract_train N : extract only the first N hdf5 files of data/ground_truth_train.zip (128 slices each; for the denoiser)
  --exp noise       : test the photon-count model on the real observations (are N0*exp(-mu*y) integers? are counts Poisson?)
  --exp baselines   : FBP (ramp), FBP (Hann, cut-off tuned on validation slices), TV (primal-dual, lambda tuned on validation
                      slices) for sparse-view / lower-dose conditions made FROM THE REAL OBSERVATIONS (angle subsampling and
                      exact binomial thinning of the photon counts); one random slice from each of n different patients, so the
                      evaluation slices are independent; 95% bootstrap CIs over slices.
Usage:  python run_stage3_step2.py --extract_train 70
        python run_stage3_step2.py --exp noise
        python run_stage3_step2.py --exp baselines --angles 60 --doses 1,0.25 --n_val 6 --n_test 40
Outputs in results/stage3: noise_check.json, baselines_slices.csv, baselines_summary.csv, baselines_examples.png"""
import argparse, json, os, time, platform
import numpy as np, pandas as pd, torch
from skimage.metrics import structural_similarity
from src.lodopab import (LoDoPaB, make_operator, gt_oriented, one_per_patient, extract_partial, IMG, N_ANGLES, N_DET)
from src.lodopab_noise import counts_from_obs, integrality, thin_obs, N0, MU_MAX
from src.tv_pdhg import tv_pdhg

p = argparse.ArgumentParser()
p.add_argument("--root", default="data")
p.add_argument("--out", default="results/stage3")
p.add_argument("--extract_train", type=int, default=0)
p.add_argument("--exp", default="", choices=["", "noise", "baselines"])
p.add_argument("--val_split", default="validation")
p.add_argument("--test_split", default="test")
p.add_argument("--angles", default="60", help="comma list of numbers of angles (1000 = native; TV is skipped above 240)")
p.add_argument("--doses", default="1,0.25", help="comma list of photon-count fractions f (1 = native 4096 photons per ray)")
p.add_argument("--n_val", type=int, default=6)
p.add_argument("--n_test", type=int, default=40)
p.add_argument("--tv_iters", type=int, default=150)
p.add_argument("--tv_lams", default="1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2", help="TV weights (normalised units, see src/tv_pdhg.py)")
p.add_argument("--hann", default="0.2,0.3,0.4,0.5,0.7,1.0")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--n_boot", type=int, default=2000)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
print(f"torch {torch.__version__} | python {platform.python_version()} | device {dev}", flush=True)

if a.extract_train:
    d, n = extract_partial(a.root, "ground_truth_train", a.extract_train)
    print(f"extracted {n} training files into {d} ({n * 128} slices)")

# =============================================================================================== noise model
if a.exp == "noise":
    ds = LoDoPaB(a.root, a.test_split)
    idx = one_per_patient(ds, 8, a.seed)
    y = torch.tensor(ds.obs(idx)).to(dev)
    res = {}
    print("\n=== integrality of recovered counts N = N0 exp(-mu y) (random reals would give 0.10) ===")
    print(f"{'N0':>8s} {'mu':>9s} {'frac integer':>13s} {'mean |dev|':>11s}  (bins with N>=3: {integrality(y)[2]})")
    for n0 in (1024.0, 2048.0, 4096.0, 8192.0):
        f, d, _ = integrality(y, N0=n0); print(f"{n0:8.0f} {MU_MAX:9.5f} {f:13.3f} {d:11.4f}"); res[f"integer_frac_N0_{int(n0)}"] = f
    for mu in (80.0, 81.35858, 82.5):
        f, d, _ = integrality(y, mu=mu); print(f"{4096:8d} {mu:9.5f} {f:13.3f} {d:11.4f}")
    N = counts_from_obs(y)
    print(f"\nrecovered counts: min {N.min().item():.0f}, max {N.max().item():.0f}, share of zero counts {(N == 0).double().mean().item():.4f}; "
          f"max y {y.max().item():.4f} (expected for the 0.1 replacement: {np.log(N0 / 0.1) / MU_MAX:.4f})")
    # Poisson dispersion against the expected counts from our forward model
    op, _ = make_operator(1000, dev)
    nb = min(4, len(idx))
    x = torch.tensor(gt_oriented(ds, idx[:nb])).to(dev)
    yp = op.forward(x).reshape(nb, N_ANGLES, N_DET)
    yo = y[:nb]
    sc = float((yp * yo).sum() / (yp * yp).sum()); lam = N0 * torch.exp(-MU_MAX * yp * sc)
    Nn = counts_from_obs(yo)
    for lo, hi in ((20, 200), (200, 1000), (1000, 5000)):
        m = (lam >= lo) & (lam < hi)
        if m.sum() > 1000:
            z = ((Nn[m].double() - lam[m].double()) ** 2 / lam[m].double()).mean().item()
            print(f"expected count in [{lo},{hi}): {int(m.sum()):9d} bins, mean (N-lambda)^2/lambda = {z:.3f} (1.0 for Poisson, > 1 = extra variance incl. model error)")
            res[f"dispersion_{lo}_{hi}"] = z
    json.dump(res, open(f"{a.out}/noise_check.json", "w"), indent=1)
    print("\nIf the 4096 / 81.35858 row is near 1.0 and the others near 0.1, the model is confirmed and exact dose thinning is valid.")

# =============================================================================================== baselines
if a.exp == "baselines":
    dv, dt = LoDoPaB(a.root, a.val_split), LoDoPaB(a.root, a.test_split)
    iv, it = one_per_patient(dv, a.n_val, a.seed), one_per_patient(dt, a.n_test, a.seed + 1)
    print(f"validation: {len(iv)} slices from {len(iv)} patients | test: {len(it)} slices from {len(it)} patients")
    GT = {"val": torch.tensor(gt_oriented(dv, iv)).to(dev), "test": torch.tensor(gt_oriented(dt, it)).to(dev)}
    OBS = {"val": torch.tensor(dv.obs(iv)), "test": torch.tensor(dt.obs(it))}
    psnr = lambda r, t: (-10 * torch.log10(((r.clamp(0, 1) - t) ** 2).mean(1).clamp_min(1e-12))).cpu().numpy()
    ssim = lambda r, t: np.array([structural_similarity(r_.clamp(0, 1).reshape(IMG, IMG).cpu().numpy(), t_.reshape(IMG, IMG).cpu().numpy(), data_range=1.0)
                                  for r_, t_ in zip(r, t)])
    lams = [float(v) for v in a.tv_lams.split(",")]; hanns = [float(v) for v in a.hann.split(",")]
    rows, examples = [], {}
    for f in [float(v) for v in a.doses.split(",")]:
        Y = {k: thin_obs(OBS[k], f, seed=a.seed + 11 + (k == "test")) for k in OBS}        # native or exactly thinned photon counts
        for K in [int(v) for v in a.angles.split(",")]:
            op, ai = make_operator(K, dev)
            y = {k: Y[k][:, ai, :].reshape(len(Y[k]), -1).to(dev) for k in Y}
            tag = f"K={K}, dose={f:g} x 4096"
            print(f"\n--- condition {tag} ---", flush=True)
            # FBP: ramp, and Hann with cut-off tuned on validation slices
            sc = {h: psnr(op.fbp(y["val"], "hann", h), GT["val"]).mean() for h in hanns}
            hb = max(sc, key=sc.get)
            print("Hann cut-off (x Nyquist) -> val PSNR: " + ", ".join(f"{h:g}: {v:.2f}" for h, v in sc.items()) + f"  => {hb:g}"
                  + ("  [WARNING: best value at the edge of the grid]" if hb in (hanns[0], hanns[-1]) else ""))
            rec = {"FBP-ramp": op.fbp(y["test"]), "FBP-Hann": op.fbp(y["test"], "hann", hb)}
            par = {"FBP-ramp": "", "FBP-Hann": hb}
            if K <= 240:
                L = op.lipschitz()
                x0v = op.fbp(y["val"], "hann", hb).clamp(0, 1)
                t = time.time(); tv_pdhg(op, y["val"][:1], lams[0], a.tv_iters, x0v[:1], L); torch.cuda.synchronize() if dev == "cuda" else None
                per = time.time() - t
                print(f"TV: {per:.1f}s per slice per solve -> tuning {len(lams) * len(iv) * per / 60:.1f} min, test {len(it) * per / 60:.1f} min", flush=True)
                ts = {}
                for lam in lams:
                    ts[lam] = psnr(tv_pdhg(op, y["val"], lam, a.tv_iters, x0v, L), GT["val"]).mean()
                lb = max(ts, key=ts.get)
                print("TV lambda -> val PSNR: " + ", ".join(f"{l:g}: {v:.2f}" for l, v in ts.items()) + f"  => {lb:g}"
                      + ("  [WARNING: best value at the edge of the grid]" if lb in (lams[0], lams[-1]) else ""), flush=True)
                x0t = rec["FBP-Hann"].clamp(0, 1)
                rec["TV"] = torch.cat([tv_pdhg(op, y["test"][b:b + 8], lb, a.tv_iters, x0t[b:b + 8], L) for b in range(0, len(it), 8)])
                par["TV"] = lb
            for m, r in rec.items():
                P, S = psnr(r, GT["test"]), ssim(r, GT["test"])
                for i in range(len(it)):
                    rows.append(dict(angles=K, dose=f, method=m, param=par[m], slice=int(it[i]), patient=int(dt.patient_ids[it[i]]), psnr=P[i], ssim=S[i]))
                print(f"  {m:9s}: PSNR {P.mean():.2f}  SSIM {S.mean():.3f}", flush=True)
            examples[tag] = {m: r[0].clamp(0, 1).reshape(IMG, IMG).cpu().numpy() for m, r in rec.items()}
    D = pd.DataFrame(rows); D.to_csv(f"{a.out}/baselines_slices.csv", index=False)
    rng = np.random.default_rng(0)
    def ci(v):
        b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(a.n_boot)]
        return np.percentile(b, 2.5), np.percentile(b, 97.5)
    S = []
    for (K, f, m), g in D.groupby(["angles", "dose", "method"], sort=False):
        lo, hi = ci(g.psnr.values); slo, shi = ci(g.ssim.values)
        S.append(dict(angles=K, dose=f, method=m, n=len(g), psnr=g.psnr.mean(), psnr_lo=lo, psnr_hi=hi, ssim=g.ssim.mean(), ssim_lo=slo, ssim_hi=shi))
    S = pd.DataFrame(S); S.to_csv(f"{a.out}/baselines_summary.csv", index=False)
    print("\n=== summary (mean over test slices, one per patient; 95% bootstrap CI over slices) ===")
    print(S.round(3).to_string(index=False))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    g0 = GT["test"][0].reshape(IMG, IMG).cpu().numpy()
    fig, ax = plt.subplots(len(examples), 4, figsize=(12, 3.1 * len(examples)), squeeze=False)
    for r, (tag, d) in enumerate(examples.items()):
        ims = [("ground truth", g0)] + list(d.items())
        for c in range(4):
            if c < len(ims):
                ax[r, c].imshow(ims[c][1], cmap="gray", vmin=0, vmax=1); ax[r, c].set_title(ims[c][0] if c == 0 else f"{ims[c][0]} ({tag})", fontsize=8)
            ax[r, c].axis("off")
    plt.tight_layout(); plt.savefig(f"{a.out}/baselines_examples.png", dpi=110); plt.close()
    print(f"\nwrote {a.out}/baselines_slices.csv, baselines_summary.csv, baselines_examples.png")
