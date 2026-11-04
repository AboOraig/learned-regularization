"""Stage 3 / step 5: do the conclusions hold over a grid of dose x number of angles?
For every condition (K angles, dose f = fraction of 4096 photons) and independently of every other condition:
  1. tune FBP-Hann, TV, TV-long, PnP-const, PnP-TVinit on --n_val validation slices (one per patient; real LoDoPaB observations, thinned and
     angle-subsampled); grids extend automatically when an optimum lands on an edge;
  2. on the same --n_ct test slices (one per patient, same as steps 2-4) simulate data on a 2x finer grid (no inverse crime, LoDoPaB Poisson model;
     validated against real observations at K=60) and reconstruct with all methods: lesion-free PSNR / SSIM per slice;
  3. with --detect 1: insert M well separated lesions (sigma_l in {2,4} px, contrast 0.06) and store the location-known detection scores and the
     false-blob counts (thresholds 0.5, 1, 1.5 x nominal lesion response) as in steps 4-4c.
Each condition is saved to <out>/step5_K<K>_dose<f>.npz and step5_params_K<K>_dose<f>.json and is skipped if present (resumable).
Statistics (paired differences, bootstrap over patients, grid figure): run_stage3_step5_summary.py
Usage: python run_stage3_step5.py --angles 60 --doses 1,0.25,0.1       (one column of the grid)
       python run_stage3_step5.py                                       (all 9 conditions)"""
import argparse, os, time, platform, json
import numpy as np, torch
from skimage.metrics import structural_similarity
from src.lodopab import LoDoPaB, make_operator, gt_oriented, one_per_patient, IMG
from src.lodopab_noise import thin_obs
from src.tv_pdhg import tv_pdhg
from src.denoiser import load_denoiser
from src.ct_sim import CTSimulator, poisson_obs
from src.tune_ct import tune, recon_all, psnr
from src import lesions as L
from src import lesions_ct as C

p = argparse.ArgumentParser()
p.add_argument("--root", default="data"); p.add_argument("--out", default="results/stage3/s5")
p.add_argument("--model", default="results/stage3/dncnn_ct.pt")
p.add_argument("--doses", default="1,0.25,0.1"); p.add_argument("--angles", default="30,60,120")
p.add_argument("--n_val", type=int, default=20); p.add_argument("--n_ct", type=int, default=40)
p.add_argument("--detect", type=int, default=1); p.add_argument("--M", type=int, default=4)
p.add_argument("--sigmas", default="2,4"); p.add_argument("--contrast", type=float, default=0.06)
p.add_argument("--tv_iters", type=int, default=150); p.add_argument("--auto_extend", type=int, default=2)
p.add_argument("--tv_lams", default="1e-3,3e-3,1e-2,3e-2"); p.add_argument("--hann", default="0.08,0.12,0.16,0.2,0.3,0.4")
p.add_argument("--sd_grid", default="0.01,0.02,0.03,0.05,0.08"); p.add_argument("--ck", default="1,2,4,8,12,20,30,45")
p.add_argument("--fidelity", type=int, default=1); p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(a.out, exist_ok=True)
print(f"torch {torch.__version__} | python {platform.python_version()} | device {dev}", flush=True)
fl = lambda s: [float(v) for v in s.split(",")]
lams, hanns, SD, CK = fl(a.tv_lams), fl(a.hann), fl(a.sd_grid), tuple(int(v) for v in a.ck.split(","))
SIG = fl(a.sigmas); c0 = a.contrast; MULT = (0.5, 1.0, 1.5); MKEY = {0.5: "fl", 1.0: "fl1", 1.5: "fl1.5"}
METHODS = ["FBP-Hann", "TV", "TV-long", "PnP-const", "PnP-TVinit"]
den = load_denoiser(a.model, dev)
dv, dt = LoDoPaB(a.root, "validation"), LoDoPaB(a.root, "test")
iv, it = one_per_patient(dv, a.n_val, a.seed), one_per_patient(dt, a.n_ct, a.seed + 1)
GTv, OBSv = torch.tensor(gt_oriented(dv, iv)).to(dev), torch.tensor(dv.obs(iv))
GTt, OBSt = torch.tensor(gt_oriented(dt, it)), None
patients = np.asarray(dt.patient_ids)[it]
masks = [C.body_mask(g.reshape(IMG, IMG)) for g in GTt.numpy()]
dmin = 10 * max(SIG) + 4
cen, val = C.sample_centers(masks, a.M, dmin, seed=a.seed + 5)
NOM = {s: L.nominal_response(s, c0, IMG) for s in SIG}
ssim = lambda r, t: np.array([structural_similarity(r_.clamp(0, 1).reshape(IMG, IMG).cpu().numpy(), t_.reshape(IMG, IMG).cpu().numpy(), data_range=1.0) for r_, t_ in zip(r, t)])
OPS = {}

for K in [int(v) for v in a.angles.split(",")]:
    for f in fl(a.doses):
        tag = f"K{K}_dose{f:g}"; fn = f"{a.out}/step5_{tag}.npz"
        if os.path.exists(fn):
            print(f"[{tag}] done, skipping", flush=True); continue
        t0 = time.time(); print(f"\n=== condition K={K}, dose={f:g} ===", flush=True)
        if K not in OPS:
            OPS[K] = make_operator(K, dev)
        op, ai = OPS[K]; L_ = op.lipschitz(); eta = 1.0 / L_; sim = CTSimulator(op, dev)
        yv = thin_obs(OBSv, f, seed=a.seed + 11)[:, ai, :].reshape(len(iv), -1).to(dev)
        prm = tune(op, den, yv, GTv, L_, eta, hanns, lams, SD, CK, a.tv_iters, a.auto_extend, log=lambda s: print(s, flush=True))
        print("parameters:", prm, f"(tuning {(time.time() - t0) / 60:.1f} min)", flush=True)
        json.dump({k: (list(v) if isinstance(v, tuple) else v) for k, v in prm.items()}, open(f"{a.out}/step5_params_{tag}.json", "w"))
        R = {}; put = lambda k, v: R.setdefault(k, []).append(v)
        t1 = time.time()
        for b in range(0, len(GTt), 5):
            x = GTt[b:b + 5].to(dev); cn, vl = cen[b:b + 5], val[b:b + 5]
            rec_a = recon_all(op, den, poisson_obs(sim.clean(x), f, seed=1000 * b + 1), prm, L_, eta, a.tv_iters)
            for m, r in rec_a.items():
                put(f"psnr|{m}", psnr(r, x)); put(f"ssim|{m}", ssim(r, x))
            if a.detect:
                for j, s in enumerate(SIG):
                    rho, nom = L.lesion_rho(s), NOM[s]
                    xl = C.insert_multi(x, cn, vl, s, c0, IMG)
                    rec_l = recon_all(op, den, poisson_obs(sim.clean(xl), f, seed=1000 * b + 2 + j), prm, L_, eta, a.tv_iters)
                    tt1, tt0 = C.responses(xl, rho, cn, IMG).cpu().numpy(), C.responses(x, rho, cn, IMG).cpu().numpy()
                    put(f"valid|{s:g}|{c0:g}", vl.numpy() & ((tt1 - tt0) >= 0.5 * nom)); put(f"truth|{s:g}|{c0:g}", tt1 - tt0)
                    for m in METHODS:
                        put(f"t1|{s:g}|{c0:g}|{m}", C.responses(rec_l[m], rho, cn, IMG).cpu().numpy())
                        put(f"t0|{s:g}|{c0:g}|{m}", C.responses(rec_a[m], rho, cn, IMG).cpu().numpy())
                        Rm, un = C.unsupported_map(rec_a[m], x, rho, nom, size=IMG)
                        for mu in MULT:
                            put(f"{MKEY[mu]}|{s:g}|{c0:g}|{m}", ((Rm >= mu * nom) & un).sum((1, 2)).cpu().numpy())
            d = min(b + 5, len(GTt)); el = time.time() - t1
            print(f"  [{tag}] {d}/{len(GTt)} slices, {el / 60:.1f} min, ETA {el / d * (len(GTt) - d) / 60:.1f} min", flush=True)
        R = {k: np.concatenate(v) for k, v in R.items()}; R["patient"] = patients
        np.savez(fn, **R)
        print(f"  [{tag}] test PSNR / SSIM: " + ", ".join(f"{m} {R[f'psnr|{m}'].mean():.2f}/{R[f'ssim|{m}'].mean():.3f}" for m in METHODS), flush=True)
        if a.fidelity:
            n = min(8, len(it)); yr = thin_obs(torch.tensor(dt.obs(it[:n])), f, seed=a.seed + 12)[:, ai, :].reshape(n, -1).to(dev)
            x0 = op.fbp(yr, "hann", prm["hann"]).clamp(0, 1); tvr = tv_pdhg(op, yr, prm["lam"], a.tv_iters, x0, L_)
            g = GTt[:n].to(dev)
            print(f"  SIMULATOR CHECK (first {n} slices): TV real {psnr(tvr, g).mean():.2f} dB vs simulated {R['psnr|TV'][:n].mean():.2f} dB", flush=True)
        print(f"  (condition took {(time.time() - t0) / 60:.1f} min)", flush=True)
print("\nall requested conditions done; now run: python run_stage3_step5_summary.py")
